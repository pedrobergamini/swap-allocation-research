"""Evaluate frozen models on held-out recordings through the Rust runner.

The skip decision and final net output come from sar-bench, including integer allocation,
actual certificate quotes and Fynd's final candidate selection. No interpolated regret curve
participates in the gate. Results and raw rows go in a new directory outside the frozen models.

Usage: python -m warm_starts.score_test --cases CASES --model-dir work/models/v1
    --recordings 'work/recordings/live/*.json.zst' --output-dir OUTPUT
"""

import argparse
import glob
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from .dataset import read_jsonl_zst
from .recorded import capture_info, capture_number

METHODS = ("water_fill", "offset_certified")
FROZEN_FILES = {"offset.json", "confidence.json", "report.json", "confidence-report.json"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(directory):
    expected = {}
    manifest = directory / "manifest.json"
    if manifest.exists():
        release = json.loads(manifest.read_text())
        if release.get("format") != "sar-release-models/1":
            raise ValueError("unsupported public model manifest")
        for artifact in release["files"]:
            name = artifact["path"]
            if name not in {"offset.json", "confidence.json"} or name in expected:
                raise ValueError("unexpected or duplicate public model artifact")
            expected[name] = artifact["sha256"]
        if set(expected) != {"offset.json", "confidence.json"}:
            raise ValueError("public manifest must identify both model artifacts")
        if any(sha256(directory / name) != digest for name, digest in expected.items()):
            raise ValueError("frozen artifact hash mismatch")
        return expected
    for line in (directory / "FROZEN.txt").read_text().splitlines()[1:]:
        digest, path = line.split(maxsplit=1)
        name = Path(path.strip()).name
        if name in expected:
            raise ValueError(f"duplicate frozen artifact: {name}")
        expected[name] = digest
    if set(expected) != FROZEN_FILES:
        raise ValueError("FROZEN.txt must identify all four selection and model artifacts")
    if any(sha256(directory / name) != digest for name, digest in expected.items()):
        raise ValueError("frozen artifact hash mismatch")
    return expected


def case_key(case):
    return case["block"], f"{case['direction']}-{case['amount_in']}"


def summarize(cases, rows):
    """Pair every held-out case, rejecting incomplete or ambiguous runner output."""
    expected = {}
    for case in cases:
        key = case_key(case)
        if key in expected:
            raise ValueError(f"duplicate test case: {key}")
        expected[key] = case
    if not expected:
        raise ValueError("no held-out test cases")

    paired = {method: {} for method in METHODS}
    for row in rows:
        method = row["method"]
        key = row["block"], row["case_id"]
        if method not in paired or key not in expected:
            raise ValueError(f"unexpected solve: {method} {key}")
        if key in paired[method]:
            raise ValueError(f"duplicate solve: {method} {key}")
        case = expected[key]
        if (
            row["repeat"] != 0
            or row["build"] != "patched"
            or row["direction"] != case["direction"]
            or row["amount_in"] != case["amount_in"]
        ):
            raise ValueError(f"solve does not match test case: {method} {key}")
        paired[method][key] = row
    if any(set(method_rows) != set(expected) for method_rows in paired.values()):
        raise ValueError("missing held-out solves")

    reference_failures = failed = quality_failures = seeded = skipped = 0
    deltas, skipped_deltas = [], []
    for key in expected:
        reference, candidate = (paired[method][key] for method in METHODS)
        if reference["gas_price_wei"] != candidate["gas_price_wei"]:
            raise ValueError(f"paired gas prices differ: {key}")
        seeds = candidate["seeds"]
        seed = seeds[0] if seeds else None
        has_proposal = seed is not None and seed["proposal"] is not None
        did_skip = has_proposal and seed["skipped"]
        seeded += has_proposal
        skipped += did_skip
        if reference["status"] != "ok":
            reference_failures += 1
            continue
        if candidate["status"] != "ok":
            failed += 1
            continue
        reference_net, net = int(reference["net_out"]), int(candidate["net_out"])
        if reference_net <= 0 or net < 0:
            raise ValueError(f"invalid net output: {key}")
        # 0.1 bp is one part in 100,000; compare integers at the gate boundary.
        quality_failures += 100_000 * (reference_net - net) > reference_net
        delta = (net - reference_net) / reference_net * 10_000
        deltas.append(delta)
        if did_skip:
            skipped_deltas.append(delta)
    return {
        "solves": len(expected),
        "successful_pairs": len(deltas),
        "reference_failures": reference_failures,
        "failed_where_fynd_solved": failed,
        "gate_failures": failed + quality_failures,
        "gate_passed": reference_failures == 0 and failed + quality_failures == 0,
        "seeded_solves": seeded,
        "skipped_solves": skipped,
        "skip_rate": skipped / seeded if seeded else None,
        "worst_final_delta_bp": min(deltas) if deltas else None,
        "worst_skipped_final_delta_bp": min(skipped_deltas) if skipped_deltas else None,
    }


def evaluate(cases_path, model_dir, recordings, output_dir, sar_bench):
    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(model_dir):
        raise ValueError("evaluation output must be outside the frozen model directory")
    artifacts = verify_frozen(model_dir)
    if (model_dir / "manifest.json").exists():
        release = json.loads((model_dir / "manifest.json").read_text())
        divisor = release["certificate_step_divisor"]
    else:
        report = json.loads((model_dir / "confidence-report.json").read_text())
        divisor = report["certificate"]["chosen_step_divisor"]
    if not isinstance(divisor, int) or divisor <= 0:
        raise ValueError("the frozen selection has no certified step to evaluate")
    cases = [case for case in read_jsonl_zst(cases_path) if case["split"] == "test"]
    if not cases:
        raise ValueError("no held-out test cases")
    captures = {}
    for recording in recordings:
        capture = capture_number(str(recording))
        if capture in captures:
            raise ValueError(f"duplicate recording for capture {capture}")
        captures[capture] = str(Path(recording).resolve())
    groups = defaultdict(list)
    for case in cases:
        groups[case["capture"], case["block"]].append(case)
    required = {capture for capture, _ in groups}
    if not required <= captures.keys():
        raise ValueError(f"missing test recordings: {sorted(required - captures.keys())}")
    capture_metadata = {}
    for capture in sorted(required):
        recording = captures[capture]
        first, last, gas = capture_info(recording)
        capture_metadata[capture] = {
            "path": recording,
            "sha256": sha256(recording),
            "gas_price_wei": str(gas),
        }
        if any(not first <= block <= last for cap, block in groups if cap == capture):
            raise ValueError(f"test block outside capture {capture}")
    # A new directory prevents an evaluation from overwriting earlier evidence.
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, row_files = [], []
    for (capture, block), group in sorted(groups.items()):
        stem = f"capture-{capture:02d}-block-{block}"
        cases_file, rows_file = output_dir / f"{stem}-cases.json", output_dir / f"{stem}.jsonl"
        inputs = [
            {
                "id": case_key(case)[1],
                "direction": case["direction"],
                "amount_in": case["amount_in"],
            }
            for case in group
        ]
        cases_file.write_text(json.dumps({"cases": inputs}) + "\n")
        command = [
            str(sar_bench),
            "--recording",
            captures[capture],
            "--blocks",
            str(block),
            "--cases",
            str(cases_file),
            "--methods",
            ",".join(METHODS),
            "--gas-price-wei",
            capture_metadata[capture]["gas_price_wei"],
            "--offset-model",
            str(model_dir / "offset.json"),
            "--confidence-model",
            str(model_dir / "confidence.json"),
            "--certificate-step-divisor",
            str(divisor),
            "--repeats",
            "1",
            "--output",
            str(rows_file),
        ]
        with (output_dir / f"{stem}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.extend(json.loads(line) for line in rows_file.read_text().splitlines())
        row_files.append({"path": rows_file.name, "sha256": sha256(rows_file)})
    verify_frozen(model_dir)
    result = {
        "format": "sar-runtime-test/1",
        "cases": str(Path(cases_path).resolve()),
        "cases_sha256": sha256(cases_path),
        "artifacts_sha256": artifacts,
        "runner_sha256": sha256(sar_bench),
        "recordings": capture_metadata,
        "rows": row_files,
        "certificate_step_divisor": divisor,
        "reference": "Patched Fynd without a seed; complete final net output per solve",
        "runtime": summarize(cases, rows),
    }
    (output_dir / "test-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--recordings", required=True, help="glob of numbered market recordings")
    parser.add_argument("--output-dir", required=True, help="new directory outside frozen models")
    parser.add_argument("--sar-bench", default="target/release/sar-bench")
    args = parser.parse_args()
    result = evaluate(
        args.cases, args.model_dir, glob.glob(args.recordings), args.output_dir, args.sar_bench
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["runtime"]["gate_passed"] else 1)


if __name__ == "__main__":
    main()
