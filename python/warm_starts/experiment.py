"""Run one bounded allocation-model experiment; raw run directories are private by default."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import zstandard

from . import dataset, model, train_offset
from .features import OFFSET_FEATURE_COUNT

RECIPE_SCHEMA = {
    "type": "object",
    "properties": {
        "hidden": {
            "type": "array",
            "items": {"type": "integer", "enum": [16, 32]},
            "minItems": 2,
            "maxItems": 2,
        },
        "epochs": {"type": "integer", "enum": [300, 1000]},
        "learning_rate": {"type": "number", "enum": [0.001, 0.003]},
    },
    "required": ["hidden", "epochs", "learning_rate"],
    "additionalProperties": False,
}
FIXED = {
    "seed": 1,
    "batch": 256,
    "chunks": 1.0,
    "feature_spec": "coarse-offset/2",
    "loss": "smooth absolute allocation error",
    "refinement": "always",
}
TIMEOUTS = {"proposal": 300, "training": 900, "evaluation": 1800}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_recipe(recipe):
    if not isinstance(recipe, dict) or set(recipe) != set(RECIPE_SCHEMA["required"]):
        raise ValueError("recipe must contain only hidden, epochs and learning_rate")
    if recipe["hidden"] not in ([16, 16], [32, 32]) or not all(
        type(width) is int for width in recipe["hidden"]
    ):
        raise ValueError("hidden must be [16, 16] or [32, 32]")
    if type(recipe["epochs"]) is not int or recipe["epochs"] not in (300, 1000):
        raise ValueError("epochs must be 300 or 1000")
    if type(recipe["learning_rate"]) not in (int, float) or recipe["learning_rate"] not in (
        0.001,
        0.003,
    ):
        raise ValueError("learning_rate must be 0.001 or 0.003")
    return recipe


def validate_cases(path):
    rows = list(dataset.read_jsonl_zst(path))
    seen, usable = set(), {"train": 0, "validation": 0}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("split"), str)
            or row["split"] not in usable
        ):
            raise ValueError("experiment inputs must contain only train and validation rows")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("case id must be a nonempty string")
        if row["id"] in seen:
            raise ValueError("duplicate case id")
        seen.add(row["id"])
        if row.get("offset") is None:
            continue
        offset = row["offset"]
        features = offset.get("features") if isinstance(offset, dict) else None
        if (
            not isinstance(features, list)
            or len(features) != OFFSET_FEATURE_COUNT
            or not all(type(v) in (int, float) and math.isfinite(v) for v in features)
        ):
            raise ValueError("invalid coarse-offset/2 features")
        label = row.get("label_share")
        if type(label) not in (int, float) or not 0 <= label <= 1 or not 0 <= features[2] <= 1:
            raise ValueError("allocation shares must be in [0, 1]")
        usable[row["split"]] += 1
    if not all(usable.values()):
        raise ValueError("both train and validation need usable rows")
    return rows, usable


def run_process(command, *, cwd, log, timeout, stdin=None):
    """Terminate the entire stage process group when its wall-clock budget expires."""
    started = time.monotonic()
    with open(log, "w") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(stdin, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            # The leader may exit before a descendant that ignores SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise TimeoutError("stage exceeded its wall-clock budget") from None
    if process.returncode:
        raise RuntimeError(
            f"stage process exited with status {process.returncode}; inspect private log"
        )
    return time.monotonic() - started


def verify_codex_events(path):
    completed, items = False, 0
    for line in Path(path).read_text().splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        item = event.get("item")
        if item:
            items += 1
            if item.get("type") not in ("agent_message", "reasoning", "error"):
                raise ValueError(
                    "Codex used a tool or emitted an unsupported item; proposal rejected"
                )
        if event.get("type") == "turn.completed":
            completed = True
        if event.get("type") in ("turn.failed", "error"):
            raise ValueError("Codex proposal failed")
    if not completed or not items:
        raise ValueError("Codex event stream did not complete")
    return {"completed": True, "tool_calls": 0}


def propose(output, counts):
    # An empty workspace avoids loading project configuration or exposing experiment files.
    with tempfile.TemporaryDirectory(prefix="allocation-proposal-") as temporary:
        workspace = Path(temporary)
        schema = workspace / "schema.json"
        response = workspace / "recipe.json"
        write_json(schema, RECIPE_SCHEMA)
        command = [
            "codex",
            "-a",
            "never",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(response),
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            'web_search="disabled"',
            "-c",
            "mcp_servers={}",
            "-c",
            "plugins={}",
            "-c",
            "features.skip_host_skill_discovery=true",
            "-c",
            "suppress_unstable_features_warning=true",
        ]
        for feature in (
            "shell_tool",
            "unified_exec",
            "apps",
            "hooks",
            "memories",
            "chronicle",
            "skill_search",
            "multi_agent",
            "multi_agent_v2",
            "browser_use",
            "computer_use",
            "image_generation",
            "view_image",
            "sleep_tool",
            "workspace_dependencies",
            "remote_plugin",
            "plugins",
            "code_mode",
            "code_mode_only",
            "code_mode_host",
        ):
            command += ["-c", f"features.{feature}=false"]
        prompt = (
            "Select one supervised training recipe for a small two-pool allocation MLP. "
            "Return only JSON matching the schema. Hidden layers must be [16,16] or [32,32]. "
            "Do not use tools. The model predicts a bounded correction of at most 1/20 to a "
            "coarse allocation using seven fixed features. Training uses smooth absolute "
            "allocation error, seed 1, batch size 256. Refinement always runs during evaluation. "
            "This is one development trial, not a held-out performance claim. "
            f"Usable rows: train={counts['train']}, validation={counts['validation']}. "
            "Prefer a modest configuration suitable for a single CPU trial."
        )
        duration = run_process(
            command + [prompt],
            cwd=workspace,
            log=output / "proposal-events.log",
            timeout=TIMEOUTS["proposal"],
        )
        audit = verify_codex_events(output / "proposal-events.log")
        recipe = validate_recipe(json.loads(response.read_text()))
        version = subprocess.run(
            ["codex", "--version"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        return recipe, {
            "seconds": duration,
            "audit": audit,
            "cli_version": version,
            "model": None,
            "model_identity_note": "Not reported by the CLI event stream.",
        }


def train_worker(cases, recipe_path, output):
    validate_cases(cases)
    recipe = validate_recipe(json.loads(Path(recipe_path).read_text()))
    data = train_offset.load(cases)
    net, inputs, mean, std, predict = train_offset.train(
        data, recipe["hidden"], 1.0, recipe["epochs"], 256, recipe["learning_rate"], 1
    )
    check = data["validation"]["x"][:512]
    artifact, drift = train_offset.export(net, inputs, mean, std, 1.0, predict, check)
    model.save(artifact, output / "candidate.json")
    initial, _, _, _, _ = train_offset.train(
        data, recipe["hidden"], 1.0, 0, 256, recipe["learning_rate"], 1
    )
    changed = any(
        not np.array_equal(a, b) for a, b in zip(net.weights, initial.weights, strict=True)
    )
    if not changed:
        raise ValueError("training did not change model weights")
    write_json(
        output / "training.json",
        {
            "reference_drift": drift,
            "weights_changed": changed,
            "start_error": train_offset.start_errors(data, predict, ["train", "validation"]),
            "start_regret": train_offset.start_regrets(data, predict, ["train", "validation"]),
        },
    )


def replay_jobs(path, cases):
    manifest = json.loads(path.read_text())
    if set(manifest) != {"split", "jobs"} or manifest["split"] != "validation":
        raise ValueError("replay manifest must explicitly select validation")
    eligible = {
        (r["block"], r["direction"], r["amount_in"])
        for r in cases
        if r["split"] == "validation" and "block" in r
    }
    jobs = manifest["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("replay requires at least one job")
    for job in jobs:
        if set(job) != {"recording", "blocks", "cases", "gas_price_wei"}:
            raise ValueError("invalid replay job fields")
        if not job["blocks"] or not job["cases"] or len(set(job["blocks"])) != len(job["blocks"]):
            raise ValueError("replay blocks and cases must be nonempty and unique")
        if len({c["id"] for c in job["cases"]}) != len(job["cases"]):
            raise ValueError("duplicate replay case id")
        if type(job["gas_price_wei"]) is not int or job["gas_price_wei"] < 0:
            raise ValueError("gas price must be a nonnegative integer")
        for block in job["blocks"]:
            for case in job["cases"]:
                if (block, case["direction"], case["amount_in"]) not in eligible:
                    raise ValueError("replay includes a case outside prepared validation data")
        job["recording"] = str((path.parent / job["recording"]).resolve())
    return jobs


def read_evaluation(path, expected, methods):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["block"], row["case_id"], row["repeat"], row["method"])
        if key in rows:
            raise ValueError("duplicate evaluation row")
        rows[key] = row
    if set(rows) != {(*key, method) for key in expected for method in methods}:
        raise ValueError("incomplete or unexpected evaluation rows")
    for (*key, _), row in rows.items():
        for name, value in expected[tuple(key)].items():
            if row.get(name) != value:
                raise ValueError(f"evaluation {name} does not match the replay manifest")
    return rows


def compare(rows, references):
    losses, ratios = [], []
    failures = 0
    for key, row in rows.items():
        reference = references[key]
        if row["status"] != "ok" or reference["status"] != "ok":
            raise ValueError("evaluation contains unsuccessful solves")
        net, reference_net = int(row["net_out"]), int(reference["net_out"])
        if reference_net <= 0 or reference["elapsed_us"] <= 0 or row["elapsed_us"] <= 0:
            raise ValueError("invalid evaluation output or timing")
        failures += 100000 * (reference_net - net) > reference_net
        losses.append((reference_net - net) / reference_net * 10000)
        ratios.append(row["elapsed_us"] / reference["elapsed_us"])
    return {
        "paired_solves": len(losses),
        "quality_failures_over_0_1bp": failures,
        "worst_loss_bp": max(losses),
        "median_time_ratio": float(np.median(ratios)),
        "p90_time_ratio": float(np.percentile(ratios, 90)),
    }


def evaluate(jobs, binary, frozen, output):
    started = time.monotonic()
    candidate_rows, frozen_rows, reference_rows = {}, {}, {}
    for index, job in enumerate(jobs):
        cases_path = output / f"replay-{index}-cases.json"
        write_json(cases_path, {"cases": job["cases"]})
        expected = {
            (block, case["id"], repeat): {
                "direction": case["direction"],
                "amount_in": case["amount_in"],
                "gas_price_wei": str(job["gas_price_wei"]),
                "build": "patched",
            }
            for block in job["blocks"]
            for case in job["cases"]
            for repeat in range(3)
        }
        base = [
            str(binary),
            "--recording",
            job["recording"],
            "--blocks",
            ",".join(map(str, job["blocks"])),
            "--cases",
            str(cases_path),
            "--gas-price-wei",
            str(job["gas_price_wei"]),
            "--repeats",
            "3",
        ]
        for name, artifact, methods in (
            ("candidate", output / "candidate.json", ["water_fill", "offset"]),
            ("frozen", frozen, ["offset"]),
        ):
            destination = output / f"replay-{index}-{name}.jsonl"
            remaining = TIMEOUTS["evaluation"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("evaluation exceeded its wall-clock budget")
            run_process(
                base
                + [
                    "--methods",
                    ",".join(methods),
                    "--offset-model",
                    str(artifact),
                    "--output",
                    str(destination),
                ],
                cwd=output,
                log=output / f"replay-{index}-{name}.log",
                timeout=remaining,
            )
            rows = read_evaluation(destination, expected, methods)
            for (*key, method), row in rows.items():
                target = (
                    reference_rows
                    if method == "water_fill"
                    else (candidate_rows if name == "candidate" else frozen_rows)
                )
                target[(index, *key)] = row
    return {
        "seconds": time.monotonic() - started,
        "candidate_vs_fynd": compare(candidate_rows, reference_rows),
        "frozen_offset_vs_fynd": compare(frozen_rows, reference_rows),
        "candidate_vs_frozen_offset": compare(candidate_rows, frozen_rows),
        "timing_note": "Candidate and frozen evaluated in separate passes; indicative local timing.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private run directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", type=Path)
    source.add_argument("--proposal", choices=["codex"])
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--bench-binary", type=Path)
    parser.add_argument("--frozen-model", type=Path)
    parser.add_argument("--_train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args._train:
        train_worker(args.cases, args.recipe, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    os.chmod(args.output, 0o700)
    result = {
        "format": "allocation-trial/1",
        "status": "running",
        "fixed": FIXED,
        "source": args.proposal or "recipe",
        "stages": {},
    }
    frozen_hash = None
    stage = "validation"
    stage_started = time.monotonic()
    try:
        frozen_hash = digest(args.frozen_model) if args.frozen_model else None
        rows, counts = validate_cases(args.cases)
        jobs = replay_jobs(args.replay.resolve(), rows) if args.replay else None
        if jobs and (not args.bench_binary or not args.frozen_model):
            raise ValueError("replay requires --bench-binary and --frozen-model")
        result["cases_sha256"] = digest(args.cases)
        result["usable_rows"] = counts
        stage = "proposal"
        stage_started = time.monotonic()
        if args.proposal:
            recipe, result["stages"][stage] = propose(args.output, counts)
        else:
            recipe = validate_recipe(json.loads(args.recipe.read_text()))
        result["recipe"] = recipe
        write_json(args.output / "recipe.json", recipe)
        stage = "training"
        stage_started = time.monotonic()
        seconds = run_process(
            [
                sys.executable,
                "-m",
                "warm_starts.experiment",
                "--_train",
                "--cases",
                str(args.cases.resolve()),
                "--recipe",
                str(args.output / "recipe.json"),
                "--output",
                str(args.output),
            ],
            cwd=Path.cwd(),
            log=args.output / "training.log",
            timeout=TIMEOUTS[stage],
        )
        result["stages"][stage] = {"seconds": seconds}
        result["candidate_sha256"] = digest(args.output / "candidate.json")
        result["training"] = json.loads((args.output / "training.json").read_text())
        if jobs:
            stage = "evaluation"
            stage_started = time.monotonic()
            result["stages"][stage] = evaluate(
                jobs, args.bench_binary.resolve(), args.frozen_model.resolve(), args.output
            )
        result["status"] = "completed" if jobs else "training_smoke_completed"
    except (
        ValueError,
        OSError,
        RuntimeError,
        TimeoutError,
        KeyError,
        TypeError,
        zstandard.ZstdError,
    ) as error:
        result.update(status="failed", failed_stage=stage, error=str(error))
        result["stages"][stage] = {"seconds": time.monotonic() - stage_started, "status": "failed"}
    finally:
        if frozen_hash is not None:
            result["frozen_sha256"] = frozen_hash
            try:
                result["frozen_unchanged"] = digest(args.frozen_model) == frozen_hash
            except OSError:
                result["frozen_unchanged"] = False
            if not result["frozen_unchanged"]:
                result.update(status="failed", error="frozen model changed during trial")
        write_json(args.output / "result.json", result)
    print(json.dumps({"status": result["status"], "result": str(args.output / "result.json")}))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
