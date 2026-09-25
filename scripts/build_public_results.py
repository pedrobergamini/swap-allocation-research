"""Build source-neutral launch summaries from complete, private evaluation inputs.

Pass bench directories and a runtime report explicitly. Only aggregates and selected
order numbers are exported; recordings, feature vectors and input paths are not.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

METHOD_LABELS = {
    "water_fill": "Fynd reference",
    "coarse": "Coarse split, always refine",
    "offset": "Neural initializer, always refine",
    "offset_certified": "Learned warm start with checked skip",
    "quadratic": "Quadratic formula",
    "offset_confident": "Prediction-only skip (disabled)",
}
GATE_BP = 0.1
GAS_PRICE = "6000000"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    return row["block"], row["case_id"], row["repeat"]


def seed(row):
    return row["seeds"][0] if row["seeds"] else {}


def eligible(row):
    return seed(row).get("proposal") is not None


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[math.ceil(position)] * fraction


def index_rows(rows, methods, expected):
    indexed = {method: {} for method in methods}
    for row in rows:
        method = row["method"]
        if method not in indexed:
            raise ValueError(f"Unexpected method: {method}")
        if row["gas_price_wei"] != GAS_PRICE:
            raise ValueError("Unexpected gas price")
        row_key = key(row)
        if row_key in indexed[method]:
            raise ValueError("Duplicate benchmark row")
        if row["status"] == "ok":
            if int(row["net_out"]) <= 0 or row["elapsed_us"] <= 0:
                raise ValueError("Successful row has invalid output or timing")
            if sum(map(int, row["pool_amounts"])) != int(row["amount_in"]):
                raise ValueError("Final allocation does not conserve input")
        indexed[method][row_key] = row
    reference_keys = indexed["water_fill"].keys()
    if len(reference_keys) != expected:
        raise ValueError(f"Expected {expected} reference rows, got {len(reference_keys)}")
    for method, method_rows in indexed.items():
        if method_rows.keys() != reference_keys:
            raise ValueError(f"Incomplete paired evaluation for {method}")
        for row_key, row in method_rows.items():
            reference = indexed["water_fill"][row_key]
            if (row["direction"], row["amount_in"]) != (
                reference["direction"],
                reference["amount_in"],
            ):
                raise ValueError("Paired rows identify different orders")
    return indexed


def delta(row, reference):
    return (int(row["net_out"]) - int(reference["net_out"])) / int(reference["net_out"]) * 1e4


def population_summary(method_rows, references, keys):
    pairs = [(method_rows[k], references[k]) for k in sorted(keys)]
    successful = [(row, ref) for row, ref in pairs if row["status"] == ref["status"] == "ok"]
    losses = [delta(row, ref) for row, ref in successful]
    ratios = [row["elapsed_us"] / ref["elapsed_us"] for row, ref in successful]
    failed = sum(row["status"] != "ok" and ref["status"] == "ok" for row, ref in pairs)
    reference_failures = sum(ref["status"] != "ok" for _, ref in pairs)
    # Compare integer output units so rounding cannot hide a loss just above 0.1 bp.
    quality_failures = (
        sum(
            100_000 * (int(ref["net_out"]) - int(row["net_out"])) > int(ref["net_out"])
            for row, ref in successful
        )
        + failed
    )
    return {
        "solves": len(pairs),
        "successful_pairs": len(successful),
        "candidate_failures": sum(row["status"] != "ok" for row, _ in pairs),
        "failed_where_fynd_solved": failed,
        "reference_failures": reference_failures,
        "gate_failures": quality_failures,
        "gate_passed": quality_failures == reference_failures == 0,
        "time_ratio_median": statistics.median(ratios) if ratios else None,
        "time_ratio_p90": percentile(ratios, 90) if ratios else None,
        "time_ratio_sum": (
            sum(row["elapsed_us"] for row, _ in successful)
            / sum(ref["elapsed_us"] for _, ref in successful)
            if successful
            else None
        ),
        "delta_bp": {
            "worst": min(losses) if losses else None,
            "median": statistics.median(losses) if losses else None,
            "mean": statistics.mean(losses) if losses else None,
        },
        "better_equal_worse": [
            sum(value > 0 for value in losses),
            sum(value == 0 for value in losses),
            sum(value < 0 for value in losses),
        ],
    }


def simulation_summary(log_path, rows):
    marker = re.compile(r'solve block=(\d+) case=(\S+) method="(\w+)" repeat=Some\((\d+)\)')
    stage = re.compile(r"(\w[\w-]*): (\d+) calls")
    active = {key(row) for row in rows if row["method"] == "offset" and eligible(row)}
    expected = {(row["method"], key(row)) for row in rows if key(row) in active}
    counts = {}
    pending = None
    for line in log_path.read_text().splitlines():
        match = marker.search(line)
        if match:
            block, case, method, repeat = match.groups()
            pending = method, (int(block), case, int(repeat))
        elif "simulation by stage" in line and pending is not None:
            if pending in expected:
                if pending in counts:
                    raise ValueError("Duplicate simulation counts")
                counts[pending] = sum(int(value) for _, value in stage.findall(line))
            pending = None
    if counts.keys() != expected:
        raise ValueError("Incomplete simulation counts")
    result = {}
    for method in METHOD_LABELS:
        values = [counts[method, row_key] for row_key in sorted(active)]
        result[method] = {
            "solves": len(values),
            "total_mean": statistics.mean(values),
            "total_median": statistics.median(values),
        }
    result["paired_offset_minus_coarse"] = {
        "solves": len(active),
        "mean": statistics.mean(counts["offset", k] - counts["coarse", k] for k in active),
        "median": statistics.median(counts["offset", k] - counts["coarse", k] for k in active),
    }
    return result


def bench_summary(directory, name, expected):
    rows_path = directory / f"{name}.jsonl"
    indexed = index_rows(read_rows(rows_path), METHOD_LABELS, expected)
    refs = indexed["water_fill"]
    active = {k for k, row in indexed["offset"].items() if eligible(row)}
    metrics = read_rows(directory / f"{name}-metrics.jsonl")
    index_rows(metrics, METHOD_LABELS, expected // 3)
    simulations = simulation_summary(directory / f"{name}-metrics.log", metrics)
    clean = index_rows(
        read_rows(directory / f"{name}-clean.jsonl"), ("water_fill",), expected // 3
    )["water_fill"]
    identical = 0
    for row_key, clean_row in clean.items():
        if row_key not in refs or clean_row["status"] != "ok":
            raise ValueError("Unpaired or unsuccessful unmodified Fynd result")
        if any(
            clean_row[field] != refs[row_key][field]
            for field in (
                "amount_in",
                "direction",
                "amount_out",
                "net_out",
                "gas",
                "pool_amounts",
                "status",
            )
        ):
            raise ValueError("Unmodified and patched seedless Fynd outputs differ")
        identical += 1
    result = {
        "solves": expected,
        "unique_orders": len({k[:2] for k in refs}),
        "repeats": sorted({k[2] for k in refs}),
        "source_rows_sha256": digest(rows_path),
        "clean_reference_equivalence": {"cases": identical, "identical": identical},
        "paired_simulations_offset_minus_coarse": simulations["paired_offset_minus_coarse"],
        "methods": {},
    }
    for method, method_rows in indexed.items():
        if method != "water_fill":
            observed = {k for k, row in method_rows.items() if eligible(row)}
            if observed != active:
                raise ValueError("Methods have different eligible populations")
        skipped = {k for k, row in method_rows.items() if seed(row).get("skipped")}
        successful_skips = [
            delta(method_rows[k], refs[k])
            for k in skipped
            if method_rows[k]["status"] == refs[k]["status"] == "ok"
        ]
        result["methods"][method] = {
            "label": METHOD_LABELS[method],
            "all": population_summary(method_rows, refs, refs.keys()),
            "two_pool": population_summary(method_rows, refs, active),
            "skipped": {
                "solves": len(skipped),
                "rate": len(skipped) / len(active),
                "worst_final_delta_bp": min(successful_skips) if successful_skips else None,
            },
            "simulations": simulations[method],
        }
    return result, indexed


def runtime_summary(report_path):
    report = json.loads(report_path.read_text())
    rows = []
    for entry in report["rows"]:
        row_path = report_path.parent / entry["path"]
        if digest(row_path) != entry["sha256"]:
            raise ValueError("Runtime source checksum mismatch")
        rows.extend(read_rows(row_path))
    indexed = index_rows(rows, ("water_fill", "offset_certified"), 6336)
    refs, candidates = indexed["water_fill"], indexed["offset_certified"]
    summary = population_summary(candidates, refs, refs.keys())
    active = {k for k, row in candidates.items() if eligible(row)}
    skipped = {k for k, row in candidates.items() if seed(row).get("skipped")}
    skipped_deltas = [
        delta(candidates[k], refs[k])
        for k in skipped
        if candidates[k]["status"] == refs[k]["status"] == "ok"
    ]
    result = {
        field: summary[field]
        for field in (
            "solves",
            "successful_pairs",
            "reference_failures",
            "failed_where_fynd_solved",
            "gate_failures",
            "gate_passed",
        )
    }
    result.update(
        {
            "seeded_solves": len(active),
            "skipped_solves": len(skipped),
            "skip_rate": len(skipped) / len(active),
            "worst_final_delta_bp": summary["delta_bp"]["worst"],
            "worst_skipped_final_delta_bp": min(skipped_deltas) if skipped_deltas else None,
        }
    )
    if result != report["runtime"]:
        raise ValueError("Recomputed runtime results disagree with saved report")
    return result


def dataset_summary(path):
    import zstandard

    splits = defaultdict(Counter)
    groups = defaultdict(set)
    with zstandard.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            splits[row["split"]]["cases"] += 1
            splits[row["split"]]["eligible" if row["offset"] is not None else "declined"] += 1
            groups[row["split"]].add(row["capture"])
    expected = {"train": 25344, "validation": 6336, "test": 6336}
    if {name: values["cases"] for name, values in splits.items()} != expected:
        raise ValueError("Dataset split counts differ from frozen experiment")
    if (
        not max(groups["train"])
        < min(groups["validation"])
        <= max(groups["validation"])
        < min(groups["test"])
    ):
        raise ValueError("Dataset groups are not chronological and disjoint")
    return {
        "scope": "Historical-state replay for two WETH/USDC Uniswap v3 pools on Base",
        "splitting": "Chronological groups: 8 training, 2 validation, 2 held-out test",
        "splits": {name: dict(splits[name]) for name in expected},
        "order_sizes": "Synthetic log-uniform size distribution, not observed order flow",
        "gas_price_wei": GAS_PRICE,
        "original_corpus_distributed": False,
        "acquisition": "Prepared training cases and compatible replay inputs from the user's data stack",
    }


def selected_examples(indexed):
    candidates = indexed["offset_certified"]
    ordered = sorted(
        candidates,
        key=lambda k: (k[0], candidates[k]["direction"], int(candidates[k]["amount_in"]), k[2]),
    )
    predicates = [
        (
            "checked-skip",
            "A two-pool split that skips refinement",
            lambda row: seed(row).get("skipped"),
        ),
        ("one-pool", "One pool is enough", lambda row: not eligible(row)),
        (
            "refinement",
            "The check retains refinement",
            lambda row: (
                eligible(row) and not seed(row).get("skipped") and seed(row).get("lookups", 0) >= 8
            ),
        ),
    ]
    examples = []
    for identifier, title, predicate in predicates:
        chosen = next(k for k in ordered if k[2] == 0 and predicate(candidates[k]))
        row = candidates[chosen]
        direction = row["direction"]
        input_symbol, output_symbol = direction.upper().split("-")
        example = {
            "id": identifier,
            "title": title,
            "description": {
                "checked-skip": "Six neighboring quotes pass the gross-output check; the disjoint candidate skips refinement.",
                "one-pool": "The learned initializer declines because only one pool is active; Fynd continues unchanged.",
                "refinement": "The neighboring-quote check does not authorize skipping; exchange refinement still runs.",
            }[identifier],
            "benchmark": "fresh",
            "repeat": 0,
            "direction": direction,
            "input_symbol": input_symbol,
            "output_symbol": output_symbol,
            "input_decimals": 18 if input_symbol == "WETH" else 6,
            "output_decimals": 18 if output_symbol == "WETH" else 6,
            "amount_in": row["amount_in"],
            "gas_price_wei": GAS_PRICE,
            "methods": {},
        }
        for method in ("water_fill", "coarse", "offset", "offset_certified"):
            value = indexed[method][chosen]
            first_seed = seed(value)
            if value["status"] != "ok":
                raise ValueError("Selected example contains a failed solve")
            proposal = first_seed.get("proposal")
            example["methods"][method] = {
                "initial_pool_amounts": proposal,
                "final_pool_amounts": value["pool_amounts"],
                "gross_out": value["amount_out"],
                "net_out": value["net_out"],
                "gas": value["gas"],
                "elapsed_us": value["elapsed_us"],
                "delta_bp": delta(value, indexed["water_fill"][chosen]),
                "seed_status": "not_registered"
                if method == "water_fill"
                else "accepted"
                if proposal
                else "declined",
                "refinement_skipped": bool(first_seed.get("skipped")),
                "certified_bound_bp": first_seed.get("certified_bound_bp"),
            }
        examples.append(example)
    return {
        "format": "sar-public-examples/1",
        "description": "Selected recorded results; no live browser execution or interpolation",
        "units": {
            "amounts": "Integer token base units as decimal strings",
            "time": "microseconds",
            "gas": "gas units",
            "delta_bp": "Final net-output difference against Fynd; positive is better",
        },
        "initial_allocation": "First seed proposal for the disjoint candidate; null when no seed ran. Final allocation is Fynd's selected result and can differ.",
        "pool_order": ["Pool 1", "Pool 2"],
        "selection": "First matching fresh-benchmark order by block, direction and amount; repeat 0. Examples illustrate behavior, not median latency.",
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    benchmarks = {}
    for name, expected in (("dev", 1800), ("fresh", 4320)):
        benchmarks[name], indexed = bench_summary(args.bench_dir, name, expected)
        if name == "fresh":
            examples = selected_examples(indexed)
    results = {
        "format": "sar-public-results/1",
        "gate_bp": GATE_BP,
        "dataset": dataset_summary(args.cases),
        "held_out_runtime": runtime_summary(args.runtime_report),
        "benchmarks": benchmarks,
        "interpretation": {
            "reference": "Patched Fynd with no seed; output identity checked separately against unmodified Fynd",
            "timing": "Paired method/reference wall time; medians are medians of per-solve ratios. Summed ratio uses total method time divided by total reference time. Timings exclude simulation-count instrumentation.",
            "populations": "All includes every paired order and repeat. Two-pool selects orders where the initializer can run. Simulation counts use a separate single-repeat instrumented run over the same unique orders.",
            "gate": "Every paired case is checked, including one-pool cases and candidate failures. Reference failures prevent a passing gate. Loss strictly greater than 0.1 bp fails.",
            "certificate": "The neighboring-quote check bounds gross output under the concavity assumption; net output after gas is measured empirically.",
            "attribution": "Compare the complete learned warm-start method with Fynd for end-to-end performance. Compare offset with coarse to isolate learned initialization; offset_certified also includes the regret estimator and refinement skip check.",
            "independence": "Orders, repeats and market states are correlated; counts do not establish future failure probabilities.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("results.json", results), ("examples.json", examples)):
        (args.output_dir / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
