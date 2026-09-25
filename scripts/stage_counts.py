"""Simulations per solve, by Water-fill stage, from a `swap-metrics` sar-bench log.

Run sar-bench built with `--features swap-metrics` and
`RUST_LOG=sar_bench=debug,fynd_core::algorithm::sim_meter=debug NO_COLOR=1`. Each timed solve's
`solve` marker is followed by Fynd's "simulation by stage" line; warm-up solves (no repeat) are
skipped. Counts are deterministic, unlike wall time, so they compare methods without timer noise.

Only solves where the method's seed ran (two active paths, per the bench rows) are counted, and
water_fill is restricted to the same solves. `paired_total` compares two methods solve by solve
(same block, case and repeat): the learned start against the coarse one, and each skip against
always refining.

Usage: python scripts/stage_counts.py LOG ROWS.jsonl
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np

MARKER = re.compile(r'solve block=(\d+) case=(\S+) method="(\w+)" repeat=Some\((\d+)\)')
STAGE = re.compile(r"(\w[\w-]*): (\d+) calls")
REPORTED = ("chunking", "exchange", "initializer")
# (method, reference): what each comparison isolates.
PAIRS = (
    ("offset", "coarse"),
    ("offset", "water_fill"),
    ("offset_confident", "offset"),
    ("offset_certified", "offset"),
)


def main():
    log_path, rows_path = sys.argv[1:]
    with open(rows_path) as file:
        rows = [json.loads(line) for line in file if line.strip()]
    seeded = {
        (row["block"], row["case_id"], row["repeat"])
        for row in rows
        if row["seeds"] and row["seeds"][0]["proposal"] is not None
    }

    counts = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(dict)
    pending = None
    with open(log_path) as file:
        for line in file:
            if marker := MARKER.search(line):
                block, case, method, repeat = marker.groups()
                pending = (method, (int(block), case, int(repeat)))
            elif "simulation by stage" in line and pending is not None:
                method, key = pending
                pending = None
                if key not in seeded:
                    continue
                stages = {name: int(calls) for name, calls in STAGE.findall(line)}
                counts[method]["total"].append(sum(stages.values()))
                totals[method][key] = sum(stages.values())
                for stage in REPORTED:
                    counts[method][stage].append(stages.get(stage, 0))

    summary = {
        method: {
            "solves": len(stages["total"]),
            **{f"{name}_median": float(np.median(values)) for name, values in stages.items()},
            "total_p90": float(np.percentile(stages["total"], 90)),
            "total_mean": round(float(np.mean(stages["total"])), 1),
        }
        for method, stages in sorted(counts.items())
    }
    summary["paired_total"] = {}
    for method, reference in PAIRS:
        shared = totals[method].keys() & totals[reference].keys()
        if not shared:
            continue
        delta = np.array([totals[method][key] - totals[reference][key] for key in shared])
        summary["paired_total"][f"{method} - {reference}"] = {
            "solves": len(shared),
            "mean": round(float(delta.mean()), 2),
            "median": float(np.median(delta)),
            "p10": float(np.percentile(delta, 10)),
            "p90": float(np.percentile(delta, 90)),
            "fewer_same_more": [
                int(np.sum(delta < 0)),
                int(np.sum(delta == 0)),
                int(np.sum(delta > 0)),
            ],
        }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
