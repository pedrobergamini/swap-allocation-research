"""Checks continuous labels from the collection grid against dense quotes and Fynd's own split.

For each (block, order) of a benchmark run, quotes both pools densely from the recording
(`sar-quote`), then compares pool 1's share at the best split found three ways: dense PCHIP
(truth), PCHIP on the collection grid only (j*Q/64 and j*Q/20), and Fynd's final `water_fill`
route. Usage: uv run python scripts/validate_labels.py RECORDING BENCH.jsonl
"""

import json
import subprocess
import sys
from collections import defaultdict

import numpy as np
from warm_starts.curves import best_share

DENSE = 1024


def main(recording, bench_path):
    with open(bench_path) as file:
        rows = [json.loads(line) for line in file]
    fynd = {
        (r["block"], r["case_id"]): r
        for r in rows
        if r["method"] == "water_fill" and r["repeat"] == 0
    }
    blocks = sorted({key[0] for key in fynd})
    orders = defaultdict(set)
    for r in fynd.values():
        orders[r["direction"]].add(int(r["amount_in"]))
    quotes = {}
    for direction, amounts in orders.items():
        dense = sorted(
            {j * q // DENSE for q in amounts for j in range(1, DENSE + 1)}
            | {j * q // 20 for q in amounts for j in range(1, 21)}
        )
        out = subprocess.run(
            [
                "target/release/sar-quote",
                "--recording",
                recording,
                "--blocks",
                ",".join(map(str, blocks)),
                "--direction",
                direction,
                "--amounts",
                ",".join(map(str, dense)),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in out.splitlines():
            q = json.loads(line)
            if q["amount_out"] is not None:
                pool = 0 if q["pool"].lower().startswith("0xd0b5") else 1
                quotes[(q["block"], direction, pool, int(q["amount_in"]))] = int(q["amount_out"])

    errors = defaultdict(list)
    for (block, case_id), r in fynd.items():
        amount, direction = int(r["amount_in"]), r["direction"]
        final = int(r["pool_amounts"][0]) / amount
        if final in (0.0, 1.0):
            continue  # single-pool orders have no split to label

        def samples(points, block=block, direction=direction):
            return [
                {
                    0: 0,
                    **{
                        x: quotes[(block, direction, p, x)]
                        for x in points
                        if (block, direction, p, x) in quotes
                    },
                }
                for p in range(2)
            ]

        dense = [j * amount // DENSE for j in range(1, DENSE + 1)]
        sparse = sorted(
            {j * amount // 64 for j in range(1, 65)} | {j * amount // 20 for j in range(1, 21)}
        )
        truth, _ = best_share(amount, samples(dense))
        label, _ = best_share(amount, samples(sparse))
        errors["sparse_label_vs_dense"].append(abs(label - truth))
        errors["dense_vs_fynd_final"].append(abs(truth - final))
        errors["sparse_label_vs_fynd_final"].append(abs(label - final))
    for name, values in errors.items():
        v = np.array(values)
        print(
            f"{name:28s} n={len(v)} median {np.median(v):.5f} p90 {np.percentile(v, 90):.5f} "
            f"max {v.max():.5f} (share of order)"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
