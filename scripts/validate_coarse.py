"""Checks the offline coarse-pass replay against Fynd's own coarse splits.

Reads a benchmark run with the `coarse` method (its seed log records `request.coarse`), quotes
both pools at every j * Q / 20 from the recording, and replays the pass.
Usage: uv run python scripts/validate_coarse.py RECORDING BENCH.jsonl
"""

import json
import subprocess
import sys
from collections import Counter, defaultdict

from warm_starts.coarse import COARSE_CHUNKS, coarse_split


def main(recording, bench_path):
    with open(bench_path) as file:
        rows = [json.loads(line) for line in file]
    coarse = {
        (r["block"], r["case_id"]): r
        for r in rows
        if r["method"] == "coarse" and r["repeat"] == 0 and r["seeds"] and r["seeds"][0]["proposal"]
    }
    blocks = sorted({key[0] for key in coarse})
    orders = defaultdict(set)
    for r in coarse.values():
        orders[r["direction"]].add(int(r["amount_in"]))
    quotes = {}
    for direction, amounts in orders.items():
        points = sorted(
            {j * q // COARSE_CHUNKS for q in amounts for j in range(1, COARSE_CHUNKS + 1)}
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
                ",".join(map(str, points)),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in out.splitlines():
            q = json.loads(line)
            pool = 0 if q["pool"].lower().startswith("0xd0b5") else 1
            value = None if q["amount_out"] is None else int(q["amount_out"])
            quotes[(q["block"], direction, pool, int(q["amount_in"]))] = value

    outcome = Counter()
    for (block, _), r in coarse.items():
        amount, direction = int(r["amount_in"]), r["direction"]

        def out(pool, x, block=block, direction=direction):
            return 0 if x == 0 else quotes.get((block, direction, pool, x))

        replayed = coarse_split(amount, out)
        actual = int(r["seeds"][0]["proposal"][0])
        chunks_off = round((replayed - actual) * COARSE_CHUNKS / amount)
        outcome[chunks_off] += 1
    print("replayed minus Fynd coarse, in chunks:", dict(sorted(outcome.items())))


if __name__ == "__main__":
    main(*sys.argv[1:])
