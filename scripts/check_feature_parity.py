"""Checks that the offset seed's Rust features equal the Python features training data uses.

For every bench order the `offset` seed handled, this quotes the same recorded state with
`sar-quote` at the amounts Water-fill's coarse pass prices (j * Q / 20) and the whole order, replays
the coarse pass in Python (`coarse.coarse_pass`), builds `offset_feature_vector`, and compares it
with the vector the seed logged. Together with `check_model_parity.py` this covers the path from
pool state to proposed split.

Every feature must match exactly except `quadratic_offset_chunks`: Fynd prices each chunk on the
pool state its earlier chunks drained, while training data differences quotes of untouched state,
and the two can differ by rounding. Its largest difference is reported, in chunks, and must stay
under `QUADRATIC_TOLERANCE_CHUNKS`.

Usage: python scripts/check_feature_parity.py BENCH.jsonl RECORDING [SAR_QUOTE]
"""

import json
import subprocess
import sys
from collections import defaultdict

from warm_starts import coarse, features, quadratic

QUADRATIC_FEATURE = 6
QUADRATIC_TOLERANCE_CHUNKS = 1e-3

DIRECTIONS = {"weth-usdc": features.WETH_USDC, "usdc-weth": features.USDC_WETH}
POOLS = {
    "0xd0b53d9277642d899df5c87a3966a349a798f224": 0,
    "0xb4cb800910b228ed3d0834cf79d697127bbb00e5": 1,
}


def quotes(sar_quote, recording, blocks, direction, amounts):
    command = [
        sar_quote,
        "--recording",
        recording,
        "--blocks",
        ",".join(map(str, sorted(blocks))),
        "--direction",
        direction,
        "--amounts",
        ",".join(map(str, sorted(amounts))),
    ]
    table = {}
    lines = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    for line in lines.splitlines():
        row = json.loads(line)
        out = None if row["amount_out"] is None else int(row["amount_out"])
        table[(row["block"], POOLS[row["pool"]], int(row["amount_in"]))] = out
    return table


def main():
    rows_path, recording, *rest = sys.argv[1:]
    sar_quote = rest[0] if rest else "target/release/sar-quote"
    seeded = []
    with open(rows_path) as file:
        for line in file:
            row = json.loads(line)
            if row["method"] == "offset" and row["seeds"] and row["seeds"][0]["proposal"]:
                seeded.append(row)

    wanted = defaultdict(lambda: (set(), set()))
    for row in seeded:
        amount = int(row["amount_in"])
        blocks, amounts = wanted[row["direction"]]
        blocks.add(row["block"])
        amounts.update(j * amount // coarse.COARSE_CHUNKS for j in range(1, 21))
    tables = {
        direction: quotes(sar_quote, recording, blocks, direction, amounts)
        for direction, (blocks, amounts) in wanted.items()
    }

    checked, mismatches, worst_gap = 0, [], 0.0
    for row in seeded:
        amount, block = int(row["amount_in"]), row["block"]
        table = tables[row["direction"]]

        def out(pool, x, table=table, block=block):
            return 0 if x == 0 else table.get((block, pool, x))

        coarse_first, gains = coarse.coarse_pass(amount, out)
        moved = quadratic.free_quadratic_offset(amount, coarse_first, gains)
        full = [out(0, amount), out(1, amount)]
        vector = features.offset_feature_vector(
            DIRECTIONS[row["direction"]], amount, coarse_first, full, moved
        )
        logged = row["seeds"][0]["features"]
        checked += 1
        gap = abs(vector[QUADRATIC_FEATURE] - logged[QUADRATIC_FEATURE])
        worst_gap = max(worst_gap, gap)
        exact = [v for i, v in enumerate(vector) if i != QUADRATIC_FEATURE]
        logged_exact = [v for i, v in enumerate(logged) if i != QUADRATIC_FEATURE]
        if exact != logged_exact or gap > QUADRATIC_TOLERANCE_CHUNKS:
            mismatches.append(
                {"case": row["case_id"], "block": block, "python": vector, "rust": logged}
            )
    print(
        json.dumps(
            {
                "checked": checked,
                "mismatches": len(mismatches),
                "worst_quadratic_gap_chunks": worst_gap,
                "examples": mismatches[:3],
            },
            indent=1,
        )
    )
    sys.exit(1 if mismatches or not checked else 0)


if __name__ == "__main__":
    main()
