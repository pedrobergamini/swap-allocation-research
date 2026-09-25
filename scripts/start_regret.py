"""How much output each seed's start leaves behind before exchange refinement.

Distance from Water-fill's final split overstates misses where the combined output is flat and
understates them where it is curved. Regret measures what the gate measures instead: the gross
output of the seed's proposed split, quoted exactly from the recording with `sar-quote`, against
the gross output of unmodified Water-fill's final split on the same block, in bp. Zero means the
start was already as good as Fynd's answer; the share under 0.1 bp is how often the start alone
would have passed the gate.

Proposals are deterministic per block and order, so repeat 0 is read. Gross output leaves gas
out; a final split on one pool saves one hop of gas the regret here does not count.

Usage: python scripts/start_regret.py BENCH.jsonl RECORDING [SAR_QUOTE]
"""

import json
import subprocess
import sys
from collections import defaultdict

import numpy as np

POOLS = {
    "0xd0b53d9277642d899df5c87a3966a349a798f224": 0,
    "0xb4cb800910b228ed3d0834cf79d697127bbb00e5": 1,
}
THRESHOLDS_BP = (0.02, 0.1)


def quotes(sar_quote, recording, block, direction, amounts):
    command = [
        sar_quote,
        "--recording",
        recording,
        "--blocks",
        str(block),
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
        table[(POOLS[row["pool"]], int(row["amount_in"]))] = out
    return table


def main():
    rows_path, recording, *rest = sys.argv[1:]
    sar_quote = rest[0] if rest else "target/release/sar-quote"
    with open(rows_path) as file:
        rows = [json.loads(line) for line in file if line.strip()]
    rows = [row for row in rows if row["repeat"] == 0]
    final = {
        (row["block"], row["case_id"]): [int(a) for a in row["pool_amounts"]]
        for row in rows
        if row["method"] == "water_fill"
    }
    starts = []
    for row in rows:
        seed = row["seeds"][0] if row["seeds"] else None
        if row["method"] != "water_fill" and seed and seed["proposal"]:
            key = (row["block"], row["case_id"])
            starts.append(
                (row["method"], key, row["direction"], [int(a) for a in seed["proposal"]])
            )

    wanted = defaultdict(set)
    for _, key, direction, proposal in starts:
        wanted[(key[0], direction)].update([*proposal, *final[key]])
    tables = {
        (block, direction): quotes(sar_quote, recording, block, direction, amounts - {0})
        for (block, direction), amounts in wanted.items()
    }

    def gross(block, direction, amounts):
        table = tables[(block, direction)]
        outs = [0 if a == 0 else table[(pool, a)] for pool, a in enumerate(amounts)]
        return None if None in outs else sum(outs)

    regret, failed = defaultdict(list), defaultdict(int)
    for method, key, direction, proposal in starts:
        start = gross(key[0], direction, proposal)
        best = gross(key[0], direction, final[key])
        if start is None or best is None:
            failed[method] += 1
            continue
        regret[method].append((best - start) / best * 1e4)

    summary = {}
    for method, values in sorted(regret.items()):
        values = np.array(values)
        summary[method] = {
            "solves": len(values),
            "unquotable": failed[method],
            "regret_bp": {
                **{f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 99)},
                "max": float(values.max()),
            },
            **{f"under_{t}bp": round(float(np.mean(values <= t)), 4) for t in THRESHOLDS_BP},
        }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
