"""Writes a quote table from a Fynd recording, in the format `warm_starts.dataset` reads.

Quotes both pools at every amount the historical collection asks for (the 64-step grid and the
20-chunk points of each order size in the dataset config) at the given blocks, through
`sar-quote`. Blocks are assigned to windows round-robin (`--windows N` -> w00..), so one
recording can feed a train/validation split for pipeline checks; its blocks are minutes apart,
so it is never evidence of generalization.

Usage: uv run python scripts/recording_quote_table.py RECORDING CONFIG OUTPUT --blocks a,b,...
"""

import argparse
import json
import subprocess
from pathlib import Path

from warm_starts.dataset import order_sizes, write_jsonl_zst

POOL_INDEX = {
    "0xd0b53d9277642d899df5c87a3966a349a798f224": 1,
    "0xb4cb800910b228ed3d0834cf79d697127bbb00e5": 2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--windows", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    blocks = [int(block) for block in args.blocks.split(",")]
    window_of = {block: f"w{index % args.windows:02d}" for index, block in enumerate(blocks)}
    rows = []
    for direction in config["order_sizes"]:
        amounts = set()
        for total in order_sizes(config, direction):
            amounts.update(
                j * total // config["grid_steps"] for j in range(1, config["grid_steps"] + 1)
            )
            amounts.update(j * total // 20 for j in range(1, 21))
        out = subprocess.run(
            [
                "target/release/sar-quote",
                "--recording",
                args.recording,
                "--blocks",
                args.blocks,
                "--direction",
                direction,
                "--amounts",
                ",".join(map(str, sorted(amounts))),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in out.splitlines():
            quote = json.loads(line)
            rows.append(
                {
                    "window": window_of[quote["block"]],
                    "block": quote["block"],
                    "pool": POOL_INDEX[quote["pool"].lower()],
                    "direction": direction,
                    "amount_in": quote["amount_in"],
                    "amount_out": quote["amount_out"],
                }
            )
    write_jsonl_zst(args.output, rows)
    print(f"{len(rows)} quotes -> {args.output}")


if __name__ == "__main__":
    main()
