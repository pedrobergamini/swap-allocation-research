"""Builds offset cases from market recordings, labelled with a local search over exact quotes.

Each recording is one capture of the two pools, numbered by its file name
(`<name>-NN.json.zst`); splits are whole captures, so no market state appears in two splits. A
case is one order: a capture, a block sampled every `block_stride` blocks, a direction and a size
Q. Each capture draws `sizes_per_capture` sizes per direction log-uniformly from `order_range`,
rounded down to a multiple of 20 base units, and every sampled block is asked all of them. Per case:

- features: logged by the `offset_features` seed inside Fynd (`sar-bench`), the same code the
  offset seed runs, so they are exact by construction. A Python replay of the coarse pass cannot
  be: Fynd charges a pool's gas, at its own derived token price, against its bids until it wins a
  chunk, which moves the coarse split of small orders. Cases the seed declines (one active path, a
  failed whole-order quote) carry no features, as at runtime.
- label: the pool 1 amount on the Q / 16384 lattice, exchange refinement's finest step, with the
  most net output among the sampled allocations, from `sar-quote` on the recording. A Q/256
  scan chooses one neighbourhood to scan at Q/16384. Gas steps can create other local maxima,
  so this does not establish the global optimum on the lattice.
- `regret_curve`: combined net output at every scanned split in bp below the label's, so selection
  and the confidence model estimate a predicted split's regret by linear interpolation. The
  runtime test gate instead compares completed Rust solves.

Gas is priced at the capture's gas price; for WETH output a wei of gas is a wei of output, for
USDC output it is converted at the order's own rate (best combined output over Q). This differs
from Fynd's derived ETH price, so these net-output labels are approximate.

Usage: python -m warm_starts.recorded --recordings 'work/inputs/*.json.zst'
         --config configs/dataset-recorded.json --output work/datasets/cases.jsonl.zst
"""

import argparse
import glob
import io
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import zstandard

from . import features

GRID_STEPS = 256
LATTICE_STEPS = 16384
POOLS = {
    "0xd0b53d9277642d899df5c87a3966a349a798f224": 0,
    "0xb4cb800910b228ed3d0834cf79d697127bbb00e5": 1,
}
DIRECTIONS = features.DIRECTION_INDEX
# Keeps each `sar-quote` command line well under the OS argument limit.
AMOUNTS_PER_CALL = 4000


def capture_number(path):
    match = re.search(r"-(\d+)\.json\.zst$", path)
    if match is None:
        raise ValueError(f"{path} is not named <name>-NN.json.zst")
    return int(match[1])


def split_of(config, capture):
    for name, (first, last) in config["splits"].items():
        if first <= capture <= last:
            return name
    return None


def capture_info(recording):
    """Read checkpoint bounds and explicit gas price from a Fynd replay input."""
    with open(recording, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        replay = json.load(reader)
    metadata = replay["metadata"]
    if metadata.get("chain") != "base" or metadata.get("schema_version") != 1:
        raise ValueError("replay input must use Base recording schema version 1")
    gas = metadata.get("gas_price_wei")
    if not isinstance(gas, str) or not gas.isascii() or not gas.isdecimal():
        raise ValueError("replay input must include gas_price_wei as a decimal string")
    blocks = [update["block_number_or_timestamp"] for update in replay["updates"]]
    if not blocks or any(type(block) is not int or block < 0 for block in blocks):
        raise ValueError("replay input must contain nonnegative integer checkpoint blocks")
    if any(first >= second for first, second in pairwise(blocks)):
        raise ValueError("replay checkpoint blocks must be strictly increasing")
    return blocks[0], blocks[-1], int(gas)


def fynd_features(sar_bench, recording, blocks, gas_price, sizes):
    """{(block, direction, amount): (offset, decline reason)} from the Fynd seed; one is None."""
    cases = [
        {"id": f"{direction}-{amount}", "direction": direction, "amount_in": str(amount)}
        for direction, amounts in sizes.items()
        for amount in amounts
    ]
    with tempfile.TemporaryDirectory() as scratch:
        cases_path, rows_path = Path(scratch, "cases.json"), Path(scratch, "rows.jsonl")
        cases_path.write_text(json.dumps({"cases": cases}))
        command = [
            sar_bench,
            "--recording",
            recording,
            "--blocks",
            ",".join(map(str, blocks)),
            "--cases",
            str(cases_path),
            "--methods",
            "offset_features",
            "--gas-price-wei",
            str(gas_price),
            "--repeats",
            "1",
            "--output",
            str(rows_path),
        ]
        subprocess.run(command, capture_output=True, text=True, check=True)
        table = {}
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            key = (row["block"], row["direction"], int(row["amount_in"]))
            # The disjoint pass asks first; that record is the start exchange refines.
            seed = row["seeds"][0] if row["seeds"] else None
            if seed and seed["proposal"]:
                offset = {"coarse_first": seed["proposal"][0], "features": seed["features"]}
                table[key] = (offset, None)
            else:
                table[key] = (None, seed["declined"] if seed else "seed not asked")
    return table


def quote(sar_quote, recording, block, direction, amounts):
    """{(pool, amount): (gross out, gas) or None} for both pools at every amount."""
    table = {}
    amounts = sorted(amounts)
    for start in range(0, len(amounts), AMOUNTS_PER_CALL):
        batch = amounts[start : start + AMOUNTS_PER_CALL]
        command = [
            sar_quote,
            "--recording",
            recording,
            "--blocks",
            str(block),
            "--direction",
            direction,
            "--amounts",
            ",".join(map(str, batch)),
        ]
        lines = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        for line in lines.splitlines():
            row = json.loads(line)
            quoted = (
                None if row["amount_out"] is None else (int(row["amount_out"]), int(row["gas"]))
            )
            table[(POOLS[row["pool"]], int(row["amount_in"]))] = quoted
    return table


def order_sizes(config, capture, direction):
    low, high = (int(x) for x in config["order_range"][direction])
    rng = np.random.default_rng([config["seed"], capture, DIRECTIONS[direction]])
    logs = rng.uniform(math.log(low), math.log(high), config["sizes_per_capture"])
    return sorted({int(math.exp(x)) // 20 * 20 for x in logs})


def combined(table, amount, first, gas_value):
    """Net output of giving pool 1 `first`: each used pool's gross less its gas, in output units."""
    legs = [(0, first), (1, amount - first)]
    quotes = [table[leg] for leg in legs if leg[1] > 0]
    if None in quotes:
        return None
    return sum(out - gas * gas_value for out, gas in quotes)


def scan(table, amount, firsts, gas_value):
    return {
        first: net
        for first in firsts
        if (net := combined(table, amount, first, gas_value)) is not None
    }


def gas_value(table, direction, amount, grid, gas_price):
    """Output units one unit of gas costs, or None when no grid split quotes."""
    if direction == "usdc-weth":
        return gas_price
    gross = [
        sum(table[leg][0] for leg in ((0, a), (1, amount - a)) if leg[1] > 0)
        for a in grid
        if all(table[leg] is not None for leg in ((0, a), (1, amount - a)) if leg[1] > 0)
    ]
    # USDC units per wei of WETH, at the order's own best rate.
    return gas_price * max(gross) / amount if gross else None


def stage_one_amounts(amount):
    grid = [k * amount // GRID_STEPS for k in range(GRID_STEPS + 1)]
    return set(grid) | {amount - a for a in grid}, grid


def lattice_window(amount, best_step):
    per_step = LATTICE_STEPS // GRID_STEPS
    first = max(0, (best_step - 1) * per_step)
    last = min(LATTICE_STEPS, (best_step + 1) * per_step)
    return [m * amount // LATTICE_STEPS for m in range(first, last + 1)]


def block_cases(sar_quote, recording, capture, split, block, direction, sizes, offsets, gas_price):
    stage_one = {amount: stage_one_amounts(amount) for amount in sizes}
    table = quote(
        sar_quote,
        recording,
        block,
        direction,
        set().union(*(a for a, _ in stage_one.values())) - {0},
    )
    grid_outputs, windows, gas_values = {}, {}, {}
    for amount in sizes:
        grid = stage_one[amount][1]
        gas_values[amount] = gas_value(table, direction, amount, grid, gas_price)
        if gas_values[amount] is None:
            continue
        outputs = scan(table, amount, grid, gas_values[amount])
        best_first = max(outputs, key=lambda first: (outputs[first], -first))
        grid_outputs[amount] = outputs
        windows[amount] = lattice_window(amount, grid.index(best_first))
    needed = {x for amount, window in windows.items() for a in window for x in (a, amount - a)}
    table.update(
        quote(sar_quote, recording, block, direction, needed - {a for _, a in table} - {0})
    )

    cases = []
    for amount, outputs in grid_outputs.items():
        outputs = {**outputs, **scan(table, amount, windows[amount], gas_values[amount])}
        # Ties go to the smallest pool 1 amount, as in the historical dataset.
        label = max(outputs, key=lambda first: (outputs[first], -first))
        best = outputs[label]
        firsts = sorted(outputs)
        cases.append(
            {
                "id": f"c{capture:02d}-{block}-{direction}-{amount}",
                "split": split,
                "capture": capture,
                "block": block,
                "direction": direction,
                "amount_in": str(amount),
                "offset": offsets[(block, direction, amount)][0],
                "declined": offsets[(block, direction, amount)][1],
                "label_share": label / amount,
                "label_first": str(label),
                "regret_curve": {
                    "share": [first / amount for first in firsts],
                    "bp": [round((best - outputs[first]) / best * 1e4, 9) for first in firsts],
                },
            }
        )
    return cases


def build(recordings, config, sar_quote, sar_bench, workers):
    jobs = []
    for recording in sorted(recordings, key=capture_number):
        capture = capture_number(recording)
        split = split_of(config, capture)
        if split is None:
            continue
        first, last, gas_price = capture_info(recording)
        blocks = list(range(first, last + 1, config["block_stride"]))
        sizes = {direction: order_sizes(config, capture, direction) for direction in DIRECTIONS}
        offsets = fynd_features(sar_bench, recording, blocks, gas_price, sizes)
        for block in blocks:
            for direction in DIRECTIONS:
                jobs.append(
                    (
                        sar_quote,
                        recording,
                        capture,
                        split,
                        block,
                        direction,
                        sizes[direction],
                        offsets,
                        gas_price,
                    )
                )
    with ThreadPoolExecutor(workers) as pool:
        for cases in pool.map(lambda job: block_cases(*job), jobs):
            yield from cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", required=True, help="glob of <name>-NN.json.zst captures")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sar-quote", default="target/release/sar-quote")
    parser.add_argument("--sar-bench", default="target/release/sar-bench")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    recordings = glob.glob(args.recordings)
    counts = defaultdict(lambda: [0, 0])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with (
        open(args.output, "wb") as raw,
        zstandard.ZstdCompressor(level=10).stream_writer(raw) as compressed,
    ):
        text = io.TextIOWrapper(compressed, encoding="utf-8")
        for case in build(recordings, config, args.sar_quote, args.sar_bench, args.workers):
            text.write(json.dumps(case) + "\n")
            counts[case["split"]][0] += 1
            counts[case["split"]][1] += case["offset"] is not None
        text.flush()
    print(f"cases (with offset features) per split: {dict(counts)} -> {args.output}")


if __name__ == "__main__":
    main()
