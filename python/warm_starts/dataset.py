"""Builds allocation cases from a quote table.

A quote table has one row per (window, block, pool, direction, amount_in) with the pool's gross
`amount_out` against untouched state, or null for a failed quote. A case is one order: a block,
a direction and an order size Q. Its probes are the four quotes per pool the runtime also sees
(Q/16, Q/4, Q/2, Q); its teacher grid is every pool's output at j * Q / 64, which only labelling
and evaluation read. The label is the best grid split by combined gross output, ties going to the
smallest pool 1 share. `label_share` is the continuous best split: PCHIP through every quote
of the order (the 64-step grid plus the 20-chunk points) maximized at 1/16384 resolution, which
is an approximate reference rather than a guarantee of agreement with Fynd's final split.
`chunks` holds each pool's output at j * Q / 20, the amounts Water-fill's coarse pass prices, and
`offset` the replayed coarse split with the `coarse-offset/2` features built from it.

Windows are numbered in chronological order, and splits are whole windows, so no market state
appears in two splits.

Usage: python -m warm_starts.dataset --quotes Q.jsonl.zst --config configs/dataset-v1.json
         --output work/datasets/v1/cases.jsonl.zst
"""

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import zstandard

from . import coarse, curves, features, interpolation, quadratic

PROBE_STEPS = (4, 16, 32, 64)  # Q/16, Q/4, Q/2 and Q on the 64-step grid
COARSE_CHUNKS = 20  # Water-fill's coarse pass prices each pool at multiples of Q/20


def read_jsonl_zst(path):
    with open(path, "rb") as file:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(file), "utf-8")
        for line in reader:
            yield json.loads(line)


def write_jsonl_zst(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file, zstandard.ZstdCompressor(level=19).stream_writer(file) as writer:
        for row in rows:
            writer.write((json.dumps(row, separators=(",", ":")) + "\n").encode())


def order_sizes(config, direction):
    spec = config["order_sizes"][direction]
    return [int(spec["base"]) << i for i in range(spec["doublings"])]


def window_index(window):
    return int(window.lstrip("w"))


def split_of(config, window):
    index = window_index(window)
    for name, (first, last) in config["splits"].items():
        if first <= index <= last:
            return name
    raise ValueError(f"window {window} is in no split")


def best_split(grid, steps):
    """Best k by combined gross output, smallest k on ties; None when no k has both quotes."""
    best = None
    for k in range(steps + 1):
        first, second = grid[0][k], grid[1][steps - k]
        if first is None or second is None:
            continue
        total = first + second
        if best is None or total > best[1]:
            best = (k, total)
    return best


def build_cases(quotes, config):
    steps = config["grid_steps"]
    table = defaultdict(dict)
    for row in quotes:
        out = row["amount_out"]
        key = (row["window"], row["block"], row["direction"], row["pool"])
        table[key][int(row["amount_in"])] = None if out is None else int(out)

    markets = sorted({key[:3] for key in table}, key=lambda m: (window_index(m[0]), m[1], m[2]))
    skipped = defaultdict(int)
    for window, block, direction in markets:
        pools = [table.get((window, block, direction, pool), {}) for pool in (1, 2)]
        for amount in order_sizes(config, direction):
            grid_amounts = [j * amount // steps for j in range(steps + 1)]
            if not all(a in pool for pool in pools for a in grid_amounts[1:]):
                skipped["incomplete grid"] += 1
                continue
            grid = [[0] + [pool[a] for a in grid_amounts[1:]] for pool in pools]
            probes = [[grid[p][j] for j in PROBE_STEPS] for p in range(2)]
            vector = features.feature_vector(direction, amount, probes)
            best = best_split(grid, steps)
            if vector is None or best is None:
                skipped["no reference price or no valid split"] += 1
                continue
            chunk_amounts = [j * amount // COARSE_CHUNKS for j in range(1, COARSE_CHUNKS + 1)]
            chunks = (
                [[0] + [pool[a] for a in chunk_amounts] for pool in pools]
                if all(a in pool for pool in pools for a in chunk_amounts)
                else None
            )
            # Continuous label from every quote of the order's grids, failed quotes left out.
            points = set(grid_amounts[1:]) | (set(chunk_amounts) if chunks else set())
            samples = [
                {0: 0, **{a: pool[a] for a in points if pool.get(a) is not None}} for pool in pools
            ]
            label_share, _ = curves.best_share(amount, samples)
            offset = None
            if chunks:
                by_amount = [dict(zip([0, *chunk_amounts], pool, strict=True)) for pool in chunks]
                replay = coarse.coarse_pass(amount, lambda p, x, table=by_amount: table[p].get(x))
                full = [grid[0][steps], grid[1][steps]]
                # The offset seed declines when a whole-order quote fails, so no case teaches it one.
                if replay is not None and None not in full:
                    coarse_first, gains = replay
                    moved = quadratic.free_quadratic_offset(amount, coarse_first, gains)
                    vector_offset = (
                        None
                        if moved is None
                        else features.offset_feature_vector(
                            direction, amount, coarse_first, full, moved
                        )
                    )
                    if vector_offset is not None:
                        offset = {"coarse_first": str(coarse_first), "features": vector_offset}
            try:
                first, _ = interpolation.interpolation_split(amount, probes)
                interpolation_k = first * steps // amount
            except ValueError:
                interpolation_k = None
            yield {
                "id": f"{window}-{block}-{direction}-{amount}",
                "split": split_of(config, window),
                "window": window,
                "block": block,
                "direction": direction,
                "amount_in": str(amount),
                "probes": [[None if v is None else str(v) for v in pool] for pool in probes],
                "features": vector,
                "grid": [[None if v is None else str(v) for v in pool] for pool in grid],
                "chunks": chunks
                and [[None if v is None else str(v) for v in pool] for pool in chunks],
                "label_share": label_share,
                "offset": offset,
                "label_k": best[0],
                "label_gross": str(best[1]),
                "interpolation_k": interpolation_k,
            }
    if skipped:
        print(f"skipped: {dict(skipped)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    counts = defaultdict(int)

    def counted(cases):
        for case in cases:
            counts[case["split"]] += 1
            yield case

    write_jsonl_zst(args.output, counted(build_cases(read_jsonl_zst(args.quotes), config)))
    print(f"cases per split: {dict(counts)} -> {args.output}")


if __name__ == "__main__":
    main()
