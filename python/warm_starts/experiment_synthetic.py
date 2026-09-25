"""Generate artificial allocation examples for an offline training smoke check, not a benchmark."""

import argparse
from pathlib import Path

import numpy as np

from . import dataset


def rows():
    rng = np.random.default_rng(71)
    for index in range(256):
        coarse = float(rng.uniform(0.1, 0.9))
        gap = float(rng.normal(0, 4))
        correction = float(0.03 * np.tanh(gap / 4))
        yield {
            "id": f"synthetic-{index}",
            "split": "train" if index < 192 else "validation",
            "window": index // 16,
            "direction": "weth-usdc",
            "amount_in": str(10**18 + index),
            "label_share": coarse + correction,
            "offset": {"features": [0.0, float(rng.uniform(2, 5)), coarse, gap, 0.0, 0.0, 0.0]},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    dataset.write_jsonl_zst(args.output, rows())


if __name__ == "__main__":
    main()
