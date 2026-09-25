"""Trains the coarse-offset model and exports a `sar-mlp/2` artifact.

The model moves Water-fill's coarse split by at most `--chunks` coarse chunks toward the best
split, reading only features the initializer gets for free (`coarse-offset/2`). The target is
each case's continuous best share (`label_share`); the loss is a smooth absolute error in coarse
chunks, so the median and tail of the start error both count and a flat curve cannot stall it.

What matters downstream is how far the start sits from the best split: exchange refinement
begins with steps of 1/256 of the order and needs about one extra round per step of distance.
The report gives that distance for the coarse split, the model and the non-learned quadratic seed
on every split, and, for cases built from recordings (`recorded`), the start's approximate net regret against the sampled reference. The final quality gate
is measured separately on complete Rust solver outputs.

Usage: python -m warm_starts.train_offset --cases CASES --output MODEL.json --report REPORT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import dataset, model, quadratic
from .features import OFFSET_FEATURE_NAMES, OFFSET_SPEC_VERSION
from .train import Mlp, sigmoid

COARSE_CHUNKS = 20
SMOOTHING_CHUNKS = 0.01


def load(path):
    splits = {}
    for case in dataset.read_jsonl_zst(path):
        if case["offset"] is None:
            continue
        rows = splits.setdefault(
            case["split"],
            {
                "x": [],
                "label": [],
                "id": [],
                "group": [],
                "cell": [],
                "quadratic": [],
                "curves": [],
            },
        )
        rows["x"].append(case["offset"]["features"])
        curve = case.get("regret_curve")
        rows["curves"].append(curve and (np.array(curve["share"]), np.array(curve["bp"])))
        rows["label"].append(case["label_share"])
        rows["id"].append(case["id"])
        # One market (capture or window), direction and size: its blocks are near-duplicates.
        market = case.get("capture", case.get("window"))
        rows["group"].append(market)
        rows["cell"].append(f"{market}-{case['direction']}-{case['amount_in']}")
        rows["quadratic"].append(quadratic.quadratic_share(case["offset"]["features"]))
    return {
        name: {
            "x": np.array(r["x"]),
            "label": np.array(r["label"]),
            "id": r["id"],
            "group": np.array(r["group"]),
            "cell": np.array(r["cell"]),
            "quadratic": np.array(r["quadratic"], dtype=float),
            # Only cases built from recordings carry regret curves.
            "curves": r["curves"] if all(c is not None for c in r["curves"]) else None,
        }
        for name, r in splits.items()
    }


def start_error(share, label):
    distance = np.abs(share - label)
    distance = distance[~np.isnan(distance)]
    return {
        "median_pct": float(np.median(distance) * 100),
        "p90_pct": float(np.percentile(distance, 90) * 100),
        "p99_pct": float(np.percentile(distance, 99) * 100),
        "max_pct": float(distance.max() * 100),
        # Distance in exchange refinement's first step size (1/256 of the order), unrounded so
        # starts closer than one step still rank.
        "mean_steps": float(np.mean(distance * 256)),
        "cases": len(distance),
    }


def regret_of(share, curves):
    """Approximate net regret against the sampled label, linearly interpolated in bp."""
    return np.array([np.interp(s, *curve) for s, curve in zip(share, curves, strict=True)])


def reachable_floor(curves, coarse, reach):
    """Least regret any start within the reach of the coarse split can have, per case.

    A label beyond the reach leaves a floor no model can remove; selection scores the regret
    above it, the part the model controls, while reports keep the whole regret.
    """
    floors = []
    for (shares, bp), center in zip(curves, coarse, strict=True):
        low, high = max(center - reach, 0.0), min(center + reach, 1.0)
        inside = bp[(shares >= low) & (shares <= high)]
        ends = np.interp([low, high], shares, bp)
        floors.append(min(ends.min(), inside.min()) if inside.size else ends.min())
    return np.array(floors)


def cell_cvar5(regret, cells):
    """Mean of the worst 5 % of cells, a cell's regret being its mean over its blocks.

    Blocks of one capture are near-duplicates, so a row-level tail is a handful of cells counted
    many times; ranking on cells keeps one bad size from looking like many independent misses.
    """
    names, index = np.unique(cells, return_inverse=True)
    per_cell = np.bincount(index, weights=regret) / np.bincount(index)
    return float(np.sort(per_cell)[-max(1, len(names) // 20) :].mean()), len(names)


def start_regret(share, curves, cells):
    regret = regret_of(share, curves)
    cvar, cell_count = cell_cvar5(regret, cells)
    return {
        "median_bp": float(np.median(regret)),
        "p90_bp": float(np.percentile(regret, 90)),
        "p99_bp": float(np.percentile(regret, 99)),
        "max_bp": float(regret.max()),
        "mean_bp": float(regret.mean()),
        # Mean of the worst 5 %: steadier than p99 on a few thousand cases.
        "cvar5_bp": float(np.sort(regret)[-max(1, len(regret) // 20) :].mean()),
        "under_0.02bp": float(np.mean(regret <= 0.02)),
        "under_0.1bp": float(np.mean(regret <= 0.1)),
        "cell_cvar5_bp": cvar,
        "cases": len(regret),
        "cells": cell_count,
    }


def train(data, hidden, chunks, epochs, batch, learning_rate, seed):
    rng = np.random.default_rng(seed)
    train_x = data["train"]["x"]
    inputs = [i for i in range(train_x.shape[1]) if train_x[:, i].std() > 0]
    mean, std = train_x[:, inputs].mean(axis=0), train_x[:, inputs].std(axis=0)

    def normalize(x):
        return (x[:, inputs] - mean) / std

    net = Mlp([len(inputs), *hidden, 1], rng)
    params = [p for pair in zip(net.weights, net.biases, strict=True) for p in pair]
    moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    x, coarse, label = normalize(train_x), train_x[:, 2], data["train"]["label"]
    reach = chunks / COARSE_CHUNKS
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            rows = order[start : start + batch]
            activations = net.forward(x[rows])
            squashed = sigmoid(activations[-1][:, 0])
            share = coarse[rows] + (2 * squashed - 1) * reach
            error = (share - label[rows]) * COARSE_CHUNKS
            # Smooth |error|. A label beyond the reach (small orders whose output is flat across
            # many chunks, so the miss costs little) just pulls the start to the edge.
            grad_share = error / np.sqrt(error**2 + SMOOTHING_CHUNKS**2) * COARSE_CHUNKS
            grad_logit = grad_share * 2 * reach * squashed * (1 - squashed) / len(rows)
            grads = net.backward(activations, grad_logit)
            step += 1
            flat = [g for pair in grads for g in pair]
            for param, (m, v), grad in zip(params, moments, flat, strict=True):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad**2
                param -= (
                    learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )

    def predict(features):
        squashed = sigmoid(net.forward(normalize(features))[-1][:, 0])
        return np.clip(features[:, 2] + (2 * squashed - 1) * reach, 0.0, 1.0)

    return net, inputs, mean, std, predict


def export(net, inputs, mean, std, chunks, predict, check):
    """The `sar-mlp/2` artifact, after checking the reference forward pass agrees with numpy."""
    layers = [
        (w.tolist(), b.tolist(), "tanh")
        for w, b in zip(net.weights[:-1], net.biases[:-1], strict=True)
    ]
    layers.append((net.weights[-1].tolist(), net.biases[-1].tolist(), "identity"))
    artifact = model.artifact(
        OFFSET_SPEC_VERSION, inputs, mean, std, layers, {"kind": "coarse_offset", "chunks": chunks}
    )
    # The reference forward pass defines what ships; numpy only differs in summation order.
    reference = np.array([model.share(artifact, list(row), row[2]) for row in check])
    drift = float(np.max(np.abs(reference - predict(check))))
    assert drift < 1e-9, f"reference and numpy predictions differ by {drift}"
    return artifact, drift


def start_regrets(data, predict, splits):
    return {
        name: {
            "coarse": start_regret(data[name]["x"][:, 2], data[name]["curves"], data[name]["cell"]),
            "offset": start_regret(
                predict(data[name]["x"]), data[name]["curves"], data[name]["cell"]
            ),
            "quadratic": start_regret(
                data[name]["quadratic"], data[name]["curves"], data[name]["cell"]
            ),
        }
        for name in splits
        if name in data and data[name]["curves"] is not None
    }


def start_errors(data, predict, splits):
    return {
        name: {
            "coarse": start_error(data[name]["x"][:, 2], data[name]["label"]),
            "offset": start_error(predict(data[name]["x"]), data[name]["label"]),
            "quadratic": start_error(data[name]["quadratic"], data[name]["label"]),
        }
        for name in splits
        if name in data
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--hidden", default="32,32")
    parser.add_argument("--chunks", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    # Test windows are scored once, for the frozen candidate (`select_offset`); a plain training
    # run reports them only when asked.
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    hidden = [int(width) for width in args.hidden.split(",")]
    data = load(args.cases)
    reach_needed = float(np.max(np.abs(data["train"]["x"][:, 2] - data["train"]["label"])))
    net, inputs, mean, std, predict = train(
        data, hidden, args.chunks, args.epochs, args.batch, args.learning_rate, args.seed
    )
    check = data.get("validation", data["train"])["x"][:512]
    artifact, drift = export(net, inputs, mean, std, args.chunks, predict, check)
    splits = ["train", "validation", *(["test"] if args.include_test else [])]
    report = {
        "config": vars(args),
        "inputs": [OFFSET_FEATURE_NAMES[i] for i in inputs],
        "reference_drift": drift,
        "max_train_distance_chunks": reach_needed * COARSE_CHUNKS,
        "start_error": start_errors(data, predict, splits),
        "start_regret": start_regrets(data, predict, splits),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("max_train_distance_chunks", "start_error", "start_regret")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
