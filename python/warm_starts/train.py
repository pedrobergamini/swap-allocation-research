"""Trains the allocation MLP with numpy and exports a `sar-mlp/1` artifact.

The network maps the probe features to pool 1's share `f` of the order. Training minimizes
regret: how many basis points of gross output the split `f` loses against the best split on the
case's 64-step teacher grid, reading the grid as a piecewise-linear curve. A short phase of
squared error toward the best grid share comes first, because regret is flat wherever the curve
is flat and gives a fresh network no direction.

The exported artifact is checked against the pure-Python reference forward pass before it is
written, so what ships is exactly what was evaluated.

Usage: python -m warm_starts.train --cases work/datasets/v1/cases.jsonl.zst
         --output models/two-pool-v1.json --report reports/train-v1.json
"""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from . import dataset, model
from .features import FEATURE_NAMES, FEATURE_SPEC_VERSION

GRID_STEPS = 64


def regret_curve(case):
    """Basis points lost at each grid split k = 0..64; infeasible splits get the worst feasible."""
    grid = [[None if v is None else int(v) for v in pool] for pool in case["grid"]]
    best = int(case["label_gross"])
    curve = []
    for k in range(GRID_STEPS + 1):
        first, second = grid[0][k], grid[1][GRID_STEPS - k]
        curve.append(
            None if first is None or second is None else 1e4 * (best - (first + second)) / best
        )
    worst = max(value for value in curve if value is not None)
    return [worst if value is None else value for value in curve]


def load(path):
    splits = {}
    for case in dataset.read_jsonl_zst(path):
        rows = splits.setdefault(case["split"], {"x": [], "regret": [], "k": [], "interp": []})
        rows["x"].append(case["features"])
        rows["regret"].append(regret_curve(case))
        rows["k"].append(case["label_k"])
        rows["interp"].append(case["interpolation_k"])
    return {
        name: {
            key: np.array(value, dtype=float) if key != "interp" else value
            for key, value in rows.items()
        }
        for name, rows in splits.items()
    }


def regret_at(curve, share):
    """Interpolated regret (bp) and its slope in share, per case."""
    position = np.clip(share, 0.0, 1.0) * GRID_STEPS
    left = np.minimum(np.floor(position).astype(int), GRID_STEPS - 1)
    rows = np.arange(len(share))
    low, high = curve[rows, left], curve[rows, left + 1]
    return low + (high - low) * (position - left), (high - low) * GRID_STEPS


class Mlp:
    def __init__(self, sizes, rng):
        self.weights = [
            rng.normal(0, np.sqrt(1 / fan_in), (fan_out, fan_in))
            for fan_in, fan_out in pairwise(sizes)
        ]
        self.biases = [np.zeros(fan_out) for fan_out in sizes[1:]]

    def forward(self, x):
        activations = [x]
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ weights.T + bias
            activations.append(z if index == len(self.weights) - 1 else np.tanh(z))
        return activations

    def backward(self, activations, grad_logit):
        grads = []
        grad = grad_logit[:, None]
        for index in reversed(range(len(self.weights))):
            grads.append((grad.T @ activations[index], grad.sum(axis=0)))
            if index:
                grad = (grad @ self.weights[index]) * (1 - activations[index] ** 2)
        return list(reversed(grads))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(data, hidden, epochs, warmup_epochs, batch, learning_rate, seed):
    rng = np.random.default_rng(seed)
    train_x = data["train"]["x"]
    inputs = [i for i in range(train_x.shape[1]) if train_x[:, i].std() > 0]
    mean, std = train_x[:, inputs].mean(axis=0), train_x[:, inputs].std(axis=0)
    normalize = lambda x: (x[:, inputs] - mean) / std
    net = Mlp([len(inputs), *hidden, 1], rng)
    params = [p for pair in zip(net.weights, net.biases) for p in pair]
    moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    x, curve, target = normalize(train_x), data["train"]["regret"], data["train"]["k"] / GRID_STEPS
    step = 0
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            rows = order[start : start + batch]
            activations = net.forward(x[rows])
            share = sigmoid(activations[-1][:, 0])
            if epoch < warmup_epochs:
                grad_share = 2 * (share - target[rows])
            else:
                _, grad_share = regret_at(curve[rows], share)
            grads = net.backward(activations, grad_share * share * (1 - share) / len(rows))
            step += 1
            for param, (m, v), grad in zip(params, moments, [g for pair in grads for g in pair]):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad**2
                param -= (
                    learning_rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
    predict = lambda x: sigmoid(net.forward(normalize(x))[-1][:, 0])
    return net, inputs, mean, std, predict


def summary(regret):
    return {
        "mean_bp": float(np.mean(regret)),
        "p95_bp": float(np.percentile(regret, 95)),
        "max_bp": float(np.max(regret)),
        "cases": len(regret),
    }


def evaluate(data, predict):
    report = {}
    for name, rows in data.items():
        share = predict(rows["x"])
        baselines = {
            "learned": share,
            "all_pool1": np.ones(len(share)),
            "all_pool2": np.zeros(len(share)),
        }
        interp = [k for k in rows["interp"] if k is not None]
        report[name] = {
            method: summary(regret_at(rows["regret"], s)[0]) for method, s in baselines.items()
        }
        if len(interp) == len(share):
            report[name]["interpolation"] = summary(
                regret_at(rows["regret"], np.array(interp) / GRID_STEPS)[0]
            )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--hidden", default="32,32")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    hidden = [int(width) for width in args.hidden.split(",")]
    data = load(args.cases)
    net, inputs, mean, std, predict = train(
        data, hidden, args.epochs, args.warmup_epochs, args.batch, args.learning_rate, args.seed
    )
    layers = [(w.tolist(), b.tolist(), "tanh") for w, b in zip(net.weights[:-1], net.biases[:-1])]
    layers.append((net.weights[-1].tolist(), net.biases[-1].tolist(), "identity"))
    artifact = model.artifact(FEATURE_SPEC_VERSION, inputs, mean, std, layers, {"kind": "share"})

    # The reference forward pass defines what ships; numpy only differs in summation order.
    check = data.get("validation", data["train"])["x"][:512]
    reference = np.array([model.share(artifact, list(row)) for row in check])
    drift = float(np.max(np.abs(reference - predict(check))))
    assert drift < 1e-9, f"reference and numpy predictions differ by {drift}"

    report = {
        "config": vars(args),
        "inputs": [FEATURE_NAMES[i] for i in inputs],
        "reference_drift": drift,
        "regret": evaluate(data, predict),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["regret"], indent=2))


if __name__ == "__main__":
    main()
