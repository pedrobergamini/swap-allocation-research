"""Chooses the v1 coarse-offset model on validation, and scores it on test only when asked.

Trains every training configuration in a small grid, each with several seeds, on the train split
and scores it on validation. Cases built from recordings carry sampled regret curves; there a
trial's score is the approximate validation net regret against the sampled label (bp)
above the least any start within the model's reach could have (`reachable_floor`), averaged over
the worst 5 % of cells, a cell being one capture, direction and size, whose blocks are
near-duplicates. The floor matters for small orders whose best split lies several chunks from the
coarse split: their regret is the reach's, not the model's, and would otherwise be most of the
tail and the same for every trial. Without curves it is the mean distance in exchange refinement's first step size
(1/256 of the order).

Seeds of one configuration differ by about as much as configurations do, so the configuration is
chosen by its mean score over seeds, and the exported model is its median seed, not its best,
which would carry that seed's luck into the reported numbers. Test cases are scored only with
`--score-test`, which the final run passes once. The Fynd bench is the final judge.

Usage: python -m warm_starts.select_offset --cases CASES --output-dir work/models/v1 [--score-test]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from . import model
from .features import OFFSET_FEATURE_NAMES
from .train_offset import (
    COARSE_CHUNKS,
    cell_cvar5,
    export,
    load,
    reachable_floor,
    regret_of,
    start_error,
    start_errors,
    start_regret,
    start_regrets,
    train,
)

HIDDEN = [32, 32]
CHUNKS = 1.0
BATCH = 256
EPOCHS = [300, 1000]
LEARNING_RATES = [1e-3, 3e-3]
SEEDS = range(10)


def score(data, predict, floor):
    validation = data["validation"]
    shares = predict(validation["x"])
    distance = start_error(shares, validation["label"])
    if validation["curves"] is None:
        return distance["mean_steps"], distance
    regret = start_regret(shares, validation["curves"], validation["cell"])
    excess, _ = cell_cvar5(regret_of(shares, validation["curves"]) - floor, validation["cell"])
    return excess, {**distance, "regret": regret, "excess_cell_cvar5_bp": excess}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-test", action="store_true")
    args = parser.parse_args()
    data = load(args.cases)
    if "validation" not in data:
        raise SystemExit("no validation cases; nothing to select on")

    floor = (
        reachable_floor(
            data["validation"]["curves"],
            data["validation"]["x"][:, 2],
            CHUNKS / COARSE_CHUNKS,
        )
        if data["validation"]["curves"] is not None
        else None
    )
    configs = []
    for epochs, learning_rate in itertools.product(EPOCHS, LEARNING_RATES):
        trials = []
        for seed in SEEDS:
            fitted = train(data, HIDDEN, CHUNKS, epochs, BATCH, learning_rate, seed)
            value, validation = score(data, fitted[4], floor)
            trials.append(
                {"seed": seed, "score": value, "validation": validation, "fitted": fitted}
            )
        scores = np.array([t["score"] for t in trials])
        configs.append(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "trials": trials,
            }
        )
        print(
            f"epochs {epochs} lr {learning_rate}: validation score {scores.mean():.5f} "
            f"+- {scores.std():.5f} (min {scores.min():.5f}, max {scores.max():.5f})",
            flush=True,
        )

    best = min(configs, key=lambda c: c["score_mean"])
    ranked = sorted(best["trials"], key=lambda t: t["score"])
    chosen = ranked[(len(ranked) - 1) // 2]
    net, inputs, mean, std, predict = chosen["fitted"]
    artifact, drift = export(net, inputs, mean, std, CHUNKS, predict, data["validation"]["x"][:512])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(artifact, output / "offset.json")

    distance_chunks = np.abs(data["train"]["x"][:, 2] - data["train"]["label"]) * COARSE_CHUNKS
    splits = ["train", "validation", *(["test"] if args.score_test else [])]
    report = {
        "cases": args.cases,
        "selection": (
            "configuration with the lowest mean validation score over seeds, then its median "
            "seed; score = "
            + (
                "cell-level CVaR5 of regret above the reachable floor (bp)"
                if data["validation"]["curves"] is not None
                else "mean distance in 1/256 steps"
            )
        ),
        "grid": {
            "hidden": HIDDEN,
            "chunks": CHUNKS,
            "batch": BATCH,
            "epochs": EPOCHS,
            "learning_rates": LEARNING_RATES,
            "seeds": list(SEEDS),
        },
        "configs": [
            {
                **{k: v for k, v in c.items() if k != "trials"},
                "trials": [{k: v for k, v in t.items() if k != "fitted"} for t in c["trials"]],
            }
            for c in configs
        ],
        "chosen": {
            "epochs": best["epochs"],
            "learning_rate": best["learning_rate"],
            "seed": chosen["seed"],
            "score": chosen["score"],
        },
        "inputs": [OFFSET_FEATURE_NAMES[i] for i in inputs],
        "reference_drift": drift,
        "max_train_distance_chunks": float(distance_chunks.max()),
        # Labels the model cannot reach; small orders whose output is flat across many chunks.
        "train_labels_beyond_reach": int(np.sum(distance_chunks > CHUNKS)),
        "cases_per_split": {name: len(rows["label"]) for name, rows in data.items()},
        "test_scored": args.score_test,
        "start_error": start_errors(data, predict, splits),
        "start_regret": start_regrets(data, predict, splits),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("chosen", "start_error", "start_regret")}, indent=1))


if __name__ == "__main__":
    main()
