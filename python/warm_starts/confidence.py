"""Train the allocation-regret estimator for the neural initializer's approximate starting loss.

Cross-fitting groups related training states before fitting offset models, so each regret
label measures an offset model on a group excluded from that model's training. Labels interpolate
sampled regret curves; they are not completed Fynd solves.

An upper-quantile loss favors conservative estimates. A training envelope limits extrapolation,
and validation chooses a threshold capped by the configured regret margin. If no threshold
passes, the artifact disables skipping. The optional hand rule is a separate appendix baseline.

The offline checked-skip diagnostic uses sampled net curves. Rust instead quotes rounded
allocations and checks a gross-output bound; these are different policies. The released rule's
final net-output quality must be evaluated through the Rust runner, not inferred from this proxy.

Rows are grouped into cells by market group, direction, and size. Cells can share market states
and are not established independent statistical samples. Test diagnostics run only when requested.

Usage: python -m warm_starts.confidence --cases CASES --offset-dir MODEL_DIR [--score-test]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import model
from .features import OFFSET_SPEC_VERSION
from .select_offset import BATCH, CHUNKS, HIDDEN
from .train import Mlp
from .train_offset import load, regret_of, train

MARGIN_BP = 0.02
MAX_SKIPPED_REGRET_BP = 0.05
MAX_SHARE_ABOVE_MARGIN = 0.01
REGRET_FLOOR_BP = 1e-4
# Finite (the Rust loader requires it) and below any estimate: skipping off.
NEVER_SKIP_LOG10_BP = -100.0
QUANTILE = 0.9
CONFIDENCE_HIDDEN = [32, 32]
CONFIDENCE_EPOCHS, CONFIDENCE_LEARNING_RATE = 300, 1e-3
# Features bounded by the training range: log10 notional, coarse share, full-order price gap.
ENVELOPE_FEATURES = [1, 2, 3]
QUADRATIC_FEATURE = 6
# The quadratic's move saturates at the reach when the best split lies beyond it.
QUADRATIC_BOUND_CHUNKS = 0.99
NOTIONAL_FEATURE = 1
CERTIFIED_MAX_BP = 0.02
CERTIFICATE_STEP_DIVISORS = [256, 512, 1024, 2048, 4096]


def artifact_predict(artifact, x):
    """The artifact's forward pass in numpy (summation order aside, the reference's)."""
    activations = (x[:, artifact["inputs"]] - artifact["mean"]) / artifact["std"]
    for layer in artifact["layers"]:
        z = activations @ np.array(layer["weights"]).T + np.array(layer["bias"])
        activations = np.tanh(z) if layer["activation"] == "tanh" else z
    return activations[:, 0]


def offset_shares(artifact, x):
    squashed = 1 / (1 + np.exp(-artifact_predict(artifact, x)))
    reach = artifact["output"]["chunks"] / 20
    return np.clip(x[:, 2] + (2 * squashed - 1) * reach, 0.0, 1.0)


def subset(rows, mask):
    return {
        key: (value[mask] if isinstance(value, np.ndarray) else None)
        for key, value in rows.items()
        if key != "curves"
    } | {"curves": [c for c, keep in zip(rows["curves"], mask, strict=True) if keep]}


def cross_fitted_regret(data, chosen, folds):
    """Each train case's regret under an offset model trained without its capture group."""
    rows = data["train"]
    groups = sorted(set(rows["group"].tolist()))
    fold_groups = np.array_split(np.array(groups), folds)
    regret = np.full(len(rows["label"]), np.nan)
    for held_out in fold_groups:
        held = np.isin(rows["group"], held_out)
        fitted = train(
            {"train": subset(rows, ~held)},
            HIDDEN,
            CHUNKS,
            chosen["epochs"],
            BATCH,
            chosen["learning_rate"],
            chosen["seed"],
        )
        predict = fitted[4]
        regret[held] = regret_of(
            predict(rows["x"][held]), [c for c, h in zip(rows["curves"], held, strict=True) if h]
        )
    return regret, [[str(g) for g in fold] for fold in fold_groups]


def fit_regression(x, target, seed):
    """A tanh MLP fitting `target`'s `QUANTILE` quantile; returns inputs, normalization, layers."""
    rng = np.random.default_rng(seed)
    inputs = [i for i in range(x.shape[1]) if x[:, i].std() > 0]
    mean, std = x[:, inputs].mean(axis=0), x[:, inputs].std(axis=0)
    normalized = (x[:, inputs] - mean) / std
    net = Mlp([len(inputs), *CONFIDENCE_HIDDEN, 1], rng)
    params = [p for pair in zip(net.weights, net.biases, strict=True) for p in pair]
    moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    step = 0
    for _ in range(CONFIDENCE_EPOCHS):
        order = rng.permutation(len(normalized))
        for start in range(0, len(normalized), BATCH):
            rows = order[start : start + BATCH]
            activations = net.forward(normalized[rows])
            under = activations[-1][:, 0] < target[rows]
            grad_logit = np.where(under, -QUANTILE, 1 - QUANTILE) / len(rows)
            grads = net.backward(activations, grad_logit)
            step += 1
            flat = [g for pair in grads for g in pair]
            for param, (m, v), grad in zip(params, moments, flat, strict=True):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad**2
                param -= (
                    CONFIDENCE_LEARNING_RATE
                    * (m / (1 - 0.9**step))
                    / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                )
    layers = [
        (w.tolist(), b.tolist(), "tanh")
        for w, b in zip(net.weights[:-1], net.biases[:-1], strict=True)
    ]
    layers.append((net.weights[-1].tolist(), net.biases[-1].tolist(), "identity"))
    return inputs, mean, std, layers


def envelope_of(x, feature_count):
    low, high = [None] * feature_count, [None] * feature_count
    for index in ENVELOPE_FEATURES:
        low[index], high[index] = float(x[:, index].min()), float(x[:, index].max())
    low[QUADRATIC_FEATURE] = -QUADRATIC_BOUND_CHUNKS * CHUNKS
    high[QUADRATIC_FEATURE] = QUADRATIC_BOUND_CHUNKS * CHUNKS
    return {"low": low, "high": high}


def inside(x, envelope):
    keep = np.ones(len(x), dtype=bool)
    for index, (low, high) in enumerate(zip(envelope["low"], envelope["high"], strict=True)):
        if low is not None:
            keep &= low <= x[:, index]
        if high is not None:
            keep &= x[:, index] <= high
    return keep


def acceptable(regret, cells):
    """No skipped order above the hard cap, and few skipped cells with an order above the margin."""
    failing = len(set(cells[regret > MARGIN_BP].tolist()))
    return regret.max() <= MAX_SKIPPED_REGRET_BP and failing <= MAX_SHARE_ABOVE_MARGIN * len(
        set(cells.tolist())
    )


def largest_threshold(scores, eligible, regret, cells):
    """Largest score cut whose skipped set (eligible and score <= cut) passes, scanning upward."""
    best = None
    for cut in np.unique(scores[eligible]):
        skipped = eligible & (scores <= cut)
        if not acceptable(regret[skipped], cells[skipped]):
            break
        best = float(cut)
    return best


def highest_passing_cut(scores, eligible, regret, cells):
    """Highest score cut whose skipped set passes, anywhere in the scan. Reported only: the
    per-cell share is not monotone in the cut (one failing cell among the first few skipped
    fails it), so when the upward scan stops early this shows whether a later cut would pass."""
    passing = [
        float(cut)
        for cut in np.unique(scores[eligible])
        if acceptable(regret[eligible & (scores <= cut)], cells[eligible & (scores <= cut)])
    ]
    return max(passing, default=None)


def certified_bound(shares, curves, divisor):
    """Approximate certificate score used by the original validation selection.

    Interpolated net curves differ from Rust's exact gross quotes and rounded allocations;
    this is not a runtime replay or a proof of net regret. Held-out evaluation uses sar-bench."""
    step = 1 / divisor
    at = regret_of(shares, curves)
    left = regret_of(np.clip(shares - step, 0.0, 1.0), curves)
    right = regret_of(np.clip(shares + step, 0.0, 1.0), curves)
    bound = np.maximum(left - at, right - at)
    return np.where((left < at) | (right < at), np.inf, bound)


def skip_summary(skipped, regret, cells):
    """Skip rate and what the skipped orders lose, per row and per cell."""
    lost = regret[skipped]
    skipped_cells = set(cells[skipped].tolist())
    failing_cells = set(cells[skipped & (regret > MARGIN_BP)].tolist())
    return {
        "skip_rate": float(skipped.mean()),
        "rows": int(skipped.size),
        "skipped_rows": int(skipped.sum()),
        "cells": len(set(cells.tolist())),
        "skipped_cells": len(skipped_cells),
        "worst_skipped_bp": float(lost.max()) if lost.size else None,
        "skipped_above_margin": int(np.sum(lost > MARGIN_BP)),
        "skipped_above_0.05bp": int(np.sum(lost > MAX_SKIPPED_REGRET_BP)),
        "skipped_above_gate_0.1bp": int(np.sum(lost > 0.1)),
        "skipped_cells_with_a_row_above_margin": len(failing_cells),
    }


def auc(scores, positive):
    """Probability a random positive scores above a random negative (ties count half)."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    positives, negatives = positive.sum(), (~positive).sum()
    if not positives or not negatives:
        return None
    return float(
        (ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--offset-dir", required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score-test", action="store_true")
    args = parser.parse_args()
    data = load(args.cases)
    offset_dir = Path(args.offset_dir)
    chosen = json.loads((offset_dir / "report.json").read_text())["chosen"]
    offset = model.load(offset_dir / "offset.json")

    oof_regret, folds = cross_fitted_regret(data, chosen, args.folds)
    train_x = data["train"]["x"]
    target = np.log10(np.maximum(oof_regret, 0.0) + REGRET_FLOOR_BP)
    inputs, mean, std, layers = fit_regression(train_x, target, args.seed)
    envelope = envelope_of(train_x, train_x.shape[1])
    # The threshold is filled in once validation sets it.
    output = {"kind": "regret_estimate", "skip_at_most_log10_bp": None, "envelope": envelope}
    artifact = model.artifact(OFFSET_SPEC_VERSION, inputs, mean, std, layers, output)

    splits = ["validation", *(["test"] if args.score_test else [])]
    scored = {}
    for name in ["train", *splits]:
        if name not in data:
            continue
        rows = data[name]
        # On train, the final offset model's regret is in sample; the cross-fitted regret is what
        # the confidence model learned from (so there it is in sample instead).
        regret = (
            oof_regret
            if name == "train"
            else regret_of(offset_shares(offset, rows["x"]), rows["curves"])
        )
        scored[name] = {
            "regret": regret,
            "estimate": artifact_predict(artifact, rows["x"]),
            "inside": inside(rows["x"], envelope),
        }

    validation = scored["validation"]
    validation_cells = data["validation"]["cell"]
    passing_threshold = largest_threshold(
        validation["estimate"], validation["inside"], validation["regret"], validation_cells
    )
    margin_cap = float(np.log10(MARGIN_BP + REGRET_FLOOR_BP))
    threshold = (
        NEVER_SKIP_LOG10_BP if passing_threshold is None else min(passing_threshold, margin_cap)
    )
    notional = {name: data[name]["x"][:, NOTIONAL_FEATURE] for name in scored}
    quadratic_free = {
        name: np.abs(data[name]["x"][:, QUADRATIC_FEATURE]) < QUADRATIC_BOUND_CHUNKS * CHUNKS
        for name in scored
    }
    notional_cap = largest_threshold(
        notional["validation"], quadratic_free["validation"], validation["regret"], validation_cells
    )

    # Certified variant: the largest-skip step whose certified skipped set passes the rule.
    learned_skip = validation["inside"] & (validation["estimate"] <= threshold)
    shares = {name: offset_shares(offset, data[name]["x"]) for name in scored}
    certificate = []
    for divisor in CERTIFICATE_STEP_DIVISORS:
        bound = certified_bound(shares["validation"], data["validation"]["curves"], divisor)
        skipped = learned_skip & (bound <= CERTIFIED_MAX_BP)
        certificate.append(
            {
                "divisor": divisor,
                "passes": bool(
                    skipped.any()
                    and acceptable(validation["regret"][skipped], validation_cells[skipped])
                ),
                "skip_rate": float(skipped.mean()),
            }
        )
    passing = [c for c in certificate if c["passes"]]
    certified_divisor = max(passing, key=lambda c: c["skip_rate"])["divisor"] if passing else None

    output["skip_at_most_log10_bp"] = threshold
    check = data["validation"]["x"][:512]
    reference = np.array([model.regret_estimate(artifact, list(row))[0] for row in check])
    drift = float(np.max(np.abs(reference - artifact_predict(artifact, check))))
    assert drift < 1e-9, f"reference and numpy estimates differ by {drift}"
    model.save(artifact, offset_dir / "confidence.json")

    report = {
        "cases": args.cases,
        "offset_model": str(offset_dir / "offset.json"),
        "rule": {
            "margin_bp": MARGIN_BP,
            "max_skipped_regret_bp": MAX_SKIPPED_REGRET_BP,
            "max_share_above_margin": MAX_SHARE_ABOVE_MARGIN,
        },
        "folds": folds,
        "threshold_log10_bp": threshold,
        "threshold_bp": 10**threshold - REGRET_FLOOR_BP,
        # The rule's own cut before the margin cap; null when none passed (skipping off).
        "passing_threshold_log10_bp": passing_threshold,
        "highest_passing_cut_log10_bp": highest_passing_cut(
            validation["estimate"], validation["inside"], validation["regret"], validation_cells
        ),
        "skipping_off": passing_threshold is None,
        "envelope": envelope,
        "reference_drift": drift,
        "test_scored": args.score_test,
        "cross_fitted_train_regret_bp": {
            "median": float(np.median(oof_regret)),
            "p90": float(np.percentile(oof_regret, 90)),
            "p99": float(np.percentile(oof_regret, 99)),
            "share_above_margin": float(np.mean(oof_regret > MARGIN_BP)),
        },
        "hand_rule": {"notional_cap_log10_usd": notional_cap},
        "certificate": {
            "max_bp": CERTIFIED_MAX_BP,
            "validation_by_step_divisor": certificate,
            "chosen_step_divisor": certified_divisor,
        },
        # train: cross-fitted offset regret, confidence in sample; validation set the threshold.
        "splits": {},
    }
    for name, values in scored.items():
        rows = data[name]
        learned = values["inside"] & (values["estimate"] <= threshold)
        hand = (
            quadratic_free[name] & (notional[name] <= notional_cap)
            if notional_cap is not None
            else np.zeros(len(learned), dtype=bool)
        )
        certified = (
            learned
            & (certified_bound(shares[name], rows["curves"], certified_divisor) <= CERTIFIED_MAX_BP)
            if certified_divisor is not None
            else np.zeros(len(learned), dtype=bool)
        )
        report["splits"][name] = {
            "estimate_auc_for_regret_above_margin": auc(
                values["estimate"], values["regret"] > MARGIN_BP
            ),
            "inside_envelope": float(values["inside"].mean()),
            "confidence": skip_summary(learned, values["regret"], rows["cell"]),
            "hand_rule": skip_summary(hand, values["regret"], rows["cell"]),
            "certified": skip_summary(certified, values["regret"], rows["cell"]),
        }
    (offset_dir / "confidence-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("threshold_bp", "hand_rule", "certificate", "splits")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
