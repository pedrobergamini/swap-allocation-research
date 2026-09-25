"""Scores sar-bench methods against seedless Water-fill on the same solves.

Each method's solve is paired with `water_fill` on the same block, case and repeat. Reported per
method, over solves where the seed ran (two active paths):

- gate failures: solves whose net output falls more than 0.1 bp below Water-fill's, plus any
  solve that fails where Water-fill succeeds, even before the seed runs;
- output delta vs Water-fill in bp (worst, p1, p50, p90, p99, mean; positive = more output) and how
  many solves came out better, equal or worse;
- time ratio (method / water_fill elapsed) at the median and p90;
- start distance: how far the seed's proposal sits from Water-fill's final split, as a percent of
  the order;
- for seeds that may skip exchange refinement: the skip rate, and over skipped solves alone the
  output delta and which candidate Fynd returned (the seed's start, the coarse split, one pool
  only, or another split, such as fill-and-spill's). Solves of one order across blocks and
  repeats are not independent, so skipped solves are also counted per cell (one order, i.e. one
  case in one recording), with the rule-of-three bound on the share of cells above the gate.

A seed is asked once per fine pass it replaces, the disjoint split first and then fill-and-spill;
start distance reads the first record, the one exchange refinement starts from.

Usage: python scripts/summarize_bench.py BENCH.jsonl [BENCH.jsonl ...]
"""

import json
import sys
from collections import defaultdict

import numpy as np

GATE_BP = 0.1
COARSE_SHARE_FEATURE = 2
# What each method does in place of Water-fill's 256-chunk fine pass; printed with the numbers so
# every table carries its own legend.
METHODS = {
    "water_fill": "patched Fynd without a seed: runs both 256-chunk passes (reference)",
    "coarse": "non-learned: starts refinement from Fynd's own 20-chunk split, 0 extra simulations",
    "interpolation": "non-learned: best split between 4 quotes per pool, interpolated",
    "offset": "learned: nudges the 20-chunk split by up to its reach in chunks, then refines as "
    "Fynd does, 0 extra simulations",
    "offset_confident": "learned, with the allocation-regret estimator: as `offset`, and on the disjoint split it "
    "skips exchange refinement when its regret estimate is at most the frozen threshold and the "
    "order is inside the training envelope",
    "offset_certified": "learned: as `offset_confident`, "
    "but skips only when quotes one step either side pass a 0.02 bp gross-output bound; net loss is measured empirically "
    "(6 extra simulations when checked)",
    "quadratic": "appendix baseline, non-learned: fits a parabola per pool to what the 20-chunk pass's "
    "last chunks added, moves to where the two sum highest, 0 extra simulations",
    "learned": "learned model (probe-impact): predicts the share from 4 probes per pool",
    "reference": "diagnostic: best of 257 exact grid splits, deliberately expensive",
}


def read(paths):
    rows = []
    for path in paths:
        with open(path) as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    return rows


def winner(row, seed):
    """Which candidate a skipped solve returned, read off its final split."""
    amounts = [int(amount) for amount in row["pool_amounts"]]
    # Fynd rebuilds the legs from split ratios, which rounds the proposal by a few units.
    tolerance = int(row["amount_in"]) // 10**12 + 1
    proposal = [int(amount) for amount in seed["proposal"]]
    if all(abs(a - p) <= tolerance for a, p in zip(amounts, proposal, strict=True)):
        return "start"
    if 0 in amounts:
        return "one_pool"
    coarse = seed["features"][COARSE_SHARE_FEATURE]
    if abs(amounts[0] / int(row["amount_in"]) - coarse) < 1e-9:
        return "coarse"
    return "other"


def skipped_summary(loss_bp, cases, winners, seeded):
    if not loss_bp:
        return {"rate": 0.0, "solves": 0}
    loss_bp = np.array(loss_bp)
    worst_by_cell = defaultdict(lambda: -np.inf)
    for case, loss in zip(cases, loss_bp, strict=True):
        worst_by_cell[case] = max(worst_by_cell[case], loss)
    cells = len(worst_by_cell)
    cells_above_gate = int(sum(loss > GATE_BP for loss in worst_by_cell.values()))
    return {
        "rate": round(len(loss_bp) / seeded, 4),
        "solves": len(loss_bp),
        "gate_failures": int(np.sum(loss_bp > GATE_BP)),
        "delta_bp": {
            "worst": float(-loss_bp.max()),
            **{f"p{q}": float(np.percentile(-loss_bp, q)) + 0.0 for q in (1, 50)},
            "mean": float(-loss_bp.mean()) + 0.0,
        },
        "better_equal_worse": [
            int(np.sum(loss_bp < 0)),
            int(np.sum(loss_bp == 0)),
            int(np.sum(loss_bp > 0)),
        ],
        "winner": dict(winners),
        "cells": cells,
        "cells_above_gate": cells_above_gate,
        # Rule of three: 0 failing cells out of n bounds the failing share below 3/n at 95 %.
        "cells_above_gate_bound_95": round(3 / cells, 4) if cells_above_gate == 0 else None,
    }


def main():
    rows = read(sys.argv[1:])
    baseline = {
        (row["block"], row["case_id"], row["repeat"]): row
        for row in rows
        if row["method"] == "water_fill"
    }
    by_method = defaultdict(list)
    for row in rows:
        if row["method"] != "water_fill":
            by_method[row["method"]].append(row)

    summary = {}
    for method, method_rows in by_method.items():
        ratios, loss_bp, distance, declined, not_ok = [], [], [], 0, 0
        skipped_loss_bp, skipped_cases, winners = [], [], defaultdict(int)
        for row in method_rows:
            reference = baseline[(row["block"], row["case_id"], row["repeat"])]
            if row["status"] != "ok" or reference["status"] != "ok":
                # Failures can happen before the seed records a proposal.
                not_ok += row["status"] != "ok" and reference["status"] == "ok"
                continue
            seed = row["seeds"][0] if row["seeds"] else None
            if seed is None or seed["proposal"] is None:
                declined += 1
                continue
            ratios.append(row["elapsed_us"] / reference["elapsed_us"])
            reference_net, net = int(reference["net_out"]), int(row["net_out"])
            loss_bp.append((reference_net - net) / reference_net * 1e4)
            amount = int(row["amount_in"])
            final_first = int(reference["pool_amounts"][0])
            distance.append(abs(int(seed["proposal"][0]) - final_first) / amount * 100)
            if seed.get("skipped"):
                skipped_loss_bp.append(loss_bp[-1])
                skipped_cases.append(row["case_id"])
                winners[winner(row, seed)] += 1
        if not ratios:
            summary[method] = {
                "what": METHODS.get(method, "?"),
                "seeded": 0,
                "declined": declined,
                "failed_where_fynd_solved": not_ok,
                "gate_failures": not_ok,
            }
            continue
        loss_bp, ratios, distance = map(np.array, (loss_bp, ratios, distance))
        summary[method] = {
            "what": METHODS.get(method, "?"),
            "seeded": len(ratios),
            "declined": declined,
            "failed_where_fynd_solved": not_ok,
            "gate_failures": int(np.sum(loss_bp > GATE_BP)) + not_ok,
            # Output delta vs water_fill in bp; positive means more output than Fynd.
            "delta_bp": {
                "worst": float(-loss_bp.max()),
                **{f"p{q}": float(np.percentile(-loss_bp, q)) + 0.0 for q in (1, 50, 90, 99)},
                # Outputs are in different tokens per direction, so this averages per-solve bp.
                "mean": float(-loss_bp.mean()) + 0.0,
            },
            "better_equal_worse": [
                int(np.sum(loss_bp < 0)),
                int(np.sum(loss_bp == 0)),
                int(np.sum(loss_bp > 0)),
            ],
            "time_ratio_median": round(float(np.median(ratios)), 3),
            "time_ratio_p90": round(float(np.percentile(ratios, 90)), 3),
            "start_distance_median_pct": round(float(np.median(distance)), 3),
            "start_distance_p90_pct": round(float(np.percentile(distance, 90)), 3),
        }
        if any("skipped" in (row["seeds"] or [{}])[0] for row in method_rows):
            summary[method]["skipped"] = skipped_summary(
                skipped_loss_bp, skipped_cases, winners, len(ratios)
            )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
