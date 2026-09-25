"""Prints the report's markdown tables from a bench summary, its stage counts and a test report.

Each table carries its method legend, following the reporting convention: seedless Fynd, the
coarse split and our model in the main table, the quadratic in the appendix.

Usage: python scripts/report_tables.py SUMMARY.json STAGES.json [TEST_REPORT.json]
"""

import json
import sys
from pathlib import Path

from summarize_bench import METHODS

LABELS = {
    "water_fill": "Patched Fynd without a seed",
    "coarse": "Coarse split",
    "offset": "Neural allocation initializer",
    "offset_confident": "Prediction-only skip (disabled)",
    "offset_certified": "Learned warm start with checked skip",
    "quadratic": "Appendix: quadratic formula",
}
MAIN = ["coarse", "offset", "offset_certified"]
APPENDIX = ["offset_confident", "quadratic"]


def bp(value):
    return f"{value:+.4f}"


def bound(share):
    return "n/a" if share is None else f"< {share:.1%}"


def bench_rows(summary, stages, methods):
    lines = []
    for method in methods:
        if method not in summary:
            continue
        row, sims = summary[method], stages.get(method, {})
        if "delta_bp" not in row:
            # No solve where the seed ran and both finished: only failures can be reported.
            lines.append(
                f"| {LABELS[method]} | {row['failed_where_fynd_solved']} "
                f"| no seeded solves ({row['declined']} declined) |" + " - |" * 8
            )
            continue
        delta = row["delta_bp"]
        lines.append(
            f"| {LABELS[method]} | {row['gate_failures']} | {bp(delta['worst'])} "
            f"| {bp(delta['p1'])} | {bp(delta['p50'])} | {bp(delta['p90'])} | {bp(delta['p99'])} "
            f"| {bp(delta['mean'])} "
            f"| {' / '.join(map(str, row['better_equal_worse']))} "
            f"| {row['time_ratio_median']}, {row['time_ratio_p90']} "
            f"| {sims.get('total_mean', '?')}, {sims.get('total_median', '?')} |"
        )
    return lines


def bench_tables(summary, stages):
    header = [
        (
            "| Method | Gate failures | Worst | p1 | p50 | p90 | p99 | Mean | Better / equal / worse "
            "| Time vs Fynd (median, p90) | Simulations (mean, median) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    fynd = stages.get("water_fill", {})
    fynd_row = (
        f"| {LABELS['water_fill']} | - | - | - | - | - | - | - | - | 1, 1 "
        f"| {fynd.get('total_mean', '?')}, {fynd.get('total_median', '?')} |"
    )
    lines = [*header, fynd_row, *bench_rows(summary, stages, MAIN)]
    lines += ["", "Appendix:", "", *header, *bench_rows(summary, stages, APPENDIX)]
    lines += [
        "",
        "Skipped solves only:",
        "",
        (
            "| Method | Skip rate | Solves | Gate failures | Cells | Cells above gate "
            "| 95 % bound on cells above gate | Worst | p1 | p50 | Returned candidate |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method in ("offset_confident", "offset_certified"):
        skipped = summary.get(method, {}).get("skipped")
        if not skipped or not skipped.get("solves"):
            continue
        delta = skipped["delta_bp"]
        winners = ", ".join(f"{k} {v}" for k, v in sorted(skipped["winner"].items()))
        lines.append(
            f"| {LABELS[method]} | {skipped['rate']:.1%} | {skipped['solves']} "
            f"| {skipped['gate_failures']} | {skipped['cells']} | {skipped['cells_above_gate']} "
            f"| {bound(skipped['cells_above_gate_bound_95'])} "
            f"| {bp(delta['worst'])} | {bp(delta['p1'])} "
            f"| {bp(delta['p50'])} | {winners} |"
        )
    lines += ["", "Legend:", ""]
    lines += [
        f"- {LABELS[m]}: {METHODS.get(m, summary[m]['what'])}"
        for m in [*MAIN, *APPENDIX]
        if m in summary
    ]
    lines += [
        "",
        "Cell bounds assume independence that these captures do not establish; treat them as descriptive.",
    ]
    return "\n".join(lines)


def test_tables(report):
    lines = [
        (
            "Original interpolated estimates against sampled net-output labels. These do not replay "
            "Rust certificate decisions or measure final output against Fynd."
        ),
        "",
        "| Start | Median | p90 | p99 | Max | Within 0.02 bp | Within 0.1 bp | Cells |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, label in (
        ("coarse", "Coarse split"),
        ("offset", LABELS["offset"]),
        ("quadratic", "Appendix: quadratic formula"),
    ):
        r = report["start_regret"][name]
        lines.append(
            f"| {label} | {r['median_bp']:.4f} | {r['p90_bp']:.4f} | {r['p99_bp']:.4f} "
            f"| {r['max_bp']:.3f} | {r['under_0.02bp']:.1%} | {r['under_0.1bp']:.1%} "
            f"| {r['cells']} |"
        )
    lines += [
        "",
        (
            "| Skip rule | Skip rate | Skipped cells | Worst skipped (bp) "
            "| Cells with an order above 0.02 | Orders above 0.02 | Orders above 0.05 "
            "| Orders above 0.1 | 95 % bound on cells above 0.1 bp |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, label in (
        ("confidence", "Confident skip"),
        ("certified", "Certified skip (offline proxy)"),
        ("hand_rule", "Appendix: hand rule"),
    ):
        s = report["skip"][name]
        cells = s["skipped_cells"]
        # Rule of three: 0 failures in n independent cells bounds the rate below 3/n at 95 %.
        bound = f"< {3 / cells:.1%}" if cells and s["skipped_above_gate_0.1bp"] == 0 else "n/a"
        worst = "-" if s["worst_skipped_bp"] is None else f"{s['worst_skipped_bp']:.4f}"
        lines.append(
            f"| {label} | {s['skip_rate']:.1%} | {cells}/{s['cells']} | {worst} "
            f"| {s['skipped_cells_with_a_row_above_margin']} "
            f"| {s['skipped_above_margin']} | {s['skipped_above_0.05bp']} "
            f"| {s['skipped_above_gate_0.1bp']} | {bound} |"
        )
    lines += ["", "Cell bounds are descriptive; different sizes share the same market states."]
    return "\n".join(lines)


def runtime_test_table(report):
    runtime = report["runtime"]
    rate = runtime["skip_rate"]
    skip = "n/a" if rate is None else f"{rate:.1%}"
    worst = runtime["worst_final_delta_bp"]
    worst_skipped = runtime["worst_skipped_final_delta_bp"]
    return "\n".join(
        [
            "### Test through the frozen Rust model",
            "",
            (
                "The seed and certificate run in sar-bench. Deltas compare completed net output with "
                "patched Fynd without a seed; positive means more output. Skip rate counts eligible "
                "two-pool orders, and the gate includes every paired case and candidate failure."
            ),
            "",
            "| Solves | Successful pairs | Eligible | Skipped | Skip rate | Gate failures | Worst delta | Worst skipped delta |",
            "|---|---|---|---|---|---|---|---|",
            (
                f"| {runtime['solves']} | {runtime['successful_pairs']} | {runtime['seeded_solves']} "
                f"| {runtime['skipped_solves']} | {skip} | {runtime['gate_failures']} "
                f"| {bp(worst) if worst is not None else 'n/a'} bp "
                f"| {bp(worst_skipped) if worst_skipped is not None else 'n/a'} bp |"
            ),
            "",
            (
                f"Reference failures: {runtime['reference_failures']}. "
                f"Gate passed: {runtime['gate_passed']}."
            ),
        ]
    )


def main():
    summary_path, stages_path, *test = sys.argv[1:]
    summary = json.loads(Path(summary_path).read_text())
    stages = json.loads(Path(stages_path).read_text())
    print(bench_tables(summary, stages))
    if test:
        print()
        report = json.loads(Path(test[0]).read_text())
        print(runtime_test_table(report) if "runtime" in report else test_tables(report))


if __name__ == "__main__":
    main()
