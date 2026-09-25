"""Render the public report from reviewed aggregate results, without private inputs."""

import argparse
import html
import json
from pathlib import Path

from build_launch_visuals import build_performance, build_results

ROOT = Path(__file__).resolve().parents[1]


def build(results, trial):
    markdown = ["# Learned warm starts for swap allocation", ""]
    document = []

    def heading(text):
        markdown.extend([f"## {text}", ""])
        document.append(f"<h2>{html.escape(text)}</h2>")

    def paragraph(text):
        markdown.extend([text, ""])
        document.append(f"<p>{html.escape(text)}</p>")

    def table(headers, rows):
        markdown.extend(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        )
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        markdown.append("")
        head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
            for row in rows
        )
        document.append(
            f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        )

    fresh = results["benchmarks"]["fresh"]
    methods = fresh["methods"]
    checked = methods["offset_certified"]
    runtime = results["held_out_runtime"]
    coarse_time = methods["coarse"]["all"]["time_ratio_sum"]
    learned_time = methods["offset"]["all"]["time_ratio_sum"]
    checked_time = checked["all"]["time_ratio_sum"]
    heading("Results")
    paragraph(
        "Learned warm starts with checked refinement skipping achieved about "
        f"{1 / checked['two_pool']['time_ratio_median']:.1f}× faster median complete solves "
        "on eligible two-pool orders than the Fynd reference. Across the full fresh benchmark, "
        f"the method used {1 - checked_time:.1%} less total solve time, including orders where "
        "the initializer declined. These measurements cover two WETH/USDC Uniswap V3 pools on Base."
    )
    paragraph(
        "A neural allocation initializer supplies a starting split that replaces Fynd's fine "
        "allocation passes. An allocation-regret estimator screens that proposal, and a numerical "
        "refinement skip check determines whether the solver can omit refinement. Timings include "
        "this work and any refinement that remains."
    )
    markdown.extend(
        ["![Fresh benchmark complete-solve performance](../assets/performance.svg)", ""]
    )
    document.append(build_performance(results))
    paragraph(
        f"On the separate held-out test, all {runtime['successful_pairs']:,} paired orders "
        f"completed within the accepted {results['gate_bp']}-basis-point loss limit for output "
        f"after gas. The worst observed loss was {-runtime['worst_final_delta_bp']:.5f} basis "
        "points. Complete-solver performance across other pools and market conditions remains untested."
    )
    heading("Methods")
    table(
        ["Method", "Behavior"],
        [
            [
                "Fynd reference",
                "Fynd with no custom initializer; runs its coarse, fine, and refinement passes.",
            ],
            ["Coarse split", "Start refinement from Fynd's 20-chunk split; no learned model."],
            [
                "Learned initialization",
                "Replace both 256-chunk fine passes with a learned correction; always refine.",
            ],
            [
                "Learned warm start with checked skip",
                "The released method. The allocation-regret estimator and six nearby quotes determine whether to skip refinement of the two-pool candidate.",
            ],
            [
                "Quadratic formula",
                "Computes a starting split from the coarse pass using a quadratic approximation.",
            ],
            [
                "Prediction-only skip",
                "Disabled: prediction alone failed the development quality gate.",
            ],
        ],
    )
    paragraph(
        "The neural allocation initializer is a multilayer perceptron (MLP), a feedforward "
        "network with fully connected layers, trained through supervised learning. Both it and "
        "the allocation-regret estimator have two hidden layers of 32 units and 1,281 learned "
        "parameters each. Each network selects five inputs from the seven-field representation: "
        "trade direction, order size, coarse allocation, the price gap between pools, and a "
        "quadratic correction. The two quote-failure flags are constant in the training data "
        "and are excluded. Both artifacts record the selected indices as [0, 1, 2, 3, 6]."
    )
    paragraph(
        "The allocation network adjusts the coarse split by at most one twentieth of the order. "
        "It reuses the coarse pass's simulations and runs inference in Rust. Fynd still constructs "
        "the candidate routes and selects the best output after gas."
    )
    paragraph(
        "The allocation-regret estimator predicts the starting allocation's output loss against "
        "the sampled training reference; it is not a probability of correctness. If its estimate "
        "is at most 0.02 basis points and the order falls within the configured limits, the "
        "refinement skip check quotes three splits on each pool: the proposed split and its "
        "neighbors, one Q/512 step away, where Q is the order size. These six quotes bound "
        "gross-output loss assuming concave pool output. Refinement runs if the check fails. "
        "The method, named offset_certified in the code, does not bound loss after gas; that "
        "is measured by evaluating the final solver output."
    )
    heading("Solve time")
    markdown.extend(["![Fresh benchmark total time comparison](../assets/results.svg)", ""])
    document.append(build_results(results))
    paragraph(
        "Compared with starting from the coarse allocation alone, learned initialization "
        f"reduced total solve time by {1 - learned_time / coarse_time:.1%}, with refinement "
        "enabled in both. Checked refinement skipping increased that reduction to "
        f"{1 - checked_time / coarse_time:.1%}. Both alternatives replace Fynd's fine passes; "
        "this comparison measures the additional contribution of the learned starting split "
        "and the skip check."
    )
    paragraph(
        f"The fresh benchmark contains {fresh['unique_orders']:,} unique orders repeated three times, or "
        f"{fresh['solves']:,} solves per method. The initializer runs on "
        f"{checked['two_pool']['solves']:,} solves. Time is the complete Solver::quote call, including "
        "discovery, coarse work, initialization, inference, refinement, and assembly. State replay "
        "occurs before timing. Methods rotate order and use fresh per-solve caches."
    )
    paragraph(
        "Each time ratio divides the method's solve time by the reference time for the same order "
        "and repeat. Median and p90 summarize those paired ratios; lower is faster. The headline "
        "speedup is the reciprocal of the eligible subset's median time ratio. Total time divides the summed "
        "method time by the summed reference time. Simulation counts come from separate "
        "instrumented runs over eligible two-pool orders, so counting does not affect timing."
    )
    ordered = [
        "water_fill",
        "coarse",
        "offset",
        "offset_certified",
        "quadratic",
        "offset_confident",
    ]
    table(
        [
            "Method",
            "All median",
            "All p90",
            "Two-pool median",
            "Total time",
            "Simulations",
            "Worst delta (bp)",
            "Gate failures",
        ],
        [
            [
                methods[name]["label"],
                f"{methods[name]['all']['time_ratio_median']:.3f}x",
                f"{methods[name]['all']['time_ratio_p90']:.3f}x",
                f"{methods[name]['two_pool']['time_ratio_median']:.3f}x",
                f"{methods[name]['all']['time_ratio_sum']:.3f}x",
                f"{methods[name]['simulations']['total_mean']:.1f}",
                f"{methods[name]['all']['delta_bp']['worst']:+.5f}",
                methods[name]["all"]["gate_failures"],
            ]
            for name in ordered
        ],
    )
    paragraph(
        "Output delta is the difference from Fynd's output after gas, in basis points. Negative "
        "values mean less output. A loss greater than 0.1 basis points fails the quality check, "
        "as does a failed solve when Fynd succeeds. Every order is checked, including those where "
        "the initializer declines. Prediction-only skipping remains disabled because it failed "
        "on development data, despite passing on this benchmark."
    )
    heading("Output quality")
    dev = results["benchmarks"]["dev"]["methods"]
    table(
        ["Evaluation", "Method", "Solves", "Gate failures", "Worst net delta (bp)"],
        [
            [
                "Development",
                dev["offset_certified"]["label"],
                dev["offset_certified"]["all"]["solves"],
                dev["offset_certified"]["all"]["gate_failures"],
                f"{dev['offset_certified']['all']['delta_bp']['worst']:+.6f}",
            ],
            [
                "Development",
                dev["offset_confident"]["label"],
                dev["offset_confident"]["all"]["solves"],
                dev["offset_confident"]["all"]["gate_failures"],
                f"{dev['offset_confident']['all']['delta_bp']['worst']:+.6f}",
            ],
            [
                "Held-out",
                "Frozen release",
                runtime["solves"],
                runtime["gate_failures"],
                f"{runtime['worst_final_delta_bp']:+.6f}",
            ],
        ],
    )
    paragraph(
        "The three development failures are one order repeated three times. The checked-skip "
        "variant refused that shortcut. In the held-out runtime evaluation, "
        f"{runtime['skipped_solves']:,} of {runtime['seeded_solves']:,} eligible orders skipped "
        f"refinement ({runtime['skip_rate']:.1%}); their worst final net delta was "
        f"{runtime['worst_skipped_final_delta_bp']:+.6f} basis points."
    )
    heading("Training and evaluation data")
    paragraph(
        "Training, validation, and test cases are split chronologically. Only orders eligible "
        "for the learned initializer produce model inputs; complete-solver evaluation also "
        "includes orders where it declines."
    )
    table(
        ["Split", "Chronological groups", "Cases", "Eligible model inputs"],
        [
            [
                name,
                groups,
                f"{results['dataset']['splits'][key]['cases']:,}",
                f"{results['dataset']['splits'][key]['eligible']:,}",
            ]
            for name, key, groups in [
                ("Training", "train", "1–8"),
                ("Validation", "validation", "9–10"),
                ("Held-out test", "test", "11–12"),
            ]
        ],
    )
    paragraph(
        "Training, validation, and final-test states span a single overnight sampling period. "
        "The fresh timing benchmark uses a later "
        "window, after the model was frozen, and a separate random seed for order sizes. Sizes "
        "are generated from a log-uniform distribution, spanning approximately 0.02–2,000 WETH "
        "and 50–5,000,000 USDC. Gas price is fixed at 6,000,000 wei. The orders do not represent "
        "observed trading activity, and nearby states and repeated solves are correlated."
    )
    paragraph(
        "Training targets come from a scan in Q/256 steps followed by a local search in Q/16384 "
        "steps. These are approximate targets because changes in gas cost can make a split "
        "outside the search window better. Model selection used validation data. The released "
        "allocation model was trained for 1,000 epochs with learning rate 0.001 and seed 1; "
        "its weights and skip parameters stayed fixed during final evaluation."
    )
    paragraph(
        "On an older market period, the learned starting split was farther from the sampled "
        "reference than the quadratic formula in 30 of 32 windows. Median distance was 0.119% "
        "of the order for the model and 0.053% for the formula. This diagnostic measures the "
        "starting allocation against an approximate target, before refinement. It leaves "
        "transfer to other market conditions unresolved; it does not measure final output quality."
    )
    heading("Reproducibility")
    paragraph(
        "Fynd is pinned to c7622b41d1eee5f18082dab8beed2ee2db5b7cde. All timing methods use the "
        "patched runner, with no custom initializer registered for the reference. That reference "
        "matched a separate unmodified Fynd build on allocation, gross output, gas, and net "
        "output for the checked development and fresh cases. The unmodified build was used "
        "for output comparison, not a separate timing benchmark."
    )
    paragraph(
        "The held-out quality test runs the complete Rust solver with rounded allocations and "
        "exact quotes. Final quality results come from these complete solves, rather than "
        "interpolated offline estimates. "
        "The published weights match the frozen originals byte for byte, and the tables and "
        "figures are generated from the public aggregate JSON."
    )
    paragraph(
        "The harness works with historical market data supplied by the user. Prepare training "
        "cases and compatible Fynd replay inputs from your own data stack to run new "
        "experiments. The repository documents the input formats and includes a synthetic "
        "example to get started. The reproduction guide specifies the inputs for each check."
    )
    heading("Codex experiment harness")
    if trial is None:
        paragraph(
            "The Codex experiment harness accepts a bounded training recipe proposed by Codex. "
            "Local code trains and evaluates the candidate. No connected trial is included "
            "in these results."
        )
    else:
        recipe = trial["recipe"]
        evaluation = trial["evaluation"]["candidate_vs_fynd"]
        paragraph(
            "The Codex experiment harness connects recipe proposals to training and evaluation; "
            "reusable autonomous orchestration remains work in progress. In a trial separate "
            "from frozen v1 development, Codex proposed a training recipe while local code kept the "
            "data splits, features, loss, and evaluation rules fixed. The proposal selected "
            f"hidden layers of {recipe['hidden'][0]} and {recipe['hidden'][1]} units, "
            f"{recipe['epochs']:,} epochs, and learning rate {recipe['learning_rate']}. "
            "Training updated the weights in "
            f"{trial['durations_seconds']['training']:.2f} seconds. The new candidate retained "
            "refinement on every eligible order and did not use the frozen allocation-regret estimator."
        )
        paragraph(
            f"Validation replay completed {evaluation['paired_solves']:,} paired solves "
            "across three repeats of 6,336 orders, with "
            f"{evaluation['quality_failures_over_0_1bp']} losses exceeding 0.1 basis points. "
            f"The worst loss against Fynd was {evaluation['worst_loss_bp']:.5f} basis points. "
            "Candidate and frozen-model timing ran in separate passes, so their difference "
            "does not establish a speed improvement. This trial demonstrates the proposal, "
            "training, and evaluation flow on validation data. It did not use the final test "
            "or replace the frozen model."
        )
    markdown.extend(
        [
            "See [the repository guide](../README.md), [reproduction instructions](../docs/reproduction.md), [machine-readable results](results.json), and [the recorded-results explorer](../demo/index.html).",
            "",
        ]
    )
    document.append(
        '<p class="links"><a href="../README.md">Repository guide</a> · <a href="../docs/reproduction.md">Reproduction instructions</a> · <a href="results.json">Machine-readable results</a> · <a href="../demo/index.html">Recorded-results explorer</a></p>'
    )
    css = """
:root{color-scheme:light;--ink:#111;--muted:#555;--line:#d0d0d0;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#eee;color:var(--ink);font:17px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
main{max-width:1180px;margin:0 auto;background:var(--paper);border-left:1px solid var(--line);border-right:1px solid var(--line)}
.report-header{padding:68px 64px 62px;background-color:#111;background-image:linear-gradient(#ffffff08 1px,transparent 1px),linear-gradient(90deg,#ffffff08 1px,transparent 1px);background-size:48px 48px;color:#fff;border-bottom:8px solid #777}
.kicker{font:500 11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.16em;text-transform:uppercase;color:#ccc}
h1{font-size:clamp(38px,5vw,66px);font-weight:600;line-height:1.03;letter-spacing:-.055em;max-width:900px;margin:36px 0 0}
.report-body{padding:34px 64px 72px;counter-reset:section}
h2{counter-increment:section;display:flex;align-items:baseline;gap:16px;font-size:27px;font-weight:600;line-height:1.25;margin:58px 0 24px;letter-spacing:-.025em;border-top:1px solid var(--ink);padding-top:20px}
h2::before{content:counter(section,decimal-leading-zero);font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);letter-spacing:0}
p{max-width:850px;overflow-wrap:anywhere;margin:20px 0}strong{font-weight:650}
svg{display:block;max-width:100%;height:auto;margin:32px 0;border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.55;font-variant-numeric:tabular-nums}
th{text-align:left;background:#111;color:#fff;font-weight:600;vertical-align:bottom}td,th{padding:13px 12px;border-bottom:1px solid var(--line)}tr:nth-child(even){background:#f4f4f4}
.table{overflow:auto;margin:28px 0;border-bottom:1px solid var(--ink)}a{color:inherit;text-decoration-thickness:1px;text-underline-offset:4px}a:hover{background:#e5e5e5}a:focus-visible{outline:2px solid currentColor;outline-offset:4px}.links{border-top:1px solid var(--ink);padding-top:24px;margin-top:48px;font-size:14px}
@media(max-width:700px){main{border:0}.report-header{padding:40px 24px 38px}.report-body{padding:20px 24px 50px}h1{margin-top:28px}h2{font-size:23px;gap:12px;margin-top:42px}body{font-size:16px}td,th{padding:11px 10px}.kicker{font-size:10px}}
@media print{body{background:#fff}main{max-width:none;border:0}.report-header{background:#fff;color:#111;border-bottom:2px solid #111;padding:20px 0}.kicker{color:#555}.report-body{padding:0}h2,svg,table{break-inside:avoid}a{color:#111}}
"""
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Swap allocation research — report</title><style>'
        + css
        + '</style><main><header class="report-header"><div class="kicker">Research report · frozen v1</div><h1>Learned warm starts for swap allocation</h1></header><div class="report-body">'
        + "\n".join(document)
        + "</div></main></html>\n"
    )
    return "\n".join(markdown), page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "reports/results.json")
    parser.add_argument("--trial", type=Path, default=ROOT / "reports/codex-trial.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    trial = json.loads(args.trial.read_text()) if args.trial.exists() else None
    markdown, document = build(results, trial)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.md").write_text(markdown)
    (args.output_dir / "report.html").write_text(document)


if __name__ == "__main__":
    main()
