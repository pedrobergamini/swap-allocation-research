# Swap Allocation Research

Learned warm starts for swap allocation, integrated into [Fynd](https://github.com/propeller-heads/fynd). A neural allocation initializer proposes a starting split between two liquidity pools; numerical checks determine when refinement can be skipped. The complete method achieved about **5.3× faster median complete solves on eligible two-pool orders** than the Fynd reference in the fresh benchmark. Across all benchmark orders, it used **56.5% less total solve time**.

This release includes frozen model weights, Rust solver integration, evaluation tooling, a recorded-results explorer, and a Codex experiment harness. The measurements include feature preparation, inference, checks, and any remaining refinement.

The experiment evaluates two WETH/USDC Uniswap V3 pools on Base. It grew out of internal dsolver work at **Dewiz**. Thanks to **Propeller Heads** for Fynd, the open-source solver that supplies the reference implementation, refinement, and replay machinery.

## Results and demo

- Read the [technical report](reports/REPORT.md) or download its [self-contained HTML version](reports/report.html).
- Download the [recorded-results explorer](demo/index.html) and open it in a browser to inspect selected orders. It does not run Fynd in the browser.
- Inspect the [frozen weights](models/v1) and [model contract](docs/models.md).

The fresh benchmark covers 1,440 unique orders, each repeated three times. The headline speedup describes the eligible two-pool subset; the total-time reduction includes orders where the initializer declines. Both compare complete solves against Fynd with its original initialization.

A separate comparison measures what learning adds beyond a simpler alternative. With refinement enabled in both, learned initialization reduced total solve time by **5.6% compared with starting from the coarse allocation alone**. Checked refinement skipping increased that reduction to **8.7%**. The report includes the full comparison.

The separate held-out test contains **6,336 successful paired orders**, all within the accepted 0.1-basis-point loss limit for output after gas. The report gives the methods, measurements, and limits of this two-pool experiment.

The harness works with historical market data you supply. Prepare training cases and compatible Fynd replay inputs from your own data stack, then use the training and evaluation workflow to test your own ideas. The [input contracts](docs/data.md) define the formats, and the synthetic example below exercises the workflow. See [reproduction and validation](docs/reproduction.md) for the inputs required by each check.

## Quickstart

Use Python 3.12 and an existing `uv` installation. Dependency versions are locked in `uv.lock`; the example needs no network after those dependencies are installed.

```sh
uv sync --locked
uv run python -m warm_starts.experiment_synthetic --output work/synthetic.jsonl.zst
uv run python -m warm_starts.experiment \
  --cases work/synthetic.jsonl.zst \
  --output work/synthetic-trial \
  --recipe configs/experiment-default.json
```

This trains an actual model on **synthetic** features and labels. It checks that training changes the weights and records validation diagnostics. It does not reproduce the market benchmark. Each run needs a new output directory.

For compatible real inputs, the [Codex experiment harness](docs/experiments.md) can train and evaluate one bounded recipe, supplied as JSON or proposed through the Codex CLI. A [real connected trial](reports/codex-trial.json) completed training and validation replay with no failures of the quality gate; it did not replace frozen v1. Broader autonomous experiment orchestration remains work in progress. New candidates always retain numerical refinement and do not reuse v1's allocation-regret estimator.

## Build the Fynd replay runners

The integration uses Fynd revision `c7622b41d1eee5f18082dab8beed2ee2db5b7cde`. Setup verifies separate clean and patched checkouts and refuses unexpected edits. Use a Rust toolchain compatible with the locked dependency graph; release validation uses Rust 1.95.0.

```sh
./scripts/setup-fynd.sh
cargo build --release --locked
cargo build --release --locked --manifest-path baseline/Cargo.toml
```

The Rust build includes native dependencies and needs the normal C/C++ build tools for your platform. The two runner binaries are `target/release/sar-bench` and `baseline/target/release/sar-bench`. See [reproduction and validation](docs/reproduction.md) for replay commands, comparisons, and checks.

## What is learned

The neural allocation initializer is a supervised multilayer perceptron (MLP), a feedforward network with fully connected layers. It learns a correction of at most one twentieth of the order around Fynd's coarse split. This warm start replaces the solver's fine allocation passes.

A separate allocation-regret estimator predicts the proposal's output shortfall against the sampled reference. A low estimate allows the refinement skip check to run; the estimate alone does not authorize a skip. The numerical check bounds gross-output loss under its assumptions, while complete-solver evaluation measures output after gas.

Each frozen network has two hidden layers of 32 units and 1,281 learned parameters. Both select five inputs from the seven-field feature representation. Supervised training changes these parameters. The Codex experiment harness proposes a training recipe, while local code performs training, evaluation, and result recording.

## License

New code and published model weights are [MIT licensed](LICENSE), copyright Pedro Bergamini. [NOTICE](NOTICE) records upstream attribution.
