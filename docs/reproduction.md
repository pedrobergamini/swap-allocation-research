# Reproduction and validation

The release includes model artifacts, aggregate results, and synthetic examples. Historical replay inputs are supplied by the user. You can exercise the public code with synthetic inputs, evaluate your own prepared market states, or regenerate the published report and figures from the aggregate results.

Recreating the published training run and market replay requires the original historical inputs, which are not included in the release. New experiments use your prepared training cases and compatible Fynd replay files, as described in the [input contracts](data.md).

## Install and build

Use Python 3.12, `uv`, Rust, and native C/C++ build tooling. Versions are locked in `uv.lock`, the root `Cargo.lock`, and `baseline/Cargo.lock`. Release validation uses Rust 1.95.0. Setup fetches the exact upstream Fynd revision from its canonical repository; it does not update dependency versions.

```sh
uv sync --locked
./scripts/setup-fynd.sh
cargo build --release --locked
cargo build --release --locked --manifest-path baseline/Cargo.toml
(cd models/v1 && shasum -a 256 -c SHA256SUMS)
```

The root binary links the patched Fynd checkout; the baseline binary links the clean checkout. Both checkouts must remain at `c7622b41d1eee5f18082dab8beed2ee2db5b7cde`. Setup verifies the entire expected patch and refuses unexpected local changes. Separate build outputs avoid mixing the two implementations.

## Train a synthetic model

```sh
uv run python -m warm_starts.experiment_synthetic --output work/synthetic.jsonl.zst
uv run python -m warm_starts.experiment \
  --cases work/synthetic.jsonl.zst --output work/synthetic-trial \
  --recipe configs/experiment-default.json
```

Inspect the candidate weights and result record in the new output directory. The [experiment documentation](experiments.md) covers Codex proposals and real replay evaluation. For a manual supervised run outside the bounded runner, `python -m warm_starts.train_offset --help` documents the training parameters.

## Evaluate compatible replay inputs

Prepare inputs using the [data contract](data.md). Replace the example file and block with values from your input; keep both runners' cases, blocks, gas, repeats, and timeout identical.

```sh
mkdir -p work/results
baseline/target/release/sar-bench \
  --recording work/inputs/market-01.json.zst --blocks 12345 \
  --cases work/inputs/orders.json --methods water_fill \
  --gas-price-wei 6000000 --repeats 3 --output work/results/clean.jsonl

target/release/sar-bench \
  --recording work/inputs/market-01.json.zst --blocks 12345 \
  --cases work/inputs/orders.json --methods water_fill,coarse,offset,offset_certified \
  --gas-price-wei 6000000 --repeats 3 \
  --offset-model models/v1/offset.json --confidence-model models/v1/confidence.json \
  --certificate-step-divisor 512 --output work/results/patched.jsonl
```

The timer measures the complete quote call; loading/replaying states is outside it. Compare clean `water_fill` with patched `water_fill` to detect behavioral drift before interpreting initializer comparisons. Outputs contain per-solve status, gas, net and gross output, integer pool allocations, elapsed microseconds, and initializer decisions. Failed solves must remain in quality accounting. Hardware and scheduling affect timing; do not expect identical historical microseconds.

For a final-test dataset with matching capture filenames, run the frozen evaluator:

```sh
uv run python -m warm_starts.score_test \
  --cases work/inputs/final-test.jsonl.zst --model-dir models/v1 \
  --recordings 'work/inputs/market-*.json.zst' \
  --output-dir work/final-evaluation
```

It pairs every requested case and rejects missing, duplicate, mismatched, or unexpected solves. Its 0.1-basis-point gate compares completed net outputs using integer arithmetic, includes cases declined by the learned initializer, and cannot pass when the reference fails. Use a new output directory. Never use this final-test result to select recipes while continuing to call it untouched evaluation.

## Checks

```sh
uv run ruff format --check python scripts
uv run ruff check python scripts
uv run pytest
cargo fmt -p swap-allocation-research -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
cargo build --release --locked
cargo fmt --manifest-path baseline/Cargo.toml -p swap-allocation-baseline -- --check
cargo clippy --locked --manifest-path baseline/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path baseline/Cargo.toml
cargo build --release --locked --manifest-path baseline/Cargo.toml
cargo test --locked --manifest-path work/fynd-patched/Cargo.toml -p fynd-core water_fill
```

Replay-dependent parity checks need your own matching inputs:

```sh
uv run python scripts/check_model_parity.py work/results/patched.jsonl offset models/v1/offset.json
uv run python scripts/check_model_parity.py work/results/patched.jsonl offset_certified models/v1/offset.json models/v1/confidence.json
uv run python scripts/check_feature_parity.py work/results/patched.jsonl work/inputs/market-01.json.zst
```

`check_feature_parity.py` checks the free quadratic feature within its documented rounding tolerance; model inference/allocation parity is checked separately. Diagnostic `swap-metrics` builds count simulations but add overhead; never use their elapsed times as performance evidence.

The public report and recorded-results explorer use aggregate results and selected recorded examples. Regenerate the HTML and SVG artifacts from those public JSON inputs:

```sh
uv run python scripts/build_launch_visuals.py
uv run python scripts/build_public_report.py
```

This regeneration does not replay the market or retrain v1. Chart PNG exports and demo screenshots are separate publication assets. The cover PNGs are editorial illustrations and are not regenerated from benchmark data.

The headline speedup is the reciprocal of the fresh benchmark's median paired time ratio on eligible two-pool orders. The total-time reduction is one minus its ratio of summed times across all orders. These measure different populations and weight orders differently; the report keeps them separate. Neither uses the 6,336-order held-out quality test as a timing benchmark.

## Export the public package

`release-files.txt` enumerates the files intended for distribution. Export them to a new directory:

```sh
python3 scripts/export_release.py --output /path/to/new-public-directory
```

The exporter rejects excluded paths, missing files, symlinks, and an existing destination. It copies only listed files and writes `SHA256SUMS.release` for the exported contents. It does not copy working data, environment files, caches, or local agent configuration. Review the file list when adding new public files.
