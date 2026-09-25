# Codex experiment harness

The Codex experiment harness connects a bounded recipe proposal to supervised training, Rust evaluation, and result recording. Supply a recipe directly or ask the installed Codex CLI to propose one. Local code owns the training procedure and evaluation rules; a recipe cannot change the evaluator. Reusable autonomous orchestration remains work in progress.

The [recorded connected trial](../reports/codex-trial.json) completed training and validation evaluation. It is separate from frozen v1 development and did not replace the released weights.

## Offline smoke check

After the Python setup in the README:

```sh
uv run python -m warm_starts.experiment_synthetic --output work/synthetic.jsonl.zst
uv run python -m warm_starts.experiment \
  --cases work/synthetic.jsonl.zst \
  --recipe configs/experiment-default.json \
  --output work/synthetic-trial
```

The generated data is artificial, with 192 training examples and 64 validation examples. It exercises parameter updates, export and reference inference. It does not exercise Fynd replay or reproduce the market results. The result status is `training_smoke_completed` rather than `completed`.

The output directory must not already exist. It contains the recipe, exported `candidate.json`, training metrics, process logs and `result.json`. Treat the entire directory as private until its contents have been reviewed.

## Recipe contract

```json
{"hidden": [16, 16], "epochs": 300, "learning_rate": 0.001}
```

Only these three keys are accepted. Hidden layers must be `[16,16]` or `[32,32]`, epochs 300 or 1,000, and learning rate 0.001 or 0.003. Seed 1, batch size 256, the seven-field `coarse-offset/2` representation, the existing smooth absolute allocation-error loss, a maximum correction of one coarse chunk, and evaluation with refinement always enabled are fixed by the runner.

The trainer receives all seven feature fields and excludes fields that are constant in the training split. Each exported artifact records its selected indices. Both frozen v1 networks select five fields; see the [model interface](models.md).

Prepared input must contain both `train` and `validation` examples. Any other split, including `test`, is rejected before training. Split labels are the dataset producer's responsibility; the runner cannot establish that incorrectly labeled data was truly held out. See [data inputs](data.md).

## Codex proposal

Install and authenticate a compatible Codex CLI yourself; the runner does not install software or configure credentials. The integration was exercised with CLI 0.153.0. The CLI must support `--ignore-user-config`, `--ignore-rules`, `--ephemeral`, JSON events, and structured output.

```sh
uv run python -m warm_starts.experiment \
  --cases work/prepared-development.jsonl.zst \
  --proposal codex \
  --output work/codex-trial
```

The proposal receives a short technical description and usable training/validation row counts. It receives no dataset paths, rows, test metrics or prior conversation. An empty temporary workspace, ignored user configuration, disabled project-document loading, host-skill discovery suppression and disabled plugin, memory, shell, browser and app features limit the invocation's context and tools. It uses a read-only sandbox with approval disabled. Its event stream must complete without any tool-call items; an unexpected tool item rejects the proposal. Raw events stay in the private run directory.

These checks constrain this integration; they are not an isolation guarantee against a modified CLI or incorrectly configured host. Use the recipe interface when your CLI does not support the required controls. Do not bypass its sandbox to make the proposal work.

## Complete Rust evaluation

Supply a patched `sar-bench`, the released frozen allocation model, and a replay manifest in addition to the arguments above:

```sh
uv run python -m warm_starts.experiment \
  --cases work/prepared-development.jsonl.zst \
  --recipe configs/experiment-default.json \
  --replay work/validation-replay.json \
  --bench-binary target/release/sar-bench \
  --frozen-model models/v1/offset.json \
  --output work/evaluated-trial
```

Replace `--recipe ...` with `--proposal codex` to connect the full trial to Codex. A replay manifest has this shape; the numbers below illustrate the format rather than identify a distributed dataset:

```json
{
  "split": "validation",
  "jobs": [{
    "recording": "states/example.json.zst",
    "blocks": [100],
    "cases": [{"id": "example", "direction": "weth-usdc", "amount_in": "1000000000000000000"}],
    "gas_price_wei": 6000000
  }]
}
```

Recording paths resolve relative to the manifest. Every block/direction/amount combination must appear in the prepared validation split. Three repeats are run for each requested case. Supply compatible replay inputs from your own historical pool states, following the [input contracts](data.md).

The candidate and frozen allocation model both use the `offset` method, which always refines. The frozen allocation-regret estimator (`confidence.json`) is not attached to a newly trained allocation model. Seedless Fynd is the reference. Summaries include every requested solve, including orders where the initializer declines. Missing rows, duplicate rows and unsuccessful solves invalidate the evaluation. Successful evaluations report losses above 0.1 basis points as quality failures rather than silently dropping cases.

The candidate and reference run together; the frozen model is measured in a separate pass. The candidate/frozen timing ratio is therefore indicative local timing and is not a tightly interleaved benchmark. Final held-out evidence requires a separate evaluation after candidate selection.

The proposal has a five-minute budget, training fifteen minutes, and the whole replay evaluation thirty minutes. Timeouts terminate the stage's process group and persist a failure result. Output includes the chosen recipe, fixed settings, input and candidate hashes, stage durations, quality/timing summaries, and a check that the frozen model did not change. No candidate is promoted automatically.
