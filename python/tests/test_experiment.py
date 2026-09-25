import json
import sys

import pytest
from warm_starts import dataset, experiment, experiment_synthetic


@pytest.fixture
def recipe():
    return {"hidden": [16, 16], "epochs": 300, "learning_rate": 0.001}


@pytest.mark.parametrize(
    "change",
    [
        {"hidden": [16, 32]},
        {"hidden": [16.0, 16]},
        {"epochs": 301},
        {"epochs": True},
        {"learning_rate": 0.1},
        {"seed": 7},
        {"command": "echo injected"},
    ],
)
def test_recipe_rejects_unbounded_changes(recipe, change):
    recipe.update(change)
    with pytest.raises(ValueError):
        experiment.validate_recipe(recipe)


def test_dataset_rejects_test_even_if_row_has_no_features(tmp_path):
    rows = list(experiment_synthetic.rows())
    rows.append({"id": "heldout", "split": "test", "offset": None})
    path = tmp_path / "cases.zst"
    dataset.write_jsonl_zst(path, rows)
    with pytest.raises(ValueError, match="only train and validation"):
        experiment.validate_cases(path)


def test_training_changes_weights_and_exports_reference_model(tmp_path, recipe):
    path, config = tmp_path / "cases.zst", tmp_path / "recipe.json"
    dataset.write_jsonl_zst(path, experiment_synthetic.rows())
    experiment.write_json(config, recipe)
    experiment.train_worker(path, config, tmp_path)
    report = json.loads((tmp_path / "training.json").read_text())
    artifact = json.loads((tmp_path / "candidate.json").read_text())
    assert report["weights_changed"] is True
    assert report["reference_drift"] < 1e-9
    assert artifact["output"] == {"kind": "coarse_offset", "chunks": 1.0}
    assert set(report["start_error"]) == {"train", "validation"}


def test_tool_call_rejected_even_if_proposal_completes(tmp_path):
    path = tmp_path / "events"
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}},
                {"type": "turn.completed"},
            ]
        )
    )
    with pytest.raises(ValueError, match="used a tool"):
        experiment.verify_codex_events(path)


@pytest.mark.parametrize("contents", ["", '{"type":"turn.failed"}', '{"type":"turn.completed"}'])
def test_incomplete_proposals_rejected(tmp_path, contents):
    path = tmp_path / "events"
    path.write_text(contents)
    with pytest.raises(ValueError):
        experiment.verify_codex_events(path)


def test_replay_cannot_include_nonvalidation_order(tmp_path):
    manifest = {
        "split": "validation",
        "jobs": [
            {
                "recording": "replay.zst",
                "blocks": [12],
                "cases": [{"id": "x", "direction": "weth-usdc", "amount_in": "1"}],
                "gas_price_wei": 6000000,
            }
        ],
    }
    path = tmp_path / "replay.json"
    experiment.write_json(path, manifest)
    rows = [{"split": "train", "block": 12, "direction": "weth-usdc", "amount_in": "1"}]
    with pytest.raises(ValueError, match="outside prepared validation"):
        experiment.replay_jobs(path, rows)


def test_evaluation_rejects_missing_repeats(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"block": 12, "case_id": "x", "repeat": 0, "method": "offset"}))
    with pytest.raises(ValueError, match="incomplete"):
        experiment.read_evaluation(path, {(12, "x", 0), (12, "x", 1)}, ["offset"])


def test_evaluation_quality_gate_includes_every_case():
    baseline = {0: {"status": "ok", "net_out": "1000000", "elapsed_us": 100}}
    candidate = {0: {"status": "ok", "net_out": "999980", "elapsed_us": 75}}
    summary = experiment.compare(candidate, baseline)
    assert summary["quality_failures_over_0_1bp"] == 1
    assert summary["median_time_ratio"] == 0.75
    candidate[0]["status"] = "timeout"
    with pytest.raises(ValueError, match="unsuccessful"):
        experiment.compare(candidate, baseline)


def test_stage_timeout_and_failure(tmp_path):
    with pytest.raises(TimeoutError):
        experiment.run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            log=tmp_path / "timeout.log",
            timeout=0.05,
        )
    with pytest.raises(RuntimeError, match="status 4"):
        experiment.run_process(
            [sys.executable, "-c", "raise SystemExit(4)"],
            cwd=tmp_path,
            log=tmp_path / "failure.log",
            timeout=5,
        )


def test_failed_stage_persists_result(tmp_path, monkeypatch, recipe):
    path, config = tmp_path / "cases.zst", tmp_path / "recipe.json"
    dataset.write_jsonl_zst(path, experiment_synthetic.rows())
    experiment.write_json(config, recipe)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment", "--cases", str(path), "--recipe", str(config), "--output", str(output)],
    )

    def failure(*args, **kwargs):
        raise TimeoutError("budget expired")

    monkeypatch.setattr(experiment, "run_process", failure)
    with pytest.raises(SystemExit):
        experiment.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_stage"] == "training"
    assert result["recipe"] == recipe


def test_gate_exact_boundary_with_large_token_amounts():
    reference = 10**27
    baseline = {0: {"status": "ok", "net_out": str(reference), "elapsed_us": 100}}
    candidate = {
        0: {"status": "ok", "net_out": str(reference - reference // 100000), "elapsed_us": 75}
    }
    assert experiment.compare(candidate, baseline)["quality_failures_over_0_1bp"] == 0
    candidate[0]["net_out"] = str(reference - reference // 100000 - 1)
    assert experiment.compare(candidate, baseline)["quality_failures_over_0_1bp"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("direction", "usdc-weth"),
        ("amount_in", "2"),
        ("gas_price_wei", "1"),
        ("build", "clean"),
    ],
)
def test_replay_output_identity_matches_manifest(tmp_path, field, value):
    identity = {
        "direction": "weth-usdc",
        "amount_in": "1",
        "gas_price_wei": "6000000",
        "build": "patched",
    }
    row = dict(identity, block=12, case_id="x", repeat=0, method="offset")
    row[field] = value
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="does not match the replay manifest"):
        experiment.read_evaluation(path, {(12, "x", 0): identity}, ["offset"])


@pytest.mark.parametrize("bad", [None, "bad", True])
def test_invalid_feature_types_rejected_at_boundary(tmp_path, bad):
    rows = list(experiment_synthetic.rows())
    rows[0]["offset"]["features"][0] = bad
    path = tmp_path / "cases.zst"
    dataset.write_jsonl_zst(path, rows)
    with pytest.raises(ValueError, match="invalid coarse-offset"):
        experiment.validate_cases(path)


def test_missing_frozen_model_persists_failure(tmp_path, monkeypatch, recipe):
    path, config = tmp_path / "cases.zst", tmp_path / "recipe.json"
    dataset.write_jsonl_zst(path, experiment_synthetic.rows())
    experiment.write_json(config, recipe)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment",
            "--cases",
            str(path),
            "--recipe",
            str(config),
            "--output",
            str(output),
            "--frozen-model",
            str(tmp_path / "missing"),
        ],
    )
    with pytest.raises(SystemExit):
        experiment.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_stage"] == "validation"


def test_corrupt_compressed_input_persists_failure(tmp_path, monkeypatch, recipe):
    cases, config = tmp_path / "cases.zst", tmp_path / "recipe.json"
    cases.write_bytes(b"not a compressed dataset")
    experiment.write_json(config, recipe)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment", "--cases", str(cases), "--recipe", str(config), "--output", str(output)],
    )
    with pytest.raises(SystemExit):
        experiment.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_stage"] == "validation"
