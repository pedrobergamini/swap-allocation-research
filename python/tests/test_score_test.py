"""Complete Rust solves, rather than a Python policy approximation, determine the test gate."""

from copy import deepcopy

import pytest
from warm_starts.score_test import summarize


def case():
    return {"block": 100, "direction": "weth-usdc", "amount_in": "1000"}


def pair(net="999990", seeds=None):
    common = {
        "block": 100,
        "case_id": "weth-usdc-1000",
        "direction": "weth-usdc",
        "amount_in": "1000",
        "repeat": 0,
        "build": "patched",
        "gas_price_wei": "6000000",
        "status": "ok",
    }
    return [
        dict(common, method="water_fill", net_out="1000000", seeds=[]),
        dict(common, method="offset_certified", net_out=net, seeds=seeds or []),
    ]


def test_gate_includes_unseeded_failures_and_loss_boundary():
    assert summarize([case()], pair())["gate_passed"]  # Exactly 0.1 bp is allowed.
    assert summarize([case()], pair("999989"))["gate_failures"] == 1
    rows = pair()
    rows[1].update(status="timeout", net_out=None)
    result = summarize([case()], rows)
    assert result["failed_where_fynd_solved"] == 1
    assert result["gate_failures"] == 1
    assert not result["gate_passed"]


def test_skip_statistics_read_the_disjoint_rust_decision():
    seeds = [
        {"proposal": ["500", "500"], "skipped": False},
        {"proposal": ["500", "500"], "skipped": True},
    ]
    result = summarize([case()], pair(seeds=seeds))
    assert result["seeded_solves"] == 1
    assert result["skipped_solves"] == 0
    assert result["skip_rate"] == 0
    seeds[0]["skipped"] = True
    result = summarize([case()], pair(seeds=seeds))
    assert result["skipped_solves"] == 1
    assert result["skip_rate"] == 1
    assert result["worst_skipped_final_delta_bp"] == -0.1


def test_reference_failure_cannot_produce_a_passing_test():
    rows = pair()
    rows[0].update(status="timeout", net_out=None)
    result = summarize([case()], rows)
    assert result["reference_failures"] == 1
    assert not result["gate_passed"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "wrong_input", "wrong_gas"])
def test_rejects_incomplete_or_mispaired_evidence(change):
    rows = pair()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif change == "extra":
        rows[-1]["block"] = 101
    elif change == "wrong_input":
        rows[-1]["amount_in"] = "1001"
    else:
        rows[-1]["gas_price_wei"] = "60000000"
    with pytest.raises(ValueError):
        summarize([case()], rows)


def test_rejects_duplicate_or_empty_test_cases():
    with pytest.raises(ValueError, match="duplicate test case"):
        summarize([case(), case()], pair())
    with pytest.raises(ValueError, match="no held-out"):
        summarize([], [])


def test_public_models_are_verified_without_private_training_reports(tmp_path):
    import hashlib
    import json

    from warm_starts.score_test import verify_frozen

    artifacts = []
    for name in ("offset.json", "confidence.json"):
        content = b'{"test": "model"}\n'
        (tmp_path / name).write_bytes(content)
        artifacts.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"format": "sar-release-models/1", "files": artifacts}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert set(verify_frozen(tmp_path)) == {"offset.json", "confidence.json"}
    (tmp_path / "offset.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen(tmp_path)


def test_public_manifest_rejects_unknown_paths_and_missing_models(tmp_path):
    import json

    from warm_starts.score_test import verify_frozen

    for files in ([{"path": "../private.json", "sha256": "0" * 64}], []):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"format": "sar-release-models/1", "files": files})
        )
        with pytest.raises(ValueError):
            verify_frozen(tmp_path)
