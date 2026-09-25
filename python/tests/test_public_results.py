"""Published quality gates include every paired case and preserve the integer boundary."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_public_results", Path(__file__).resolve().parents[2] / "scripts/build_public_results.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def solve(method, *, status="ok", net=1_000_000, seeds=None):
    return {
        "block": 1,
        "case_id": "case",
        "repeat": 0,
        "method": method,
        "gas_price_wei": "6000000",
        "status": status,
        "net_out": str(net) if status == "ok" else None,
        "elapsed_us": 10,
        "pool_amounts": ["4", "6"] if status == "ok" else None,
        "amount_in": "10",
        "direction": "usdc-weth",
        "seeds": seeds or [],
    }


def summarize(reference, candidate):
    rows = module.index_rows([reference, candidate], ("water_fill", "offset_certified"), 1)
    return module.population_summary(
        rows["offset_certified"], rows["water_fill"], rows["water_fill"].keys()
    )


@pytest.mark.parametrize(
    "problem", ["missing_candidate", "duplicate_candidate", "missing_reference"]
)
def test_pairing_rejects_incomplete_or_duplicate_evidence(problem):
    reference, candidate = solve("water_fill"), solve("offset_certified")
    rows = {
        "missing_candidate": [reference],
        "duplicate_candidate": [reference, candidate, candidate],
        "missing_reference": [candidate],
    }[problem]
    with pytest.raises(ValueError):
        module.index_rows(rows, ("water_fill", "offset_certified"), 1)


def test_candidate_failure_before_seeding_fails_gate():
    summary = summarize(solve("water_fill"), solve("offset_certified", status="error: timeout"))
    assert summary["solves"] == 1
    assert summary["successful_pairs"] == 0
    assert summary["failed_where_fynd_solved"] == summary["gate_failures"] == 1
    assert not summary["gate_passed"]


def test_declined_candidate_still_has_output_quality_checked():
    summary = summarize(
        solve("water_fill"),
        solve("offset_certified", net=999_989, seeds=[{"proposal": None}]),
    )
    assert summary["successful_pairs"] == 1
    assert summary["gate_failures"] == 1
    assert not summary["gate_passed"]


@pytest.mark.parametrize("extra_loss,failures", [(-1, 0), (0, 0), (1, 1)])
def test_quality_gate_uses_exact_integer_boundary(extra_loss, failures):
    reference_net = 10**30
    candidate_net = reference_net - reference_net // 100_000 - extra_loss
    summary = summarize(
        solve("water_fill", net=reference_net), solve("offset_certified", net=candidate_net)
    )
    assert summary["gate_failures"] == failures
    assert summary["gate_passed"] == (failures == 0)


def test_failed_reference_prevents_passing_gate():
    summary = summarize(solve("water_fill", status="error: timeout"), solve("offset_certified"))
    assert summary["reference_failures"] == 1
    assert summary["gate_failures"] == 0
    assert not summary["gate_passed"]
