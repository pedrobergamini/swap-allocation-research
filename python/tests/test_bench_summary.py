"""Bench failures must not disappear when a solve has no usable seed record."""

import json
import runpy
import sys
from pathlib import Path

import pytest

SUMMARY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "summarize_bench.py"


def solve(method, case="case", *, status="ok", seeds=None, net=1_000_000):
    return {
        "block": 1,
        "case_id": case,
        "repeat": 0,
        "method": method,
        "status": status,
        "seeds": seeds or [],
        "elapsed_us": 100,
        "amount_in": "1000",
        "net_out": str(net) if status == "ok" else None,
        "pool_amounts": ["500", "500"] if status == "ok" else None,
    }


def summarize(rows, tmp_path, monkeypatch, capsys):
    bench = tmp_path / "bench.jsonl"
    bench.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(sys, "argv", [str(SUMMARY_SCRIPT), str(bench)])
    runpy.run_path(str(SUMMARY_SCRIPT), run_name="__main__")
    return json.loads(capsys.readouterr().out)["offset_certified"]


@pytest.mark.parametrize("seeds", [[], [{"proposal": None}], [{"proposal": ["500", "500"]}]])
def test_failure_counts_with_any_seed_record(seeds, tmp_path, monkeypatch, capsys):
    summary = summarize(
        [
            solve("water_fill"),
            solve("offset_certified", status="error: timeout", seeds=seeds),
        ],
        tmp_path,
        monkeypatch,
        capsys,
    )
    assert summary["gate_failures"] == 1
    assert summary["failed_where_fynd_solved"] == 1
    assert summary["seeded"] == 0
    assert summary["declined"] == 0
    assert "delta_bp" not in summary


@pytest.mark.parametrize("candidate_status", ["ok", "error: timeout"])
def test_failed_reference_is_not_a_gate_failure(candidate_status, tmp_path, monkeypatch, capsys):
    summary = summarize(
        [
            solve("water_fill", status="error: timeout"),
            solve("offset_certified", status=candidate_status),
        ],
        tmp_path,
        monkeypatch,
        capsys,
    )
    assert summary["gate_failures"] == 0
    assert summary["failed_where_fynd_solved"] == 0
    assert summary["seeded"] == 0


def test_failures_add_to_quality_gate_without_changing_valid_statistics(
    tmp_path, monkeypatch, capsys
):
    valid_rows = [
        solve("water_fill", "seeded"),
        solve("offset_certified", "seeded", seeds=[{"proposal": ["500", "500"]}], net=999_980),
        solve("water_fill", "declined"),
        solve("offset_certified", "declined", seeds=[{"proposal": None}]),
    ]
    original = summarize(valid_rows, tmp_path, monkeypatch, capsys)
    with_failure = summarize(
        valid_rows
        + [
            solve("water_fill", "failed"),
            solve("offset_certified", "failed", status="error: timeout"),
        ],
        tmp_path,
        monkeypatch,
        capsys,
    )
    assert original["gate_failures"] == 1
    assert original["seeded"] == 1
    assert original["declined"] == 1
    assert with_failure == original | {"gate_failures": 2, "failed_where_fynd_solved": 1}
