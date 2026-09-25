"""Headline metrics must keep the eligible median and whole-benchmark total distinct."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_launch_visuals", Path(__file__).resolve().parents[2] / "scripts/build_launch_visuals.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_headline_uses_eligible_median_and_full_benchmark_total():
    results = {
        "benchmarks": {
            "fresh": {
                "methods": {
                    "offset_certified": {
                        "two_pool": {"time_ratio_median": 0.25, "time_ratio_sum": 0.9},
                        "all": {"time_ratio_median": 0.5, "time_ratio_sum": 0.4},
                    }
                }
            }
        }
    }
    speedup, reduction = module.performance_metrics(results)
    assert speedup == pytest.approx(4.0)
    assert reduction == pytest.approx(0.6)
