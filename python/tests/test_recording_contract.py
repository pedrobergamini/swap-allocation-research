"""Replay preparation must not depend on the producer's private acquisition logs."""

import json

import pytest
import zstandard
from warm_starts.recorded import capture_info


def write_replay(path, *, gas="6000000", blocks=(10, 20), chain="base", schema=1):
    replay = {
        "metadata": {"chain": chain, "schema_version": schema, "gas_price_wei": gas},
        "updates": [{"block_number_or_timestamp": block} for block in blocks],
    }
    path.write_bytes(zstandard.ZstdCompressor().compress(json.dumps(replay).encode()))


def test_reads_bounds_and_gas_without_producer_log(tmp_path):
    path = tmp_path / "market-01.json.zst"
    write_replay(path, gas="1208925819614629174706176")
    assert capture_info(path) == (10, 20, 1208925819614629174706176)


@pytest.mark.parametrize(
    "changes",
    [
        {"gas": None},
        {"gas": 123},
        {"gas": "-1"},
        {"gas": "1.5"},
        {"blocks": []},
        {"blocks": [20, 10]},
        {"blocks": [10, 10]},
        {"blocks": [True]},
        {"chain": "ethereum"},
        {"schema": 2},
    ],
)
def test_rejects_unsupported_or_ambiguous_replay_metadata(tmp_path, changes):
    path = tmp_path / "market-01.json.zst"
    write_replay(path, **changes)
    with pytest.raises(ValueError):
        capture_info(path)
