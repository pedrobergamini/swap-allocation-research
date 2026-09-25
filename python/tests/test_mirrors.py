"""Pins the Python mirrors to the cases the Rust unit tests pin.

Cross-language parity on real data (identical feature vectors, predictions and splits) is checked
by `scripts/check_model_parity.py` against `sar-bench` seed logs; these tests keep each mirror honest
on its own.
"""

import math

import pytest
from warm_starts import allocation, coarse, features, interpolation, model, quadratic


def outputs(first, second):
    return [list(first), list(second)]


def test_probe_amounts_round_down():
    assert features.probe_amounts(1_000_001) == [62500, 250000, 500000, 1000001]


def test_failed_probes_are_flagged_not_zero():
    probes = [
        [36_000_000_000_000_000, None, None, None],
        [36_100_000_000_000_000, None, None, None],
    ]
    vector = features.feature_vector(features.USDC_WETH, 1_600_000_000, probes)
    assert vector[0] == 1.0
    assert vector[2 + 4] == 0.0
    assert vector[2] < 0.0
    assert vector[10 + 1] == 1.0
    assert vector[10] == 0.0


def test_no_reference_price_declines():
    assert features.feature_vector(features.WETH_USDC, 64, [[None] * 4, [None] * 4]) is None


def test_feature_names_match_rust_layout():
    assert features.FEATURE_COUNT == 18
    assert features.FEATURE_NAMES[2] == "impact_bps_p1_q16"
    assert features.FEATURE_NAMES[17] == "failed_p2_q1"


def test_split_conserves_huge_amounts_and_endpoints():
    amount = 340282366920938463463374607431768211457
    for numerator in (0, 1, 12_345, allocation.DENOMINATOR - 1, allocation.DENOMINATOR):
        first, second = allocation.split(amount, numerator)
        assert first + second == amount
    assert allocation.split(amount, 0)[0] == 0
    assert allocation.split(amount, allocation.DENOMINATOR)[1] == 0


def test_numerator_rounds_and_clamps():
    assert allocation.numerator(-0.2) == 0
    assert allocation.numerator(1.7) == allocation.DENOMINATOR
    assert allocation.numerator(0.5) == allocation.DENOMINATOR // 2
    assert allocation.numerator(1.5 / allocation.DENOMINATOR) == 2
    with pytest.raises(ValueError):
        allocation.numerator(float("nan"))


def test_interpolation_ties_go_to_the_smallest_step():
    split = interpolation.interpolation_split(
        6_400, outputs([800, 3200, 6400, 12800], [800, 3200, 6400, 12800])
    )
    assert split == (0, 6_400)


def test_interpolation_moves_flow_to_the_pool_that_saturates_later():
    split = interpolation.interpolation_split(
        6_400, outputs([400, 1600, 3200, 6400], [500, 2000, 2000, 2000])
    )
    assert split == (4_800, 1_600)


def test_interpolation_never_extrapolates_past_failed_probes():
    split = interpolation.interpolation_split(
        6_400, outputs([400, 1600, 3200, 6400], [500, 2000, None, None])
    )
    assert split[1] <= 1_600


def test_reference_share_matches_the_rust_unit_case(tmp_path):
    artifact = model.artifact(
        features.FEATURE_SPEC_VERSION,
        [1],
        [0.0],
        [2.0],
        [([[1.0]], [0.5], "identity")],
        {"kind": "share"},
    )
    path = tmp_path / "model.json"
    model.save(artifact, path)
    loaded = model.load(path)
    vector = [0.0] * features.FEATURE_COUNT
    vector[1] = 3.0
    assert model.share(loaded, vector) == 1.0 / (1.0 + math.exp(-(3.0 / 2.0 + 0.5)))


def test_coarse_offset_output_stays_within_its_chunks():
    artifact = model.artifact(
        features.OFFSET_SPEC_VERSION,
        [2],
        [0.0],
        [1.0],
        [([[1000.0]], [0.0], "identity")],
        {"kind": "coarse_offset", "chunks": 1.0},
    )
    vector = [0.0] * features.OFFSET_FEATURE_COUNT
    vector[2] = 0.5
    assert model.share(artifact, vector, coarse_share=0.5) == 0.5 + 1.0 / 20
    vector[2] = -0.5
    assert model.share(artifact, vector, coarse_share=0.0) == 0.0


def test_offset_features_measure_full_order_price_gap():
    vector = features.offset_feature_vector(
        features.WETH_USDC, 10**18, 6 * 10**17, [2_000_000_000, 1_000_000_000], -0.25
    )
    assert vector[2] == 0.6
    assert abs(vector[3] - 10_000 * math.log(2.0)) < 1e-9
    assert vector[4:] == [0.0, 0.0, -0.25]
    assert features.offset_feature_vector(features.WETH_USDC, 10**18, 0, [None, None], 0.0) is None


def test_coarse_replay_gives_each_chunk_to_the_better_marginal():
    # Pool 1 pays 2 per unit up to 50 then nothing; pool 2 pays 1 per unit: 10 chunks of 5 each.
    def out(pool, x):
        return min(x, 50) * 2 if pool == 0 else x

    assert coarse.coarse_split(100, out) == 50


def parabola(slope, first):
    """Pins `parabola` in the Rust tests: out(t) = slope * t - t^2 with knots from `first`."""

    def out(t):
        return slope * t - t * t

    return first, out(first + 1) - out(first), out(first + 2) - 2 * out(first + 1) + out(first)


def test_quadratic_offset_matches_rust_cases():
    for first, second in [(-1, -1), (0, -1), (-2, 0)]:
        offset = quadratic.best_offset(parabola(1.0, first), parabola(0.2, second), -1.0, 1.0)
        assert abs(offset - 0.2) < 1e-12
    assert quadratic.best_offset(parabola(10.0, -1), parabola(0.2, -1), -1.0, 1.0) == 1.0
    assert quadratic.best_offset(parabola(-10.0, 0), parabola(0.2, -1), 0.0, 1.0) == 0.0


def test_coarse_pass_records_each_pools_chunk_gains():
    # Pool 1 pays 2 per unit up to 50 then nothing; pool 2 pays 1 per unit: 10 chunks of 5 each.
    def out(pool, x):
        return min(x, 50) * 2 if pool == 0 else x

    first, gains = coarse.coarse_pass(100, out)
    assert first == 50
    # Pool 1 won chunks 1-10 (10 each); pool 2 won the rest and so the last chunk: no bid.
    assert gains[0] == ([10] * 10, 0)
    assert gains[1] == ([5] * 10, None)


def test_free_quadratic_finds_the_balanced_split():
    # Identical concave pools and a 60/40 coarse split of 1,000 (chunk 50): the best split is
    # 50/50, two chunks away, so the quadratics through the chunk gains move the cap, one chunk.
    def out(x):
        return 4000 * x - x * x

    def gains(start, count):
        return [out(start + 50 * (k + 1)) - out(start + 50 * k) for k in range(count)]

    pool1 = (gains(0, 12), out(650) - out(600))
    pool2 = (gains(0, 8), None)
    assert quadratic.free_quadratic_offset(1000, 600, [pool1, pool2]) == -1.0


def confidence_artifact():
    """The Rust `confidence_model`: reads coarse_share / 2, bounds only log10 notional to [2, 4]."""
    low, high = [None] * features.OFFSET_FEATURE_COUNT, [None] * features.OFFSET_FEATURE_COUNT
    low[1], high[1] = 2.0, 4.0
    return model.artifact(
        features.OFFSET_SPEC_VERSION,
        [2],
        [0.0],
        [2.0],
        [([[1.0]], [0.0], "identity")],
        {
            "kind": "regret_estimate",
            "skip_at_most_log10_bp": 1.5,
            "envelope": {"low": low, "high": high},
        },
    )


def test_skips_only_at_or_below_the_threshold_inside_the_envelope():
    artifact = confidence_artifact()

    def skips(notional, coarse_share):
        vector = [0.0] * features.OFFSET_FEATURE_COUNT
        vector[1], vector[2] = notional, coarse_share
        return model.regret_estimate(artifact, vector)[1]

    assert skips(3.0, 3.0), "an estimate equal to the threshold skips"
    assert not skips(3.0, 3.0 + 1e-9), "an estimate above the threshold refines"
    assert skips(2.0, 0.0) and skips(4.0, 0.0), "the envelope is inclusive"
    assert not skips(2.0 - 1e-9, 0.0) and not skips(4.0 + 1e-9, 0.0), "outside it refines"
    assert not skips(math.nan, 0.0) and not skips(3.0, math.nan), "NaN never skips"
