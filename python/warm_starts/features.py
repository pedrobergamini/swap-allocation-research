"""Probe schedule and the v1 feature vector.

Mirrors `src/features.rs` operation for operation: same constants, same order of floating-point
steps, platform `log`. Amounts stay Python ints until the one `float(int)` conversion Rust also
makes, so the offline dataset and the Rust runner produce identical vectors.
"""

import math

WETH_USDC = "weth-usdc"
USDC_WETH = "usdc-weth"
DIRECTION_INDEX = {WETH_USDC: 0, USDC_WETH: 1}

PROBE_FRACTIONS = ((1, 16), (1, 4), (1, 2), (1, 1))
PROBES_PER_POOL = len(PROBE_FRACTIONS)
FEATURE_SPEC_VERSION = "probe-impact/1"
FEATURE_NAMES = (
    "direction",
    "log10_notional_usdc",
    *(f"impact_bps_p{pool}_{tag}" for pool in (1, 2) for tag in ("q16", "q4", "q2", "q1")),
    *(f"failed_p{pool}_{tag}" for pool in (1, 2) for tag in ("q16", "q4", "q2", "q1")),
)
FEATURE_COUNT = len(FEATURE_NAMES)

# Output-per-input in base units to whole tokens, written out as in Rust.
_UNIT_SCALE = {WETH_USDC: 1e12, USDC_WETH: 1e-12}


def probe_amounts(amount):
    """Probe input amounts for an order of `amount`, rounded down."""
    return [amount * numerator // denominator for numerator, denominator in PROBE_FRACTIONS]


def feature_vector(direction, amount, outputs):
    """Feature vector for `outputs[pool][probe]` (gross output ints, `None` for a failed quote).

    Returns `None` when neither pool quoted its smallest probe.
    """
    amounts = probe_amounts(amount)
    scale = _UNIT_SCALE[direction]
    prices = [[None] * PROBES_PER_POOL for _ in range(2)]
    for pool in range(2):
        for probe in range(PROBES_PER_POOL):
            out = outputs[pool][probe]
            if out is not None:
                amount_in = float(amounts[probe])
                if amount_in > 0.0:
                    prices[pool][probe] = float(out) / amount_in * scale

    first, second = prices[0][0], prices[1][0]
    if first is None and second is None:
        return None
    reference = (
        max(first, second)
        if first is not None and second is not None
        else (first if first is not None else second)
    )
    if reference <= 0.0:
        return None

    features = [0.0] * FEATURE_COUNT
    features[0] = float(DIRECTION_INDEX[direction])
    features[1] = math.log10(_notional_usdc(direction, amount, reference))
    for pool in range(2):
        for probe in range(PROBES_PER_POOL):
            slot = pool * PROBES_PER_POOL + probe
            price = prices[pool][probe]
            if price is not None and price > 0.0:
                features[2 + slot] = 10_000.0 * math.log(price / reference)
            else:
                features[2 + 2 * PROBES_PER_POOL + slot] = 1.0
    return features


def _notional_usdc(direction, amount, reference):
    if direction == WETH_USDC:
        return float(amount) / 1e18 * reference
    return float(amount) / 1e6


# Coarse-offset features: only what the initializer knows without a new simulation. It gets the
# coarse split in the request, and each pool's quote for the whole order is already in the solve's
# swap cache from discovery, so probing it is a cache hit.
OFFSET_SPEC_VERSION = "coarse-offset/2"
OFFSET_FEATURE_NAMES = (
    "direction",
    "log10_notional_usdc",
    "coarse_share",
    "full_price_gap_bps",
    "failed_p1_full",
    "failed_p2_full",
    "quadratic_offset_chunks",
)
OFFSET_FEATURE_COUNT = len(OFFSET_FEATURE_NAMES)


def offset_feature_vector(direction, amount, coarse_first, full_outputs, quadratic_offset_chunks):
    """Features from the coarse split and each pool's full-order output (None if failed).

    `full_price_gap_bps` is `10_000 * ln(price1 / price2)` for the full order through each pool.
    `quadratic_offset_chunks` is the free quadratic's move of pool 1 from the coarse split, in
    chunks (`quadratic.free_quadratic_offset`). Returns None when neither pool quoted the full
    order.
    """
    scale = _UNIT_SCALE[direction]
    prices = [
        None if out is None or out <= 0 else float(out) / float(amount) * scale
        for out in full_outputs
    ]
    known = [price for price in prices if price is not None]
    if not known:
        return None
    reference = max(known)
    features = [0.0] * OFFSET_FEATURE_COUNT
    features[0] = float(DIRECTION_INDEX[direction])
    features[1] = math.log10(_notional_usdc(direction, amount, reference))
    features[2] = float(coarse_first) / float(amount)
    if prices[0] is not None and prices[1] is not None:
        features[3] = 10_000.0 * math.log(prices[0] / prices[1])
    features[4] = 1.0 if prices[0] is None else 0.0
    features[5] = 1.0 if prices[1] is None else 0.0
    features[6] = quadratic_offset_chunks
    return features
