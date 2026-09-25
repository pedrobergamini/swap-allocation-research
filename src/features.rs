//! Probe schedule and the v1 feature vector.
//!
//! A probe is one exact-input quote of a single pool at a fixed fraction of the order. The model
//! and the interpolation baseline see the same four probes per pool and nothing else about the
//! market. `python/warm_starts/features.py` mirrors this file operation for operation (same
//! constants, same order of floating-point steps, platform `log`), so the offline dataset and the
//! Rust runner produce identical vectors.

use num_bigint::BigUint;
use num_traits::ToPrimitive;

use crate::pools::Direction;

/// Probe amounts as `(numerator, denominator)` of the order: Q/16, Q/4, Q/2 and Q, rounded down.
pub const PROBE_FRACTIONS: [(u32, u32); 4] = [(1, 16), (1, 4), (1, 2), (1, 1)];
pub const PROBES_PER_POOL: usize = PROBE_FRACTIONS.len();

/// Version of the feature layout below; a model artifact names the version it was trained on.
pub const FEATURE_SPEC_VERSION: &str = "probe-impact/1";

/// Feature names in vector order.
pub const FEATURE_NAMES: [&str; 18] = [
    "direction",
    "log10_notional_usdc",
    "impact_bps_p1_q16",
    "impact_bps_p1_q4",
    "impact_bps_p1_q2",
    "impact_bps_p1_q1",
    "impact_bps_p2_q16",
    "impact_bps_p2_q4",
    "impact_bps_p2_q2",
    "impact_bps_p2_q1",
    "failed_p1_q16",
    "failed_p1_q4",
    "failed_p1_q2",
    "failed_p1_q1",
    "failed_p2_q16",
    "failed_p2_q4",
    "failed_p2_q2",
    "failed_p2_q1",
];
pub const FEATURE_COUNT: usize = FEATURE_NAMES.len();

/// The probe input amounts for an order of `amount`.
pub fn probe_amounts(amount: &BigUint) -> [BigUint; PROBES_PER_POOL] {
    PROBE_FRACTIONS.map(|(numerator, denominator)| amount * numerator / denominator)
}

/// Gross outputs of every probe, pool 1 first; `None` is a failed quote, never zero output.
pub type ProbeOutputs = [[Option<BigUint>; PROBES_PER_POOL]; 2];

/// Builds the feature vector, or `None` when neither pool quoted its smallest probe, which leaves
/// no reference price to measure impact against.
///
/// `impact_bps` is `10_000 * ln(price / reference)`, where `price` is a probe's output per input
/// in whole tokens and `reference` is the better of the two pools' Q/16 prices. It is zero or
/// negative for a sell order and grows in magnitude with price impact. A failed probe contributes
/// impact 0 and failure flag 1.
pub fn feature_vector(
    direction: Direction,
    amount: &BigUint,
    outputs: &ProbeOutputs,
) -> Option<[f64; FEATURE_COUNT]> {
    let amounts = probe_amounts(amount);
    let scale = unit_scale(direction);
    let mut prices = [[None; PROBES_PER_POOL]; 2];
    for pool in 0..2 {
        for probe in 0..PROBES_PER_POOL {
            if let Some(out) = &outputs[pool][probe] {
                let amount_in = to_f64(&amounts[probe]);
                if amount_in > 0.0 {
                    prices[pool][probe] = Some(to_f64(out) / amount_in * scale);
                }
            }
        }
    }
    let reference = match (prices[0][0], prices[1][0]) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return None,
    };
    if reference <= 0.0 {
        return None;
    }

    let mut features = [0.0; FEATURE_COUNT];
    features[0] = f64::from(direction.index());
    features[1] = notional_usdc(direction, amount, reference).log10();
    for (pool, pool_prices) in prices.iter().enumerate() {
        for (probe, price) in pool_prices.iter().copied().enumerate() {
            let slot = pool * PROBES_PER_POOL + probe;
            match price {
                Some(price) if price > 0.0 => {
                    features[2 + slot] = 10_000.0 * (price / reference).ln();
                }
                _ => features[2 + 2 * PROBES_PER_POOL + slot] = 1.0,
            }
        }
    }
    Some(features)
}

/// Converts output-per-input in base units to whole tokens: `10^(decimals_in - decimals_out)`,
/// written out because `powi` is not guaranteed exact.
fn unit_scale(direction: Direction) -> f64 {
    match direction {
        Direction::WethUsdc => 1e12,
        Direction::UsdcWeth => 1e-12,
    }
}

/// The order's size in whole USDC, using the reference price for WETH.
fn notional_usdc(direction: Direction, amount: &BigUint, reference: f64) -> f64 {
    match direction {
        Direction::WethUsdc => to_f64(amount) / 1e18 * reference,
        Direction::UsdcWeth => to_f64(amount) / 1e6,
    }
}

/// Version of the coarse-offset layout below.
pub const OFFSET_SPEC_VERSION: &str = "coarse-offset/2";

/// Coarse-offset feature names in vector order.
pub const OFFSET_FEATURE_NAMES: [&str; 7] = [
    "direction",
    "log10_notional_usdc",
    "coarse_share",
    "full_price_gap_bps",
    "failed_p1_full",
    "failed_p2_full",
    "quadratic_offset_chunks",
];
pub const OFFSET_FEATURE_COUNT: usize = OFFSET_FEATURE_NAMES.len();

/// Features from what the initializer knows without a new simulation: the coarse split, and each
/// pool's output for the whole order (`None` if it failed), which discovery already cached.
///
/// `full_price_gap_bps` is `10_000 * ln(price1 / price2)` for the whole order through each pool.
/// `quadratic_offset_chunks` is where the quadratics through the coarse pass's chunk gains put the
/// best split, as pool 1's move from the coarse split in chunks (within one chunk); the model
/// learns its correction on top of that estimate. Returns `None` when neither pool quoted the whole
/// order.
pub fn offset_feature_vector(
    direction: Direction,
    amount: &BigUint,
    coarse_first: &BigUint,
    full_outputs: &[Option<BigUint>; 2],
    quadratic_offset_chunks: f64,
) -> Option<[f64; OFFSET_FEATURE_COUNT]> {
    let scale = unit_scale(direction);
    let amount_f = to_f64(amount);
    let prices = full_outputs.clone().map(|out| {
        out.map(|out| to_f64(&out)).filter(|out| *out > 0.0).map(|out| out / amount_f * scale)
    });
    let reference = match prices {
        [Some(a), Some(b)] => a.max(b),
        [Some(a), None] | [None, Some(a)] => a,
        [None, None] => return None,
    };
    let mut features = [0.0; OFFSET_FEATURE_COUNT];
    features[0] = f64::from(direction.index());
    features[1] = notional_usdc(direction, amount, reference).log10();
    features[2] = to_f64(coarse_first) / amount_f;
    if let [Some(first), Some(second)] = prices {
        features[3] = 10_000.0 * (first / second).ln();
    }
    features[4] = if prices[0].is_none() { 1.0 } else { 0.0 };
    features[5] = if prices[1].is_none() { 1.0 } else { 0.0 };
    features[6] = quadratic_offset_chunks;
    Some(features)
}

fn to_f64(value: &BigUint) -> f64 {
    value.to_f64().expect("BigUint always converts to f64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_amounts_round_down() {
        let amounts = probe_amounts(&BigUint::from(1_000_001u32));
        assert_eq!(amounts.map(|a| a.to_string()), ["62500", "250000", "500000", "1000001"]);
    }

    #[test]
    fn failed_probes_are_flagged_not_zero() {
        let amount = BigUint::from(1_600_000_000u64);
        let mut outputs: ProbeOutputs = Default::default();
        outputs[0][0] = Some(BigUint::from(36_000_000_000_000_000u64));
        outputs[1][0] = Some(BigUint::from(36_100_000_000_000_000u64));
        let features = feature_vector(Direction::UsdcWeth, &amount, &outputs).unwrap();
        assert_eq!(features[0], 1.0);
        assert_eq!(features[2 + 4], 0.0, "pool 2's Q/16 price is the reference");
        assert!(features[2] < 0.0);
        assert_eq!(features[10 + 1], 1.0, "pool 1 Q/4 failed");
        assert_eq!(features[10], 0.0);
        assert_eq!(features[2 + 1], 0.0);
    }

    #[test]
    fn no_reference_price_declines() {
        let outputs: ProbeOutputs = Default::default();
        assert!(feature_vector(Direction::WethUsdc, &BigUint::from(64u8), &outputs).is_none());
    }
}
