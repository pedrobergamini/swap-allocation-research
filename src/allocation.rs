//! Turns a predicted fraction into an exact integer split.
//!
//! The order amount never passes through floating point: the fraction becomes an integer
//! numerator over [`DENOMINATOR`], pool 1 gets `amount * numerator / DENOMINATOR` rounded down,
//! and pool 2 gets the exact remainder. The Python exporter implements the same rule, and the
//! parity tests hold the two together.

use num_bigint::BigUint;

/// Resolution of an allocation fraction. Finer than exchange refinement's smallest step
/// (`amount / 16384`), so rounding the fraction never decides where refinement starts.
pub const DENOMINATOR: u32 = 1 << 16;

/// The numerator nearest to `fraction`, clamped to `[0, DENOMINATOR]`. Half-way cases round
/// away from zero, like Python's `math.floor(x + 0.5)` for non-negative `x`.
///
/// # Panics
///
/// When `fraction` is not finite; the model format guarantees a finite output.
pub fn numerator(fraction: f64) -> u32 {
    assert!(fraction.is_finite(), "allocation fraction must be finite, got {fraction}");
    let scaled = (fraction.clamp(0.0, 1.0) * f64::from(DENOMINATOR) + 0.5).floor();
    scaled as u32
}

/// Pool 1's and pool 2's amounts; they always sum to `amount`.
pub fn split(amount: &BigUint, numerator: u32) -> [BigUint; 2] {
    assert!(numerator <= DENOMINATOR, "numerator {numerator} exceeds {DENOMINATOR}");
    let first = amount * numerator / DENOMINATOR;
    let second = amount - &first;
    [first, second]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_conserves_huge_amounts_and_endpoints() {
        let amount: BigUint = "340282366920938463463374607431768211457".parse().unwrap();
        for numerator in [0, 1, 12_345, DENOMINATOR - 1, DENOMINATOR] {
            let [first, second] = split(&amount, numerator);
            assert_eq!(&first + &second, amount);
        }
        assert_eq!(split(&amount, 0)[0], BigUint::from(0u8));
        assert_eq!(split(&amount, DENOMINATOR)[1], BigUint::from(0u8));
    }

    #[test]
    fn numerator_rounds_and_clamps() {
        assert_eq!(numerator(-0.2), 0);
        assert_eq!(numerator(1.7), DENOMINATOR);
        assert_eq!(numerator(0.5), DENOMINATOR / 2);
        assert_eq!(numerator(1.5 / f64::from(DENOMINATOR)), 2);
    }
}
