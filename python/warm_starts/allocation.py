"""Turns a predicted fraction into an exact integer split.

Mirrors `src/allocation.rs`: pool 1 gets `amount * numerator // DENOMINATOR`, pool 2 the exact
remainder, and the order amount never passes through floating point.
"""

import math

DENOMINATOR = 1 << 16


def numerator(fraction):
    """The numerator nearest to `fraction`, clamped to [0, DENOMINATOR]; halves round up."""
    if not math.isfinite(fraction):
        raise ValueError(f"allocation fraction must be finite, got {fraction}")
    return math.floor(min(max(fraction, 0.0), 1.0) * float(DENOMINATOR) + 0.5)


def split(amount, numerator):
    if not 0 <= numerator <= DENOMINATOR:
        raise ValueError(f"numerator {numerator} outside [0, {DENOMINATOR}]")
    first = amount * numerator // DENOMINATOR
    return first, amount - first
