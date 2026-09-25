"""Continuous best split from sampled quote curves.

Each pool's gross output curve `out(x)` is known at a set of input amounts. A shape-preserving
cubic (PCHIP, Fritsch-Carlson) through those samples keeps the curve monotone and never
overshoots between samples, so the combined output `out1(x) + out2(Q - x)` can be maximized
over a continuous `x` instead of only at sampled splits. This is the label resolution the
coarse-correction model needs: its target moves by fractions of one 1/20 chunk.
"""

import numpy as np


def pchip_slopes(x, y):
    """Fritsch-Carlson derivative estimates at each knot."""
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    if len(x) == 2:
        slopes[:] = delta[0]
        return slopes
    for i in range(1, len(x) - 1):
        if delta[i - 1] * delta[i] <= 0:
            slopes[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    slopes[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return slopes


def _end_slope(h0, h1, d0, d1):
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(slope) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(slope) > abs(3 * d0):
        return 3 * d0
    return slope


def pchip(x, y, query):
    """Evaluates the PCHIP interpolant of (x, y) at `query` (inside [x[0], x[-1]])."""
    x, y, query = np.asarray(x, float), np.asarray(y, float), np.asarray(query, float)
    slopes = pchip_slopes(x, y)
    index = np.clip(np.searchsorted(x, query, side="right") - 1, 0, len(x) - 2)
    h = x[index + 1] - x[index]
    t = (query - x[index]) / h
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return (
        h00 * y[index] + h10 * h * slopes[index] + h01 * y[index + 1] + h11 * h * slopes[index + 1]
    )


def best_share(amount, samples, resolution=1 << 14):
    """Pool 1's share of `amount` maximizing interpolated combined output.

    `samples[p]` maps input amounts (ints, including 0 and `amount`) to pool p's output. Failed
    quotes are left out of `samples`; the search stays inside amounts both pools have covered.
    Returns (share, interpolated combined output at that share).
    """
    curves = []
    for pool in samples:
        xs = sorted(pool)
        curves.append((np.array(xs, float) / amount, np.array([pool[x] for x in xs], float)))
    (x1, y1), (x2, y2) = curves
    shares = np.linspace(0.0, 1.0, resolution + 1)
    feasible = (shares <= x1[-1]) & (1 - shares <= x2[-1])
    shares = shares[feasible]
    totals = pchip(x1, y1, shares) + pchip(x2, y2, 1 - shares)
    best = int(np.argmax(totals))
    return float(shares[best]), float(totals[best])
