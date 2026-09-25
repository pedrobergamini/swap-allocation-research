"""Probe interpolation baseline, mirroring `interpolation_split` in `src/seeds.rs`.

Each pool's output curve is the piecewise-linear line through (0, 0) and its successful probes.
The split is the best of `k / INTERPOLATION_STEPS` of the order by estimated gross output, ties
going to the smallest `k`; a split beyond a pool's last probe is never extrapolated.
"""

from itertools import pairwise

from .features import probe_amounts

INTERPOLATION_STEPS = 64


def interpolation_split(amount, outputs):
    """Returns (pool1, pool2) amounts, or raises ValueError like the Rust `Err` cases."""
    amounts = probe_amounts(amount)
    knots = []
    for pool in range(2):
        points = [(0.0, 0.0)]
        for probe, amount_in in enumerate(amounts):
            if outputs[pool][probe] is not None:
                points.append((float(amount_in), float(outputs[pool][probe])))
        points.sort(key=lambda point: point[0])
        deduped = [points[0]]
        for point in points[1:]:
            if point[0] != deduped[-1][0]:
                deduped.append(point)
        knots.append(deduped)
    if any(len(pool) < 2 for pool in knots):
        raise ValueError("a pool has no successful probe")

    best = None
    for step in range(INTERPOLATION_STEPS + 1):
        first = amount * step // INTERPOLATION_STEPS
        second = amount - first
        estimates = [_interpolate(knots[0], float(first)), _interpolate(knots[1], float(second))]
        if None in estimates:
            continue
        estimate = estimates[0] + estimates[1]
        if best is None or estimate > best[0]:
            best = (estimate, (first, second))
    if best is None:
        raise ValueError("no grid split falls inside both pools' probes")
    return best[1]


def _interpolate(knots, amount_in):
    if amount_in == 0.0:
        return 0.0
    for (x0, y0), (x1, y1) in pairwise(knots):
        if amount_in <= x1:
            return y0 + (y1 - y0) * (amount_in - x0) / (x1 - x0)
    return None
