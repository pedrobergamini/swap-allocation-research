"""The non-learned `quadratic` seed and the `quadratic_offset_chunks` feature, from chunk gains.

Mirrors `free_quadratic_offset` in `src/seeds.rs`: each pool's output near its coarse amount is the
quadratic through three points one coarse chunk apart, read off what the coarse pass's chunks
added to it (`coarse.coarse_pass`), so it costs no simulation. Pool 1 then moves, within one
chunk, to where the two quadratics' sum peaks.
"""


def from_gains(paid, next_gain):
    """(first point offset, step, curvature) from a pool's chunk gains, or None if it won none.

    With a standing bid the points are the last chunk won and the next one (first = -1); without
    one, the last two chunks won (first = -2); one chunk and no bid is taken as linear.
    """
    if not paid:
        return None
    if next_gain is not None:
        first, earlier, later = -1, paid[-1], next_gain
    elif len(paid) >= 2:
        first, earlier, later = -2, paid[-2], paid[-1]
    else:
        first, earlier, later = -1, paid[0], paid[0]
    # Exact integer difference first, as in Rust, before any rounding to float.
    return float(first), float(earlier), float(later - earlier)


def gain(quadratic, t):
    first, step, curvature = quadratic
    u = t - first
    return u * step + u * (u - 1.0) * curvature / 2.0


def best_offset(first, second, low, high):
    """Move of pool 1, in chunks within [low, high], maximizing gain1(s) + gain2(-s)."""
    candidates = [low, 0.0, high]
    concavity = first[2] + second[2]
    if concavity < 0.0:
        slope_at_zero = (
            first[1]
            - second[1]
            - first[2] * (2.0 * first[0] + 1.0) / 2.0
            + second[2] * (2.0 * second[0] + 1.0) / 2.0
        )
        candidates.append(min(max(-slope_at_zero / concavity, low), high))
    best, best_total = 0.0, gain(first, 0.0) + gain(second, 0.0)
    for s in candidates:
        total = gain(first, s) + gain(second, -s)
        if total > best_total:
            best, best_total = s, total
    return best


def free_quadratic_offset(amount, coarse_first, gains):
    """Pool 1's move in chunks; `gains` is `coarse.coarse_pass`'s per-pool chunk gains."""
    fits = [from_gains(paid, next_gain) for paid, next_gain in gains]
    if None in fits:
        return None
    amount_f = float(amount)
    chunk_f = amount_f / 20.0
    coarse_f = float(coarse_first)
    low = max(-1.0, -coarse_f / chunk_f)
    high = min(1.0, (amount_f - coarse_f) / chunk_f)
    return best_offset(fits[0], fits[1], low, high)


def quadratic_share(features):
    """The quadratic seed's pool 1 share from a `coarse-offset/2` feature vector."""
    return min(max(features[2] + features[6] / 20.0, 0.0), 1.0)
