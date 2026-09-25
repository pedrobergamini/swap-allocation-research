"""The certificate replay `confidence.py` uses to choose the certified seed's step, and its threshold scans."""

import numpy as np
from warm_starts.confidence import certified_bound, highest_passing_cut, largest_threshold


def parabola_curve(best):
    """A regret curve in bp over pool 1's share: 1e4 * (share - best)^2, on a fine grid."""
    share = np.linspace(0.0, 1.0, 100_001)
    return share, 1e4 * (share - best) ** 2


def test_certificate_bounds_a_start_within_one_step():
    curve = parabola_curve(0.5)
    start = np.array([0.5 + 1 / 1024])
    bound = certified_bound(start, [curve], 256)
    regret = 1e4 * (1 / 1024) ** 2
    # The drop to the worse neighbour bounds the start's regret from above.
    assert bound[0] >= regret
    assert np.isclose(bound[0], 1e4 * ((1 / 256 + 1 / 1024) ** 2 - (1 / 1024) ** 2), rtol=1e-3)


def test_certificate_refuses_when_a_neighbour_is_better():
    curve = parabola_curve(0.5)
    start = np.array([0.5 + 1 / 128])
    assert np.isinf(certified_bound(start, [curve], 256)[0])


def test_one_failing_cell_first_stops_the_scan_but_a_later_cut_passes():
    # 200 cells of one order each; the lowest-scored cell loses 0.03 bp, the rest nothing.
    scores = np.arange(200, dtype=float)
    regret = np.where(scores == 0, 0.03, 0.0)
    cells = np.arange(200)
    eligible = np.ones(200, dtype=bool)
    # Skipping only the first cell fails 1 of 1 cells, so the upward scan skips nothing...
    assert largest_threshold(scores, eligible, regret, cells) is None
    # ...while from 100 skipped cells on, one failing cell is within the 1 % share.
    assert highest_passing_cut(scores, eligible, regret, cells) == 199.0
