"""Water-fill's 20-chunk coarse pass, replayed from quote curves.

Mirrors `disjoint_waterfill` in Fynd for two single-hop paths: chunk 0 is `Q // 20` plus the
remainder, the rest `Q // 20`; each chunk goes to the path whose next chunk adds the most gross
output, first path on ties. Fynd simulates each chunk against the path's own post-swap state;
here a chunk's output is `out(x + chunk) - out(x)` from untouched-state quotes, which for a
Uniswap V3 pool differs by rounding only. Fynd also charges a path's gas once, on the first chunk
it wins; at Base gas prices that is far below one chunk's output for the orders we model, and
`scripts/validate_coarse.py` checks the replay against Fynd's own coarse splits.
"""

COARSE_CHUNKS = 20


def coarse_amounts(amount):
    base = amount // COARSE_CHUNKS
    return [base + (amount - COARSE_CHUNKS * base)] + [base] * (COARSE_CHUNKS - 1)


def coarse_pass(amount, out):
    """Replays the pass: (pool 1's coarse amount, each pool's chunk gains), or None.

    A pool's chunk gains are `(paid, next)`: what each chunk it won added, in order, and what its
    next chunk would add when the pass ends, None for the pool that won the last chunk (Fynd
    forgets the winner's price) or a failed quote. Mirrors `ChunkGains` in the Fynd patch.
    `out(pool, x)` is pool's gross output for input `x` (None fails).
    """
    allocated = [0, 0]
    paid = [[], []]
    gains = [None, None]
    winner = None
    for chunk in coarse_amounts(amount):
        gains = []
        for pool in range(2):
            before, after = out(pool, allocated[pool]), out(pool, allocated[pool] + chunk)
            gains.append(None if before is None or after is None else after - before)
        if gains[0] is None and gains[1] is None:
            return None
        winner = 0 if gains[1] is None or (gains[0] is not None and gains[0] >= gains[1]) else 1
        allocated[winner] += chunk
        paid[winner].append(gains[winner])
    standing = [None if pool == winner else gains[pool] for pool in range(2)]
    return allocated[0], [(paid[pool], standing[pool]) for pool in range(2)]


def coarse_split(amount, out):
    """Pool 1's coarse amount; `out(pool, x)` is pool's gross output for input `x` (None fails)."""
    replay = coarse_pass(amount, out)
    return None if replay is None else replay[0]
