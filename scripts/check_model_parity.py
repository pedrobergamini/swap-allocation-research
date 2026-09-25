"""Replays a learned seed's predictions in the Python reference and compares them bit for bit.

sar-bench logs, for every solve a learned seed handled, the feature vector it read, the share it
chose and its proposed split. This recomputes the share with `warm_starts.model.share` from the
same features and the same artifact, then the integer split, and reports every solve whose split
differs from the seed's logged proposal. With a confidence artifact it also recomputes the
logged regret estimate (exactly equal) and the skip decision (for `offset_certified`, given the
logged certificate bound). It covers inference and allocation;
feature extraction parity is checked separately against quote tables.

Usage: python scripts/check_model_parity.py BENCH.jsonl METHOD MODEL.json [CONFIDENCE.json]
"""

import json
import sys

from warm_starts import allocation, model

COARSE_SHARE_FEATURE = 2
# `CERTIFIED_MAX_BP` in src/seeds.rs.
CERTIFIED_MAX_BP = 0.02


def main():
    rows_path, method, model_path, *confidence_path = sys.argv[1:]
    artifact = model.load(model_path)
    confidence = model.load(confidence_path[0]) if confidence_path else None
    offset = artifact["output"]["kind"] == "coarse_offset"
    checked, mismatches = 0, []
    with open(rows_path) as file:
        for line in file:
            row = json.loads(line)
            if row["method"] != method or not row["seeds"]:
                continue
            # One record per seeded fine pass: the disjoint split first, then fill-and-spill.
            for seed in row["seeds"]:
                if seed["proposal"] is None:
                    continue
                features = seed["features"]
                coarse = features[COARSE_SHARE_FEATURE] if offset else None
                numerator = allocation.numerator(model.share(artifact, features, coarse))
                first, second = allocation.split(int(row["amount_in"]), numerator)
                checked += 1
                python = {"proposal": [str(first), str(second)]}
                rust = {"proposal": seed["proposal"]}
                if confidence is not None:
                    estimate, skip = model.regret_estimate(confidence, features)
                    # The certified seed quotes its neighbours only when the rule says skip, and
                    # then skips only on a small enough bound; the bound itself needs the quotes.
                    bound = seed.get("certified_bound_bp")
                    if method == "offset_certified":
                        skip = skip and bound is not None and bound <= CERTIFIED_MAX_BP
                    elif bound is not None:
                        skip = None
                    python |= {"estimate": estimate, "skipped": skip}
                    rust |= {
                        "estimate": seed["regret_estimate_log10_bp"],
                        "skipped": seed["skipped"],
                    }
                if python != rust:
                    mismatches.append(
                        {
                            "case": row["case_id"],
                            "block": row["block"],
                            "python": python,
                            "rust": rust,
                        }
                    )
    print(
        json.dumps(
            {"checked": checked, "mismatches": len(mismatches), "examples": mismatches[:5]},
            indent=1,
        )
    )
    sys.exit(1 if mismatches or not checked else 0)


if __name__ == "__main__":
    main()
