# Input contracts

The harness starts with prepared training cases and Fynd-compatible replay inputs from your own data stack. You choose how to source and reconstruct historical pool states. The formats below define what the training and evaluation tools consume. Supplied data must preserve these feature, label, and pool definitions.

## Prepared training cases

Input is Zstandard-compressed UTF-8 JSON Lines, one order per line. `warm_starts.dataset.read_jsonl_zst` reads this format. A usable allocation-training case contains:

| Field | Meaning |
|---|---|
| `id` | Unique string identifying the order at its checkpoint. |
| `split` | `train` or `validation` for experiment runs. `test` is reserved for a separate final evaluation and rejected by the experiment runner. |
| `capture` or `window` | Group identifier for related market states. Keep each group in one chronological split. |
| `block` | Integer checkpoint identifier. Required for Rust replay. |
| `direction` | `weth-usdc` or `usdc-weth`. |
| `amount_in` | Positive integer input amount in token base units, encoded as a decimal string. |
| `offset` | Object with `features` (seven finite numbers in the order below), or `null` if the initializer declined. Prepared replay cases can also carry `coarse_first` as a decimal-string diagnostic; training does not require it. |
| `label_share` | Target pool-1 share, from 0 to 1. |
| `regret_curve` | Optional object with increasing `share` coordinates and matching `bp` values. It describes sampled approximate net-output loss against the selected label, not the final Rust quality gate. |
| `declined` | Optional explanation for a case without usable offset features. |

The `coarse-offset/2` representation contains seven fields in a fixed order:

1. Direction, 0 for WETH to USDC and 1 for USDC to WETH.
2. Base-10 logarithm of USDC notional.
3. Pool-1 share from the coarse allocation.
4. `10,000 * ln(pool1 full-order price / pool2 full-order price)`.
5. Pool-1 whole-order quote failure flag.
6. Pool-2 whole-order quote failure flag.
7. The quadratic initializer's proposed correction in coarse chunks.

Both frozen v1 networks select indices `[0, 1, 2, 3, 6]`, giving five network inputs. The two failure flags remain part of the input contract even though those networks do not use them. The trainer receives all seven fields, then excludes those that are constant in its training split; exported artifacts record the selected indices.

Use `warm_starts.features.offset_feature_vector` and the Rust implementation in `src/features.rs` as the operational definitions. Pool identity and order are fixed by `src/pools.rs`. WETH uses 18 decimals and USDC uses 6. A model trained on a different pool order or meaning of these fields is not interchangeable with v1.

The included synthetic generator creates made-up feature/label pairs to exercise training. Synthetic validation results say nothing about market performance.

## Fynd replay inputs

The Rust runner reads the pinned Fynd `MarketRecording` format: Zstandard-compressed JSON with `metadata` and ordered `updates`. It is a serialized replay interface, not a required collection technique. Supply a complete starting state and updates sufficient to reconstruct both configured pools at every requested checkpoint. Fynd's own types at the pinned revision define the nested protocol-state schema; do not approximate those states with the public demo's numerical examples.

The Python preparation/evaluation tools require `metadata.chain` to be `base`, `metadata.schema_version` to be `1`, and `metadata.gas_price_wei` to be a nonnegative decimal string. Update checkpoint blocks must be nonnegative integers in strictly increasing order. The Rust runner requires the requested block to exist and both pools to have state by that point. Gas price is passed explicitly to each solve. No acquisition log or provider credential is required.

`warm_starts.recorded` builds training features and sampled allocation labels from prepared replay files. It and `warm_starts.score_test` associate captures by filenames ending in `-NN.json.zst`; for example, `market-01.json.zst` corresponds to `capture: 1`. The prepared-data builder uses `configs/dataset-recorded.json` to choose whole-capture splits, checkpoint stride, and synthetic order sizes. Its label is the best **sampled** split from a coarse grid and a local finer scan. It does not establish a global optimum, and its gas conversion is an approximation.

Direct `sar-bench` usage accepts any input filename. Its case JSON is:

```json
{"cases":[{"id":"example-order","direction":"weth-usdc","amount_in":"1000000000000000000"}]}
```

Use a nonempty case list with unique identities and nonzero amounts, and request checkpoint blocks contained in the replay. The [experiment replay manifest](experiments.md) connects validation cases to replay files without exposing their acquisition source.

## Split ownership

Choose chronological train, validation, and final-test groups before selecting model recipes. Do not distribute neighboring states from one capture between these splits. The experiment runner refuses final-test rows; the frozen evaluation command intentionally selects them. A held-out result becomes development evidence once used to choose the next candidate, so a newly selected candidate requires new untouched final evidence.
