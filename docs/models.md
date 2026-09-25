# Frozen allocation networks

`models/v1/offset.json` contains the neural allocation initializer; `models/v1/confidence.json` contains its allocation-regret estimator. These are the released numerical weights. Their bytes match the original frozen models. The adjacent `manifest.json` uses format `sar-release-models/1`, identifies both files by relative path and SHA-256, and records the fixed certificate step divisor. `SHA256SUMS` permits verification without Python:

```sh
(cd models/v1 && shasum -a 256 -c SHA256SUMS)
```

The release manifest is sufficient to load and verify the public model package.

## Numerical interface

Both model files use `sar-mlp/2` and `coarse-offset/2`. Each is a multilayer perceptron (MLP) with dimensions 5–32–32–1 and 1,281 learned parameters, including biases. The feature names and order are embedded in each artifact. Both artifacts select indices `[0, 1, 2, 3, 6]` from the seven-field representation; the two quote-failure flags are not network inputs. `mean` and `std` normalize the five selected inputs. Each layer stores weights in output-by-input order, a bias vector, and an activation. Hidden layers use `tanh`; the last layer is linear before the output transformation.

The allocation network's `coarse_offset` output applies a sigmoid, maps it into a bounded correction around the coarse share, and clamps the result to [0, 1]. V1's reach is one of the 20 coarse chunks. Runtime allocation converts the share to a numerator with denominator 65,536 and uses integer arithmetic so the two pool amounts sum to the exact input. The Python reference and Rust implementation are tested together.

The allocation-regret estimator uses the `regret_estimate` output to predict the allocation network's starting output loss in log10 basis points against the sampled reference. This is a loss estimate, not a probability of correctness. It is specific to these allocation weights and their recorded feature envelope. Do not attach it to a newly trained allocation model. New experiment candidates run refinement unconditionally.

## Runtime methods

| Method | Behavior |
|---|---|
| `water_fill` | Fynd without an initializer override; the reference behavior. |
| `coarse` | Reuses the coarse allocation for the fine initialization, then refines. |
| `offset` | Uses the neural allocation initializer, then refines. |
| `offset_confident` | Skips based on the regret estimate alone; disabled after a development quality failure and retained as a comparison. |
| `offset_certified` | Uses the allocation-regret estimator and refinement skip check before omitting refinement. |
| `quadratic` | Uses an analytic coarse correction as another non-learned comparison. |

The refinement skip check uses nearby pool quotes to bound **gross-output** loss. Its existing method identifier is `offset_certified`. It is not a certificate of net output after gas, generalization, or global optimality. The step divisor is 512 for frozen v1. An empirical final net-output comparison against completed Fynd solves is a separate check.

The released method was evaluated on two WETH/USDC pools on Base. The [report](../reports/REPORT.md) describes the measured performance and its limits; broader market coverage and production execution are outside this evaluation.
