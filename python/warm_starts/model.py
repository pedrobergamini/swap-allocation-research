"""The `sar-mlp/2` artifact and a pure-Python reference forward pass.

An artifact names the feature spec it reads and how its single output becomes pool 1's share:

- `{"kind": "share"}`: the share is `sigmoid(z)`.
- `{"kind": "coarse_offset", "chunks": c}`: the share is the coarse split's share moved by
  `(2 * sigmoid(z) - 1) * c / 20`, clamped to [0, 1], so the model can only nudge the coarse
  split by up to `c` chunks either way.
- `{"kind": "regret_estimate", "skip_at_most_log10_bp": t, "envelope": {"low": [...],
  "high": [...]}}`: the allocation-regret estimator. Its output `z` estimates log10 of the offset
  model's start regret in bp (plus 1e-4 bp); Fynd may skip exchange refinement when `z <= t` and
  every bounded feature lies inside the envelope (bounds per feature of the full vector, null
  for none, inclusive), the range the model was trained on.

`share` follows `AllocationModel::share` in `src/model.rs` step for step (normalize, then per unit
sum left to right from the bias, then the activation, then the output transform), so parity
tests can compare the two bit for bit. Training uses numpy; only this function defines what
ships.
"""

import json
import math

from .allocation import DENOMINATOR
from .features import (
    FEATURE_NAMES,
    FEATURE_SPEC_VERSION,
    OFFSET_FEATURE_NAMES,
    OFFSET_SPEC_VERSION,
)

MODEL_FORMAT = "sar-mlp/2"
COARSE_CHUNKS = 20
SPECS = {FEATURE_SPEC_VERSION: FEATURE_NAMES, OFFSET_SPEC_VERSION: OFFSET_FEATURE_NAMES}
ACTIVATIONS = {
    "relu": lambda value: max(value, 0.0),
    "tanh": math.tanh,
    "identity": lambda value: value,
}


def artifact(spec, inputs, mean, std, layers, output):
    """Builds an artifact; `layers` is a list of (weights[out][in], bias[out], activation)."""
    return {
        "format": MODEL_FORMAT,
        "feature_spec": spec,
        "feature_names": list(SPECS[spec]),
        "inputs": [int(index) for index in inputs],
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "layers": [
            {
                "weights": [[float(weight) for weight in row] for row in weights],
                "bias": [float(value) for value in bias],
                "activation": activation,
            }
            for weights, bias, activation in layers
        ],
        "output": output,
        "denominator": DENOMINATOR,
    }


def save(model, path):
    # json writes the shortest float string that round-trips, which serde_json reads back to the
    # same f64.
    with open(path, "w") as file:
        json.dump(model, file, indent=1, allow_nan=False)
        file.write("\n")


def load(path):
    with open(path) as file:
        model = json.load(file)
    if model["format"] != MODEL_FORMAT or model["feature_spec"] not in SPECS:
        raise ValueError("model format or feature spec mismatch")
    names = SPECS[model["feature_spec"]]
    if model["feature_names"] != list(names) or model["denominator"] != DENOMINATOR:
        raise ValueError("model feature names or denominator mismatch")
    if not all(0 <= index < len(names) for index in model["inputs"]):
        raise ValueError("an input index is outside the feature vector")
    return model


def logit(model, features):
    activations = [
        (features[index] - mean) / std
        for index, mean, std in zip(model["inputs"], model["mean"], model["std"], strict=True)
    ]
    for layer in model["layers"]:
        activate = ACTIVATIONS[layer["activation"]]
        outputs = []
        for row, bias in zip(layer["weights"], layer["bias"], strict=True):
            total = bias
            for weight, value in zip(row, activations, strict=True):
                total += weight * value
            outputs.append(activate(total))
        activations = outputs
    return activations[0]


def sigmoid(value):
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        # Rust's exp overflows to infinity here, giving exactly 0.0.
        return 0.0


def regret_estimate(model, features):
    """(estimated log10 start regret in bp, whether to skip refinement), as Rust decides it."""
    output = model["output"]
    if output["kind"] != "regret_estimate":
        raise ValueError("not a regret_estimate model")
    estimate = logit(model, features)
    envelope = output["envelope"]
    inside = all(
        (low is None or low <= value) and (high is None or value <= high)
        for value, low, high in zip(features, envelope["low"], envelope["high"], strict=True)
    )
    return estimate, estimate <= output["skip_at_most_log10_bp"] and inside


def share(model, features, coarse_share=None):
    """Pool 1's share of the order, in [0, 1]."""
    squashed = sigmoid(logit(model, features))
    output = model["output"]
    if output["kind"] == "regret_estimate":
        raise ValueError("a regret_estimate model proposes no share")
    if output["kind"] == "share":
        return squashed
    offset = (2.0 * squashed - 1.0) * output["chunks"] / COARSE_CHUNKS
    return min(max(coarse_share + offset, 0.0), 1.0)
