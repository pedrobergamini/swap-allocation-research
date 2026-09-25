//! Inference for the neural allocation initializer and allocation-regret estimator.
//!
//! A multilayer perceptron (MLP) is a stack of fully connected layers: each multiplies its input
//! by a weight matrix, adds a bias and applies an activation. The last layer produces one number
//! `z`, and the artifact's output rule turns it into pool 1's share of the order: either
//! `sigmoid(z)` directly, or the coarse split's share moved by at most a few coarse chunks. A
//! regret estimator's `z` is instead its estimate of the allocation network's starting loss, in log10 bp,
//! with the rule for when the seed may skip exchange refinement.
//! Artifacts are JSON written by `python/warm_starts/model.py`; nothing executable is loaded.
//! Everything is validated once, at load, so inference itself has no failure path.

use std::path::Path;

use anyhow::{bail, ensure, Context};
use serde::Deserialize;

use crate::{allocation, features};

pub const MODEL_FORMAT: &str = "sar-mlp/2";
/// Chunks in Water-fill's coarse pass, the unit of a `coarse_offset` output.
pub const COARSE_CHUNKS: f64 = 20.0;

/// Which feature vector the model reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSpec {
    /// Four probes per pool (`features::feature_vector`).
    ProbeImpact,
    /// The coarse split and cached whole-order quotes (`features::offset_feature_vector`).
    CoarseOffset,
}

impl FeatureSpec {
    fn parse(version: &str) -> anyhow::Result<Self> {
        match version {
            features::FEATURE_SPEC_VERSION => Ok(Self::ProbeImpact),
            features::OFFSET_SPEC_VERSION => Ok(Self::CoarseOffset),
            other => bail!("unknown feature spec `{other}`"),
        }
    }

    fn names(self) -> &'static [&'static str] {
        match self {
            Self::ProbeImpact => &features::FEATURE_NAMES,
            Self::CoarseOffset => &features::OFFSET_FEATURE_NAMES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Relu,
    Tanh,
    Identity,
}

/// How the network's output is read.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// Pool 1's share is `sigmoid(z)`.
    Share,
    /// Pool 1's share is `coarse + (2 * sigmoid(z) - 1) * chunks / 20`, clamped to `[0, 1]`.
    CoarseOffset { chunks: f64 },
    /// `z` estimates log10 of the offset model's start regret in bp. The seed may skip exchange
    /// refinement when `z <= skip_at_most_log10_bp` and every bounded feature lies within
    /// `[low, high]`, inclusive. Bounds index the full feature vector; `None` leaves a side open.
    RegretEstimate { skip_at_most_log10_bp: f64, low: Vec<Option<f64>>, high: Vec<Option<f64>> },
}

/// Which rule an artifact's output follows, for loaders that need one kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    Share,
    CoarseOffset,
    RegretEstimate,
}

/// The artifact's `output` object. Read as a plain struct: serde_json's `arbitrary_precision`,
/// which recordings need, breaks numbers inside internally tagged enums.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputField {
    kind: String,
    chunks: Option<f64>,
    skip_at_most_log10_bp: Option<f64>,
    envelope: Option<EnvelopeField>,
}

/// Per-feature bounds of the training range; `null` for no bound on that side.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvelopeField {
    low: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
}

impl OutputField {
    fn parse(self) -> anyhow::Result<Output> {
        match (self.kind.as_str(), self.chunks, self.skip_at_most_log10_bp, self.envelope) {
            ("share", None, None, None) => Ok(Output::Share),
            ("coarse_offset", Some(chunks), None, None) => Ok(Output::CoarseOffset { chunks }),
            ("regret_estimate", None, Some(skip_at_most_log10_bp), Some(envelope)) => {
                Ok(Output::RegretEstimate {
                    skip_at_most_log10_bp,
                    low: envelope.low,
                    high: envelope.high,
                })
            }
            (kind, chunks, threshold, envelope) => bail!(
                "output kind `{kind}` with chunks {chunks:?}, skip_at_most_log10_bp \
                 {threshold:?} and envelope {envelope:?} is not valid"
            ),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Layer {
    /// `weights[out][in]`.
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    activation: Activation,
}

#[derive(Debug, Deserialize)]
struct Artifact {
    format: String,
    feature_spec: String,
    feature_names: Vec<String>,
    /// Indices into the full feature vector that the model reads, in input order.
    inputs: Vec<usize>,
    mean: Vec<f64>,
    std: Vec<f64>,
    layers: Vec<Layer>,
    output: OutputField,
    denominator: u32,
}

/// A confidence model's verdict on one solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegretEstimate {
    /// The network's raw output: estimated log10 of the start regret in bp.
    pub log10_bp: f64,
    /// Whether the seed may skip exchange refinement.
    pub skip: bool,
}

/// A validated model ready for inference.
#[derive(Debug)]
pub struct AllocationModel {
    spec: FeatureSpec,
    inputs: Vec<usize>,
    mean: Vec<f64>,
    std: Vec<f64>,
    layers: Vec<Layer>,
    output: Output,
}

impl AllocationModel {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading model {}", path.display()))?;
        Self::from_json(&text).with_context(|| format!("invalid model {}", path.display()))
    }

    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let artifact: Artifact = serde_json::from_str(text)?;
        ensure!(
            artifact.format == MODEL_FORMAT,
            "format `{}` is not {MODEL_FORMAT}",
            artifact.format
        );
        let spec = FeatureSpec::parse(&artifact.feature_spec)?;
        ensure!(
            artifact.feature_names == spec.names(),
            "feature names differ from {}",
            artifact.feature_spec
        );
        ensure!(
            artifact.denominator == allocation::DENOMINATOR,
            "denominator {} is not {}",
            artifact.denominator,
            allocation::DENOMINATOR
        );
        let width = artifact.inputs.len();
        ensure!(width > 0, "the model reads no features");
        ensure!(
            artifact.inputs.iter().all(|&index| index < spec.names().len()),
            "an input index is outside the feature vector"
        );
        ensure!(artifact.mean.len() == width && artifact.std.len() == width, "normalization width");
        ensure!(artifact.mean.iter().all(|v| v.is_finite()), "non-finite mean");
        ensure!(artifact.std.iter().all(|v| v.is_finite() && *v > 0.0), "std must be positive");
        ensure!(!artifact.layers.is_empty(), "no layers");
        let output = artifact.output.parse()?;
        match &output {
            Output::Share => {}
            Output::CoarseOffset { chunks } => {
                ensure!(chunks.is_finite() && *chunks > 0.0, "offset chunks must be positive");
                ensure!(
                    spec == FeatureSpec::CoarseOffset,
                    "a coarse offset needs {}",
                    features::OFFSET_SPEC_VERSION
                );
            }
            Output::RegretEstimate { skip_at_most_log10_bp, low, high } => {
                ensure!(
                    spec == FeatureSpec::CoarseOffset,
                    "a regret estimate needs {}",
                    features::OFFSET_SPEC_VERSION
                );
                ensure!(skip_at_most_log10_bp.is_finite(), "the skip threshold must be finite");
                let count = spec.names().len();
                ensure!(
                    low.len() == count && high.len() == count,
                    "the envelope must bound all {count} features"
                );
                for (index, (low, high)) in low.iter().zip(high).enumerate() {
                    ensure!(
                        low.is_none_or(f64::is_finite) && high.is_none_or(f64::is_finite),
                        "feature {index} has a non-finite envelope bound"
                    );
                    if let (Some(low), Some(high)) = (low, high) {
                        ensure!(low <= high, "feature {index} has an empty envelope");
                    }
                }
                // The estimate is `z` itself, so a squashing last layer would cap it.
                let last = artifact.layers.last().map(|layer| layer.activation);
                ensure!(
                    last == Some(Activation::Identity),
                    "a regret estimate's last layer must be identity"
                );
            }
        }

        let mut fan_in = width;
        for (index, layer) in artifact.layers.iter().enumerate() {
            let fan_out = layer.bias.len();
            ensure!(fan_out > 0, "layer {index} has no outputs");
            ensure!(layer.weights.len() == fan_out, "layer {index} weight rows != bias length");
            ensure!(
                layer.weights.iter().all(|row| row.len() == fan_in),
                "layer {index} weight columns != {fan_in}"
            );
            let finite = layer.weights.iter().flatten().chain(&layer.bias).all(|v| v.is_finite());
            ensure!(finite, "layer {index} has a non-finite parameter");
            fan_in = fan_out;
        }
        if fan_in != 1 {
            bail!("the last layer has {fan_in} outputs, expected 1");
        }
        Ok(Self {
            spec,
            inputs: artifact.inputs,
            mean: artifact.mean,
            std: artifact.std,
            layers: artifact.layers,
            output,
        })
    }

    pub fn spec(&self) -> FeatureSpec {
        self.spec
    }

    pub fn output_kind(&self) -> OutputKind {
        match self.output {
            Output::Share => OutputKind::Share,
            Output::CoarseOffset { .. } => OutputKind::CoarseOffset,
            Output::RegretEstimate { .. } => OutputKind::RegretEstimate,
        }
    }

    /// Pool 1's share of the order, in `[0, 1]`. `features` must be the vector of
    /// [`Self::spec`]; `coarse_share` is read only by a coarse-offset output.
    ///
    /// # Panics
    ///
    /// On a regret-estimate model, which proposes no share. Loaders check [`Self::output_kind`].
    pub fn share(&self, features: &[f64], coarse_share: f64) -> f64 {
        let squashed = sigmoid(self.logit(features));
        match self.output {
            Output::Share => squashed,
            Output::CoarseOffset { chunks } => {
                let offset = (2.0 * squashed - 1.0) * chunks / COARSE_CHUNKS;
                (coarse_share + offset).clamp(0.0, 1.0)
            }
            Output::RegretEstimate { .. } => {
                unreachable!("a regret-estimate model proposes no share; loaders check the kind")
            }
        }
    }

    /// The estimated start regret and the skip decision, or `None` unless this is a
    /// regret-estimate model. `features` must be the full vector of [`Self::spec`].
    pub fn regret_estimate(&self, features: &[f64]) -> Option<RegretEstimate> {
        let Output::RegretEstimate { skip_at_most_log10_bp, low, high } = &self.output else {
            return None;
        };
        let log10_bp = self.logit(features);
        // Comparisons with NaN are false, so a NaN estimate or feature never skips.
        let inside_envelope =
            features.iter().zip(low.iter().zip(high)).all(|(value, (low, high))| {
                low.is_none_or(|low| low <= *value) && high.is_none_or(|high| *value <= high)
            });
        Some(RegretEstimate {
            log10_bp,
            skip: log10_bp <= *skip_at_most_log10_bp && inside_envelope,
        })
    }

    fn logit(&self, features: &[f64]) -> f64 {
        let mut activations: Vec<f64> = self
            .inputs
            .iter()
            .zip(self.mean.iter().zip(&self.std))
            .map(|(&index, (mean, std))| (features[index] - mean) / std)
            .collect();
        for layer in &self.layers {
            activations = layer
                .weights
                .iter()
                .zip(&layer.bias)
                .map(|(row, bias)| {
                    // Summed left to right from the bias, the order the Python reference uses.
                    let mut sum = *bias;
                    for (weight, input) in row.iter().zip(&activations) {
                        sum += weight * input;
                    }
                    match layer.activation {
                        Activation::Relu => sum.max(0.0),
                        Activation::Tanh => sum.tanh(),
                        Activation::Identity => sum,
                    }
                })
                .collect();
        }
        activations[0]
    }
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(spec: &str, names: &[&str], input: usize, layers: &str, output: &str) -> String {
        let names = serde_json::to_string(names).unwrap();
        format!(
            r#"{{"format":"sar-mlp/2","feature_spec":"{spec}","feature_names":{names},
            "inputs":[{input}],"mean":[0.0],"std":[2.0],"layers":{layers},"output":{output},
            "denominator":65536}}"#
        )
    }

    fn share_model(layers: &str) -> anyhow::Result<AllocationModel> {
        AllocationModel::from_json(&artifact(
            features::FEATURE_SPEC_VERSION,
            &features::FEATURE_NAMES,
            1,
            layers,
            r#"{"kind":"share"}"#,
        ))
    }

    #[test]
    fn predicts_through_normalization_and_sigmoid() {
        let model =
            share_model(r#"[{"weights":[[1.0]],"bias":[0.5],"activation":"identity"}]"#).unwrap();
        let mut features = [0.0; features::FEATURE_COUNT];
        features[1] = 3.0;
        assert_eq!(model.share(&features, 0.0), sigmoid(3.0 / 2.0 + 0.5));
    }

    #[test]
    fn coarse_offset_moves_at_most_its_chunks() {
        let model = AllocationModel::from_json(&artifact(
            features::OFFSET_SPEC_VERSION,
            &features::OFFSET_FEATURE_NAMES,
            2,
            r#"[{"weights":[[1000.0]],"bias":[0.0],"activation":"identity"}]"#,
            r#"{"kind":"coarse_offset","chunks":1.0}"#,
        ))
        .unwrap();
        let mut features = [0.0; features::OFFSET_FEATURE_COUNT];
        features[2] = 0.5;
        assert_eq!(model.share(&features, 0.5), 0.5 + 1.0 / 20.0);
        features[2] = -0.5;
        assert_eq!(model.share(&features, 0.0), 0.0);
    }

    #[test]
    fn rejects_malformed_artifacts() {
        for layers in [
            r#"[]"#,
            r#"[{"weights":[[1.0,2.0]],"bias":[0.0],"activation":"relu"}]"#,
            r#"[{"weights":[[1.0]],"bias":[0.0],"activation":"relu"},
                {"weights":[[1.0],[1.0]],"bias":[0.0,0.0],"activation":"identity"}]"#,
            r#"[{"weights":[[1e999]],"bias":[0.0],"activation":"relu"}]"#,
        ] {
            assert!(share_model(layers).is_err(), "{layers}");
        }
        let offset_on_probe_spec = artifact(
            features::FEATURE_SPEC_VERSION,
            &features::FEATURE_NAMES,
            1,
            r#"[{"weights":[[1.0]],"bias":[0.0],"activation":"identity"}]"#,
            r#"{"kind":"coarse_offset","chunks":1.0}"#,
        );
        assert!(AllocationModel::from_json(&offset_on_probe_spec).is_err());
    }

    const IDENTITY_LAYER: &str = r#"[{"weights":[[1.0]],"bias":[0.0],"activation":"identity"}]"#;

    /// A confidence artifact reading `coarse_share` (index 2) with `envelope` as its bounds.
    fn confidence_artifact(spec: &str, names: &[&str], layers: &str, envelope: &str) -> String {
        let output = format!(
            r#"{{"kind":"regret_estimate","skip_at_most_log10_bp":1.5,"envelope":{envelope}}}"#
        );
        artifact(spec, names, 2, layers, &output)
    }

    /// Bounds only `log10_notional_usdc` (index 1), which the model does not read.
    fn confidence_model() -> AllocationModel {
        let envelope = r#"{"low":[null,2.0,null,null,null,null,null],
                           "high":[null,4.0,null,null,null,null,null]}"#;
        AllocationModel::from_json(&confidence_artifact(
            features::OFFSET_SPEC_VERSION,
            &features::OFFSET_FEATURE_NAMES,
            IDENTITY_LAYER,
            envelope,
        ))
        .unwrap()
    }

    #[test]
    fn regret_estimate_is_the_raw_network_output() {
        let model = confidence_model();
        assert_eq!(model.output_kind(), OutputKind::RegretEstimate);
        let mut features = [0.0; features::OFFSET_FEATURE_COUNT];
        features[1] = 3.0;
        features[2] = -5.0;
        let estimate = model.regret_estimate(&features).unwrap();
        assert_eq!(estimate.log10_bp, -5.0 / 2.0);

        let share = share_model(IDENTITY_LAYER).unwrap();
        assert_eq!(share.regret_estimate(&[0.0; features::FEATURE_COUNT]), None);
    }

    #[test]
    fn skips_only_at_or_below_the_threshold_inside_the_envelope() {
        let model = confidence_model();
        let skips = |notional: f64, coarse_share: f64| {
            let mut features = [0.0; features::OFFSET_FEATURE_COUNT];
            features[1] = notional;
            features[2] = coarse_share;
            model.regret_estimate(&features).unwrap().skip
        };
        // The estimate is coarse_share / 2 against a threshold of 1.5.
        assert!(skips(3.0, 3.0), "an estimate equal to the threshold skips");
        assert!(!skips(3.0, 3.0 + 1e-9), "an estimate above the threshold refines");
        assert!(skips(2.0, 0.0) && skips(4.0, 0.0), "the envelope is inclusive");
        assert!(!skips(2.0 - 1e-9, 0.0) && !skips(4.0 + 1e-9, 0.0), "outside it refines");
        assert!(!skips(f64::NAN, 0.0) && !skips(3.0, f64::NAN), "NaN never skips");
    }

    fn open_envelope(feature_count: usize) -> String {
        let nulls = vec!["null"; feature_count].join(",");
        format!(r#"{{"low":[{nulls}],"high":[{nulls}]}}"#)
    }

    #[test]
    fn rejects_malformed_regret_estimates() {
        let open = &open_envelope(features::OFFSET_FEATURE_COUNT);
        let offset = |layers: &str, envelope: &str| {
            AllocationModel::from_json(&confidence_artifact(
                features::OFFSET_SPEC_VERSION,
                &features::OFFSET_FEATURE_NAMES,
                layers,
                envelope,
            ))
        };
        assert!(offset(IDENTITY_LAYER, open).is_ok());
        for envelope in [
            r#"{"low":[null,null,null,null,null,null],"high":[null,null,null,null,null,null,null]}"#,
            r#"{"low":[null,null,null,null,null,null,null],"high":[null,null,null,null,null,null]}"#,
            r#"{"low":[null,1e999,null,null,null,null,null],
                "high":[null,null,null,null,null,null,null]}"#,
            r#"{"low":[null,2.0,null,null,null,null,null],
                "high":[null,1.0,null,null,null,null,null]}"#,
            r#"{"low":[null,null,null,null,null,null,null]}"#,
        ] {
            assert!(offset(IDENTITY_LAYER, envelope).is_err(), "{envelope}");
        }
        let tanh_last = r#"[{"weights":[[1.0]],"bias":[0.0],"activation":"tanh"}]"#;
        assert!(offset(tanh_last, open).is_err(), "a squashing last layer caps the estimate");

        let on_probe_spec = confidence_artifact(
            features::FEATURE_SPEC_VERSION,
            &features::FEATURE_NAMES,
            IDENTITY_LAYER,
            &open_envelope(features::FEATURE_COUNT),
        );
        assert!(AllocationModel::from_json(&on_probe_spec).is_err());
        for output in [
            r#"{"kind":"regret_estimate","envelope":{"low":[],"high":[]}}"#,
            r#"{"kind":"regret_estimate","skip_at_most_log10_bp":1e999,
                "envelope":{"low":[null,null,null,null,null,null,null],
                            "high":[null,null,null,null,null,null,null]}}"#,
            r#"{"kind":"coarse_offset","chunks":1.0,"skip_at_most_log10_bp":1.0}"#,
        ] {
            let text = artifact(
                features::OFFSET_SPEC_VERSION,
                &features::OFFSET_FEATURE_NAMES,
                2,
                IDENTITY_LAYER,
                output,
            );
            assert!(AllocationModel::from_json(&text).is_err(), "{output}");
        }
    }

    #[test]
    #[should_panic(expected = "proposes no share")]
    fn a_regret_estimate_model_proposes_no_share() {
        confidence_model().share(&[0.0; features::OFFSET_FEATURE_COUNT], 0.5);
    }
}
