// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hook system for activation capture and intervention.
//!
//! Provides [`HookPoint`] (named locations in a forward pass),
//! [`HookSpec`] (what to capture and where to intervene), and
//! [`HookCache`] (captured tensors from a forward pass).
//!
//! See `design/hook-system.md` for the design rationale.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

use candle_core::Tensor;

use crate::error::{MIError, Result};
use crate::interp::intervention::{StateKnockoutSpec, StateSteeringSpec};

// ---------------------------------------------------------------------------
// HookPoint
// ---------------------------------------------------------------------------

/// Named location in a forward pass where activations can be captured
/// or interventions applied.
///
/// Mirrors the `TransformerLens` hook point naming convention via
/// [`Display`](std::fmt::Display) and [`FromStr`].
///
/// # String conversion
///
/// ```
/// use candle_mi::HookPoint;
///
/// let hook = HookPoint::AttnPattern(5);
/// assert_eq!(hook.to_string(), "blocks.5.attn.hook_pattern");
///
/// let parsed: HookPoint = "blocks.5.attn.hook_pattern".parse().unwrap();
/// assert_eq!(parsed, hook);
/// ```
///
/// Unknown strings parse as [`HookPoint::Custom`], providing an escape
/// hatch for backend-specific hook points.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookPoint {
    // -- Embedding --
    /// After token embedding (`hook_embed`).
    Embed,

    // -- Per-layer: transformer --
    /// Residual stream before layer `i` (`blocks.{i}.hook_resid_pre`).
    ResidPre(usize),
    /// Query vectors in layer `i` (`blocks.{i}.attn.hook_q`).
    AttnQ(usize),
    /// Key vectors in layer `i` (`blocks.{i}.attn.hook_k`).
    AttnK(usize),
    /// Value vectors in layer `i` (`blocks.{i}.attn.hook_v`).
    AttnV(usize),
    /// Pre-softmax attention scores in layer `i` (`blocks.{i}.attn.hook_scores`).
    AttnScores(usize),
    /// Post-softmax attention pattern in layer `i` (`blocks.{i}.attn.hook_pattern`).
    AttnPattern(usize),
    /// Attention output in layer `i` (`blocks.{i}.hook_attn_out`).
    AttnOut(usize),
    /// Residual stream between attention and MLP in layer `i`
    /// (`blocks.{i}.hook_resid_mid`).
    ResidMid(usize),
    /// MLP pre-activation in layer `i` (`blocks.{i}.mlp.hook_pre`).
    MlpPre(usize),
    /// MLP post-activation in layer `i` (`blocks.{i}.mlp.hook_post`).
    MlpPost(usize),
    /// MLP output in layer `i` (`blocks.{i}.hook_mlp_out`).
    MlpOut(usize),
    /// Residual stream after full layer `i` (`blocks.{i}.hook_resid_post`).
    ResidPost(usize),

    // -- Final --
    /// After final layer norm (`hook_final_norm`).
    FinalNorm,

    // -- RWKV-specific --
    /// RWKV recurrent state at layer `i` (`blocks.{i}.rwkv.hook_state`).
    RwkvState(usize),
    /// RWKV decay vector at layer `i` (`blocks.{i}.rwkv.hook_decay`).
    RwkvDecay(usize),
    /// RWKV effective attention at layer `i` (`blocks.{i}.rwkv.hook_effective_attn`).
    ///
    /// Shape: `[batch, heads, seq_query, seq_source]`.
    /// Derived from the WKV recurrence by computing how much each
    /// source position contributes to each query position's output.
    /// Normalised via `ReLU` + L1.
    RwkvEffectiveAttn(usize),

    // -- Escape hatch --
    /// Backend-specific hook point not covered by the standard enum.
    Custom(String),
}

impl fmt::Display for HookPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embed => write!(f, "hook_embed"),
            Self::ResidPre(i) => write!(f, "blocks.{i}.hook_resid_pre"),
            Self::AttnQ(i) => write!(f, "blocks.{i}.attn.hook_q"),
            Self::AttnK(i) => write!(f, "blocks.{i}.attn.hook_k"),
            Self::AttnV(i) => write!(f, "blocks.{i}.attn.hook_v"),
            Self::AttnScores(i) => write!(f, "blocks.{i}.attn.hook_scores"),
            Self::AttnPattern(i) => write!(f, "blocks.{i}.attn.hook_pattern"),
            Self::AttnOut(i) => write!(f, "blocks.{i}.hook_attn_out"),
            Self::ResidMid(i) => write!(f, "blocks.{i}.hook_resid_mid"),
            Self::MlpPre(i) => write!(f, "blocks.{i}.mlp.hook_pre"),
            Self::MlpPost(i) => write!(f, "blocks.{i}.mlp.hook_post"),
            Self::MlpOut(i) => write!(f, "blocks.{i}.hook_mlp_out"),
            Self::ResidPost(i) => write!(f, "blocks.{i}.hook_resid_post"),
            Self::FinalNorm => write!(f, "hook_final_norm"),
            Self::RwkvState(i) => write!(f, "blocks.{i}.rwkv.hook_state"),
            Self::RwkvDecay(i) => write!(f, "blocks.{i}.rwkv.hook_decay"),
            Self::RwkvEffectiveAttn(i) => write!(f, "blocks.{i}.rwkv.hook_effective_attn"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Parse a `TransformerLens`-style string into a [`HookPoint`].
///
/// Unknown strings produce [`HookPoint::Custom`] rather than an error.
impl FromStr for HookPoint {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(parse_hook_string(s))
    }
}

/// Allow `hooks.capture("blocks.5.attn.hook_pattern")` via `Into<HookPoint>`.
impl From<&str> for HookPoint {
    fn from(s: &str) -> Self {
        parse_hook_string(s)
    }
}

/// Parse a hook string, falling back to [`HookPoint::Custom`] for unknown patterns.
fn parse_hook_string(s: &str) -> HookPoint {
    match s {
        "hook_embed" => return HookPoint::Embed,
        "hook_final_norm" => return HookPoint::FinalNorm,
        _ => {}
    }

    // Try "blocks.{layer}.{suffix}" pattern.
    if let Some(rest) = s.strip_prefix("blocks.") {
        if let Some((layer_str, suffix)) = rest.split_once('.') {
            if let Ok(layer) = layer_str.parse::<usize>() {
                return match suffix {
                    "hook_resid_pre" => HookPoint::ResidPre(layer),
                    "attn.hook_q" => HookPoint::AttnQ(layer),
                    "attn.hook_k" => HookPoint::AttnK(layer),
                    "attn.hook_v" => HookPoint::AttnV(layer),
                    "attn.hook_scores" => HookPoint::AttnScores(layer),
                    "attn.hook_pattern" => HookPoint::AttnPattern(layer),
                    "hook_attn_out" => HookPoint::AttnOut(layer),
                    "hook_resid_mid" => HookPoint::ResidMid(layer),
                    "mlp.hook_pre" => HookPoint::MlpPre(layer),
                    "mlp.hook_post" => HookPoint::MlpPost(layer),
                    "hook_mlp_out" => HookPoint::MlpOut(layer),
                    "hook_resid_post" => HookPoint::ResidPost(layer),
                    "rwkv.hook_state" => HookPoint::RwkvState(layer),
                    "rwkv.hook_decay" => HookPoint::RwkvDecay(layer),
                    "rwkv.hook_effective_attn" => HookPoint::RwkvEffectiveAttn(layer),
                    _ => HookPoint::Custom(s.to_string()),
                };
            }
        }
    }

    HookPoint::Custom(s.to_string())
}

// ---------------------------------------------------------------------------
// Intervention
// ---------------------------------------------------------------------------

/// An intervention to apply at a hook point during the forward pass.
///
/// Interventions modify activations in-place as they flow through the model.
/// They are specified as part of a [`HookSpec`] and applied by the backend
/// at the corresponding [`HookPoint`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Intervention {
    /// Replace the tensor entirely with a provided value.
    Replace(Tensor),

    /// Add a vector to the activation (e.g., residual stream steering).
    Add(Tensor),

    /// Apply a pre-softmax knockout mask.
    ///
    /// The mask tensor contains `0.0` for positions to keep and
    /// `-inf` for positions to knock out. Added to attention scores.
    Knockout(Tensor),

    /// Scale attention weights by a constant factor.
    Scale(f64),

    /// Zero the tensor at this hook point.
    Zero,

    /// Add embedding vectors to hidden states at specific token positions only.
    ///
    /// Implements the K-BERT visible matrix pattern (Liu et al., 2019): only
    /// entity tokens attend to their structural embeddings; non-entity tokens
    /// are unchanged.
    ///
    /// Equivalent to: `hidden[:, pos, :] += vector * scale` for each `(pos, vector)`.
    ///
    /// # Arguments
    ///
    /// * `positions` — Vec of `(token_index, embedding_vector)` pairs. Each vector
    ///   should have shape `[d_model]` or be broadcastable to it.
    /// * `scale` — Scaling factor applied to each vector before addition. Defaults to
    ///   `0.1` to match the K-BERT original paper (Liu et al., 2019).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use candle_mi::{HookPoint, HookSpec, Intervention};
    ///
    /// # fn main() -> candle_mi::Result<()> {
    /// # let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
    /// # let entity_embedding = candle_core::Tensor::zeros(2048usize, candle_core::DType::F32, model.device())?;
    /// # let other_embedding = candle_core::Tensor::zeros(2048usize, candle_core::DType::F32, model.device())?;
    /// let mut hooks = HookSpec::new();
    /// hooks.intervene(
    ///     HookPoint::ResidPost(28),
    ///     Intervention::AddAtPositions {
    ///         positions: vec![(3, entity_embedding), (7, other_embedding)],
    ///         scale: 0.1,
    ///     },
    /// );
    /// # Ok(())
    /// # }
    /// ```
    AddAtPositions {
        /// `(token_index, embedding_vector)` pairs to inject.
        positions: Vec<(usize, Tensor)>,
        /// Scaling factor applied to each vector before addition.
        scale: f32,
    },
}

// ---------------------------------------------------------------------------
// Intervention application
// ---------------------------------------------------------------------------

/// Apply a single [`Intervention`] to a tensor.
///
/// Used by backend implementations at each hook point that supports
/// interventions (e.g., Embed, `AttnScores`, `AttnPattern`).
///
/// # Shapes
/// - `tensor`: any shape — the activation at the hook point.
/// - returns: same shape as `tensor`.
///
/// # Errors
///
/// Returns [`MIError::Model`] if the underlying tensor operation fails.
#[cfg(any(feature = "transformer", feature = "rwkv"))]
pub(crate) fn apply_intervention(tensor: &Tensor, intervention: &Intervention) -> Result<Tensor> {
    match intervention {
        Intervention::Replace(replacement) => Ok(replacement.clone()),
        Intervention::Add(delta) => {
            // Convert delta to tensor's dtype if mismatched (e.g., F32 injection
            // into BF16 forward pass). This supports CLT injection where steering
            // vectors are accumulated in F32 for numerical stability.
            let delta = if delta.dtype() == tensor.dtype() {
                delta
            } else {
                &delta.to_dtype(tensor.dtype())?
            };
            Ok(tensor.broadcast_add(delta)?)
        }
        Intervention::Knockout(mask) => Ok(tensor.broadcast_add(mask)?),
        Intervention::Scale(factor) => Ok((tensor * *factor)?),
        Intervention::Zero => Ok(tensor.zeros_like()?),
        Intervention::AddAtPositions { positions, scale } => {
            // hidden has shape [batch, seq_len, d_model].
            // For each (pos, emb), build a sparse delta tensor that is zero
            // everywhere except at `hidden[:, pos, :]`, then broadcast-add.
            // This is the K-BERT visible matrix pattern (Liu et al., 2019):
            // only entity-token positions receive their structural embeddings.
            let seq_len = tensor.dim(1)?;
            let d_model = tensor.dim(2)?;
            let device = tensor.device();

            let mut result = tensor.clone();
            for (pos, emb) in positions {
                // Build a [1, seq_len, d_model] delta tensor: zero everywhere
                // except at row `pos`, which carries `emb * scale`.
                let mut delta_data = vec![0.0_f32; seq_len * d_model];
                // Promote to F32 for host-side arithmetic (emb may be BF16).
                let vec_f32: Vec<f32> = emb
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1()?;
                let start = pos * d_model;
                let dest = delta_data.get_mut(start..start + d_model).ok_or_else(|| {
                    MIError::Intervention(format!(
                        "AddAtPositions: position {pos} out of bounds (seq_len={seq_len})"
                    ))
                })?;
                for (dst, &src) in dest.iter_mut().zip(vec_f32.iter()) {
                    *dst = src * scale;
                }
                let delta =
                    Tensor::from_vec(delta_data, (1_usize, seq_len, d_model), device)?
                        .to_dtype(result.dtype())?;
                result = result.broadcast_add(&delta)?;
            }
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// HookSpec
// ---------------------------------------------------------------------------

/// Declares which activations to capture and which interventions to apply.
///
/// Passed to [`MIBackend::forward`](crate::MIBackend::forward). When empty,
/// the forward pass has zero overhead (no clones, no extra allocations).
///
/// # Example
///
/// ```
/// use candle_mi::{HookPoint, HookSpec};
///
/// let mut hooks = HookSpec::new();
/// hooks.capture(HookPoint::AttnPattern(5))
///      .capture("blocks.5.hook_resid_post");
/// ```
#[derive(Debug, Clone, Default)]
pub struct HookSpec {
    /// Hook points to capture during the forward pass.
    captures: HashSet<HookPoint>,
    /// Interventions to apply, stored as (`hook_point`, intervention) pairs.
    interventions: Vec<(HookPoint, Intervention)>,
    /// RWKV state knockout specification (skip kv write at specified positions).
    state_knockout: Option<StateKnockoutSpec>,
    /// RWKV state steering specification (scale kv write at specified positions).
    state_steering: Option<StateSteeringSpec>,
}

impl HookSpec {
    /// Create an empty hook specification (no captures, no interventions).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request capture of the activation at the given hook point.
    pub fn capture<H: Into<HookPoint>>(&mut self, hook: H) -> &mut Self {
        self.captures.insert(hook.into());
        self
    }

    /// Register an intervention at the given hook point.
    pub fn intervene<H: Into<HookPoint>>(
        &mut self,
        hook: H,
        intervention: Intervention,
    ) -> &mut Self {
        self.interventions.push((hook.into(), intervention));
        self
    }

    /// Check whether a specific hook point should be captured.
    #[must_use]
    pub fn is_captured(&self, hook: &HookPoint) -> bool {
        self.captures.contains(hook)
    }

    /// Check whether this spec has no captures, no interventions, and no
    /// state specs (knockout/steering).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.captures.is_empty()
            && self.interventions.is_empty()
            && self.state_knockout.is_none()
            && self.state_steering.is_none()
    }

    /// Number of requested captures.
    #[must_use]
    pub fn num_captures(&self) -> usize {
        self.captures.len()
    }

    /// Number of registered interventions.
    #[must_use]
    pub const fn num_interventions(&self) -> usize {
        self.interventions.len()
    }

    /// Iterate over interventions registered at a specific hook point.
    pub fn interventions_at(&self, hook: &HookPoint) -> impl Iterator<Item = &Intervention> {
        self.interventions
            .iter()
            .filter(move |(h, _)| h == hook)
            .map(|(_, intervention)| intervention)
    }

    /// Check whether any intervention targets the given hook point.
    #[must_use]
    pub fn has_intervention_at(&self, hook: &HookPoint) -> bool {
        self.interventions.iter().any(|(h, _)| h == hook)
    }

    /// Set an RWKV state knockout specification.
    ///
    /// At specified token positions, the WKV recurrence skips the kv write,
    /// effectively making those tokens invisible to all future positions.
    pub fn set_state_knockout(&mut self, spec: StateKnockoutSpec) -> &mut Self {
        self.state_knockout = Some(spec);
        self
    }

    /// Set an RWKV state steering specification.
    ///
    /// At specified token positions, the WKV recurrence scales the kv write
    /// by the given factor, amplifying or dampening the token's contribution.
    pub fn set_state_steering(&mut self, spec: StateSteeringSpec) -> &mut Self {
        self.state_steering = Some(spec);
        self
    }

    /// Get the state knockout specification, if any.
    #[must_use]
    pub const fn state_knockout(&self) -> Option<&StateKnockoutSpec> {
        self.state_knockout.as_ref()
    }

    /// Get the state steering specification, if any.
    #[must_use]
    pub const fn state_steering(&self) -> Option<&StateSteeringSpec> {
        self.state_steering.as_ref()
    }

    /// Merge all captures and interventions from another [`HookSpec`] into this one.
    ///
    /// Useful for combining multiple intervention sources (e.g., suppress +
    /// inject in CLT steering).
    pub fn extend(&mut self, other: &Self) -> &mut Self {
        self.captures.extend(other.captures.iter().cloned());
        self.interventions
            .extend(other.interventions.iter().cloned());
        self
    }
}

// ---------------------------------------------------------------------------
// HookCache
// ---------------------------------------------------------------------------

/// Tensors captured during a forward pass, plus the output logits.
///
/// Returned by [`MIBackend::forward`](crate::MIBackend::forward). Use
/// [`get`](Self::get) to retrieve activations at specific hook points.
///
/// # Example
///
/// ```
/// use candle_mi::{HookCache, HookPoint};
/// use candle_core::{Device, Tensor};
///
/// let logits = Tensor::zeros((1, 10, 32000), candle_core::DType::F32, &Device::Cpu).unwrap();
/// let mut cache = HookCache::new(logits);
///
/// // Store a captured activation
/// let pattern = Tensor::zeros((1, 8, 10, 10), candle_core::DType::F32, &Device::Cpu).unwrap();
/// cache.store(HookPoint::AttnPattern(5), pattern);
///
/// // Retrieve captured activations
/// let output = cache.output();
/// let attn = cache.get(&HookPoint::AttnPattern(5)).unwrap();
/// ```
#[derive(Debug)]
pub struct HookCache {
    /// Output tensor from the forward pass (typically logits).
    output: Tensor,
    /// Captured activations keyed by hook point.
    captures: HashMap<HookPoint, Tensor>,
}

impl HookCache {
    /// Create a new cache with the given output tensor and no captures.
    #[must_use]
    pub fn new(output: Tensor) -> Self {
        Self {
            output,
            captures: HashMap::new(),
        }
    }

    /// The output tensor from the forward pass.
    #[must_use]
    pub const fn output(&self) -> &Tensor {
        &self.output
    }

    /// Consume the cache and return the output tensor.
    #[must_use]
    pub fn into_output(self) -> Tensor {
        self.output
    }

    /// Retrieve a captured tensor by hook point.
    #[must_use]
    pub fn get(&self, hook: &HookPoint) -> Option<&Tensor> {
        self.captures.get(hook)
    }

    /// Retrieve a captured tensor, returning an error if not found.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the hook point was not captured.
    pub fn require(&self, hook: &HookPoint) -> Result<&Tensor> {
        self.captures
            .get(hook)
            .ok_or_else(|| MIError::Hook(format!("hook point `{hook}` was not captured")))
    }

    /// Store a captured activation. Called by backend implementations.
    pub fn store(&mut self, hook: HookPoint, tensor: Tensor) {
        self.captures.insert(hook, tensor);
    }

    /// Replace the output tensor (e.g., after computing final logits).
    ///
    /// This allows the forward pass to collect captures into a cache
    /// initialized with a placeholder, then set the real output at the end.
    pub fn set_output(&mut self, output: Tensor) {
        self.output = output;
    }

    /// Number of captured tensors (excludes the output).
    #[must_use]
    pub fn num_captures(&self) -> usize {
        self.captures.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn hook_point_display_roundtrip() {
        let cases: Vec<(HookPoint, &str)> = vec![
            (HookPoint::Embed, "hook_embed"),
            (HookPoint::FinalNorm, "hook_final_norm"),
            (HookPoint::ResidPre(0), "blocks.0.hook_resid_pre"),
            (HookPoint::AttnQ(3), "blocks.3.attn.hook_q"),
            (HookPoint::AttnK(3), "blocks.3.attn.hook_k"),
            (HookPoint::AttnV(3), "blocks.3.attn.hook_v"),
            (HookPoint::AttnScores(7), "blocks.7.attn.hook_scores"),
            (HookPoint::AttnPattern(5), "blocks.5.attn.hook_pattern"),
            (HookPoint::AttnOut(2), "blocks.2.hook_attn_out"),
            (HookPoint::ResidMid(11), "blocks.11.hook_resid_mid"),
            (HookPoint::MlpPre(1), "blocks.1.mlp.hook_pre"),
            (HookPoint::MlpPost(1), "blocks.1.mlp.hook_post"),
            (HookPoint::MlpOut(4), "blocks.4.hook_mlp_out"),
            (HookPoint::ResidPost(9), "blocks.9.hook_resid_post"),
            (HookPoint::RwkvState(6), "blocks.6.rwkv.hook_state"),
            (HookPoint::RwkvDecay(6), "blocks.6.rwkv.hook_decay"),
            (
                HookPoint::RwkvEffectiveAttn(6),
                "blocks.6.rwkv.hook_effective_attn",
            ),
        ];

        for (hook, expected_str) in cases {
            // Display
            assert_eq!(
                hook.to_string(),
                expected_str,
                "Display failed for {hook:?}"
            );
            // FromStr roundtrip
            let parsed: HookPoint = expected_str.parse().unwrap();
            assert_eq!(parsed, hook, "FromStr failed for {expected_str:?}");
            // From<&str>
            let from_str: HookPoint = HookPoint::from(expected_str);
            assert_eq!(from_str, hook, "From<&str> failed for {expected_str:?}");
        }
    }

    #[test]
    fn unknown_string_becomes_custom() {
        let hook: HookPoint = "some.unknown.hook".parse().unwrap();
        assert_eq!(hook, HookPoint::Custom("some.unknown.hook".to_string()));
    }

    #[test]
    fn hook_spec_capture_and_query() {
        let mut spec = HookSpec::new();
        assert!(spec.is_empty());

        spec.capture(HookPoint::AttnPattern(5));
        spec.capture("blocks.3.hook_resid_post");

        assert!(!spec.is_empty());
        assert_eq!(spec.num_captures(), 2);
        assert!(spec.is_captured(&HookPoint::AttnPattern(5)));
        assert!(spec.is_captured(&HookPoint::ResidPost(3)));
        assert!(!spec.is_captured(&HookPoint::Embed));
    }

    // apply_intervention is only compiled with transformer or rwkv features.
    #[cfg(any(feature = "transformer", feature = "rwkv"))]
    #[test]
    fn add_at_positions_selective() {
        use candle_core::IndexOp as _;

        // Build a [1, 5, 4] hidden tensor (batch=1, seq=5, d_model=4), all zeros.
        let hidden = Tensor::zeros(
            (1_usize, 5_usize, 4_usize),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        // Intervene at position 2 only with vector [1.0, 2.0, 3.0, 4.0], scale=1.0.
        let emb = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0],
            (4_usize,),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let intervention = Intervention::AddAtPositions {
            positions: vec![(2, emb)],
            scale: 1.0,
        };

        let result = apply_intervention(&hidden, &intervention).unwrap();

        // Positions 0, 1, 3, 4 must be unchanged (all zeros).
        for pos in [0_usize, 1, 3, 4] {
            let row: Vec<f32> = result
                .i((.., pos, ..))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert!(
                row.iter().all(|&v| v == 0.0),
                "position {pos} should be unchanged, got {row:?}"
            );
        }

        // Position 2 must equal [1.0, 2.0, 3.0, 4.0].
        let pos2: Vec<f32> = result
            .i((.., 2_usize, ..))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            pos2,
            vec![1.0_f32, 2.0, 3.0, 4.0],
            "position 2 should have the injected vector"
        );
    }

    #[test]
    fn hook_spec_intervention_query() {
        let mut spec = HookSpec::new();
        spec.intervene(HookPoint::AttnScores(5), Intervention::Zero);
        spec.intervene(HookPoint::AttnScores(5), Intervention::Scale(2.0));
        spec.intervene(HookPoint::ResidPost(10), Intervention::Zero);

        assert_eq!(spec.num_interventions(), 3);
        assert!(spec.has_intervention_at(&HookPoint::AttnScores(5)));
        assert!(!spec.has_intervention_at(&HookPoint::Embed));

        let at_5: Vec<_> = spec.interventions_at(&HookPoint::AttnScores(5)).collect();
        assert_eq!(at_5.len(), 2);
    }
}
