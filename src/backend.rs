// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core backend trait and model wrapper.
//!
//! [`MIBackend`] is the trait that every model backend implements.
//! [`MIModel`] wraps a backend with device metadata and convenience methods.

use candle_core::{DType, Device, Tensor};

use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookSpec};

// ---------------------------------------------------------------------------
// MIBackend trait
// ---------------------------------------------------------------------------

/// Unified interface for model backends with hook-aware forward passes.
///
/// Implementing this trait is the only requirement for adding a new model
/// to candle-mi.  The single [`forward`](Self::forward) method replaces
/// plip-rs's proliferation of `forward_with_*` variants: the caller
/// specifies captures and interventions via [`HookSpec`], and the backend
/// returns a [`HookCache`] containing the output plus any requested
/// activations.
///
/// Optional capabilities (chat template, embedding access) have default
/// implementations that return `None` or an error.
pub trait MIBackend: Send + Sync {
    // --- Metadata --------------------------------------------------------

    /// Number of layers (transformer blocks or RWKV blocks).
    fn num_layers(&self) -> usize;

    /// Hidden dimension (`d_model`).
    fn hidden_size(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Number of attention heads (or RWKV heads).
    fn num_heads(&self) -> usize;

    // --- Core forward pass -----------------------------------------------

    /// Unified forward pass with optional hook capture and interventions.
    ///
    /// When `hooks` is empty, this must be equivalent to a plain forward
    /// pass with **zero extra allocations** (see `design/hook-overhead.md`).
    ///
    /// The returned [`HookCache`] always contains the output tensor
    /// (logits or hidden states, depending on the backend) and any
    /// activations requested via [`HookSpec::capture`].
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures and
    /// [`MIError::Intervention`] if an intervention is invalid for
    /// the current model dimensions.
    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>;

    // --- Logit projection ------------------------------------------------

    /// Project a hidden-state tensor to vocabulary logits.
    ///
    /// `hidden` has shape `[batch, hidden_size]`; returns `[batch, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on shape mismatch or tensor operation failure.
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>;

    // --- Optional capabilities -------------------------------------------

    /// Format a prompt with the model's chat template, if any.
    ///
    /// Returns `None` for base (non-instruct) models.
    fn chat_template(&self, _prompt: &str, _system_prompt: Option<&str>) -> Option<String> {
        None
    }

    /// Return the raw embedding vector for a single token.
    ///
    /// Shape: `[hidden_size]`.  For models with tied embeddings this is
    /// also the unembedding direction.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the backend does not support this.
    fn embedding_vector(&self, _token_id: u32) -> Result<Tensor> {
        Err(MIError::Hook(
            "embedding_vector not supported for this backend".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// MIModel
// ---------------------------------------------------------------------------

/// High-level model wrapper combining a backend with device metadata.
///
/// `MIModel` delegates to the wrapped [`MIBackend`] and adds convenience
/// methods.  Full model loading (`from_pretrained`) will be available
/// once concrete backends are implemented (Phase 1+).
pub struct MIModel {
    backend: Box<dyn MIBackend>,
    device: Device,
}

impl MIModel {
    /// Wrap an existing backend.
    #[must_use]
    pub fn new(backend: Box<dyn MIBackend>, device: Device) -> Self {
        Self { backend, device }
    }

    /// The device this model lives on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.backend.num_layers()
    }

    /// Hidden dimension.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.backend.hidden_size()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.backend.vocab_size()
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.backend.num_heads()
    }

    /// Run a forward pass with the given hook specification.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying backend.
    pub fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        self.backend.forward(input_ids, hooks)
    }

    /// Project hidden states to vocabulary logits.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying backend.
    pub fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.backend.project_to_vocab(hidden)
    }

    /// Access the underlying backend (e.g., for backend-specific methods).
    #[must_use]
    pub fn backend(&self) -> &dyn MIBackend {
        &*self.backend
    }
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Sample a token from logits using the given temperature.
///
/// When `temperature <= 0.0`, performs greedy (argmax) decoding.
///
/// # Errors
///
/// Returns [`MIError::Model`] if the logits tensor is empty or
/// cannot be converted to `f32`.
pub fn sample_token(logits: &Tensor, temperature: f32) -> Result<u32> {
    if temperature <= 0.0 {
        argmax(logits)
    } else {
        sample_with_temperature(logits, temperature)
    }
}

/// Greedy (argmax) sampling.
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    let (max_idx, _) = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| MIError::Model(candle_core::Error::Msg("empty logits".into())))?;

    #[allow(clippy::cast_possible_truncation)]
    Ok(max_idx as u32)
}

/// Temperature-scaled softmax sampling.
fn sample_with_temperature(logits: &Tensor, temperature: f32) -> Result<u32> {
    use rand::Rng;

    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    // Scale by temperature.
    let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();

    // Numerically stable softmax.
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

    // Sample from the categorical distribution.
    let mut rng = rand::thread_rng();
    let r: f32 = rng.r#gen();
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            #[allow(clippy::cast_possible_truncation)]
            return Ok(idx as u32);
        }
    }

    // Fallback to last token (floating-point rounding edge case).
    #[allow(clippy::cast_possible_truncation)]
    Ok((probs.len() - 1) as u32)
}

// ---------------------------------------------------------------------------
// GenerationResult
// ---------------------------------------------------------------------------

/// Output of a text generation run with token-level details.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Original prompt text.
    pub prompt: String,
    /// Full output (prompt + generated).
    pub full_text: String,
    /// Only the generated portion.
    pub generated_text: String,
    /// Token IDs from the prompt.
    pub prompt_tokens: Vec<u32>,
    /// Token IDs that were generated.
    pub generated_tokens: Vec<u32>,
    /// Total token count (prompt + generated).
    pub total_tokens: usize,
}
