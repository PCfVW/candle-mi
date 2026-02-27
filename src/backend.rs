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
/// plip-rs's (frozen predecessor project, v1.4.0) proliferation of `forward_with_*` variants: the caller
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
    /// # Shapes
    /// - `input_ids`: `[batch, seq]` -- token IDs
    /// - returns: [`HookCache`] containing `logits` at `[batch, seq, vocab_size]`
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
    /// # Shapes
    /// - `hidden`: `[batch, hidden_size]` -- hidden states
    /// - returns: `[batch, vocab_size]`
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
    /// For models with tied embeddings this is also the unembedding direction.
    ///
    /// # Shapes
    /// - returns: `[hidden_size]`
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
    /// The underlying model backend.
    // TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch
    backend: Box<dyn MIBackend>,
    /// The device this model lives on.
    device: Device,
}

impl MIModel {
    /// Load a model from a `HuggingFace` model ID or local path.
    ///
    /// Checks local `HuggingFace` cache first, then downloads if necessary.
    /// Automatically selects the appropriate backend based on `model_type`
    /// in the model's `config.json`.
    ///
    /// # `DType` selection
    ///
    /// - **CUDA**: `BF16` (matches training dtype; `F16` causes issues on some models)
    /// - **CPU**: `F32` for full precision
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the model type is unsupported, or
    /// [`MIError::Model`] if weight loading fails.
    #[cfg(feature = "transformer")]
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use crate::config::TransformerConfig;
        use crate::transformer::GenericTransformer;

        // --- Device and dtype ---
        let device = Self::select_device()?;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        // --- Download / resolve local files ---
        let files = hf_fetch_model::download_files_blocking(model_id.to_owned())
            .map_err(|e| MIError::Download(e.to_string()))?;

        let config_path = files
            .get("config.json")
            .ok_or_else(|| MIError::Config("config.json not found in downloaded files".into()))?;
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| MIError::Config(format!("read config.json: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| MIError::Config(format!("parse config.json: {e}")))?;

        let config = TransformerConfig::from_hf_config(&json)?;

        // --- Resolve safetensors paths ---
        let weights_paths = resolve_safetensors_paths(&files)?;

        // --- Create VarBuilder ---
        let vb = create_var_builder(&weights_paths, dtype, &device)?;

        // --- Load model ---
        let transformer = GenericTransformer::load(config, &device, dtype, vb)?;

        Ok(Self::new(Box::new(transformer), device))
    }

    /// Select the best available device (CUDA GPU 0, or CPU fallback).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on device detection failure.
    #[cfg(feature = "transformer")]
    fn select_device() -> Result<Device> {
        match Device::cuda_if_available(0) {
            Ok(dev) => Ok(dev),
            Err(e) => Err(MIError::Model(e)),
        }
    }

    /// Wrap an existing backend.
    // TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch
    #[must_use]
    pub fn new(backend: Box<dyn MIBackend>, device: Device) -> Self {
        Self { backend, device }
    }

    /// The device this model lives on.
    #[must_use]
    pub const fn device(&self) -> &Device {
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
    /// # Shapes
    /// - `input_ids`: `[batch, seq]` -- token IDs
    /// - returns: [`HookCache`] containing `logits` at `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying backend.
    pub fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        self.backend.forward(input_ids, hooks)
    }

    /// Project hidden states to vocabulary logits.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, hidden_size]` -- hidden states
    /// - returns: `[batch, vocab_size]`
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying backend.
    pub fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.backend.project_to_vocab(hidden)
    }

    /// Access the underlying backend (e.g., for backend-specific methods).
    // TRAIT_OBJECT: caller needs dynamic dispatch for backend-specific methods
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
/// # Shapes
/// - `logits`: `[vocab_size]` -- logit scores for each vocabulary token
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

    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
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
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            return Ok(idx as u32);
        }
    }

    // Fallback to last token (floating-point rounding edge case).
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
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

// ---------------------------------------------------------------------------
// Weight loading helpers (used by from_pretrained)
// ---------------------------------------------------------------------------

/// Index structure for sharded safetensors models.
#[cfg(feature = "transformer")]
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    /// Maps weight name → shard filename.
    weight_map: std::collections::HashMap<String, String>,
}

/// Resolve safetensors file paths from a downloaded file map.
///
/// Tries `model.safetensors.index.json` first (sharded), falls back to
/// single `model.safetensors`.
#[cfg(feature = "transformer")]
fn resolve_safetensors_paths(
    files: &std::collections::HashMap<String, std::path::PathBuf>,
) -> Result<Vec<std::path::PathBuf>> {
    // Try sharded first
    if let Some(index_path) = files.get("model.safetensors.index.json") {
        let index_str = std::fs::read_to_string(index_path)
            .map_err(|e| MIError::Model(candle_core::Error::Msg(format!("read index: {e}"))))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| MIError::Config(format!("parse index: {e}")))?;

        // Collect unique shard filenames
        let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
        shard_names.sort();
        shard_names.dedup();

        let mut paths = Vec::with_capacity(shard_names.len());
        for shard_name in &shard_names {
            let path = files.get(shard_name.as_str()).ok_or_else(|| {
                MIError::Model(candle_core::Error::Msg(format!(
                    "shard {shard_name} not found in downloaded files"
                )))
            })?;
            // BORROW: explicit .clone() — PathBuf from HashMap value
            paths.push(path.clone());
        }
        return Ok(paths);
    }

    // Single file
    let path = files
        .get("model.safetensors")
        .ok_or_else(|| {
            MIError::Model(candle_core::Error::Msg(
                "model.safetensors not found in downloaded files".into(),
            ))
        })?;
    // BORROW: explicit .clone() — PathBuf from HashMap value
    Ok(vec![path.clone()])
}

/// Create a `VarBuilder` from safetensors file paths.
///
/// Uses buffered (safe) loading by default. With the `mmap` feature,
/// uses memory-mapped loading for reduced memory overhead on large models.
#[cfg(feature = "transformer")]
fn create_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    #[cfg(feature = "mmap")]
    {
        mmap_var_builder(paths, dtype, device)
    }
    #[cfg(not(feature = "mmap"))]
    {
        buffered_var_builder(paths, dtype, device)
    }
}

/// Load weights via buffered (safe) reading — reads all data into RAM.
///
/// Only supports single-file models. For sharded models (7B+), enable
/// the `mmap` feature.
#[cfg(all(feature = "transformer", not(feature = "mmap")))]
fn buffered_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    if paths.len() > 1 {
        return Err(MIError::Config(
            "sharded models require the `mmap` feature: \
             candle-mi = { features = [\"mmap\"] }"
                .into(),
        ));
    }
    let path = paths
        .first()
        .ok_or_else(|| MIError::Model(candle_core::Error::Msg("no safetensors files".into())))?;
    let data = std::fs::read(path).map_err(|e| {
        MIError::Model(candle_core::Error::Msg(format!(
            "read {}: {e}",
            path.display()
        )))
    })?;
    let vb = candle_nn::VarBuilder::from_buffered_safetensors(data, dtype, device)?;
    Ok(vb)
}

/// Load weights via memory-mapped files — minimal RAM overhead for large models.
///
/// # Safety
///
/// The safetensors files must not be modified while the model is loaded.
/// This is the standard invariant for memory-mapped files.
#[cfg(all(feature = "transformer", feature = "mmap"))]
#[allow(unsafe_code)]
fn mmap_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    // SAFETY: safetensors files must not be modified while loaded.
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
    Ok(vb)
}
