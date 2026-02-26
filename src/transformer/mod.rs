// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic transformer implementation.
//!
//! A single forward pass covers `LLaMA`, `Qwen2`, Gemma 2, `Phi-3`,
//! `StarCoder2`, Mistral, and more — parameterized by
//! [`TransformerConfig`](crate::config::TransformerConfig).

pub(crate) mod attention;
pub(crate) mod mlp;
pub(crate) mod norm;
pub(crate) mod rope;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::backend::MIBackend;
use crate::config::TransformerConfig;
use crate::error::Result;
use crate::hooks::{HookCache, HookPoint, HookSpec};
use crate::util::masks;

use self::attention::Attention;
use self::mlp::Mlp;
use self::norm::{Norm, create_norm};
use self::rope::RopeCache;

// ---------------------------------------------------------------------------
// TransformerLayer
// ---------------------------------------------------------------------------

/// A single transformer decoder layer.
struct TransformerLayer {
    /// Pre-attention norm (always present).
    input_norm: Norm,
    /// Self-attention block.
    attention: Attention,
    /// Post-attention norm (Gemma 2 only; `None` for standard 2-norm models).
    post_attention_norm: Option<Norm>,
    /// Pre-MLP norm (standard models: `post_attention_layernorm`;
    /// Gemma 2: `pre_feedforward_layernorm`).
    mid_norm: Norm,
    /// Post-MLP norm (Gemma 2 only; `None` for standard 2-norm models).
    post_feedforward_norm: Option<Norm>,
    /// MLP block.
    mlp: Mlp,
}

impl TransformerLayer {
    /// Load a single decoder layer from weights.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if weight loading fails.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let input_norm = create_norm(
            config.norm_type,
            config.hidden_size,
            config.norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let attention = Attention::load(config, vb.pp("self_attn"))?;

        let post_attention_norm = if config.use_post_norms {
            Some(create_norm(
                config.norm_type,
                config.hidden_size,
                config.norm_eps,
                vb.pp("post_attention_layernorm"),
            )?)
        } else {
            None
        };

        // For Gemma 2 (4-norm): this is "pre_feedforward_layernorm".
        // For standard models (2-norm): this is "post_attention_layernorm".
        let mid_norm_name = if config.use_post_norms {
            "pre_feedforward_layernorm"
        } else {
            "post_attention_layernorm"
        };
        let mid_norm = create_norm(
            config.norm_type,
            config.hidden_size,
            config.norm_eps,
            vb.pp(mid_norm_name),
        )?;

        let post_feedforward_norm = if config.use_post_norms {
            Some(create_norm(
                config.norm_type,
                config.hidden_size,
                config.norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?)
        } else {
            None
        };

        let mlp = Mlp::load(config, vb.pp("mlp"))?;

        Ok(Self {
            input_norm,
            attention,
            post_attention_norm,
            mid_norm,
            post_feedforward_norm,
            mlp,
        })
    }
}

// ---------------------------------------------------------------------------
// GenericTransformer
// ---------------------------------------------------------------------------

/// Config-driven generic transformer backend.
///
/// One implementation covers `LLaMA`, `Qwen2`, Gemma 2, `Phi-3`,
/// `StarCoder2`, Mistral, and more.  Architecture differences are
/// captured in [`TransformerConfig`] fields; the forward pass branches
/// on these at runtime with zero overhead when the branch is not taken.
pub struct GenericTransformer {
    /// Token embedding matrix.
    embed_tokens: Embedding,
    /// Decoder layers.
    layers: Vec<TransformerLayer>,
    /// Final normalization before LM head.
    final_norm: Norm,
    /// LM head (vocabulary projection).  `None` when tied to `embed_tokens`.
    lm_head: Option<Linear>,
    /// Pre-computed `RoPE` cos/sin cache.
    rope_cache: RopeCache,
    /// Model configuration.
    config: TransformerConfig,
}

impl GenericTransformer {
    /// Load a generic transformer from a [`VarBuilder`].
    ///
    /// The caller constructs the `VarBuilder` (safe or mmap) and provides
    /// the parsed `TransformerConfig`.  This function loads all weights
    /// and pre-computes the `RoPE` cache.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if weight loading fails or dimensions
    /// are inconsistent.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(
        config: TransformerConfig,
        device: &Device,
        dtype: DType,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let vb_model = vb.pp("model");

        // --- Embedding ---
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb_model.pp("embed_tokens"),
        )?;

        // --- Layers ---
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let vb_layer = vb_model.pp(format!("layers.{i}"));
            let layer = TransformerLayer::load(&config, vb_layer)?;
            layers.push(layer);
        }

        // --- Final norm ---
        let final_norm = create_norm(
            config.norm_type,
            config.hidden_size,
            config.norm_eps,
            vb_model.pp("norm"),
        )?;

        // --- LM head ---
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(candle_nn::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        // --- RoPE cache ---
        let rope_cache = RopeCache::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            rope_cache,
            config,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Project hidden states to vocabulary logits (internal helper).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]` or `[batch, hidden_size]`
    /// - returns: `[batch, seq, vocab_size]` or `[batch, vocab_size]`
    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        if let Some(head) = &self.lm_head {
            Ok(head.forward(hidden)?)
        } else {
            // Tied embeddings: logits = hidden @ embed_tokens^T
            // broadcast_matmul handles [batch, seq, d] @ [d, vocab] → [batch, seq, vocab]
            let embed_weight = self.embed_tokens.embeddings();
            let logits = hidden.broadcast_matmul(&embed_weight.t()?)?;
            Ok(logits)
        }
    }

    /// Create the appropriate attention mask for a given layer.
    ///
    /// Returns a causal mask with optional sliding window.
    fn mask_for_layer(
        &self,
        layer_idx: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let use_sliding = match (
            self.config.sliding_window,
            self.config.alternating_sliding_window,
        ) {
            (Some(_), true) => layer_idx.is_multiple_of(2), // Gemma 2: even layers
            (Some(_), false) => true,                       // Mistral: all layers
            (None, _) => false,
        };

        if use_sliding {
            if let Some(window) = self.config.sliding_window {
                return create_sliding_window_mask(seq_len, window, device, dtype);
            }
        }

        masks::create_causal_mask(seq_len, device, dtype)
    }
}

// ---------------------------------------------------------------------------
// MIBackend implementation
// ---------------------------------------------------------------------------

impl MIBackend for GenericTransformer {
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        // --- Embedding ---
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        // Optional embedding scale (Gemma)
        if let Some(scale) = self.config.embedding_scale {
            hidden = (hidden * scale)?;
        }

        // Capture cache — collects hook captures; output set at the end.
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        // Hook: Embed
        if hooks.is_captured(&HookPoint::Embed) {
            cache.store(HookPoint::Embed, hidden.clone());
        }
        for intervention in hooks.interventions_at(&HookPoint::Embed) {
            hidden = crate::hooks::apply_intervention(&hidden, intervention)?;
        }

        let (_, seq_len, _) = hidden.dims3()?;

        // --- Layer loop ---
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Hook: ResidPre
            if hooks.is_captured(&HookPoint::ResidPre(layer_idx)) {
                cache.store(HookPoint::ResidPre(layer_idx), hidden.clone());
            }

            let residual = hidden.clone();

            // Pre-attention norm
            hidden = layer.input_norm.forward(&hidden)?;

            // Attention mask for this layer
            let mask = self.mask_for_layer(layer_idx, seq_len, device, dtype)?;

            // Self-attention (hooks for Q, K, V, Scores, Pattern handled inside)
            hidden = layer.attention.forward(
                &hidden,
                &mask,
                &self.rope_cache,
                layer_idx,
                hooks,
                &mut cache,
            )?;

            // Optional post-attention norm (Gemma 2)
            if let Some(ref norm) = layer.post_attention_norm {
                hidden = norm.forward(&hidden)?;
            }

            // Hook: AttnOut
            if hooks.is_captured(&HookPoint::AttnOut(layer_idx)) {
                cache.store(HookPoint::AttnOut(layer_idx), hidden.clone());
            }

            // Residual connection after attention
            hidden = (residual + &hidden)?;

            // Hook: ResidMid
            if hooks.is_captured(&HookPoint::ResidMid(layer_idx)) {
                cache.store(HookPoint::ResidMid(layer_idx), hidden.clone());
            }

            let residual = hidden.clone();

            // Pre-MLP norm
            hidden = layer.mid_norm.forward(&hidden)?;

            // Hook: MlpPre
            if hooks.is_captured(&HookPoint::MlpPre(layer_idx)) {
                cache.store(HookPoint::MlpPre(layer_idx), hidden.clone());
            }

            // MLP
            hidden = layer.mlp.forward(&hidden)?;

            // Hook: MlpPost
            if hooks.is_captured(&HookPoint::MlpPost(layer_idx)) {
                cache.store(HookPoint::MlpPost(layer_idx), hidden.clone());
            }

            // Optional post-feedforward norm (Gemma 2)
            if let Some(ref norm) = layer.post_feedforward_norm {
                hidden = norm.forward(&hidden)?;
            }

            // Hook: MlpOut
            if hooks.is_captured(&HookPoint::MlpOut(layer_idx)) {
                cache.store(HookPoint::MlpOut(layer_idx), hidden.clone());
            }

            // Residual connection after MLP
            hidden = (residual + &hidden)?;

            // Hook: ResidPost
            if hooks.is_captured(&HookPoint::ResidPost(layer_idx)) {
                cache.store(HookPoint::ResidPost(layer_idx), hidden.clone());
            }
        }

        // --- Final norm ---
        hidden = self.final_norm.forward(&hidden)?;

        // Hook: FinalNorm
        if hooks.is_captured(&HookPoint::FinalNorm) {
            cache.store(HookPoint::FinalNorm, hidden.clone());
        }

        // --- LM head ---
        let mut logits = self.project_logits(&hidden)?;

        // Optional final logit soft-capping (Gemma 2)
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = ((logits / cap)?.tanh()? * cap)?;
        }

        // Set the real output (logits) on the capture cache.
        cache.set_output(logits);

        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.project_logits(hidden)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a sliding window causal mask.
///
/// # Shapes
/// - returns: `[1, 1, seq_len, seq_len]`
///
/// Positions beyond the window or in the future are set to `-inf`.
fn create_sliding_window_mask(
    seq_len: usize,
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask_data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j > i || i.saturating_sub(j) > window {
                // idx is always < seq_len * seq_len by construction
                if let Some(cell) = mask_data.get_mut(idx) {
                    *cell = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?)
}
