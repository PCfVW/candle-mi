// SPDX-License-Identifier: MIT OR Apache-2.0

//! `AlgZoo` model backends — stoicheia (στοιχεῖα, "elements").
//!
//! Two [`MIBackend`] implementations for ARC's
//! [`AlgZoo`](https://github.com/alignment-research-center/alg-zoo) tiny models:
//!
//! - [`StoicheiaRnn`] — single-layer `ReLU` RNN (continuous input)
//! - [`StoicheiaTransformer`] — attention-only transformer (discrete input)
//!
//! These models have 8–1,408 parameters and solve algorithmic tasks.
//! They are designed as "model organisms" for exhaustive mechanistic
//! interpretability.
//!
//! # Weight loading
//!
//! Weights are loaded from `safetensors` files (converted from `PyTorch` `.pth`
//! via [anamnesis](https://crates.io/crates/anamnesis)):
//!
//! ```text
//! amn remember model.pth --to safetensors
//! ```

pub mod config;
pub mod tasks;

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::backend::MIBackend;
use crate::error::Result;
use crate::hooks::{HookCache, HookPoint, HookSpec};

pub use config::{StoicheiaArch, StoicheiaConfig, StoicheiaOutput, StoicheiaTask};

// ---------------------------------------------------------------------------
// StoicheiaRnn
// ---------------------------------------------------------------------------

/// Single-layer `ReLU` RNN backend for `AlgZoo` continuous tasks.
///
/// Architecture (from `AlgZoo`'s `architectures.py`):
///
/// ```text
/// For each timestep t:
///     pre_act_t = W_ih @ x_t + W_hh @ h_{t-1}    // [batch, H]
///     h_t = relu(pre_act_t)                        // [batch, H]
/// output = W_oh @ h_final                          // [batch, output_size]
/// ```
///
/// Where `x_t` is scalar (`input_size = 1`), so `W_ih` has shape `[H, 1]`.
///
/// # Hook points
///
/// | Hook | Shape | Description |
/// |------|-------|-------------|
/// | `Custom("rnn.hook_pre_activation.{t}")` | `[batch, H]` | Before ReLU at timestep `t` |
/// | `Custom("rnn.hook_hidden.{t}")` | `[batch, H]` | Hidden state after timestep `t` |
/// | `Custom("rnn.hook_final_state")` | `[batch, H]` | Final hidden state |
/// | `Custom("rnn.hook_output")` | `[batch, output_size]` | After output projection |
pub struct StoicheiaRnn {
    /// Input-to-hidden weights: `[H, 1]`.
    weight_ih: Tensor,
    /// Hidden-to-hidden weights: `[H, H]`.
    weight_hh: Tensor,
    /// Output projection weights: `[output_size, H]`.
    weight_oh: Tensor,
    /// Model configuration.
    config: StoicheiaConfig,
}

impl StoicheiaRnn {
    /// Load an `AlgZoo` RNN from a safetensors file.
    ///
    /// The safetensors file must contain:
    /// - `rnn.weight_ih_l0`: `[H, 1]`
    /// - `rnn.weight_hh_l0`: `[H, H]`
    /// - `linear.weight`: `[output_size, H]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if weights are missing or have wrong shapes.
    #[allow(clippy::similar_names)]
    pub fn load(
        config: StoicheiaConfig,
        safetensors_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let buffer = std::fs::read(safetensors_path.as_ref())?;
        let vb = VarBuilder::from_buffered_safetensors(buffer, DType::F32, device)?;

        let weight_ih = vb.get((config.hidden_size, 1), "rnn.weight_ih_l0")?;
        let weight_hh = vb.get((config.hidden_size, config.hidden_size), "rnn.weight_hh_l0")?;
        let weight_oh = vb.get((config.output_size(), config.hidden_size), "linear.weight")?;

        Ok(Self {
            weight_ih,
            weight_hh,
            weight_oh,
            config,
        })
    }
}

impl MIBackend for StoicheiaRnn {
    fn num_layers(&self) -> usize {
        1
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.output_size()
    }

    fn num_heads(&self) -> usize {
        0
    }

    fn forward(&self, input: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input.device();
        let (batch_size, seq_len) = input.dims2()?;
        let h = self.config.hidden_size;

        // Initialize hidden state to zeros: [batch, H]
        let mut hidden = Tensor::zeros((batch_size, h), DType::F32, device)?;

        // Placeholder output — replaced at the end
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        // Pre-scan: determine which timesteps need hook capture.
        // This avoids per-timestep String allocation when hooks are empty
        // (zero-overhead guarantee) or when only specific timesteps are
        // captured (one allocation per timestep at scan time, not per
        // forward pass iteration).
        let has_hooks = !hooks.is_empty();
        let (captured_pre_act, captured_hidden) = if has_hooks {
            let pre_act: std::collections::HashSet<usize> = (0..seq_len)
                .filter(|t| {
                    hooks.is_captured(&HookPoint::Custom(format!("rnn.hook_pre_activation.{t}")))
                })
                .collect();
            let hid: std::collections::HashSet<usize> = (0..seq_len)
                .filter(|t| hooks.is_captured(&HookPoint::Custom(format!("rnn.hook_hidden.{t}"))))
                .collect();
            (pre_act, hid)
        } else {
            (
                std::collections::HashSet::new(),
                std::collections::HashSet::new(),
            )
        };

        // RNN loop: one timestep at a time
        for t in 0..seq_len {
            // x_t: [batch, 1] — scalar input per timestep
            // INDEX: t is bounded by seq_len from dims2()
            let x_t = input.i((.., t..t + 1))?;

            // pre_act = x_t @ W_ih^T + h_{t-1} @ W_hh^T
            // x_t @ W_ih^T: [batch, 1] @ [1, H] → [batch, H]
            let ih = x_t.matmul(&self.weight_ih.t()?)?;
            // h_{t-1} @ W_hh^T: [batch, H] @ [H, H] → [batch, H]
            let hh = hidden.matmul(&self.weight_hh.t()?)?;
            let pre_act = (ih + hh)?;

            // Hook: pre-activation at timestep t (no allocation if not captured)
            if captured_pre_act.contains(&t) {
                cache.store(
                    HookPoint::Custom(format!("rnn.hook_pre_activation.{t}")),
                    pre_act.clone(),
                );
            }

            // h_t = relu(pre_act)
            hidden = pre_act.relu()?;

            // Hook: hidden state at timestep t (no allocation if not captured)
            if captured_hidden.contains(&t) {
                cache.store(
                    HookPoint::Custom(format!("rnn.hook_hidden.{t}")),
                    hidden.clone(),
                );
            }
        }

        // Hook: final hidden state (allocated only when captured)
        if has_hooks {
            let final_hook = HookPoint::Custom("rnn.hook_final_state".into());
            if hooks.is_captured(&final_hook) {
                cache.store(final_hook, hidden.clone());
            }
        }

        // Output projection: [batch, H] @ [H, output_size] → [batch, output_size]
        let output = hidden.matmul(&self.weight_oh.t()?)?;

        // Hook: output (allocated only when captured)
        if has_hooks {
            let output_hook = HookPoint::Custom("rnn.hook_output".into());
            if hooks.is_captured(&output_hook) {
                cache.store(output_hook, output.clone());
            }
        }

        // Unsqueeze to [batch, 1, output_size] to match MIBackend convention
        let output_3d = output.unsqueeze(1)?;
        cache.set_output(output_3d);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        // Apply output linear projection: [batch, H] → [batch, output_size]
        Ok(hidden.matmul(&self.weight_oh.t()?)?)
    }
}

// ---------------------------------------------------------------------------
// StoicheiaTransformer
// ---------------------------------------------------------------------------

/// Attention layer for `AlgZoo`'s attention-only transformer.
///
/// `PyTorch` packs Q, K, V into a single `in_proj_weight` of shape `[3*H, H]`.
struct AttentionLayer {
    /// Packed Q, K, V projection: `[3*H, H]`.
    in_proj_weight: Tensor,
    /// Output projection: `[H, H]`.
    out_proj_weight: Tensor,
    /// Hidden size.
    hidden_size: usize,
}

impl AttentionLayer {
    /// Run one attention layer (full bidirectional, single head, no causal mask).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, H]`
    /// - returns: `(attn_output, scores, pattern)` where
    ///   - `attn_output`: `[batch, seq, H]`
    ///   - `scores`: `[batch, 1, seq, seq]` (pre-softmax)
    ///   - `pattern`: `[batch, 1, seq, seq]` (post-softmax)
    fn forward(&self, hidden: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let dim = self.hidden_size;

        // Project Q, K, V from packed in_proj_weight
        // hidden @ in_proj_weight^T: [batch, seq, H] @ [H, 3H] → [batch, seq, 3H]
        let qkv = hidden.broadcast_matmul(&self.in_proj_weight.t()?)?;

        // Split into Q, K, V each [batch, seq, H]
        let query = qkv.narrow(2, 0, dim)?;
        let key = qkv.narrow(2, dim, dim)?;
        let value = qkv.narrow(2, 2 * dim, dim)?;

        // Attention scores: Q @ K^T / sqrt(H)
        // [batch, seq, H] @ [batch, H, seq] → [batch, seq, seq]
        // CAST: usize → f64, hidden_size for attention scale sqrt
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let scale = (dim as f64).sqrt();
        let scores_2d = (query.matmul(&key.t()?)? / scale)?;

        // Unsqueeze to [batch, 1, seq, seq] for head dimension
        let scores = scores_2d.unsqueeze(1)?;

        // Softmax over last dimension (no causal mask — full bidirectional)
        let pattern = candle_nn::ops::softmax_last_dim(&scores)?;

        // Weighted sum: pattern @ V
        // [batch, 1, seq, seq] → squeeze → [batch, seq, seq]
        let pattern_2d = pattern.squeeze(1)?;
        // [batch, seq, seq] @ [batch, seq, H] → [batch, seq, H]
        let attn_out = pattern_2d.matmul(&value)?;

        // Output projection: [batch, seq, H] @ [H, H] → [batch, seq, H]
        let projected = attn_out.broadcast_matmul(&self.out_proj_weight.t()?)?;

        Ok((projected, scores, pattern))
    }
}

/// Attention-only transformer backend for `AlgZoo` discrete tasks.
///
/// Architecture (from `AlgZoo`'s `architectures.py`):
///
/// ```text
/// x = embed(input) + pos_embed(positions)
/// for each attention layer:
///     x = x + attention(x, x, x)       // residual, full bidirectional
/// output = unembed(x[:, -1])            // last position only
/// ```
///
/// No MLP blocks, no layer normalization, no causal mask.
///
/// # Hook points
///
/// | Hook | Shape | Description |
/// |------|-------|-------------|
/// | `Embed` | `[batch, seq, H]` | After token + positional embedding |
/// | `ResidPre(i)` | `[batch, seq, H]` | Before attention layer `i` |
/// | `AttnScores(i)` | `[batch, 1, seq, seq]` | Pre-softmax attention |
/// | `AttnPattern(i)` | `[batch, 1, seq, seq]` | Post-softmax attention |
/// | `AttnOut(i)` | `[batch, seq, H]` | Attention output (before residual add) |
/// | `ResidPost(i)` | `[batch, seq, H]` | After residual add |
pub struct StoicheiaTransformer {
    /// Token embedding.
    embed: Embedding,
    /// Positional embedding.
    pos_embed: Embedding,
    /// Attention layers.
    attns: Vec<AttentionLayer>,
    /// Unembedding weights: `[output_size, H]`.
    unembed_weight: Tensor,
    /// Model configuration.
    config: StoicheiaConfig,
}

impl StoicheiaTransformer {
    /// Load an `AlgZoo` attention-only transformer from a safetensors file.
    ///
    /// The safetensors file must contain:
    /// - `embed.weight`: `[input_range, H]`
    /// - `pos_embed.weight`: `[seq_len, H]`
    /// - `attns.{i}.in_proj_weight`: `[3*H, H]` for each layer
    /// - `attns.{i}.out_proj.weight`: `[H, H]` for each layer
    /// - `unembed.weight`: `[output_size, H]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if weights are missing or have wrong shapes.
    pub fn load(
        config: StoicheiaConfig,
        safetensors_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> {
        let buffer = std::fs::read(safetensors_path.as_ref())?;
        let vb = VarBuilder::from_buffered_safetensors(buffer, DType::F32, device)?;

        let h = config.hidden_size;

        let embed = Embedding::new(vb.get((config.input_range, h), "embed.weight")?, h);
        let pos_embed = Embedding::new(vb.get((config.seq_len, h), "pos_embed.weight")?, h);

        let mut attns = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let in_proj_weight = vb.get((3 * h, h), &format!("attns.{i}.in_proj_weight"))?;
            let out_proj_weight = vb.get((h, h), &format!("attns.{i}.out_proj.weight"))?;
            attns.push(AttentionLayer {
                in_proj_weight,
                out_proj_weight,
                hidden_size: h,
            });
        }

        let unembed_weight = vb.get((config.output_size(), h), "unembed.weight")?;

        Ok(Self {
            embed,
            pos_embed,
            attns,
            unembed_weight,
            config,
        })
    }
}

impl MIBackend for StoicheiaTransformer {
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.output_size()
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();
        let (batch, seq_len) = input_ids.dims2()?;

        // Placeholder output — replaced at the end
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        // Token embedding + positional embedding
        let token_emb = self.embed.forward(input_ids)?;
        // CAST: usize → u32, seq_len fits in u32 (`AlgZoo` max seq_len = 10)
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_ids = Tensor::new(&positions[..], device)?
            .unsqueeze(0)?
            .expand((batch, seq_len))?;
        let pos_emb = self.pos_embed.forward(&pos_ids)?;
        let mut hidden = (token_emb + pos_emb)?;

        let has_hooks = !hooks.is_empty();

        // Hook: Embed
        if has_hooks && hooks.is_captured(&HookPoint::Embed) {
            cache.store(HookPoint::Embed, hidden.clone());
        }

        // Attention layers with residual connections
        for (i, attn) in self.attns.iter().enumerate() {
            // Hook: ResidPre
            if has_hooks && hooks.is_captured(&HookPoint::ResidPre(i)) {
                cache.store(HookPoint::ResidPre(i), hidden.clone());
            }

            let (attn_out, scores, pattern) = attn.forward(&hidden)?;

            // Hook: AttnScores
            if has_hooks && hooks.is_captured(&HookPoint::AttnScores(i)) {
                cache.store(HookPoint::AttnScores(i), scores);
            }

            // Hook: AttnPattern
            if has_hooks && hooks.is_captured(&HookPoint::AttnPattern(i)) {
                cache.store(HookPoint::AttnPattern(i), pattern);
            }

            // Hook: AttnOut
            if has_hooks && hooks.is_captured(&HookPoint::AttnOut(i)) {
                cache.store(HookPoint::AttnOut(i), attn_out.clone());
            }

            // Residual connection
            hidden = (hidden + attn_out)?;

            // Hook: ResidPost
            if has_hooks && hooks.is_captured(&HookPoint::ResidPost(i)) {
                cache.store(HookPoint::ResidPost(i), hidden.clone());
            }
        }

        // Unembed last position only: [batch, H] → [batch, output_size]
        // INDEX: seq_len-1 is valid because seq_len >= 1 from dims2()
        let last_hidden = hidden.i((.., seq_len - 1, ..))?;
        let output = last_hidden.matmul(&self.unembed_weight.t()?)?;

        // Unsqueeze to [batch, 1, output_size] to match MIBackend convention
        let output_3d = output.unsqueeze(1)?;
        cache.set_output(output_3d);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        // Apply unembedding projection: [batch, H] → [batch, output_size]
        Ok(hidden.matmul(&self.unembed_weight.t()?)?)
    }
}
