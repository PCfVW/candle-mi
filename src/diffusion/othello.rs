// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OthelloGpt`: a plain GPT-2-style bidirectional backbone.
//!
//! A faithful Rust port of the `OthelloMDLM` world model (a nanoGPT/minGPT
//! lineage transformer) used in masked-diffusion Othello probing studies.
//! Unlike the [`GenericMdlm`](super::mdlm::GenericMdlm) `DiT`, this backbone
//! has **no** `adaLN` conditioning, **no** rotary embeddings, and **no**
//! weight-only `LayerNorm`.  Instead it is the classic GPT-2 recipe:
//!
//! - **learned absolute** positional embeddings (`nn.Embedding`), added to the
//!   token embedding — there is no `RoPE`;
//! - **full** `LayerNorm` (weight *and* bias) at every norm site;
//! - **with-bias** fused QKV, attention output, and both MLP linears;
//! - an exact (erf) `GELU` MLP — *not* the tanh approximation;
//! - an **untied** vocabulary head with no bias.
//!
//! Attention is bidirectional by default (`causal = false`, the masked-diffusion
//! setting); the [`causal`](OthelloGptConfig::causal) flag enables a causal mask
//! so the same module can also load an autoregressive Othello-GPT control.
//!
//! The backbone is feature-gated behind `diffusion` because its reason to exist
//! is the masked-diffusion `SUBS` sampler and the diffusion logit-lens in
//! [`sample`](super::sample), both of which operate on any
//! [`MIBackend`].

#[cfg(test)]
use std::collections::HashMap;

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use serde_json::Value;

use crate::backend::MIBackend;
use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookPoint, HookSpec};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `OthelloGpt` model configuration (a GPT-2-style backbone).
///
/// Mirrors the upstream `OthelloMDLMConfig`.  The released world model is
/// `vocab_size = 62`, `block_size = 60`, `n_layer = 8`, `n_head = 8`,
/// `n_embd = 512`, `causal = false`.
#[derive(Debug, Clone)]
pub struct OthelloGptConfig {
    /// Vocabulary size (60 move cells + pad + `[MASK]` for the world model).
    pub vocab_size: usize,
    /// Maximum sequence length covered by the learned positional embedding.
    pub block_size: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Hidden dimension (`d_model`); must be divisible by `n_head`.
    pub n_embd: usize,
    /// Per-head dimension (`n_embd / n_head`).
    pub head_dim: usize,
    /// Feed-forward expansion ratio (GPT-2 uses `4`; not stored upstream).
    pub mlp_ratio: usize,
    /// `LayerNorm` epsilon (GPT-2 uses `1e-5`).
    pub norm_eps: f64,
    /// Whether to apply a causal attention mask.  `false` (bidirectional) for
    /// the masked-diffusion world model; `true` for an autoregressive control.
    pub causal: bool,
}

impl OthelloGptConfig {
    /// Construct a config from explicit dimensions, deriving `head_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `n_head`
    /// is zero or `n_embd` is not divisible by `n_head`.
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        causal: bool,
    ) -> Result<Self> {
        if n_head == 0 || !n_embd.is_multiple_of(n_head) {
            return Err(MIError::Config(format!(
                "n_embd {n_embd} not divisible by n_head {n_head}"
            )));
        }
        Ok(Self {
            vocab_size,
            block_size,
            n_layer,
            n_head,
            n_embd,
            head_dim: n_embd / n_head,
            mlp_ratio: 4,
            norm_eps: 1e-5,
            causal,
        })
    }

    /// Parse an [`OthelloGptConfig`] from a companion `config.json` value.
    ///
    /// The converter (`scripts/convert_othello_mdlm.py`) writes this file from
    /// the checkpoint's `config` dict, so the keys mirror `OthelloMDLMConfig`:
    /// `vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, and the
    /// optional `causal` (default `false`).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a required
    /// key is missing or not a non-negative integer, or if `n_embd` is not
    /// divisible by `n_head`.
    pub fn from_hf_config(config: &Value) -> Result<Self> {
        let vocab_size = get_usize(config, "vocab_size")?;
        let block_size = get_usize(config, "block_size")?;
        let n_layer = get_usize(config, "n_layer")?;
        let n_head = get_usize(config, "n_head")?;
        let n_embd = get_usize(config, "n_embd")?;
        let causal = get_bool_or(config, "causal", false);
        Self::new(vocab_size, block_size, n_layer, n_head, n_embd, causal)
    }
}

/// Read a required non-negative integer config field as `usize`.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the key is
/// absent or not a `u64`.
fn get_usize(config: &Value, key: &str) -> Result<usize> {
    let value = config.get(key).and_then(Value::as_u64).ok_or_else(|| {
        MIError::Config(format!(
            "missing or non-integer `{key}` in OthelloGpt config"
        ))
    })?;
    // CAST: u64 → usize, model dimensions fit in usize on 64-bit targets
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    Ok(value as usize)
}

/// Read an optional boolean config field, falling back to `default` when absent.
fn get_bool_or(config: &Value, key: &str, default: bool) -> bool {
    config.get(key).and_then(Value::as_bool).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Hook helper
// ---------------------------------------------------------------------------

/// Apply the standard capture-then-intervene hook protocol at `point`.
///
/// Mirrors the helper used in
/// [`mdlm`](super::mdlm): the activation is cloned into the cache when
/// captured, then each registered intervention is applied in turn (mutating
/// `tensor` in place).
///
/// # Errors
///
/// Returns [`MIError::Model`] if an
/// intervention's tensor operation fails.
// The by-value `HookPoint` lets call sites pass a freshly-built variant without
// `&`; capturing still needs one clone either way.
#[allow(clippy::needless_pass_by_value)]
fn hook_point(
    tensor: &mut Tensor,
    point: HookPoint,
    hooks: &HookSpec,
    cache: &mut HookCache,
) -> Result<()> {
    if hooks.is_captured(&point) {
        cache.store(point.clone(), tensor.clone());
    }
    for intervention in hooks.interventions_at(&point) {
        *tensor = crate::hooks::apply_intervention(tensor, intervention)?;
    }
    Ok(())
}

/// Build an additive causal mask of shape `[1, 1, seq_len, seq_len]`.
///
/// Entries above the diagonal are `f32::NEG_INFINITY` (forbidden); entries on
/// or below the diagonal are `0.0`.  Broadcast-added to the attention scores.
///
/// # Shapes
/// - returns: `[1, 1, seq_len, seq_len]`
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failures.
fn causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let tensor = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?;
    Ok(tensor.to_dtype(dtype)?)
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// A single GPT-2-style block: pre-`LayerNorm`, bidirectional (or causal)
/// self-attention, pre-`LayerNorm`, and an exact-`GELU` MLP, each with a
/// residual connection.
struct OthelloBlock {
    /// Pre-attention `LayerNorm` (weight + bias).
    ln1: LayerNorm,
    /// Pre-MLP `LayerNorm` (weight + bias).
    ln2: LayerNorm,
    /// Fused QKV projection: `n_embd → 3 * n_embd` (with bias).
    qkv: Linear,
    /// Attention output projection: `n_embd → n_embd` (with bias).
    proj: Linear,
    /// MLP up-projection: `n_embd → mlp_ratio * n_embd` (with bias).
    mlp_fc: Linear,
    /// MLP down-projection: `mlp_ratio * n_embd → n_embd` (with bias).
    mlp_proj: Linear,
    /// Number of attention heads.
    n_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// Hidden dimension (`n_heads * head_dim`).
    hidden_dim: usize,
    /// Attention scale `1 / sqrt(head_dim)`.
    scale: f64,
    /// Whether to apply a causal attention mask.
    causal: bool,
}

impl OthelloBlock {
    /// Load a block from the `blocks.{i}` namespace.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if any weight
    /// fails to load.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &OthelloGptConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.n_embd;
        let inter = config.mlp_ratio * h;
        // CAST: usize → f64, head_dim fits in f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        let ln_cfg = candle_nn::LayerNormConfig {
            eps: config.norm_eps,
            ..Default::default()
        };

        Ok(Self {
            ln1: candle_nn::layer_norm(h, ln_cfg, vb.pp("ln1"))?,
            ln2: candle_nn::layer_norm(h, ln_cfg, vb.pp("ln2"))?,
            qkv: candle_nn::linear(h, 3 * h, vb.pp("attn").pp("qkv"))?,
            proj: candle_nn::linear(h, h, vb.pp("attn").pp("proj"))?,
            mlp_fc: candle_nn::linear(h, inter, vb.pp("mlp").pp("0"))?,
            mlp_proj: candle_nn::linear(inter, h, vb.pp("mlp").pp("2"))?,
            n_heads: config.n_head,
            head_dim: config.head_dim,
            hidden_dim: h,
            scale,
            causal: config.causal,
        })
    }

    /// Self-attention sublayer (bidirectional unless `causal`).
    ///
    /// Captures `AttnQ`/`AttnK`/`AttnV` (post-reshape), `AttnScores`
    /// (post-scale, post-mask), and `AttnPattern` (post-softmax).  No `RoPE`
    /// is applied — positional information is carried by the learned absolute
    /// embedding added at the model input.
    ///
    /// # Shapes
    /// - `xs`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn attention(
        &self,
        xs: &Tensor,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let hidden = self.hidden_dim;

        let qkv = self.qkv.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, hidden)?;
        let k = qkv.narrow(D::Minus1, hidden, hidden)?;
        let v = qkv.narrow(D::Minus1, 2 * hidden, hidden)?;

        // [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
        let mut q = q
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        hook_point(&mut q, HookPoint::AttnQ(layer_idx), hooks, cache)?;
        hook_point(&mut k, HookPoint::AttnK(layer_idx), hooks, cache)?;
        hook_point(&mut v, HookPoint::AttnV(layer_idx), hooks, cache)?;

        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let k_t = k.contiguous()?.transpose(2, 3)?;
        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let q = q.contiguous()?;
        let mut scores = (q.matmul(&k_t)? * self.scale)?;
        if self.causal {
            let mask = causal_mask(seq_len, scores.device(), scores.dtype())?;
            scores = scores.broadcast_add(&mask)?;
        }
        hook_point(&mut scores, HookPoint::AttnScores(layer_idx), hooks, cache)?;

        // Softmax in F32 (no-op promote on the F32 default path; defensive for
        // lower-precision loads).
        let original_dtype = scores.dtype();
        let scores_f32 = if original_dtype == DType::F32 {
            scores
        } else {
            // PROMOTE: softmax over a lower-precision dtype can produce NaN; compute in F32
            scores.to_dtype(DType::F32)?
        };
        // Backward-safe dispatch: fused kernel for inference, composed form
        // when the graph is tracked (training over a `VarMap`).
        let mut pattern = crate::nn_ops::softmax_last_dim(&scores_f32)?;
        if original_dtype != DType::F32 {
            pattern = pattern.to_dtype(original_dtype)?;
        }
        hook_point(
            &mut pattern,
            HookPoint::AttnPattern(layer_idx),
            hooks,
            cache,
        )?;

        // CONTIGUOUS: ensure contiguous layout for the pattern·value matmul
        let v = v.contiguous()?;
        let attn = pattern.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, hidden))?;

        Ok(self.proj.forward(&attn)?)
    }

    /// Exact-`GELU` MLP: `proj(gelu_erf(fc(x)))`.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.mlp_fc.forward(x)?;
        // Exact (erf) GELU — PyTorch `nn.GELU()` default. `Tensor::gelu()` is the
        // tanh approximation and must NOT be used here.
        let act = up.gelu_erf()?;
        Ok(self.mlp_proj.forward(&act)?)
    }

    /// Run the full block forward with hook support.
    ///
    /// Hook semantics follow the `TransformerLens` "added to the residual
    /// stream" convention: `AttnOut` / `MlpOut` capture the (ungated) sublayer
    /// contributions actually summed into the residual.
    ///
    /// # Shapes
    /// - `hidden_in`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn forward(
        &self,
        hidden_in: &Tensor,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut hidden = hidden_in.clone();
        hook_point(&mut hidden, HookPoint::ResidPre(layer_idx), hooks, cache)?;

        // --- attention sublayer ---
        let residual = hidden.clone();
        // Backward-safe dispatch: with-bias `LayerNorm` takes candle's fused
        // (no-backward) kernel; route through `nn_ops` so training works.
        let normed1 = crate::nn_ops::layer_norm(&self.ln1, &residual)?;
        let mut attn = self.attention(&normed1, layer_idx, hooks, cache)?;
        hook_point(&mut attn, HookPoint::AttnOut(layer_idx), hooks, cache)?;
        hidden = (residual + attn)?;
        hook_point(&mut hidden, HookPoint::ResidMid(layer_idx), hooks, cache)?;

        // --- MLP sublayer ---
        let residual2 = hidden.clone();
        let mut normed2 = crate::nn_ops::layer_norm(&self.ln2, &hidden)?;
        hook_point(&mut normed2, HookPoint::MlpPre(layer_idx), hooks, cache)?;
        let mut mlp_out = self.mlp(&normed2)?;
        hook_point(&mut mlp_out, HookPoint::MlpPost(layer_idx), hooks, cache)?;
        // No gating: the MLP output IS the residual contribution.
        hook_point(&mut mlp_out, HookPoint::MlpOut(layer_idx), hooks, cache)?;
        hidden = (residual2 + mlp_out)?;
        hook_point(&mut hidden, HookPoint::ResidPost(layer_idx), hooks, cache)?;

        Ok(hidden)
    }
}

// ---------------------------------------------------------------------------
// OthelloGpt
// ---------------------------------------------------------------------------

/// `OthelloGpt` backend: a plain GPT-2-style bidirectional transformer with
/// full hook support.
///
/// Load via [`OthelloGpt::load`] from a [`VarBuilder`] over a converted
/// `safetensors` checkpoint (see `scripts/convert_othello_mdlm.py`).  The
/// weight keys match the upstream `OthelloMDLM` state dict verbatim, so no
/// transposes or renames happen at load time.
pub struct OthelloGpt {
    /// Token embedding (`tok_emb.weight`).
    tok_emb: Embedding,
    /// Learned absolute positional embedding (`pos_emb.weight`):
    /// `[block_size, n_embd]`.
    pos_emb: Tensor,
    /// Transformer blocks.
    blocks: Vec<OthelloBlock>,
    /// Final `LayerNorm` (`ln_f`, weight + bias).
    ln_f: LayerNorm,
    /// Untied vocabulary head (`head.weight`, no bias).
    head: Linear,
    /// Model configuration.
    config: OthelloGptConfig,
}

impl OthelloGpt {
    /// Load an `OthelloGpt` model from a [`VarBuilder`].
    ///
    /// The caller constructs the `VarBuilder` (buffered or mmap) and provides
    /// the parsed [`OthelloGptConfig`].  Weight keys are read verbatim from the
    /// upstream state dict: `tok_emb.weight`, `pos_emb.weight`,
    /// `blocks.{i}.{ln1,ln2,attn.qkv,attn.proj,mlp.0,mlp.2}`, `ln_f`, and
    /// `head.weight`.
    ///
    /// # Shapes
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` token ids to `[batch, seq, vocab_size]` logits.
    ///
    /// # Memory
    /// Loads the single converted `safetensors` (~100 MB for the released
    /// 25.3 M-param world model) through the caller's `VarBuilder`: near-zero
    /// extra copy with the `mmap` feature, or ~100 MB CPU when buffered.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if any weight
    /// fails to load or a dimension is inconsistent with the checkpoint.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(config: OthelloGptConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.n_embd;

        let tok_emb = Embedding::new(vb.pp("tok_emb").get((config.vocab_size, h), "weight")?, h);
        let pos_emb = vb.pp("pos_emb").get((config.block_size, h), "weight")?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(OthelloBlock::load(&config, vb.pp(format!("blocks.{i}")))?);
        }

        let ln_cfg = candle_nn::LayerNormConfig {
            eps: config.norm_eps,
            ..Default::default()
        };
        let ln_f = candle_nn::layer_norm(h, ln_cfg, vb.pp("ln_f"))?;
        let head = candle_nn::linear_no_bias(h, config.vocab_size, vb.pp("head"))?;

        Ok(Self {
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            head,
            config,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &OthelloGptConfig {
        &self.config
    }

    /// Final `LayerNorm` then untied head, with the `FinalNorm` hook.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn head_forward(
        &self,
        hidden: &Tensor,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut xs = crate::nn_ops::layer_norm(&self.ln_f, hidden)?;
        hook_point(&mut xs, HookPoint::FinalNorm, hooks, cache)?;
        Ok(self.head.forward(&xs)?)
    }
}

impl MIBackend for OthelloGpt {
    fn num_layers(&self) -> usize {
        self.config.n_layer
    }

    fn hidden_size(&self) -> usize {
        self.config.n_embd
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_heads(&self) -> usize {
        self.config.n_head
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();
        let (_batch, seq_len) = input_ids.dims2()?;
        if seq_len > self.config.block_size {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "seq_len {seq_len} exceeds block_size {} (no positional embedding)",
                self.config.block_size
            ))));
        }

        // Token embedding + learned absolute positions (broadcast over batch).
        let mut hidden = self.tok_emb.forward(input_ids)?;
        let pos = self.pos_emb.narrow(0, 0, seq_len)?;
        hidden = hidden.broadcast_add(&pos)?;

        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);
        hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, layer_idx, hooks, &mut cache)?;
        }

        let logits = self.head_forward(&hidden, hooks, &mut cache)?;
        cache.set_output(logits);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        let xs = crate::nn_ops::layer_norm(&self.ln_f, hidden)?;
        Ok(self.head.forward(&xs)?)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Tensor> {
        let device = self.tok_emb.embeddings().device();
        let ids = Tensor::new(&[token_id], device)?;
        let emb = self.tok_emb.forward(&ids)?; // [1, hidden]
        Ok(emb.squeeze(0)?) // [hidden]
    }
}

// ---------------------------------------------------------------------------
// Synthetic VarBuilder for tests
// ---------------------------------------------------------------------------

/// Build an in-memory [`VarBuilder`] of zero-initialised weights matching the
/// `OthelloGpt` layout for `config`.  Test-only: exercises the full load and
/// forward path without any download.
#[cfg(test)]
fn synthetic_var_builder(
    config: &OthelloGptConfig,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let h = config.n_embd;
    let inter = config.mlp_ratio * h;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();

    let mut put = |name: String, dims: Vec<usize>| -> Result<()> {
        tensors.insert(name, Tensor::zeros(dims, DType::F32, device)?);
        Ok(())
    };

    put("tok_emb.weight".to_string(), vec![config.vocab_size, h])?;
    put("pos_emb.weight".to_string(), vec![config.block_size, h])?;
    for i in 0..config.n_layer {
        put(format!("blocks.{i}.ln1.weight"), vec![h])?;
        put(format!("blocks.{i}.ln1.bias"), vec![h])?;
        put(format!("blocks.{i}.ln2.weight"), vec![h])?;
        put(format!("blocks.{i}.ln2.bias"), vec![h])?;
        put(format!("blocks.{i}.attn.qkv.weight"), vec![3 * h, h])?;
        put(format!("blocks.{i}.attn.qkv.bias"), vec![3 * h])?;
        put(format!("blocks.{i}.attn.proj.weight"), vec![h, h])?;
        put(format!("blocks.{i}.attn.proj.bias"), vec![h])?;
        put(format!("blocks.{i}.mlp.0.weight"), vec![inter, h])?;
        put(format!("blocks.{i}.mlp.0.bias"), vec![inter])?;
        put(format!("blocks.{i}.mlp.2.weight"), vec![h, inter])?;
        put(format!("blocks.{i}.mlp.2.bias"), vec![h])?;
    }
    put("ln_f.weight".to_string(), vec![h])?;
    put("ln_f.bias".to_string(), vec![h])?;
    put("head.weight".to_string(), vec![config.vocab_size, h])?;

    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn tiny_config() -> OthelloGptConfig {
        // 2 layers, 2 heads, hidden 8 — small enough to run instantly.
        OthelloGptConfig::new(12, 6, 2, 2, 8, false).unwrap()
    }

    #[test]
    fn config_derives_head_dim() {
        let cfg = OthelloGptConfig::new(62, 60, 8, 8, 512, false).unwrap();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.mlp_ratio, 4);
        assert!(!cfg.causal);
    }

    #[test]
    fn config_rejects_indivisible_head_count() {
        assert!(OthelloGptConfig::new(62, 60, 8, 7, 512, false).is_err());
    }

    #[test]
    fn config_parses_companion_json() {
        let json = serde_json::json!({
            "vocab_size": 62,
            "block_size": 60,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "dropout": 0.0,
            "causal": false
        });
        let cfg = OthelloGptConfig::from_hf_config(&json).unwrap();
        assert_eq!(cfg.vocab_size, 62);
        assert_eq!(cfg.block_size, 60);
        assert_eq!(cfg.n_layer, 8);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn config_missing_key_errors() {
        let json = serde_json::json!({ "vocab_size": 62 });
        assert!(OthelloGptConfig::from_hf_config(&json).is_err());
    }

    #[test]
    fn forward_runs_and_shapes_match() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.hidden_size(), 8);
        assert_eq!(model.vocab_size(), 12);
        assert_eq!(model.num_heads(), 2);

        let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let hooks = HookSpec::new();
        let cache = model.forward(&input, &hooks).unwrap();
        let (batch, seq, vocab) = cache.output().dims3().unwrap();
        assert_eq!((batch, seq, vocab), (1, 4, 12));
    }

    #[test]
    fn causal_mask_is_upper_triangular() {
        let device = Device::Cpu;
        let mask = causal_mask(3, &device, DType::F32).unwrap();
        assert_eq!(mask.dims4().unwrap(), (1, 1, 3, 3));
        let values: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        let neg = f32::NEG_INFINITY;
        // Row i forbids columns j > i (strictly-upper triangle is -inf).
        assert_eq!(values, vec![0.0, neg, neg, 0.0, 0.0, neg, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn causal_model_runs_and_shapes_match() {
        // The autoregressive Othello-GPT control loads via the same module with
        // `causal = true`, exercising the causal-mask path.
        let device = Device::Cpu;
        let cfg = OthelloGptConfig::new(12, 6, 2, 2, 8, true).unwrap();
        assert!(cfg.causal);
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let cache = model.forward(&input, &HookSpec::new()).unwrap();
        assert_eq!(cache.output().dims3().unwrap(), (1, 4, 12));
    }

    #[test]
    fn hooks_capture_standard_points() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
        let mut hooks = HookSpec::new();
        hooks
            .capture(HookPoint::Embed)
            .capture(HookPoint::ResidPost(0))
            .capture(HookPoint::ResidPost(1))
            .capture(HookPoint::AttnOut(0))
            .capture(HookPoint::MlpOut(1))
            .capture(HookPoint::FinalNorm);
        let cache = model.forward(&input, &hooks).unwrap();

        assert!(cache.get(&HookPoint::Embed).is_some());
        assert!(cache.get(&HookPoint::ResidPost(0)).is_some());
        assert!(cache.get(&HookPoint::ResidPost(1)).is_some());
        assert!(cache.get(&HookPoint::AttnOut(0)).is_some());
        assert!(cache.get(&HookPoint::MlpOut(1)).is_some());
        assert!(cache.get(&HookPoint::FinalNorm).is_some());

        // ResidPost has the residual-stream shape [batch, seq, hidden].
        let rp = cache.get(&HookPoint::ResidPost(0)).unwrap();
        assert_eq!(rp.dims3().unwrap(), (1, 3, 8));
    }

    #[test]
    fn intervention_add_propagates_downstream() {
        // The P4 pattern: add a steering vector at ResidPost(layer) and verify
        // it flows into the next block. We capture ResidPre(1) — the residual
        // entering block 1, i.e. block 0's output *after* the ResidPost(0)
        // intervention — and compare it to the un-steered baseline.
        let device = Device::Cpu;
        let cfg = tiny_config();
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();

        let mut base_hooks = HookSpec::new();
        base_hooks.capture(HookPoint::ResidPre(1));
        let base = model.forward(&input, &base_hooks).unwrap();
        let base_pre: Vec<f32> = base
            .get(&HookPoint::ResidPre(1))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let steer = Tensor::ones(8, DType::F32, &device).unwrap();
        let mut hooks = HookSpec::new();
        hooks.capture(HookPoint::ResidPre(1)).intervene(
            HookPoint::ResidPost(0),
            crate::hooks::Intervention::Add(steer),
        );
        let steered = model.forward(&input, &hooks).unwrap();
        let steered_pre: Vec<f32> = steered
            .get(&HookPoint::ResidPre(1))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // With zero weights the residual is 0 until the +1 steer at ResidPost(0),
        // so block 1 sees 1.0 everywhere while the baseline sees 0.0.
        for (b, s) in base_pre.iter().zip(steered_pre.iter()) {
            assert!((b - 0.0).abs() < 1e-6, "baseline ResidPre(1) should be 0");
            assert!((s - 1.0).abs() < 1e-6, "steered ResidPre(1) should be 1");
        }
    }
}
