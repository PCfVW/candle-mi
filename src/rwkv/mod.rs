// SPDX-License-Identifier: MIT OR Apache-2.0

//! RWKV gated-linear RNN backend.
//!
//! Covers RWKV-6 (Finch) and will cover RWKV-7 (Goose) linear RNN models.
//! Architecture differences are captured in [`RwkvConfig`] fields and
//! version-specific submodules.
//!
//! # Architecture
//!
//! ```text
//! Input → Embedding → [Pre-LN (layer 0 only)]
//!   → for each layer:
//!       → LN1 → TimeMix (recurrence) → residual add
//!       → LN2 → ChannelMix (FFN) → residual add
//!   → Final LN → LM Head → logits
//! ```

pub mod config;
pub(crate) mod norm;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::backend::MIBackend;
use crate::error::Result;
use crate::hooks::{HookCache, HookPoint, HookSpec};

use self::norm::LayerNorm;
pub use config::{RwkvConfig, RwkvLoraDims, RwkvVersion, SUPPORTED_RWKV_MODEL_TYPES};

// ---------------------------------------------------------------------------
// RwkvState
// ---------------------------------------------------------------------------

/// Recurrent state for all RWKV layers.
///
/// Each layer maintains three state tensors that carry information across
/// timesteps.  The state starts as `None` (zero-initialized on first use)
/// and is updated after each forward pass.
struct RwkvState {
    /// Previous hidden for attention token-shift: `[batch, hidden_size]` per layer.
    attn_x: Vec<Option<Tensor>>,
    /// Accumulated WKV state: `[batch, num_heads, head_dim, head_dim]` per layer.
    attn_kv: Vec<Option<Tensor>>,
    /// Previous hidden for FFN token-shift: `[batch, hidden_size]` per layer.
    ffn_x: Vec<Option<Tensor>>,
}

impl RwkvState {
    /// Create a fresh (empty) state for `n_layers` layers.
    fn new(n_layers: usize) -> Self {
        Self {
            attn_x: vec![None; n_layers],
            attn_kv: vec![None; n_layers],
            ffn_x: vec![None; n_layers],
        }
    }
}

// ---------------------------------------------------------------------------
// RwkvBlock
// ---------------------------------------------------------------------------

/// A single RWKV block: `LN1 → TimeMix → residual → LN2 → ChannelMix → residual`.
struct RwkvBlock {
    /// Pre-layer norm (only present on block 0).
    pre_ln: Option<LayerNorm>,
    /// Normalization before time-mix.
    ln1: LayerNorm,
    /// Normalization before channel-mix.
    ln2: LayerNorm,
    /// Time-mix (recurrence) sub-block.
    time_mix: TimeMixV6,
    /// Channel-mix (FFN) sub-block.
    channel_mix: ChannelMixV6,
}

impl RwkvBlock {
    /// Load a single RWKV block from weights.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::error::MIError::Model) if weight
    /// loading fails.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    fn load(config: &RwkvConfig, vb: VarBuilder<'_>, layer_id: usize) -> Result<Self> {
        let eps = config.norm_eps;
        let h = config.hidden_size;

        let pre_ln = if layer_id == 0 {
            Some(LayerNorm::load(h, eps, vb.pp("pre_ln"))?)
        } else {
            None
        };

        let ln1 = LayerNorm::load(h, eps, vb.pp("ln1"))?;
        let ln2 = LayerNorm::load(h, eps, vb.pp("ln2"))?;
        let time_mix = TimeMixV6::load(config, vb.pp("attention"))?;
        let channel_mix = ChannelMixV6::load(config, vb.pp("feed_forward"))?;

        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            time_mix,
            channel_mix,
        })
    }

    /// Forward pass for a single RWKV block.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - returns: `(hidden, new_attn_x, new_attn_kv, new_ffn_x)`
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        ffn_x_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Pre-LayerNorm (only first block)
        let hidden = if let Some(ref pre_ln) = self.pre_ln {
            pre_ln.forward(hidden)?
        } else {
            hidden.clone()
        };

        // Time-mix with residual
        let (attn_out, new_attn_x, new_attn_kv) =
            self.time_mix
                .forward(&self.ln1.forward(&hidden)?, attn_x_state, attn_kv_state)?;
        let hidden = (&hidden + attn_out)?;

        // Channel-mix with residual
        let (ffn_out, new_ffn_x) = self
            .channel_mix
            .forward(&self.ln2.forward(&hidden)?, ffn_x_state)?;
        let hidden = (&hidden + ffn_out)?;

        Ok((hidden, new_attn_x, new_attn_kv, new_ffn_x))
    }
}

// ---------------------------------------------------------------------------
// TimeMixV6 (RWKV-6 time-mix / attention)
// ---------------------------------------------------------------------------

/// RWKV-6 time-mix block: data-dependent token shift + WKV recurrence.
///
/// This is the core recurrence of RWKV-6 — the "attention" equivalent.
/// Unlike transformer attention, it uses a linear recurrent state that
/// is updated at each timestep via the WKV formula.
struct TimeMixV6 {
    // --- Data-dependent mixing parameters ---
    /// Base mixing coefficient for initial projection.
    time_maa_x: Tensor, // [1, 1, hidden_size]
    /// Mixing for time-decay path.
    time_maa_w: Tensor,
    /// Mixing for key path.
    time_maa_k: Tensor,
    /// Mixing for value path.
    time_maa_v: Tensor,
    /// Mixing for receptance path.
    time_maa_r: Tensor,
    /// Mixing for gate path.
    time_maa_g: Tensor,

    // --- Low-rank mixing projections ---
    /// First `LoRA` projection: `[hidden_size, time_mix_extra_dim * 5]`.
    time_maa_w1: Tensor,
    /// Second `LoRA` projection: `[5, time_mix_extra_dim, hidden_size]`.
    time_maa_w2: Tensor,

    // --- Time decay ---
    /// Learned time-decay bias: `[1, 1, attention_hidden_size]`.
    time_decay: Tensor,
    /// First decay `LoRA` projection: `[hidden_size, time_decay_extra_dim]`.
    time_decay_w1: Tensor,
    /// Second decay `LoRA` projection: `[time_decay_extra_dim, attention_hidden_size]`.
    time_decay_w2: Tensor,

    // --- Per-head current-position bonus ---
    /// Bonus weight for the current-position kv product: `[num_heads, head_dim]`.
    time_faaaa: Tensor,

    // --- Linear projections (no bias) ---
    /// Receptance projection.
    receptance: Linear,
    /// Key projection.
    key: Linear,
    /// Value projection.
    value: Linear,
    /// Gate projection.
    gate: Linear,
    /// Output projection.
    output: Linear,

    // --- GroupNorm parameters ---
    /// `GroupNorm` scale: `[attention_hidden_size]`.
    ln_x_weight: Tensor,
    /// `GroupNorm` bias: `[attention_hidden_size]`.
    ln_x_bias: Tensor,

    // --- Dimensions ---
    /// Number of heads.
    num_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// Epsilon for `GroupNorm`.
    group_norm_eps: f64,
    /// Low-rank dimension for time-mix (5-component projection).
    time_mix_extra_dim: usize,
}

impl TimeMixV6 {
    /// Load the RWKV-6 time-mix block from weights.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::error::MIError::Model) if weights
    /// cannot be loaded.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    fn load(config: &RwkvConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.hidden_size;
        let ah = config.num_heads * config.head_dim; // attention_hidden_size
        let nh = config.num_heads;
        let hs = config.head_dim;
        let mix_extra = config.lora_dims.time_mix_extra_dim;
        let decay_extra = config.lora_dims.time_decay_extra_dim;

        let time_maa_x = vb.get((1, 1, h), "time_maa_x")?;
        let time_maa_w = vb.get((1, 1, h), "time_maa_w")?;
        let time_maa_k = vb.get((1, 1, h), "time_maa_k")?;
        let time_maa_v = vb.get((1, 1, h), "time_maa_v")?;
        let time_maa_r = vb.get((1, 1, h), "time_maa_r")?;
        let time_maa_g = vb.get((1, 1, h), "time_maa_g")?;

        let time_maa_w1 = vb.get((h, mix_extra * 5), "time_maa_w1")?;
        let time_maa_w2 = vb.get((5, mix_extra, h), "time_maa_w2")?;

        let time_decay = vb.get((1, 1, ah), "time_decay")?;
        let time_decay_w1 = vb.get((h, decay_extra), "time_decay_w1")?;
        let time_decay_w2 = vb.get((decay_extra, ah), "time_decay_w2")?;

        let time_faaaa = vb.get((nh, hs), "time_faaaa")?;

        let receptance = candle_nn::linear_no_bias(h, ah, vb.pp("receptance"))?;
        let key = candle_nn::linear_no_bias(h, ah, vb.pp("key"))?;
        let value = candle_nn::linear_no_bias(h, ah, vb.pp("value"))?;
        let gate = candle_nn::linear_no_bias(h, ah, vb.pp("gate"))?;
        let output = candle_nn::linear_no_bias(ah, h, vb.pp("output"))?;

        let ln_x_weight = vb.get(ah, "ln_x.weight")?;
        let ln_x_bias = vb.get(ah, "ln_x.bias")?;

        Ok(Self {
            time_maa_x,
            time_maa_w,
            time_maa_k,
            time_maa_v,
            time_maa_r,
            time_maa_g,
            time_maa_w1,
            time_maa_w2,
            time_decay,
            time_decay_w1,
            time_decay_w2,
            time_faaaa,
            receptance,
            key,
            value,
            gate,
            output,
            ln_x_weight,
            ln_x_bias,
            num_heads: nh,
            head_dim: hs,
            group_norm_eps: config.group_norm_eps(),
            time_mix_extra_dim: mix_extra,
        })
    }

    /// Forward pass for the RWKV-6 time-mix block.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - `attn_x_state`: `[batch, hidden_size]` or `None`
    /// - `attn_kv_state`: `[batch, num_heads, head_dim, head_dim]` or `None`
    /// - returns: `(output, new_attn_x_state, new_attn_kv_state)`
    ///   - `output`: `[batch, seq, hidden_size]`
    ///   - `new_attn_x_state`: `[batch, hidden_size]`
    ///   - `new_attn_kv_state`: `[batch, num_heads, head_dim, head_dim]`
    #[allow(clippy::many_single_char_names)] // r, k, v are standard names in RWKV papers
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, channels) = hidden.dims3()?;
        let nh = self.num_heads;
        let hs = self.head_dim;

        // --- Token shift ---
        let shifted = token_shift(hidden, attn_x_state, batch, seq_len, channels)?;

        // Save last token as new attn_x_state
        let new_attn_x_state = hidden.i((.., seq_len - 1, ..))?;

        // --- Data-dependent mixing ---
        let xx = shifted.broadcast_sub(hidden)?; // [batch, seq_len, channels]

        // Step 1: Base mixing coefficient
        let xxx = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_x)?)?;

        // Step 2: Project through w1 → tanh → reshape for 5 components
        let bt = batch * seq_len;
        let xxx_flat = xxx.reshape((bt, channels))?;
        let projected = xxx_flat.matmul(&self.time_maa_w1)?; // [bt, mix_extra*5]
        let projected = projected.tanh()?;
        let projected = projected.reshape((bt, 5, self.time_mix_extra_dim))?;
        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let projected = projected.transpose(0, 1)?.contiguous()?; // [5, bt, mix_extra]

        // Step 3: Back-project through w2
        let mixed = projected.matmul(&self.time_maa_w2)?; // [5, bt, channels]
        let mixed = mixed.reshape((5, batch, seq_len, channels))?;

        let mw = mixed.i(0)?; // [batch, seq_len, channels]
        let mk = mixed.i(1)?;
        let mv = mixed.i(2)?;
        let mr = mixed.i(3)?;
        let mg = mixed.i(4)?;

        // Step 4: Apply mixing to produce inputs for each projection
        let time_decay_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_w.broadcast_add(&mw)?)?)?;
        let key_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_k.broadcast_add(&mk)?)?)?;
        let value_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_v.broadcast_add(&mv)?)?)?;
        let receptance_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_r.broadcast_add(&mr)?)?)?;
        let gate_input =
            hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_g.broadcast_add(&mg)?)?)?;

        // --- Project to R, K, V, gate ---
        let rec = self.receptance.forward(&receptance_input)?; // [batch, seq_len, ah]
        let key = self.key.forward(&key_input)?;
        let val = self.value.forward(&value_input)?;
        let gate_val = candle_nn::ops::silu(&self.gate.forward(&gate_input)?)?;

        // --- Data-dependent time decay ---
        let td_flat = time_decay_input.reshape((bt, channels))?;
        let td_proj = td_flat.matmul(&self.time_decay_w1)?.tanh()?;
        let td_proj = td_proj.matmul(&self.time_decay_w2)?; // [bt, ah]
        let td_proj = td_proj.reshape((batch, seq_len, nh * hs))?;
        let decay_raw = self.time_decay.broadcast_add(&td_proj)?; // [batch, seq_len, ah]

        // PROMOTE: WKV recurrence must be in F32 for numerical stability
        // decay = exp(-exp(w))
        let decay = decay_raw.to_dtype(DType::F32)?.exp()?.neg()?.exp()?;

        // --- Reshape for per-head computation ---
        // rec, key, val: [batch, seq_len, ah] → [batch, seq_len, nh, hs]
        let rec = rec
            .to_dtype(DType::F32)?
            .reshape((batch, seq_len, nh, hs))?;
        let key = key
            .to_dtype(DType::F32)?
            .reshape((batch, seq_len, nh, hs))?;
        let val = val
            .to_dtype(DType::F32)?
            .reshape((batch, seq_len, nh, hs))?;
        let decay = decay.reshape((batch, seq_len, nh, hs))?;

        // time_faaaa: [nh, hs] → used as current-position bonus
        let time_first = self.time_faaaa.to_dtype(DType::F32)?;
        let time_first = time_first.reshape((1, 1, nh, hs))?;

        // --- WKV Recurrence Loop ---
        // EXPLICIT: WKV recurrence is stateful; .map() would hide the state update
        let mut state = match attn_kv_state {
            Some(prev) => prev.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch, nh, hs, hs), DType::F32, hidden.device())?,
        };

        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for ti in 0..seq_len {
            // Current timestep values: [batch, nh, hs]
            let r_t = rec.i((.., ti, .., ..))?;
            let k_t = key.i((.., ti, .., ..))?;
            let v_t = val.i((.., ti, .., ..))?;
            let decay_t = decay.i((.., ti, .., ..))?;
            let time_first_t = time_first.i((.., 0, .., ..))?; // [1, nh, hs]

            // kv = k_t^T @ v_t: outer product [batch, nh, hs, 1] x [batch, nh, 1, hs]
            let k_col = k_t.unsqueeze(candle_core::D::Minus1)?;
            let v_row = v_t.unsqueeze(2)?;
            let kv = k_col.matmul(&v_row)?; // [batch, nh, hs, hs]

            // out_t = r_t @ (time_first * kv + state)
            let time_first_expanded = time_first_t.unsqueeze(candle_core::D::Minus1)?;
            let weighted_kv = kv.broadcast_mul(&time_first_expanded)?;
            let combined = (&weighted_kv + &state)?;

            let r_row = r_t.unsqueeze(2)?; // [batch, nh, 1, hs]
            let out_t = r_row.matmul(&combined)?; // [batch, nh, 1, hs]
            let out_t = out_t.squeeze(2)?; // [batch, nh, hs]

            outputs.push(out_t);

            // State update: state = kv + decay * state
            let decay_expanded = decay_t.unsqueeze(candle_core::D::Minus1)?;
            state = (kv + state.broadcast_mul(&decay_expanded)?)?;
        }

        // Stack outputs: [batch, seq_len, nh, hs]
        let out = Tensor::stack(&outputs, 1)?;
        let new_attn_kv_state = state; // [batch, nh, hs, hs]

        // --- GroupNorm per head ---
        let out = out.reshape((bt, nh * hs))?;
        // PROMOTE: GroupNorm weights to F32 for consistency with F32 WKV output
        let out = norm::group_norm(
            &out,
            self.num_heads,
            &self.ln_x_weight.to_dtype(DType::F32)?,
            &self.ln_x_bias.to_dtype(DType::F32)?,
            self.group_norm_eps,
        )?;
        let out = out
            .reshape((batch, seq_len, nh * hs))?
            .to_dtype(hidden.dtype())?;

        // --- Apply gate and output projection ---
        let out = (out * gate_val)?;
        let out = self.output.forward(&out)?;

        Ok((out, new_attn_x_state, new_attn_kv_state))
    }
}

// ---------------------------------------------------------------------------
// ChannelMixV6 (RWKV-6 channel-mix / FFN)
// ---------------------------------------------------------------------------

/// RWKV-6 channel-mix block: receptance-gated squared-ReLU FFN.
///
/// Structure: `sigmoid(receptance(x_r)) * value(relu(key(x_k))^2)`
struct ChannelMixV6 {
    /// Token-shift mixing parameter for key path: `[1, 1, hidden_size]`.
    time_maa_k: Tensor,
    /// Token-shift mixing parameter for receptance path: `[1, 1, hidden_size]`.
    time_maa_r: Tensor,
    /// Key projection: `hidden → intermediate` (no bias).
    key: Linear,
    /// Receptance projection: `hidden → hidden` (no bias).
    receptance: Linear,
    /// Value projection: `intermediate → hidden` (no bias).
    value: Linear,
}

impl ChannelMixV6 {
    /// Load the RWKV-6 channel-mix block from weights.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::error::MIError::Model) if weights
    /// cannot be loaded.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    fn load(config: &RwkvConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.hidden_size;
        let intermediate = config.intermediate_size;

        let time_maa_k = vb.get((1, 1, h), "time_maa_k")?;
        let time_maa_r = vb.get((1, 1, h), "time_maa_r")?;

        let key = candle_nn::linear_no_bias(h, intermediate, vb.pp("key"))?;
        let receptance = candle_nn::linear_no_bias(h, h, vb.pp("receptance"))?;
        let value = candle_nn::linear_no_bias(intermediate, h, vb.pp("value"))?;

        Ok(Self {
            time_maa_k,
            time_maa_r,
            key,
            receptance,
            value,
        })
    }

    /// Forward pass for the channel-mix (FFN) block.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - `ffn_x_state`: `[batch, hidden_size]` or `None`
    /// - returns: `(output, new_ffn_x_state)`
    ///   - `output`: `[batch, seq, hidden_size]`
    ///   - `new_ffn_x_state`: `[batch, hidden_size]`
    fn forward(&self, hidden: &Tensor, ffn_x_state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, channels) = hidden.dims3()?;

        // --- Token shift ---
        let shifted = token_shift(hidden, ffn_x_state, batch, seq_len, channels)?;

        // Save last token as new ffn_x_state
        let new_ffn_x_state = hidden.i((.., seq_len - 1, ..))?;

        // --- Mixing ---
        let xx = shifted.broadcast_sub(hidden)?;
        let key_input = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_k)?)?;
        let rec_input = hidden.broadcast_add(&xx.broadcast_mul(&self.time_maa_r)?)?;

        // key = relu(key_proj(key_input))^2 (squared ReLU)
        let key_out = self.key.forward(&key_input)?.relu()?.sqr()?;

        // value = value_proj(key_out) — note: "key" output feeds into "value" projection
        let val_out = self.value.forward(&key_out)?;

        // receptance = sigmoid(receptance_proj(rec_input))
        let rec_gate = candle_nn::ops::sigmoid(&self.receptance.forward(&rec_input)?)?;

        // Output: receptance * value
        let out = (rec_gate * val_out)?;

        Ok((out, new_ffn_x_state))
    }
}

// ---------------------------------------------------------------------------
// Token shift helper
// ---------------------------------------------------------------------------

/// Compute the token-shifted input for RWKV blocks.
///
/// For multi-token inputs, shifts by one position (zero-padded or state-filled).
/// For single-token inputs, uses the previous state directly.
///
/// # Shapes
/// - `hidden`: `[batch, seq, hidden_size]`
/// - `state`: `[batch, hidden_size]` or `None` (zero on first call)
/// - returns: `[batch, seq, hidden_size]` (shifted version of `hidden`)
fn token_shift(
    hidden: &Tensor,
    state: Option<&Tensor>,
    batch: usize,
    seq_len: usize,
    channels: usize,
) -> Result<Tensor> {
    if seq_len == 1 {
        // Single token: use state directly
        match state {
            Some(prev) => Ok(prev.unsqueeze(1)?),
            None => Ok(Tensor::zeros(
                (batch, 1, channels),
                hidden.dtype(),
                hidden.device(),
            )?),
        }
    } else {
        // Multi-token: shift by padding top, cropping bottom
        let zeros = Tensor::zeros((batch, 1, channels), hidden.dtype(), hidden.device())?;
        let prev_tokens = hidden.i((.., ..seq_len - 1, ..))?;
        let shifted = Tensor::cat(&[&zeros, &prev_tokens], 1)?;

        // If we have state, replace the first token's shift
        if let Some(prev) = state {
            let state_expanded = prev.unsqueeze(1)?;
            let rest = shifted.i((.., 1.., ..))?;
            Ok(Tensor::cat(&[&state_expanded, &rest], 1)?)
        } else {
            Ok(shifted)
        }
    }
}

// ---------------------------------------------------------------------------
// GenericRwkv
// ---------------------------------------------------------------------------

/// Config-driven generic RWKV backend.
///
/// Implements the RWKV gated-linear RNN architecture with hook points
/// for mechanistic interpretability.
pub struct GenericRwkv {
    /// Token embedding matrix.
    embeddings: Embedding,
    /// RWKV blocks (layers).
    blocks: Vec<RwkvBlock>,
    /// Final normalization before LM head.
    ln_out: LayerNorm,
    /// LM head (vocabulary projection).
    lm_head: Linear,
    /// Model configuration.
    config: RwkvConfig,
}

impl GenericRwkv {
    /// Load a generic RWKV model from a [`VarBuilder`].
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::error::MIError::Model) if weight
    /// loading fails or dimensions are inconsistent.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    pub fn load(
        config: RwkvConfig,
        _device: &Device,
        _dtype: DType,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        // RWKV-6 weight prefix: "rwkv.*"
        let vb_rwkv = vb.pp("rwkv");

        // --- Embedding ---
        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb_rwkv.pp("embeddings"),
        )?;

        // --- Blocks ---
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = RwkvBlock::load(&config, vb_rwkv.pp(format!("blocks.{i}")), i)?;
            blocks.push(block);
        }

        // --- Final norm ---
        let ln_out = LayerNorm::load(config.hidden_size, config.norm_eps, vb_rwkv.pp("ln_out"))?;

        // --- LM head ---
        let lm_head = if config.tie_word_embeddings {
            let head_weight = embeddings.embeddings().clone();
            Linear::new(head_weight, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("head"))?
        };

        Ok(Self {
            embeddings,
            blocks,
            ln_out,
            lm_head,
            config,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &RwkvConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// MIBackend implementation
// ---------------------------------------------------------------------------

impl MIBackend for GenericRwkv {
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
        self.config.num_heads
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();

        // --- Embedding ---
        let mut hidden = self.embeddings.forward(input_ids)?;

        // Capture cache — collects hook captures; output set at the end.
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        // Hook: Embed
        if hooks.is_captured(&HookPoint::Embed) {
            cache.store(HookPoint::Embed, hidden.clone());
        }
        for intervention in hooks.interventions_at(&HookPoint::Embed) {
            hidden = crate::hooks::apply_intervention(&hidden, intervention)?;
        }

        // --- Layer loop ---
        let mut state = RwkvState::new(self.config.num_layers);

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // Hook: ResidPre
            if hooks.is_captured(&HookPoint::ResidPre(layer_idx)) {
                cache.store(HookPoint::ResidPre(layer_idx), hidden.clone());
            }

            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x) = block.forward(
                &hidden,
                state.attn_x.get(layer_idx).and_then(Option::as_ref),
                state.attn_kv.get(layer_idx).and_then(Option::as_ref),
                state.ffn_x.get(layer_idx).and_then(Option::as_ref),
            )?;

            hidden = new_hidden;

            // Hook: RwkvState — capture the WKV state after update
            if hooks.is_captured(&HookPoint::RwkvState(layer_idx)) {
                cache.store(HookPoint::RwkvState(layer_idx), new_attn_kv.clone());
            }

            // Store updated state
            if let Some(slot) = state.attn_x.get_mut(layer_idx) {
                *slot = Some(new_attn_x);
            }
            if let Some(slot) = state.attn_kv.get_mut(layer_idx) {
                *slot = Some(new_attn_kv);
            }
            if let Some(slot) = state.ffn_x.get_mut(layer_idx) {
                *slot = Some(new_ffn_x);
            }

            // Hook: ResidPost
            if hooks.is_captured(&HookPoint::ResidPost(layer_idx)) {
                cache.store(HookPoint::ResidPost(layer_idx), hidden.clone());
            }
        }

        // --- Final norm ---
        hidden = self.ln_out.forward(&hidden)?;

        // Hook: FinalNorm
        if hooks.is_captured(&HookPoint::FinalNorm) {
            cache.store(HookPoint::FinalNorm, hidden.clone());
        }

        // --- LM head ---
        let logits = self.lm_head.forward(&hidden)?;

        cache.set_output(logits);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.lm_head.forward(hidden)?)
    }
}
