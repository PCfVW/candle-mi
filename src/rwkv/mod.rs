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

/// Version-dispatched time-mix block.
enum TimeMix {
    /// RWKV-6 Finch time-mix.
    V6(TimeMixV6),
    /// RWKV-7 Goose time-mix.
    V7(TimeMixV7),
}

/// Version-dispatched channel-mix block.
enum ChannelMix {
    /// RWKV-6 Finch channel-mix.
    V6(ChannelMixV6),
    /// RWKV-7 Goose channel-mix.
    V7(ChannelMixV7),
}

/// A single RWKV block: `LN1 → TimeMix → residual → LN2 → ChannelMix → residual`.
struct RwkvBlock {
    /// Pre-layer norm (only present on block 0).
    pre_ln: Option<LayerNorm>,
    /// Normalization before time-mix.
    ln1: LayerNorm,
    /// Normalization before channel-mix.
    ln2: LayerNorm,
    /// Time-mix (recurrence) sub-block.
    time_mix: TimeMix,
    /// Channel-mix (FFN) sub-block.
    channel_mix: ChannelMix,
}

impl RwkvBlock {
    /// Load a single RWKV-6 block from weights.
    ///
    /// Weight prefix: `rwkv.blocks.{i}`.
    #[allow(clippy::needless_pass_by_value)]
    fn load_v6(config: &RwkvConfig, vb: VarBuilder<'_>, layer_id: usize) -> Result<Self> {
        let eps = config.norm_eps;
        let h = config.hidden_size;

        let pre_ln = if layer_id == 0 {
            Some(LayerNorm::load(h, eps, vb.pp("pre_ln"))?)
        } else {
            None
        };

        let ln1 = LayerNorm::load(h, eps, vb.pp("ln1"))?;
        let ln2 = LayerNorm::load(h, eps, vb.pp("ln2"))?;
        let time_mix = TimeMix::V6(TimeMixV6::load(config, vb.pp("attention"))?);
        let channel_mix = ChannelMix::V6(ChannelMixV6::load(config, vb.pp("feed_forward"))?);

        Ok(Self {
            pre_ln,
            ln1,
            ln2,
            time_mix,
            channel_mix,
        })
    }

    /// Load a single RWKV-7 block from weights.
    ///
    /// Weight prefix: `model.layers.{i}`.
    #[allow(clippy::needless_pass_by_value)]
    fn load_v7(config: &RwkvConfig, vb: VarBuilder<'_>, layer_id: usize) -> Result<Self> {
        let eps = config.norm_eps;
        let h = config.hidden_size;

        let pre_ln = if layer_id == 0 {
            Some(LayerNorm::load(h, eps, vb.pp("pre_norm"))?)
        } else {
            None
        };

        let ln1 = LayerNorm::load(h, eps, vb.pp("attn_norm"))?;
        let ln2 = LayerNorm::load(h, eps, vb.pp("ffn_norm"))?;
        let time_mix = TimeMix::V7(TimeMixV7::load(config, vb.pp("attn"), layer_id)?);
        let channel_mix = ChannelMix::V7(ChannelMixV7::load(config, vb.pp("ffn"))?);

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
    /// - `v_first`: `[batch, seq, hidden_size]` or `None` (V7 value residual)
    /// - returns: `(hidden, new_attn_x, new_attn_kv, new_ffn_x, v_out, decay)`
    ///   - `hidden`: `[batch, seq, hidden_size]`
    ///   - `new_attn_x`: `[batch, hidden_size]`
    ///   - `new_attn_kv`: `[batch, num_heads, head_dim, head_dim]`
    ///   - `new_ffn_x`: `[batch, hidden_size]`
    ///   - `v_out`: `[batch, seq, hidden_size]` (V7: raw v or mixed v; V6: dummy)
    ///   - `decay`: `[batch, seq, num_heads, head_dim]`
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        ffn_x_state: Option<&Tensor>,
        v_first: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        // Pre-LayerNorm (only first block)
        let hidden = if let Some(ref pre_ln) = self.pre_ln {
            pre_ln.forward(hidden)?
        } else {
            hidden.clone()
        };

        // Time-mix with residual
        let (attn_out, new_attn_x, new_attn_kv, v_out, decay) = match &self.time_mix {
            TimeMix::V6(tm) => {
                let (out, ax, akv, d) =
                    tm.forward(&self.ln1.forward(&hidden)?, attn_x_state, attn_kv_state)?;
                // V6 has no v_first concept; return a zero-sized dummy
                let dummy_v = Tensor::zeros(1, DType::F32, hidden.device())?;
                (out, ax, akv, dummy_v, d)
            }
            TimeMix::V7(tm) => tm.forward(
                &self.ln1.forward(&hidden)?,
                attn_x_state,
                attn_kv_state,
                v_first,
            )?,
        };
        let hidden = (&hidden + attn_out)?;

        // Channel-mix with residual
        let (ffn_out, new_ffn_x) = match &self.channel_mix {
            ChannelMix::V6(cm) => cm.forward(&self.ln2.forward(&hidden)?, ffn_x_state)?,
            ChannelMix::V7(cm) => cm.forward(&self.ln2.forward(&hidden)?, ffn_x_state)?,
        };
        let hidden = (&hidden + ffn_out)?;

        Ok((hidden, new_attn_x, new_attn_kv, new_ffn_x, v_out, decay))
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
    /// - returns: `(output, new_attn_x_state, new_attn_kv_state, decay)`
    ///   - `output`: `[batch, seq, hidden_size]`
    ///   - `new_attn_x_state`: `[batch, hidden_size]`
    ///   - `new_attn_kv_state`: `[batch, num_heads, head_dim, head_dim]`
    ///   - `decay`: `[batch, seq, num_heads, head_dim]` (per-timestep decay)
    #[allow(clippy::many_single_char_names)] // r, k, v are standard names in RWKV papers
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
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

        Ok((out, new_attn_x_state, new_attn_kv_state, decay))
    }
}

// ---------------------------------------------------------------------------
// LoraBlock (shared helper for RWKV-7 LoRA projections)
// ---------------------------------------------------------------------------

/// A low-rank projection pair: `down → activation → up`.
///
/// Used throughout RWKV-7 for data-dependent decay, rank-1 gate,
/// output gate, and value-residual projections.
struct LoraBlock {
    /// Down-projection: `[low_rank, input_dim]` (no bias).
    down: Linear,
    /// Up-projection: `[output_dim, low_rank]` (may have bias).
    up: Linear,
}

impl LoraBlock {
    /// Load a `LoRA` block from weights.
    ///
    /// Expects `vb` scoped to `{name}_lora.lora`:
    /// - `0.weight` — down (no bias)
    /// - `2.weight` / `2.bias` — up (bias optional via `has_up_bias`)
    #[allow(clippy::needless_pass_by_value)]
    fn load(
        input_dim: usize,
        low_rank: usize,
        output_dim: usize,
        has_up_bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let down = candle_nn::linear_no_bias(input_dim, low_rank, vb.pp("0"))?;
        let up = if has_up_bias {
            candle_nn::linear(low_rank, output_dim, vb.pp("2"))?
        } else {
            candle_nn::linear_no_bias(low_rank, output_dim, vb.pp("2"))?
        };
        Ok(Self { down, up })
    }

    /// Forward pass with tanh activation between down and up.
    ///
    /// `down(x).tanh() → up`
    fn forward_tanh(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.up.forward(&self.down.forward(x)?.tanh()?)?)
    }

    /// Forward pass with sigmoid activation between down and up.
    ///
    /// `down(x).sigmoid() → up`
    fn forward_sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self
            .up
            .forward(&candle_nn::ops::sigmoid(&self.down.forward(x)?)?)?)
    }

    /// Forward pass without intermediate activation.
    ///
    /// `down(x) → up`
    fn forward_linear(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.up.forward(&self.down.forward(x)?)?)
    }
}

// ---------------------------------------------------------------------------
// TimeMixV7 (RWKV-7 time-mix / attention)
// ---------------------------------------------------------------------------

/// RWKV-7 time-mix block: static lerp token shift + WKV-7 recurrence.
///
/// The WKV-7 recurrence uses a generalized delta rule with both diagonal
/// decay and rank-1 state transition:
///
/// ```text
/// S_t = diag(exp(w_t)) * S_{t-1} + b_t^T @ (a_t @ S_{t-1}) + k_t^T @ v_t
/// y_t = r_t @ S_t
/// ```
struct TimeMixV7 {
    // --- Static lerp mixing parameters: [1, 1, hidden_size] ---
    /// Receptance mixing: `[1, 1, hidden_size]`.
    x_r: Tensor,
    /// Decay mixing: `[1, 1, hidden_size]`.
    x_w: Tensor,
    /// Key mixing: `[1, 1, hidden_size]`.
    x_k: Tensor,
    /// Value mixing: `[1, 1, hidden_size]`.
    x_v: Tensor,
    /// Rank-1 gate mixing: `[1, 1, hidden_size]`.
    x_a: Tensor,
    /// Output gate mixing: `[1, 1, hidden_size]`.
    x_g: Tensor,

    // --- Key modification parameters ---
    /// Key normalization scale: `[hidden_size]`.
    k_k: Tensor,
    /// Key modification scale: `[hidden_size]`.
    k_a: Tensor,
    /// Output correction scale: `[num_heads, head_dim]`.
    r_k: Tensor,

    // --- Linear projections (no bias) ---
    /// Receptance projection: `hidden → hidden`.
    r_proj: Linear,
    /// Key projection: `hidden → hidden`.
    k_proj: Linear,
    /// Value projection: `hidden → hidden`.
    v_proj: Linear,
    /// Output projection: `hidden → hidden`.
    o_proj: Linear,

    // --- LoRA blocks ---
    /// Decay `LoRA` (tanh activation): `[decay_low_rank_dim, hidden] → [hidden, decay_low_rank_dim]`.
    w_lora: LoraBlock,
    /// Rank-1 gate `LoRA` (linear, sigmoid applied externally).
    a_lora: LoraBlock,
    /// Output gate `LoRA` (linear, sigmoid applied externally, no up bias).
    g_lora: LoraBlock,
    /// Value residual `LoRA` (linear, sigmoid applied externally). `None` for layer 0.
    v_lora: Option<LoraBlock>,

    // --- GroupNorm parameters ---
    /// `GroupNorm` weight: `[hidden_size]`.
    g_norm_weight: Tensor,
    /// `GroupNorm` bias: `[hidden_size]`.
    g_norm_bias: Tensor,

    // --- Dimensions ---
    /// Number of attention heads.
    num_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// Layer norm epsilon (used for `GroupNorm` eps = `head_dim * norm_eps`).
    norm_eps: f64,
    /// Layer index (0-based; layer 0 sets `v_first`).
    layer_idx: usize,
}

impl TimeMixV7 {
    /// Load the RWKV-7 time-mix block from weights.
    ///
    /// `vb` should be scoped to `model.layers.{i}.attn`.
    #[allow(clippy::needless_pass_by_value)]
    fn load(config: &RwkvConfig, vb: VarBuilder<'_>, layer_idx: usize) -> Result<Self> {
        let h = config.hidden_size;
        let nh = config.num_heads;
        let hd = config.head_dim;
        let lora = &config.lora_dims;

        // Static lerp parameters
        let x_r = vb.get((1, 1, h), "x_r")?;
        let x_w = vb.get((1, 1, h), "x_w")?;
        let x_k = vb.get((1, 1, h), "x_k")?;
        let x_v = vb.get((1, 1, h), "x_v")?;
        let x_a = vb.get((1, 1, h), "x_a")?;
        let x_g = vb.get((1, 1, h), "x_g")?;

        // Key modification
        let k_k = vb.get(h, "k_k")?;
        let k_a = vb.get(h, "k_a")?;
        let r_k = vb.get((nh, hd), "r_k")?;

        // Linear projections
        let r_proj = candle_nn::linear_no_bias(h, h, vb.pp("r_proj"))?;
        let k_proj = candle_nn::linear_no_bias(h, h, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(h, h, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(h, h, vb.pp("o_proj"))?;

        // LoRA blocks
        let w_lora = LoraBlock::load(h, lora.decay_low_rank_dim, h, true, vb.pp("w_lora.lora"))?;
        let a_lora = LoraBlock::load(h, lora.a_low_rank_dim, h, true, vb.pp("a_lora.lora"))?;
        let g_lora = LoraBlock::load(h, lora.gate_low_rank_dim, h, false, vb.pp("g_lora.lora"))?;

        let v_lora = if layer_idx > 0 {
            Some(LoraBlock::load(
                h,
                lora.v_low_rank_dim,
                h,
                true,
                vb.pp("v_lora.lora"),
            )?)
        } else {
            None
        };

        // GroupNorm
        let g_norm_weight = vb.get(h, "g_norm.weight")?;
        let g_norm_bias = vb.get(h, "g_norm.bias")?;

        Ok(Self {
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            k_k,
            k_a,
            r_k,
            r_proj,
            k_proj,
            v_proj,
            o_proj,
            w_lora,
            a_lora,
            g_lora,
            v_lora,
            g_norm_weight,
            g_norm_bias,
            num_heads: nh,
            head_dim: hd,
            norm_eps: config.norm_eps,
            layer_idx,
        })
    }

    /// Forward pass for the RWKV-7 time-mix block.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - `attn_x_state`: `[batch, hidden_size]` or `None`
    /// - `attn_kv_state`: `[batch, num_heads, head_dim, head_dim]` or `None`
    /// - `v_first`: `[batch, seq, hidden_size]` or `None` (set by layer 0)
    /// - returns: `(output, new_attn_x, new_attn_kv, v_out, decay)`
    ///   - `output`: `[batch, seq, hidden_size]`
    ///   - `new_attn_x`: `[batch, hidden_size]`
    ///   - `new_attn_kv`: `[batch, num_heads, head_dim, head_dim]`
    ///   - `v_out`: `[batch, seq, hidden_size]` (raw v from layer 0, or mixed v)
    ///   - `decay`: `[batch, seq, num_heads, head_dim]`
    #[allow(
        clippy::many_single_char_names,
        clippy::too_many_lines,
        clippy::similar_names,
        clippy::cast_precision_loss,
        clippy::as_conversions
    )]
    fn forward(
        &self,
        hidden: &Tensor,
        attn_x_state: Option<&Tensor>,
        attn_kv_state: Option<&Tensor>,
        v_first: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _channels) = hidden.dims3()?;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let h = nh * hd;
        let bt = batch * seq_len;

        // --- Token shift: static lerp ---
        let shifted = token_shift(hidden, attn_x_state, batch, seq_len, h)?;
        let new_attn_x = hidden.i((.., seq_len - 1, ..))?;
        let delta = shifted.broadcast_sub(hidden)?;

        // Static lerp for each component: x + delta * x_param
        let xr = hidden.broadcast_add(&delta.broadcast_mul(&self.x_r)?)?;
        let xw = hidden.broadcast_add(&delta.broadcast_mul(&self.x_w)?)?;
        let xk = hidden.broadcast_add(&delta.broadcast_mul(&self.x_k)?)?;
        let xv = hidden.broadcast_add(&delta.broadcast_mul(&self.x_v)?)?;
        let xa = hidden.broadcast_add(&delta.broadcast_mul(&self.x_a)?)?;
        let xg = hidden.broadcast_add(&delta.broadcast_mul(&self.x_g)?)?;

        // --- Project R, K, V ---
        let r = self.r_proj.forward(&xr)?; // [batch, seq, h]
        let k = self.k_proj.forward(&xk)?; // [batch, seq, h]
        let v = self.v_proj.forward(&xv)?; // [batch, seq, h]

        // --- Decay: w = -0.6065306597126334 * sigmoid(w_lora(xw)) ---
        let w_lora_out = self.w_lora.forward_tanh(&xw)?; // [batch, seq, h]
        let w = (candle_nn::ops::sigmoid(&w_lora_out.to_dtype(DType::F32)?)?
            * (-0.606_530_659_712_633_4_f64))?; // [batch, seq, h]

        // --- Value residual (layers > 0) ---
        let v_out = if let Some(v_lora) = &self.v_lora {
            // v = lerp(v, v_first, sigmoid(v_lora(xv)))
            let v_lora_out = v_lora.forward_linear(&xv)?;
            let mix = candle_nn::ops::sigmoid(&v_lora_out)?;
            let v_first_t = v_first.ok_or_else(|| {
                crate::error::MIError::Config("v_first required for layer > 0".into())
            })?;
            // lerp(v, v_first, mix) = v + (v_first - v) * mix
            (&v + &(&(v_first_t - &v)? * mix)?)?
        } else {
            // Layer 0: v_out IS v (will become v_first for subsequent layers)
            v.clone()
        };

        // --- Rank-1 gate: a = sigmoid(a_lora(xa)) ---
        let a = candle_nn::ops::sigmoid(&self.a_lora.forward_linear(&xa)?)?; // [batch, seq, h]

        // --- Output gate: g = g_lora(xg) with sigmoid middle activation ---
        // NOTE: sigmoid is applied BETWEEN down and up projections (inside the LoRA),
        // NOT after the full LoRA output. This matches fla's LoRA(activation='sigmoid').
        let g = self.g_lora.forward_sigmoid(&xg)?; // [batch, seq, h]

        // --- Key normalization: kk = l2_norm(k * k_k) ---
        // k_k is [h], broadcast over [batch, seq, h]
        let k_scaled = k.broadcast_mul(&self.k_k)?; // [batch, seq, h]
        let k_scaled_4d = k_scaled.reshape((batch, seq_len, nh, hd))?;
        let kk = l2_norm(&k_scaled_4d)?; // [batch, seq, nh, hd]

        // --- Key modification: k = k + k * (a - 1) * k_a ---
        // k_a is [h], a is [batch, seq, h]
        let a_minus_1 = (a.clone() - 1.0_f64)?;
        let k_mod = (&k + &(&k * &a_minus_1)?.broadcast_mul(&self.k_a)?)?; // [batch, seq, h]

        // --- Reshape for WKV recurrence ---
        let r_4d = r.to_dtype(DType::F32)?.reshape((batch, seq_len, nh, hd))?;
        let k_4d = k_mod
            .to_dtype(DType::F32)?
            .reshape((batch, seq_len, nh, hd))?;
        let v_4d = v_out
            .to_dtype(DType::F32)?
            .reshape((batch, seq_len, nh, hd))?;
        let w_4d = w.reshape((batch, seq_len, nh, hd))?; // already F32
        let kk_f32 = kk.to_dtype(DType::F32)?;
        let a_4d = a.to_dtype(DType::F32)?.reshape((batch, seq_len, nh, hd))?;

        // --- WKV-7 Recurrence Loop (F32) ---
        let mut state = match attn_kv_state {
            Some(prev) => prev.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch, nh, hd, hd), DType::F32, hidden.device())?,
        };

        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq_len);

        for ti in 0..seq_len {
            let r_t = r_4d.i((.., ti, .., ..))?; // [batch, nh, hd]
            let k_t = k_4d.i((.., ti, .., ..))?; // [batch, nh, hd]
            let v_t = v_4d.i((.., ti, .., ..))?; // [batch, nh, hd]
            let w_t = w_4d.i((.., ti, .., ..))?; // [batch, nh, hd]
            let kk_t = kk_f32.i((.., ti, .., ..))?; // [batch, nh, hd]
            let a_t = a_4d.i((.., ti, .., ..))?; // [batch, nh, hd]

            // act_a = -kk_t: [batch, nh, hd]
            let act_a = kk_t.neg()?;
            // b = kk_t * a_t: [batch, nh, hd]
            let b_t = (&kk_t * &a_t)?;

            // State transition:
            // S_t = diag(exp(w_t)) * S_{t-1} + b_t^T @ (act_a @ S_{t-1}) + k_t^T @ v_t
            //
            // Term 1: diag(exp(w_t)) * S_{t-1}
            let exp_w = w_t.exp()?; // [batch, nh, hd]
            let exp_w_col = exp_w.unsqueeze(candle_core::D::Minus1)?; // [batch, nh, hd, 1]
            let term1 = state.broadcast_mul(&exp_w_col)?; // [batch, nh, hd, hd]

            // Term 2: b_t^T @ (act_a @ S_{t-1})
            // act_a @ S: [batch, nh, 1, hd] @ [batch, nh, hd, hd] = [batch, nh, 1, hd]
            let act_a_row = act_a.unsqueeze(2)?; // [batch, nh, 1, hd]
            let a_times_s = act_a_row.matmul(&state)?; // [batch, nh, 1, hd]
            // b_t^T @ result: [batch, nh, hd, 1] @ [batch, nh, 1, hd] = [batch, nh, hd, hd]
            let b_col = b_t.unsqueeze(candle_core::D::Minus1)?; // [batch, nh, hd, 1]
            let term2 = b_col.matmul(&a_times_s)?; // [batch, nh, hd, hd]

            // Term 3: k_t^T @ v_t
            let k_col = k_t.unsqueeze(candle_core::D::Minus1)?; // [batch, nh, hd, 1]
            let v_row = v_t.unsqueeze(2)?; // [batch, nh, 1, hd]
            let term3 = k_col.matmul(&v_row)?; // [batch, nh, hd, hd]

            state = ((&term1 + &term2)? + &term3)?;

            // Output: y_t = r_t @ S_t (uses UPDATED state)
            let r_row = r_t.unsqueeze(2)?; // [batch, nh, 1, hd]
            let out_t = r_row.matmul(&state)?; // [batch, nh, 1, hd]
            let out_t = out_t.squeeze(2)?; // [batch, nh, hd]

            outputs.push(out_t);
        }

        let out = Tensor::stack(&outputs, 1)?; // [batch, seq_len, nh, hd]
        let new_attn_kv = state; // [batch, nh, hd, hd]
        let decay = w_4d; // [batch, seq, nh, hd] — the raw decay values

        // --- GroupNorm per head ---
        // V7 uses eps = head_dim * norm_eps (per fla code)
        let gn_eps = (self.head_dim as f64) * self.norm_eps;
        let out_flat = out.reshape((bt, h))?;
        let out_gn = norm::group_norm(
            &out_flat,
            self.num_heads,
            &self.g_norm_weight.to_dtype(DType::F32)?,
            &self.g_norm_bias.to_dtype(DType::F32)?,
            gn_eps,
        )?;
        let out_gn = out_gn.reshape((batch, seq_len, h))?;

        // --- Gate output correction ---
        // correction = (r * k * r_k).sum(-1, keepdim=True) * v
        // r_k: [nh, hd] → [1, 1, nh, hd] → [1, 1, h]
        let r_k_flat = self.r_k.to_dtype(DType::F32)?.reshape((1, 1, h))?;
        // r, k_mod are [batch, seq, h] in original dtype; need F32
        let r_f32 = r.to_dtype(DType::F32)?;
        let k_mod_f32 = k_mod.to_dtype(DType::F32)?;

        // (r * k * r_k): [batch, seq, h]
        let rkrk = (&r_f32 * &k_mod_f32)?.broadcast_mul(&r_k_flat)?;
        // .reshape to [batch, seq, nh, hd] → sum over hd → [batch, seq, nh, 1] → * v
        let rkrk_4d = rkrk.reshape((batch, seq_len, nh, hd))?;
        let v_f32 = v_out.to_dtype(DType::F32)?;
        let rkrk_sum_4d = rkrk_4d
            .sum_keepdim(candle_core::D::Minus1)?
            .reshape((batch, seq_len, nh, 1))?;
        let v_4d_corr = v_f32.reshape((batch, seq_len, nh, hd))?;
        let correction = rkrk_sum_4d
            .broadcast_mul(&v_4d_corr)?
            .reshape((batch, seq_len, h))?;

        // out = (gn_out + correction) * g
        let g_f32 = g.to_dtype(DType::F32)?;
        let out_corrected = ((&out_gn + &correction)? * &g_f32)?;
        let out_final = out_corrected.to_dtype(hidden.dtype())?;

        // --- Output projection ---
        let out_final = self.o_proj.forward(&out_final)?;

        // Return v for v_first threading (layer 0 provides raw v; others pass through)
        let v_for_first = if self.layer_idx == 0 { v } else { v_out };

        Ok((out_final, new_attn_x, new_attn_kv, v_for_first, decay))
    }
}

// ---------------------------------------------------------------------------
// ChannelMixV7 (RWKV-7 channel-mix / FFN)
// ---------------------------------------------------------------------------

/// RWKV-7 channel-mix block: plain squared-ReLU MLP (no receptance gate).
///
/// Structure: `value(sqrelu(key(x + delta * x_k)))`
struct ChannelMixV7 {
    /// Token-shift mixing parameter for key path: `[hidden_size]` (1D, not `[1,1,h]`).
    x_k: Tensor,
    /// Up projection: `hidden → intermediate` (no bias).
    key: Linear,
    /// Down projection: `intermediate → hidden` (no bias).
    value: Linear,
}

impl ChannelMixV7 {
    /// Load the RWKV-7 channel-mix block from weights.
    ///
    /// `vb` should be scoped to `model.layers.{i}.ffn`.
    #[allow(clippy::needless_pass_by_value)]
    fn load(config: &RwkvConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.hidden_size;
        let intermediate = config.intermediate_size;

        let x_k = vb.get(h, "x_k")?;
        let key = candle_nn::linear_no_bias(h, intermediate, vb.pp("key"))?;
        let value = candle_nn::linear_no_bias(intermediate, h, vb.pp("value"))?;

        Ok(Self { x_k, key, value })
    }

    /// Forward pass for the V7 channel-mix (FFN) block.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - `ffn_x_state`: `[batch, hidden_size]` or `None`
    /// - returns: `(output, new_ffn_x_state)`
    fn forward(&self, hidden: &Tensor, ffn_x_state: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, channels) = hidden.dims3()?;

        let shifted = token_shift(hidden, ffn_x_state, batch, seq_len, channels)?;
        let new_ffn_x = hidden.i((.., seq_len - 1, ..))?;

        let delta = shifted.broadcast_sub(hidden)?;
        // x_k is [h] (1D), need to broadcast against [batch, seq, h]
        let key_input = hidden.broadcast_add(&delta.broadcast_mul(&self.x_k)?)?;

        // sqrelu: relu(x)^2
        let key_out = self.key.forward(&key_input)?.relu()?.sqr()?;
        let out = self.value.forward(&key_out)?;

        Ok((out, new_ffn_x))
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
// Helpers
// ---------------------------------------------------------------------------

/// L2-normalize the last dimension of a tensor.
///
/// # Shapes
/// - `x`: `[..., dim]`
/// - returns: `[..., dim]` (unit-length along last dim)
fn l2_norm(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq_sum = x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let norm = (sq_sum + 1e-12_f64)?.sqrt()?;
    Ok(x_f32.broadcast_div(&norm)?)
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
    /// Dispatches to version-specific weight loading based on `config.version`.
    ///
    /// # Weight paths
    ///
    /// | Component | RWKV-6 | RWKV-7 |
    /// |-----------|--------|--------|
    /// | Embeddings | `rwkv.embeddings` | `model.embeddings` |
    /// | Blocks | `rwkv.blocks.{i}` | `model.layers.{i}` |
    /// | Final norm | `rwkv.ln_out` | `model.norm` |
    /// | LM head | `head` | `lm_head` |
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
        match config.version {
            RwkvVersion::V6 => Self::load_v6(config, vb),
            RwkvVersion::V7 => Self::load_v7(config, vb),
        }
    }

    /// Load RWKV-6 "Finch" model weights.
    ///
    /// Weight prefix: `rwkv.*` for most components, `head.*` for LM head.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    fn load_v6(config: RwkvConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let vb_rwkv = vb.pp("rwkv");

        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb_rwkv.pp("embeddings"),
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(RwkvBlock::load_v6(
                &config,
                vb_rwkv.pp(format!("blocks.{i}")),
                i,
            )?);
        }

        let ln_out = LayerNorm::load(config.hidden_size, config.norm_eps, vb_rwkv.pp("ln_out"))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
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

    /// Load RWKV-7 "Goose" model weights (fla / `HuggingFace` format).
    ///
    /// Weight prefix: `model.*` for most components, `lm_head.*` for LM head.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder convention
    fn load_v7(config: RwkvConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let vb_model = vb.pp("model");

        let embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb_model.pp("embeddings"),
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(RwkvBlock::load_v7(
                &config,
                vb_model.pp(format!("layers.{i}")),
                i,
            )?);
        }

        let ln_out = LayerNorm::load(config.hidden_size, config.norm_eps, vb_model.pp("norm"))?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
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
        let mut v_first: Option<Tensor> = None;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            // Hook: ResidPre
            if hooks.is_captured(&HookPoint::ResidPre(layer_idx)) {
                cache.store(HookPoint::ResidPre(layer_idx), hidden.clone());
            }

            let (new_hidden, new_attn_x, new_attn_kv, new_ffn_x, v_out, decay) = block.forward(
                &hidden,
                state.attn_x.get(layer_idx).and_then(Option::as_ref),
                state.attn_kv.get(layer_idx).and_then(Option::as_ref),
                state.ffn_x.get(layer_idx).and_then(Option::as_ref),
                v_first.as_ref(),
            )?;

            // Thread v_first for V7 value residual
            if layer_idx == 0 && self.config.version == RwkvVersion::V7 {
                v_first = Some(v_out);
            }

            hidden = new_hidden;

            // Hook: RwkvState — capture the WKV state after update
            if hooks.is_captured(&HookPoint::RwkvState(layer_idx)) {
                cache.store(HookPoint::RwkvState(layer_idx), new_attn_kv.clone());
            }

            // Hook: RwkvDecay — capture the per-timestep decay tensor
            if hooks.is_captured(&HookPoint::RwkvDecay(layer_idx)) {
                cache.store(HookPoint::RwkvDecay(layer_idx), decay);
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
