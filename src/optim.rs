// SPDX-License-Identifier: MIT OR Apache-2.0

//! `AdamW` with **checkpointable state** — the one thing stock `candle_nn::AdamW`
//! cannot do.
//!
//! # Why this module exists
//!
//! `candle_nn::AdamW` keeps its per-parameter moments in a private `VarAdamW`
//! and its step counter in a private field; `impl AdamW` exposes only `new_lr`,
//! `params` and `set_params` (verified against `candle-nn` 0.9.2 and 0.11.0 —
//! `src/optim.rs` is byte-identical between them, so this is settled API rather
//! than a moving target). The optimizer's state therefore cannot be read, cannot
//! be saved, and cannot be restored.
//!
//! That is not an encapsulation decision. `SGD`, in the same 201-line file, does
//! expose `into_inner()`; `AdamW` simply never got an accessor, because
//! **`candle-nn` has no notion of training state that outlives a process** —
//! `VarMap::save`/`VarMap::load` serializes model weights and nothing else. An
//! accessor with no consumer does not get written.
//!
//! It matters as soon as a run is staged. Adam divides its moments by a
//! bias-correction factor `1 - beta^t`, which is far from 1 on early steps by
//! design, to compensate for moments that started at zero. Restart a process
//! without that state and `t` resets to 1, so the optimizer applies a full
//! warm-up correction to a model already thousands of steps in. On the run that
//! motivated this module (40 epochs, ~21 h, taken in ~5 h stages) that is one
//! shock per stage boundary, landing exactly where an analysis reads a quantity
//! off consecutive checkpoints.
//!
//! # What is ours and what is candle's
//!
//! **The update rule in [`AdamW::step`] is candle's**, transcribed from
//! `candle-nn` 0.11.0 `src/optim.rs::<AdamW as Optimizer>::step` (© the candle
//! authors, `MIT OR Apache-2.0`, the same licence as this crate). None of the
//! arithmetic is re-derived; only the state's ownership changes, from private
//! `Var`s to a named, serializable map. `tests/validate_optim_parity.rs` holds
//! the two to the same trajectory (`< 1e-6` over 1, 2, 17 and 120 steps) and
//! carries a power control proving that silently losing the moments *would* be
//! visible.
//!
//! If candle-nn ever gains its own state accessors, this module should be
//! deleted in favour of stock `AdamW`. See
//! `docs/upstream/candle-adamw-state-accessors.md` for the proposed patch.
//!
//! # Feature
//!
//! Behind the default-off `training` feature: candle-mi is an interpretability
//! crate first, and an inference-only consumer should not compile an optimizer.

use std::collections::BTreeMap;

use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};

use crate::error::Result;

pub use candle_nn::ParamsAdamW;

/// Prefix under which first moments are serialized.
const FIRST: &str = "adamw.first.";
/// Prefix under which second moments are serialized.
const SECOND: &str = "adamw.second.";

/// `AdamW`, with its moments and step counter reachable.
///
/// Construct with [`AdamW::new`], drive with [`AdamW::step`], and checkpoint
/// with [`AdamW::state`] plus [`AdamW::steps_taken`]. Restore both together via
/// [`AdamW::restore`]: the moments alone are not enough, because the step
/// counter drives bias correction.
#[derive(Debug)]
pub struct AdamW {
    /// The parameters being optimized, paired with the names their state is
    /// keyed by. Names are the caller's; they must be stable across a resume.
    vars: Vec<(String, Var)>,
    /// First moments (`m`), by parameter name.
    first: BTreeMap<String, Tensor>,
    /// Second moments (`v`), by parameter name.
    second: BTreeMap<String, Tensor>,
    /// Steps taken — drives Adam's bias correction, so it MUST survive a resume.
    step: usize,
    /// Learning rate, betas, epsilon, decoupled weight decay.
    params: ParamsAdamW,
    /// The hot-path cache: both moment sets flattened into one buffer each, in
    /// [`Self::vars`] order. `None` until the first full-gradient step builds it,
    /// and dropped whenever the named maps are written from outside
    /// ([`Self::restore`]) or a step must fall back to the per-parameter path.
    ///
    /// **Why it exists.** The per-parameter update launches ~13 small kernels per
    /// parameter per step; on a 29-parameter model that is ~380 launches doing
    /// what 13 launches over one flat buffer do. This is the flat-buffer
    /// equivalent of `PyTorch`'s `foreach` multi-tensor path (see
    /// `ForeachFunctors.cuh`'s `TensorListMetadata`, which batches many tensors
    /// per launch; a concatenation achieves the same launch count without a
    /// custom kernel). Elementwise arithmetic is position-blind, so the flat
    /// trajectory is bit-identical to the per-parameter one — held by
    /// `tests/validate_optim_parity.rs` against stock candle either way.
    flat: Option<FlatMoments>,
    /// How many steps ran on the single-launch multi-tensor path — a diagnostic
    /// counter, so a test asserting that path ran can distinguish it from a
    /// silent fall-through to [`Self::step_flat`] (which computes the same
    /// numbers and would otherwise pass every value comparison vacuously).
    /// Always zero off-cuda.
    mt_steps: usize,
}

/// The flattened moments plus the layout needed to split them back by name.
#[derive(Debug)]
struct FlatMoments {
    /// First moments, one flat `f32` buffer in [`AdamW::vars`] order.
    m: Tensor,
    /// Second moments, same layout.
    v: Tensor,
    /// `(offset, len)` per parameter, in [`AdamW::vars`] order — the shapes come
    /// from the live `Var`s, so they are not duplicated here.
    spans: Vec<(usize, usize)>,
}

impl AdamW {
    /// Creates an optimizer over named parameters, with zeroed moments.
    ///
    /// Non-float parameters are skipped, matching candle's implementation, so
    /// the optimizer may hold fewer entries than were passed in. Use
    /// [`AdamW::len`] rather than the caller's own count when checking a resume.
    ///
    /// ```
    /// use candle_core::{Device, Var};
    /// use candle_mi::optim::{AdamW, ParamsAdamW};
    ///
    /// let w = Var::from_slice(&[3f32, -4.0], (2,), &Device::Cpu)?;
    /// let mut opt = AdamW::new(
    ///     vec![("w".to_owned(), w.clone())],
    ///     ParamsAdamW { lr: 0.1, ..ParamsAdamW::default() },
    /// )?;
    /// // Descend on `w^2`: 200 steps take both coordinates towards zero.
    /// for _ in 0..200 {
    ///     let loss = w.as_tensor().sqr()?.sum_all()?;
    ///     opt.step(&loss.backward()?)?;
    /// }
    /// let settled = w.as_tensor().abs()?.max(0)?.to_scalar::<f32>()?;
    /// assert!(settled < 0.2, "expected descent towards 0, got {settled}");
    /// assert_eq!(opt.steps_taken(), 200);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if a moment buffer
    /// cannot be allocated.
    pub fn new(vars: Vec<(String, Var)>, params: ParamsAdamW) -> Result<Self> {
        let mut first = BTreeMap::new();
        let mut second = BTreeMap::new();
        let mut kept = Vec::with_capacity(vars.len());
        for (name, var) in vars {
            if !var.dtype().is_float() {
                continue;
            }
            let zeros = || Tensor::zeros(var.shape(), var.dtype(), var.device());
            first.insert(name.clone(), zeros()?);
            second.insert(name.clone(), zeros()?);
            kept.push((name, var));
        }
        Ok(Self {
            vars: kept,
            first,
            second,
            step: 0,
            params,
            flat: None,
            mt_steps: 0,
        })
    }

    /// Steps taken so far — Adam's `t`, which drives bias correction.
    #[must_use]
    pub const fn steps_taken(&self) -> usize {
        self.step
    }

    /// Sets the learning rate, leaving the rest of the schedule alone.
    pub const fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    /// The schedule in force: learning rate, betas, epsilon, weight decay.
    #[must_use]
    pub const fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    /// Applies one `AdamW` update.
    ///
    /// Transcribed from `candle-nn` 0.11.0 `src/optim.rs`, with one addition:
    /// the stored moments are **detached**. A gradient carries the backward
    /// graph it was computed from, so an undetached moment would retain that
    /// graph; since the next step folds the previous moment in, the graph would
    /// chain across steps and retain every step's graph for the run's lifetime.
    /// candle's own implementation is not exposed to this because it writes
    /// through a `Var`, which drops the intermediate immediately.
    ///
    /// Parameters absent from `grads` are skipped, so a partially-connected
    /// graph updates what it can rather than erroring.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on any tensor failure.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step += 1;
        let every_grad_present = self
            .vars
            .iter()
            .all(|(_, var)| grads.get(var.as_tensor()).is_some());
        if every_grad_present && !self.vars.is_empty() {
            #[cfg(feature = "cuda")]
            if self.try_step_multi_tensor(grads)? {
                self.mt_steps += 1;
                return Ok(());
            }
            self.step_flat(grads)
        } else {
            // A partially-connected graph keeps the historical semantics exactly:
            // parameters without a gradient keep their moments untouched. The flat
            // cache cannot express "skip", so it is synced back into the named maps
            // and dropped before the per-parameter path runs.
            self.sync_named_from_flat()?;
            self.step_per_param(grads)
        }
    }

    /// The hot path: both moment sets and the whole update as flat buffers.
    ///
    /// One `cat` per input stream and ~13 elementwise launches over a single
    /// buffer replace ~13 launches PER parameter. Elementwise arithmetic is
    /// position-blind, so every parameter element sees bit-for-bit the update the
    /// per-parameter path computes with the same scalars.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on any tensor failure.
    fn step_flat(&mut self, grads: &GradStore) -> Result<()> {
        let (lr, beta1, beta2) = (self.params.lr, self.params.beta1, self.params.beta2);
        let lr_lambda = lr * self.params.weight_decay;
        // `usize` → `i32` for `powi`. Deliberately `try_from` and not an `as`
        // cast, which the conventions reserve for when truncation is the
        // intent: a step count stays far below `i32::MAX`, and saturating there
        // keeps the bias-correction factor finite rather than wrapping negative.
        let exponent = i32::try_from(self.step).unwrap_or(i32::MAX);
        let scale_m = 1.0 / (1.0 - beta1.powi(exponent));
        let scale_v = 1.0 / (1.0 - beta2.powi(exponent));

        if self.flat.is_none() {
            self.flat = Some(self.build_flat()?);
        }
        let Some(flat) = self.flat.as_ref() else {
            // EXPLICIT: unreachable -- the branch above just filled the cache;
            // structured as `let else` so no panic path exists even in theory.
            return Ok(());
        };

        let mut grad_parts = Vec::with_capacity(self.vars.len());
        let mut param_parts = Vec::with_capacity(self.vars.len());
        for (_, var) in &self.vars {
            let Some(grad) = grads.get(var.as_tensor()) else {
                // The caller checked every gradient is present; a disappearance
                // between the check and here would be a logic error upstream.
                return Err(crate::MIError::Model(candle_core::Error::Msg(
                    "gradient vanished between presence check and flat gather".to_string(),
                )));
            };
            grad_parts.push(grad.flatten_all()?);
            param_parts.push(var.as_tensor().flatten_all()?);
        }
        let g = Tensor::cat(&grad_parts, 0)?;
        let p = Tensor::cat(&param_parts, 0)?;

        let next_m = ((&flat.m * beta1)? + (&g * (1.0 - beta1))?)?;
        let next_v = ((&flat.v * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
        let m_hat = (&next_m * scale_m)?;
        let v_hat = (&next_v * scale_v)?;
        let decayed = (p * (1.0 - lr_lambda))?;
        let adjusted = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
        let updated = (decayed - (adjusted * lr)?)?;

        let spans = flat.spans.clone();
        for ((_, var), (offset, len)) in self.vars.iter().zip(spans) {
            var.set(&updated.narrow(0, offset, len)?.reshape(var.shape())?)?;
        }
        self.flat = Some(FlatMoments {
            m: next_m.detach(),
            v: next_v.detach(),
            spans: self.flat.take().map_or_else(Vec::new, |cache| cache.spans),
        });
        Ok(())
    }

    /// The historical per-parameter path, kept verbatim for partially-connected
    /// graphs; see [`AdamW::step`] for when it runs.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on any tensor failure.
    fn step_per_param(&mut self, grads: &GradStore) -> Result<()> {
        let (lr, beta1, beta2) = (self.params.lr, self.params.beta1, self.params.beta2);
        let lr_lambda = lr * self.params.weight_decay;
        // CAST rationale as in `step_flat`.
        let exponent = i32::try_from(self.step).unwrap_or(i32::MAX);
        let scale_m = 1.0 / (1.0 - beta1.powi(exponent));
        let scale_v = 1.0 / (1.0 - beta2.powi(exponent));

        // EXPLICIT: the parameter list is cloned (a `Var` is a cheap handle) so
        // the moment maps can be written inside the loop without holding a
        // borrow of `self.vars`.
        let vars = self.vars.clone();
        for (name, var) in vars {
            let Some(grad) = grads.get(var.as_tensor()) else {
                continue;
            };
            let (Some(m), Some(v)) = (self.first.get(&name), self.second.get(&name)) else {
                continue;
            };
            let next_m = ((m * beta1)? + (grad * (1.0 - beta1))?)?;
            let next_v = ((v * beta2)? + (grad.sqr()? * (1.0 - beta2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let decayed = (var.as_tensor() * (1.0 - lr_lambda))?;
            let adjusted = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
            var.set(&(decayed - (adjusted * lr)?)?)?;
            self.first.insert(name.clone(), next_m.detach());
            self.second.insert(name, next_v.detach());
        }
        Ok(())
    }

    /// Concatenates the named moments into flat buffers, in [`Self::vars`] order.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if a moment is missing
    /// or a tensor operation fails.
    fn build_flat(&self) -> Result<FlatMoments> {
        let mut m_parts = Vec::with_capacity(self.vars.len());
        let mut v_parts = Vec::with_capacity(self.vars.len());
        let mut spans = Vec::with_capacity(self.vars.len());
        let mut offset = 0_usize;
        for (name, var) in &self.vars {
            let len = var.as_tensor().elem_count();
            let (Some(m), Some(v)) = (self.first.get(name), self.second.get(name)) else {
                return Err(crate::MIError::Model(candle_core::Error::Msg(format!(
                    "moment missing for parameter {name} while building the flat cache"
                ))));
            };
            m_parts.push(m.flatten_all()?);
            v_parts.push(v.flatten_all()?);
            spans.push((offset, len));
            offset += len;
        }
        Ok(FlatMoments {
            m: Tensor::cat(&m_parts, 0)?,
            v: Tensor::cat(&v_parts, 0)?,
            spans,
        })
    }

    /// Writes the flat moments back into the named maps and drops the cache.
    ///
    /// A no-op when the cache is empty. Called before anything that reads or
    /// writes the maps directly: the per-parameter fallback here, and
    /// [`AdamW::state`] / [`AdamW::restore`] on their own paths.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on any tensor failure.
    fn sync_named_from_flat(&mut self) -> Result<()> {
        let Some(flat) = self.flat.take() else {
            return Ok(());
        };
        for ((name, var), (offset, len)) in self.vars.iter().zip(&flat.spans) {
            let shape = var.as_tensor().shape();
            self.first.insert(
                name.clone(),
                flat.m.narrow(0, *offset, *len)?.reshape(shape)?,
            );
            self.second.insert(
                name.clone(),
                flat.v.narrow(0, *offset, *len)?.reshape(shape)?,
            );
        }
        Ok(())
    }

    /// The moments, keyed for serialization alongside the model weights.
    ///
    /// Keys are `adamw.first.<name>` and `adamw.second.<name>`, so the map drops
    /// straight into `candle_core::safetensors::save` next to the weights
    /// without a second file.
    ///
    /// The step counter is deliberately **not** here: it is a scalar, and it
    /// belongs somewhere readable without loading tensors. Save it beside the
    /// checkpoint and hand it back to [`AdamW::restore`].
    ///
    /// # Shapes
    /// - returns: one entry per moment, each the **same shape and dtype as its
    ///   parameter** (Adam keeps one running average per weight element), so
    ///   the map is exactly twice the model's parameter footprint.
    #[must_use]
    pub fn state(&self) -> BTreeMap<String, Tensor> {
        // With the flat cache active the named maps are stale by design (the hot
        // path never touches them); the checkpoint view is carved out of the flat
        // buffers instead, so the KEYS and shapes on disk are identical either
        // way. `&self` is kept -- this is a read, and the published signature
        // must not change. A narrow that fails here would mean the cache and the
        // parameter list disagree on layout, which `build_flat` makes impossible;
        // the fallback to the (stale) map entry keeps this method total anyway.
        if let Some(flat) = self.flat.as_ref() {
            let mut out = BTreeMap::new();
            for ((name, var), (offset, len)) in self.vars.iter().zip(&flat.spans) {
                let shape = var.as_tensor().shape();
                if let (Ok(m), Ok(v)) = (
                    flat.m
                        .narrow(0, *offset, *len)
                        .and_then(|t| t.reshape(shape)),
                    flat.v
                        .narrow(0, *offset, *len)
                        .and_then(|t| t.reshape(shape)),
                ) {
                    out.insert(format!("{FIRST}{name}"), m);
                    out.insert(format!("{SECOND}{name}"), v);
                }
            }
            return out;
        }
        let mut out = BTreeMap::new();
        for (name, tensor) in &self.first {
            out.insert(format!("{FIRST}{name}"), tensor.clone());
        }
        for (name, tensor) in &self.second {
            out.insert(format!("{SECOND}{name}"), tensor.clone());
        }
        out
    }

    /// Restores moments and step counter saved by [`AdamW::state`].
    ///
    /// A parameter whose moments are absent keeps its zeroed buffers, so a
    /// checkpoint written by an older architecture degrades to a partial warm
    /// start rather than failing outright. The count of parameters whose first
    /// moment was restored is returned, so a caller that would rather refuse a
    /// partial resume can compare it against [`AdamW::len`] and do so.
    ///
    /// # Shapes
    /// - `state`: as produced by [`AdamW::state`] — each entry the same shape
    ///   and dtype as the parameter it belongs to. A shape mismatch is not
    ///   checked here; it surfaces on the next [`AdamW::step`].
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if a restored moment
    /// cannot be moved onto the device.
    pub fn restore(&mut self, state: &BTreeMap<String, Tensor>, step: usize) -> Result<usize> {
        self.step = step;
        // The named maps become the source of truth again. SYNC, not drop: a
        // checkpoint may restore only a subset of parameters, and the ones it
        // does not name must keep their LATEST moments -- which live in the flat
        // cache whenever it is active, not in the maps it left stale.
        self.sync_named_from_flat()?;
        let mut restored = 0_usize;
        for (name, _) in &self.vars {
            if let Some(tensor) = state.get(&format!("{FIRST}{name}")) {
                self.first.insert(name.clone(), tensor.detach());
                restored += 1;
            }
            if let Some(tensor) = state.get(&format!("{SECOND}{name}")) {
                self.second.insert(name.clone(), tensor.detach());
            }
        }
        Ok(restored)
    }

    /// How many parameters this optimizer owns, after the non-float filter.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.vars.len()
    }

    /// Whether the optimizer owns no parameters.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }

    /// How many steps ran on the single-launch multi-tensor path.
    ///
    /// Diagnostic only: the multi-tensor path computes the same numbers as the
    /// flat path, so a value-comparison test cannot tell them apart — this
    /// counter can, and the parity test asserts it to avoid gating vacuously on
    /// a silent fall-through. Always zero off-cuda.
    #[must_use]
    pub const fn multi_tensor_steps(&self) -> usize {
        self.mt_steps
    }
}

/// The single-launch `AdamW` step: every parameter, gradient and moment updated
/// by ONE kernel walking a chunk table of raw device pointers — `PyTorch`'s
/// fused/`foreach` shape (`ForeachFunctors.cuh`), which the flat-buffer path
/// above approximates but cannot reach: its `cat` gathers and `Var::set`
/// scatters cost one launch per parameter each, and on the model that motivated
/// this they ate exactly the launches the flattening saved.
///
/// The kernel (`adamw_mt_f32`, candle-kernels `reduce.cu` on the experiment
/// branch) writes parameters and moments IN PLACE through raw pointers. For the
/// parameters this is `Var::set`'s own semantics — `set` also writes into the
/// var's existing storage — reached without the intermediate tensor. Every
/// arithmetic step uses explicitly-rounded intrinsics so the trajectory is
/// bit-identical to the composed path's op-by-op f32 rounding; held by
/// `tests/optim_multi_tensor.rs` across CPU and CUDA bitwise.
///
/// Stock candle-kernels does not carry the kernel: the first step probes
/// `get_or_load_func` once and permanently falls back to [`AdamW::step_flat`]
/// when the symbol is absent, so against crates.io candle this module degrades
/// to slower, never to wrong.
#[cfg(feature = "cuda")]
impl AdamW {
    /// Attempts the multi-tensor step; `Ok(false)` means "not applicable, use
    /// the flat path" (non-cuda device, non-f32 dtype, a non-contiguous tensor,
    /// or a candle-kernels build without the kernel).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) only for failures past
    /// the preconditions — a table upload or launch error. Nothing has been
    /// mutated at that point: all writes happen inside the one kernel.
    fn try_step_multi_tensor(&mut self, grads: &GradStore) -> Result<bool> {
        static KERNEL_PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

        let Some((_, head)) = self.vars.first() else {
            return Ok(false);
        };
        let candle_core::Device::Cuda(dev) = head.device() else {
            return Ok(false);
        };
        if self.flat.is_none() {
            self.flat = Some(self.build_flat()?);
        }
        let Some(flat) = self.flat.as_ref() else {
            // EXPLICIT: unreachable — the branch above just filled the cache;
            // `let else` keeps the no-panic guarantee anyway.
            return Ok(false);
        };

        // Precondition phase: collect raw pointers WITHOUT mutating anything, so
        // a fall-through leaves the optimizer exactly as it was. All tensors
        // share the device's single stream, so an address collected here stays
        // ordered with respect to the launch below.
        let Some(m_base) = mt::f32_base_ptr(&flat.m, dev) else {
            return Ok(false);
        };
        let Some(v_base) = mt::f32_base_ptr(&flat.v, dev) else {
            return Ok(false);
        };
        let mut lens = Vec::with_capacity(self.vars.len());
        let mut param_ptrs = Vec::with_capacity(self.vars.len());
        let mut grad_ptrs = Vec::with_capacity(self.vars.len());
        for (_, var) in &self.vars {
            let Some(grad) = grads.get(var.as_tensor()) else {
                // The caller checked presence; vanishing here is a logic error.
                return Err(crate::MIError::Model(candle_core::Error::Msg(
                    "gradient vanished between presence check and pointer gather".to_string(),
                )));
            };
            let (Some(p), Some(g)) = (
                mt::f32_base_ptr(var.as_tensor(), dev),
                mt::f32_base_ptr(grad, dev),
            ) else {
                return Ok(false);
            };
            lens.push(var.as_tensor().elem_count());
            param_ptrs.push(p);
            grad_ptrs.push(g);
        }
        if !KERNEL_PRESENT.get_or_init(|| mt::kernel_present(dev)) {
            return Ok(false);
        }

        // The chunk table: (param, grad, m, v, len) per chunk, pointers
        // pre-offset to the chunk start so the kernel does no indexing at all.
        let chunks = mt::chunk_spans(&lens, mt::CHUNK);
        let mut table = Vec::with_capacity(chunks.len() * 5);
        let mut span_offset = 0_usize;
        let mut spans = Vec::with_capacity(lens.len());
        for len in &lens {
            spans.push(span_offset);
            span_offset += len;
        }
        for (tensor, offset, len) in &chunks {
            let (Some(&p_base), Some(&g_base), Some(&span)) = (
                param_ptrs.get(*tensor),
                grad_ptrs.get(*tensor),
                spans.get(*tensor),
            ) else {
                // `chunk_spans` only emits indices below `lens.len()`, which is
                // also every other vector's length; reaching here is a logic
                // error, surfaced rather than indexed into a panic.
                return Err(crate::MIError::Model(candle_core::Error::Msg(
                    "chunk table references a tensor out of range".to_string(),
                )));
            };
            let bytes = u64::try_from(*offset).unwrap_or(u64::MAX) * mt::F32_BYTES;
            let flat_bytes = u64::try_from(span + *offset).unwrap_or(u64::MAX) * mt::F32_BYTES;
            table.push(p_base + bytes);
            table.push(g_base + bytes);
            table.push(m_base + flat_bytes);
            table.push(v_base + flat_bytes);
            table.push(u64::try_from(*len).unwrap_or(u64::MAX));
        }

        // Scalars mirror `step_flat` exactly: every factor computed in f64 —
        // in the SAME operation order, so no `mul_add` fusions — and rounded
        // ONCE to f32, the single rounding candle's scalar ops apply when they
        // cast their f64 argument at the kernel boundary.
        let (lr, beta1, beta2) = (self.params.lr, self.params.beta1, self.params.beta2);
        let lr_lambda = lr * self.params.weight_decay;
        // CAST rationale as in `step_flat`.
        let exponent = i32::try_from(self.step).unwrap_or(i32::MAX);
        let scale_m = 1.0 / (1.0 - beta1.powi(exponent));
        let scale_v = 1.0 / (1.0 - beta2.powi(exponent));
        let one_minus_lr_lambda = 1.0 - lr_lambda;

        let table_dev = dev
            .cuda_stream()
            .clone_htod(&table)
            .map_err(candle_core::Error::wrap)?;
        let scalars = mt::Scalars {
            beta1: mt::kernel_scalar(beta1),
            one_minus_beta1: mt::kernel_scalar(1.0 - beta1),
            beta2: mt::kernel_scalar(beta2),
            one_minus_beta2: mt::kernel_scalar(1.0 - beta2),
            scale_m: mt::kernel_scalar(scale_m),
            scale_v: mt::kernel_scalar(scale_v),
            one_minus_lr_lambda: mt::kernel_scalar(one_minus_lr_lambda),
            eps: mt::kernel_scalar(self.params.eps),
            lr: mt::kernel_scalar(lr),
        };
        mt::launch(dev, chunks.len(), &table_dev, &scalars)?;
        Ok(true)
    }
}

/// Pointer plumbing and the kernel launch for the multi-tensor step, kept in
/// ONE module so the update logic above reads as the algorithm — and so the
/// crate's single `training`-side `unsafe` (the launch ffi) has exactly one
/// home, per the conventions' dedicated-module rule.
#[cfg(feature = "cuda")]
mod mt {
    use candle_core::Tensor;
    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDevice, kernels};

    /// The kernel this module launches — candle-kernels' multi-tensor `AdamW`
    /// step, present only on the experiment branch this crate is measured with.
    const KERNEL: &str = "adamw_mt_f32";

    /// Whether the linked candle-kernels carries [`KERNEL`] — the probe behind
    /// the once-per-process dispatch decision, and the reason stock candle
    /// degrades to the flat path instead of erroring.
    pub(super) fn kernel_present(dev: &CudaDevice) -> bool {
        dev.get_or_load_func(KERNEL, &kernels::REDUCE).is_ok()
    }

    /// Elements per launch chunk. Small enough that the largest parameter of
    /// the motivating model splits into dozens of blocks (load balance), large
    /// enough that the table upload stays a few tens of KB.
    pub(super) const CHUNK: usize = 16384;

    /// Bytes per `f32` element — the factor turning chunk-table element
    /// offsets into device-pointer byte offsets.
    pub(super) const F32_BYTES: u64 = 4;

    /// The nine `f32` scalars of one `AdamW` update, in KERNEL ARGUMENT ORDER —
    /// the struct exists so the call site cannot scramble nine same-typed
    /// positional floats.
    pub(super) struct Scalars {
        /// First-moment decay `β₁`.
        pub beta1: f32,
        /// `1 − β₁`, rounded from the f64 the composed path computes.
        pub one_minus_beta1: f32,
        /// Second-moment decay `β₂`.
        pub beta2: f32,
        /// `1 − β₂`, rounded from the f64 the composed path computes.
        pub one_minus_beta2: f32,
        /// First-moment bias correction `1 / (1 − β₁^t)`.
        pub scale_m: f32,
        /// Second-moment bias correction `1 / (1 − β₂^t)`.
        pub scale_v: f32,
        /// `1 − lr·λ`, the decoupled weight-decay factor.
        pub one_minus_lr_lambda: f32,
        /// Adam's `ε`, added to the root of the corrected second moment.
        pub eps: f32,
        /// Learning rate.
        pub lr: f32,
    }

    /// The one f64 → f32 rounding a scalar takes on its way into the kernel.
    ///
    /// This is the SAME single rounding candle's own scalar ops apply when they
    /// cast their f64 argument at the kernel boundary, which is why the cast is
    /// the intent here and not an accident of convenience — the multi-tensor
    /// trajectory must be bit-identical to the composed one.
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    pub(super) const fn kernel_scalar(x: f64) -> f32 {
        // CAST: f64 → f32, the deliberate single rounding at the kernel
        // boundary; see the doc comment.
        x as f32
    }

    /// Launches [`KERNEL`] over the uploaded chunk table.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the kernel cannot be loaded or the driver
    /// rejects the launch.
    pub(super) fn launch(
        dev: &CudaDevice,
        n_chunks: usize,
        table: &CudaSlice<u64>,
        scalars: &Scalars,
    ) -> candle_core::Result<()> {
        let func = dev.get_or_load_func(KERNEL, &kernels::REDUCE)?;
        let cfg = LaunchConfig {
            grid_dim: (
                u32::try_from(n_chunks).unwrap_or(u32::MAX).min(1 << 20),
                1,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        // Argument order mirrors the kernel signature exactly: n_chunks, table,
        // then the nine scalars in `Scalars`' declaration order.
        candle_core::builder_arg!(builder, n_chunks);
        builder.arg(table);
        candle_core::builder_arg!(
            builder,
            scalars.beta1,
            scalars.one_minus_beta1,
            scalars.beta2,
            scalars.one_minus_beta2,
            scalars.scale_m,
            scalars.scale_v,
            scalars.one_minus_lr_lambda,
            scalars.eps,
            scalars.lr
        );
        // SAFETY: ffi. Every pointer in the table derives from a live tensor
        // held by the calling optimizer or its grad store for the duration of
        // the call, offset within its own length by `chunk_spans`; table and
        // tensors all live on the device's one stream, the same stream the
        // launch goes to, so the upload is ordered before the kernel.
        #[allow(unsafe_code)]
        {
            unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        }
        Ok(())
    }

    /// The raw f32 device address of a tensor's first element, or `None` when
    /// the tensor is not a contiguous f32 cuda tensor on `dev`'s stream —
    /// `None` routes the caller to the tensor-level fallback.
    pub(super) fn f32_base_ptr(t: &Tensor, dev: &CudaDevice) -> Option<u64> {
        let (storage, layout) = t.storage_and_layout();
        let (start, _end) = layout.contiguous_offsets()?;
        let candle_core::Storage::Cuda(cs) = &*storage else {
            return None;
        };
        let slice = cs.as_cuda_slice::<f32>().ok()?;
        let view = slice.slice(start..);
        // The sync guard is dropped on return: everything here lives on the
        // device's ONE stream, so ordering needs no cross-stream event.
        let stream = dev.cuda_stream();
        let (ptr, _same_stream) =
            candle_core::cuda_backend::cudarc::driver::DevicePtr::device_ptr(&view, &stream);
        Some(ptr)
    }

    /// Splits per-tensor lengths into `(tensor, offset, len)` launch chunks of
    /// at most `chunk` elements, in tensor order. Every element lands in
    /// exactly one chunk; a zero-length tensor contributes none.
    pub(super) fn chunk_spans(lens: &[usize], chunk: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        for (tensor, &len) in lens.iter().enumerate() {
            let mut offset = 0_usize;
            while offset < len {
                let take = chunk.min(len - offset);
                out.push((tensor, offset, take));
                offset += take;
            }
        }
        out
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::mt::chunk_spans;

    #[test]
    fn chunk_spans_covers_every_element_exactly_once() {
        // One tensor under the chunk, one exactly at it, one spanning three.
        let chunks = chunk_spans(&[5, 8, 20], 8);
        assert_eq!(
            chunks,
            vec![(0, 0, 5), (1, 0, 8), (2, 0, 8), (2, 8, 8), (2, 16, 4)]
        );
        let total: usize = chunks.iter().map(|(_, _, len)| len).sum();
        assert_eq!(total, 5 + 8 + 20);
    }

    #[test]
    fn chunk_spans_skips_empty_tensors() {
        assert_eq!(chunk_spans(&[0, 3, 0], 8), vec![(1, 0, 3)]);
    }
}
