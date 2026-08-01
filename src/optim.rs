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
        let (lr, beta1, beta2) = (self.params.lr, self.params.beta1, self.params.beta2);
        let lr_lambda = lr * self.params.weight_decay;
        // CAST: usize → i32 for `powi`; a step count stays far below `i32::MAX`,
        // and saturating there keeps the correction factor finite regardless.
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
}
