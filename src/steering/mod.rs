// SPDX-License-Identifier: MIT OR Apache-2.0

//! Steering-vector **construction** methods.
//!
//! This module is a sibling of [`crate::interp::steering`], which handles
//! steering-vector **calibration** (dose-response curves, absorption-boundary
//! detection) for an already-built vector.  `crate::steering` covers how the
//! vector is constructed in the first place from model activations.
//!
//! ## Submodules
//!
//! - [`contrastive`] — Maar et al. (2026) "What's the plan?"
//!   mean-of-differences contrastive activation steering:
//!   `d = mean(positive_residuals) − mean(negative_residuals)`,
//!   optionally L2-normalised, applied additively via
//!   [`Intervention::Add`](crate::Intervention::Add).

pub mod contrastive;

// Re-export the general-purpose intervention-payload builders at the module
// root. `position_delta` and `contrastive_intervention` are not specific to the
// contrastive method — they are generic payload builders that happen to live in
// the [`contrastive`] submodule — so surface them here for discoverability
// (e.g. a single-position residual edit needs `steering::position_delta`).
pub use contrastive::{build_contrastive_direction, contrastive_intervention, position_delta};
