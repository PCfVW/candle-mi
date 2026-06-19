// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diffusion-language-model backends.
//!
//! Currently hosts [`MDLM`](mdlm::GenericMdlm) (masked diffusion; Sahoo et al.,
//! `NeurIPS` 2024), a bidirectional `DiT` with `adaLN` conditioning.  The module
//! is named for the architecture *class* so that other discrete-diffusion
//! families (`SEDD`, `LLaDA`, Dream) can join it without a rename.
//!
//! Enabled by the `diffusion` feature.  Reuses the shared
//! [`MIBackend`](crate::MIBackend) trait, hook system, and `VarBuilder` weight
//! loading; it does **not** depend on the `transformer` feature.

pub mod config;
pub mod mdlm;
pub mod rope;

pub use config::MdlmConfig;
pub use mdlm::GenericMdlm;

/// `model_type` strings handled by the diffusion backend dispatch in
/// [`MIModel::from_pretrained`](crate::MIModel::from_pretrained).
pub const SUPPORTED_DIFFUSION_MODEL_TYPES: &[&str] = &["mdlm"];
