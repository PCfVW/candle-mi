// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diffusion-language-model backends.
//!
//! Hosts [`MDLM`](mdlm::GenericMdlm) (masked diffusion; Sahoo et al.,
//! `NeurIPS` 2024), a bidirectional `DiT` with `adaLN` conditioning, and
//! [`OthelloGpt`], a plain GPT-2-style bidirectional
//! backbone (learned absolute positions, full `LayerNorm`, exact-`GELU` MLP)
//! used as a masked-diffusion world model.  The module is named for the
//! architecture *class* so that other discrete-diffusion families (`SEDD`,
//! `LLaDA`, Dream) can join it without a rename.
//!
//! Enabled by the `diffusion` feature.  Reuses the shared
//! [`MIBackend`](crate::MIBackend) trait, hook system, and `VarBuilder` weight
//! loading; it does **not** depend on the `transformer` feature.
//!
//! Beyond the backend, [`sample`] provides the masked-diffusion `SUBS` ancestral
//! sampler ([`sample::generate`]) and the per-step token trajectory
//! ([`sample::generate_trajectory`]) — the denoising-step axis the diffusion-MI
//! examples (`diffusion_logit_lens`, `diffusion_decoding_order`) consume.

pub mod config;
pub mod mdlm;
pub mod othello;
pub mod rope;
pub mod sample;

pub use config::MdlmConfig;
pub use mdlm::GenericMdlm;
pub use othello::{OthelloGpt, OthelloGptConfig};
pub use sample::{DiffusionSamplingConfig, generate, generate_trajectory};

/// `model_type` strings handled by the diffusion backend dispatch in
/// [`MIModel::from_pretrained`](crate::MIModel::from_pretrained).
pub const SUPPORTED_DIFFUSION_MODEL_TYPES: &[&str] = &["mdlm"];
