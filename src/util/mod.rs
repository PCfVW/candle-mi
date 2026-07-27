// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared utilities: attention masks, character-to-token positioning, PCA,
//! and seeded Gaussian sampling.

pub mod masks;
pub mod pca;
pub mod positioning;
#[cfg(any(feature = "transformer", feature = "diffusion"))]
pub mod randn;
