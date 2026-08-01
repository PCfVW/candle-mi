// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared utilities: attention masks, character-to-token positioning, PCA,
//! seeded Gaussian sampling, and the frozen seeded generator behind it.

pub mod masks;
pub mod pca;
pub mod positioning;
#[cfg(any(feature = "transformer", feature = "diffusion"))]
pub mod randn;
#[cfg(any(feature = "transformer", feature = "diffusion"))]
pub mod rng;
