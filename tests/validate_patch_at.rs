// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `Intervention::PatchAt` through a real forward pass.
//!
//! The unit tests in `src/hooks.rs` exercise `apply_intervention` directly.
//! This one exercises the path a consumer actually takes: capture a donor
//! activation, hand one of its rows to `PatchAt`, and run a second forward.
//! That difference is what matters, because the donor row a consumer has is a
//! **view** into the captured activation, not a freshly allocated tensor, and
//! views are what broke.
//!
//! ## What it guards
//!
//! `PatchAt` was first implemented with `Tensor::slice_scatter`. On CUDA that
//! silently overwrote every position *after* the patch site, because candle's
//! `copy_strided_src` sizes its CUDA copy from the source's whole storage rather
//! than from the view (candle#3940; CPU sizes it from the view and is correct).
//! Every unit test passed on both devices, because they built the value with
//! `Tensor::new`, which owns its storage exactly. It took a real causal trace on
//! Llama-3.2-1B to notice, and the symptom was a plausible number rather than a
//! failure.
//!
//! ## The invariant
//!
//! `ResidPost(n_layer - 1)` feeds only the final `LayerNorm` and the unembedding
//! projection, both of which are position-wise. So patching position `p` there
//! can move the logits at `p` and **nothing else**, whatever the attention
//! pattern earlier in the network. The overrun bug violates that directly:
//! positions after `p` change too.
//!
//! Both directions are asserted, so the test cannot pass by the intervention
//! quietly failing to apply: position `p` must change, every other position must
//! not.
//!
//! ## What each case actually reaches
//!
//! The two are not redundant, and neither subsumes the other. Measured against
//! a deliberately reinstated `slice_scatter` implementation:
//!
//! - **CPU still passes.** candle's CPU backend sizes the copy from the view, so
//!   it never had the bug. This case guards *our* logic instead: patching the
//!   wrong axis, the wrong position, or not at all.
//! - **CUDA fails**, at the position immediately after the patch site, which is
//!   the overrun's signature.
//!
//! ## Cost
//!
//! `OthelloGpt::init` builds a two-layer model from a seed, so this downloads
//! nothing and runs in milliseconds. The CPU case therefore runs in CI. The CUDA
//! case is `#[ignore]`d like the rest of the GPU suite and is registered in
//! `scripts/resurrect.ps1` as entry `patchat`.

#![cfg(feature = "diffusion")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use candle_core::{Device, Tensor};
use candle_nn::VarMap;

use candle_mi::{HookPoint, HookSpec, Intervention, MIBackend, OthelloGpt, OthelloGptConfig};

/// Two layers, two heads, hidden 8: small enough to be instant, deep enough to
/// have a non-final layer and several positions after the patch site.
const N_LAYER: usize = 2;
/// Sequence length of both the recipient and the donor prompt.
const SEQ_LEN: usize = 5;
/// Position to patch. Must be neither the first nor the last, so that an
/// overrun in either direction is visible.
const PATCH_POS: usize = 1;

/// Assert that patching one position at the last layer moves that position's
/// logits and leaves every other position bit-identical.
fn patch_at_moves_only_its_own_position(device: &Device) {
    let config = OthelloGptConfig::new(12, 8, N_LAYER, 2, 8, false).unwrap();
    let varmap = VarMap::new();
    let model = OthelloGpt::init(config, &varmap, device, 20_260_902).unwrap();

    let recipient = Tensor::new(&[[1_u32, 2, 3, 4, 5]], device).unwrap();
    let donor = Tensor::new(&[[6_u32, 7, 8, 9, 10]], device).unwrap();
    let last_layer = N_LAYER - 1;

    // The donor row, taken the way a consumer takes it: a view into a captured
    // activation, whose storage still holds every other position.
    let mut capture = HookSpec::new();
    capture.capture(HookPoint::ResidPost(last_layer));
    let donor_cache = MIBackend::forward(&model, &donor, &capture).unwrap();
    let donor_row = donor_cache
        .require(&HookPoint::ResidPost(last_layer))
        .unwrap()
        .get(0)
        .unwrap()
        .get(PATCH_POS)
        .unwrap(); // [hidden], offset view

    let baseline = MIBackend::forward(&model, &recipient, &HookSpec::new())
        .unwrap()
        .into_output();

    let mut hooks = HookSpec::new();
    hooks.intervene(
        HookPoint::ResidPost(last_layer),
        Intervention::PatchAt {
            position: PATCH_POS,
            value: donor_row,
        },
    );
    let patched = MIBackend::forward(&model, &recipient, &hooks)
        .unwrap()
        .into_output();

    assert_eq!(baseline.dims(), patched.dims());

    for pos in 0..SEQ_LEN {
        let before: Vec<f32> = baseline
            .get(0)
            .unwrap()
            .get(pos)
            .unwrap()
            .to_vec1()
            .unwrap();
        let after: Vec<f32> = patched.get(0).unwrap().get(pos).unwrap().to_vec1().unwrap();

        if pos == PATCH_POS {
            // Positive control: without this, the test would also pass if the
            // intervention were silently dropped.
            assert_ne!(
                before, after,
                "{device:?}: position {pos} was patched and must change"
            );
        } else {
            assert_eq!(
                before, after,
                "{device:?}: position {pos} was not patched and must not move; \
                 a difference here is the copy overrunning the patch site"
            );
        }
    }
}

#[test]
fn patch_at_moves_only_its_own_position_cpu() {
    patch_at_moves_only_its_own_position(&Device::Cpu);
}

#[test]
#[ignore = "requires a CUDA device; run with --ignored (or scripts/resurrect.ps1 -Only patchat)"]
fn cuda_patch_at_moves_only_its_own_position() {
    let device = Device::new_cuda(0).expect("no CUDA device");
    patch_at_moves_only_its_own_position(&device);
}
