#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Gemma 2 2B GemmaScope PLT reference encodings via SAE Lens.

**Independent-library oracle** for the candle-mi GemmaScope loader (audit §1.11).
Unlike a from-first-principles script that re-derives the same
``W_enc.T @ x + b_enc`` + JumpReLU algebra candle-mi implements — which cannot
catch a *shared* conceptual error (wrong transpose, `>` vs `>=`, hook point) —
this generator loads each GemmaScope transcoder through **SAE Lens**
(`SAE.from_pretrained`) and encodes via **SAE Lens's own** `sae.encode()`. The
transpose convention, the JumpReLU threshold gate, and the bias handling are
therefore an *independent* implementation; agreement validates candle-mi against
a second codebase, not against our own re-derivation.

Provenance note: the previous from-first-principles generator (in git history)
produced numerically identical results — top-10 indices 10/10, activations within
< 1e-4, `n_active` exact across all 9 cases — so the migration is a strict
provenance upgrade, not a change in the validated numbers.

candle-mi still loads the *same underlying weights* independently: the Rust test
opens the transcoder via the `mntss/gemma-scope-transcoders` curation repo (which
resolves the same `google/gemma-scope-2b-pt-transcoders` NPZs SAE Lens loads —
confirmed: `n_active` matches exactly, so both sides load the same `average_l0`
variant per layer).

Test layers `{0, 12, 25}` cover the ends and middle of the 26-layer stack. Three
seeds per layer, deterministic via `torch.manual_seed(seed_idx * 100 + layer)`,
total 9 test cases. Synthetic `torch.randn` residuals — SAE Lens's `encode()`
takes an activation directly, so no base-model forward pass is needed (the
encoder is what §1.11 validates).

Acceptance bar (in `tests/validate_plt_gemma.rs`): top-10 feature indices match
exactly, `n_active` exact, activation magnitudes within abs-diff < 1e-4 (F32).

Dependencies: `torch`, `sae_lens` (validation tooling only; NOT a crate dep).

Usage:
    python scripts/plt_gemma_validation.py

Output:
    scripts/plt_gemma_reference.json
"""

import json
import os
import platform
from pathlib import Path

import sae_lens
import torch
from sae_lens import SAE
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

CURATION_REPO = "mntss/gemma-scope-transcoders"
WEIGHTS_REPO = "google/gemma-scope-2b-pt-transcoders"
SAE_LENS_RELEASE = "gemma-scope-2b-pt-transcoders"
# Ends + middle of Gemma 2 2B's 26-layer stack, mirroring plip-rs's [0, 12, 25].
TEST_LAYERS = [0, 12, 25]
N_SEEDS_PER_LAYER = 3
TOP_K = 10


def main() -> None:
    # Determinism — CPU-only script; set the CUBLAS knob anyway for parity with
    # the historical generator.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)

    print(f"GemmaScope PLT reference generation (SAE Lens oracle) for {WEIGHTS_REPO}")
    print(f"SAE Lens release: {SAE_LENS_RELEASE} (sae_lens {sae_lens.__version__})")
    print(f"Test layers: {TEST_LAYERS}, seeds per layer: {N_SEEDS_PER_LAYER}")
    print(f"torch {torch.__version__} on {platform.platform()}")
    print()

    # Map layer index -> SAE Lens sae_id (one canonical average_l0 per layer).
    directory = get_pretrained_saes_directory()[SAE_LENS_RELEASE]
    layer_to_sae_id: dict[int, str] = {}
    for sae_id in directory.saes_map:
        layer = int(sae_id.split("/")[0].removeprefix("layer_"))
        layer_to_sae_id[layer] = sae_id

    results: dict = {
        "weights_repo": WEIGHTS_REPO,
        "curation_repo": CURATION_REPO,
        "methodology": "independent oracle via SAE Lens SAE.from_pretrained().encode()",
        "sae_lens_version": sae_lens.__version__,
        "oracle_release": SAE_LENS_RELEASE,
        "oracle_sae_ids": {},
        "schema": "GemmaScopeNpz",
        "encoder_formula": "sae_lens JumpReLU transcoder encode() (independent impl)",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "d_model": None,
        "n_features_per_layer": None,
        "test_cases": [],
    }

    for layer in TEST_LAYERS:
        sae_id = layer_to_sae_id[layer]
        loaded = SAE.from_pretrained(SAE_LENS_RELEASE, sae_id)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        sae = sae.to("cpu")
        dtype = next(sae.parameters()).dtype
        d_model = int(sae.cfg.d_in)
        n_features = int(sae.cfg.d_sae)
        results["oracle_sae_ids"][str(layer)] = sae_id

        print(
            f"Layer {layer} ({sae_id}): d_in={d_model}, d_sae={n_features}, dtype={dtype}"
        )

        if results["d_model"] is None:
            results["d_model"] = d_model
            results["n_features_per_layer"] = n_features
        else:
            assert results["d_model"] == d_model, "d_model drifted across layers"
            assert results["n_features_per_layer"] == n_features, (
                "n_features drifted across layers"
            )

        for seed_idx in range(N_SEEDS_PER_LAYER):
            seed = seed_idx * 100 + layer
            torch.manual_seed(seed)
            residual = torch.randn(d_model)

            with torch.no_grad():
                acts = sae.encode(residual.to(dtype)).detach().float()

            n_active = int((acts > 0).sum())
            top_vals, top_idx = acts.topk(min(TOP_K, n_active))

            test_case = {
                "layer": layer,
                "seed": seed,
                "residual": residual.tolist(),
                "n_active": n_active,
                "top_10": [
                    {"index": int(idx), "activation": float(val)}
                    for idx, val in zip(top_idx, top_vals, strict=False)
                ],
            }
            results["test_cases"].append(test_case)

            top_feat = f"L{layer}:{int(top_idx[0])}" if len(top_idx) > 0 else "none"
            top_act = f"{float(top_vals[0]):.4f}" if len(top_vals) > 0 else "N/A"
            print(
                f"  seed={seed:4d}: {n_active:6d} active / {n_features} features, "
                f"top={top_feat} ({top_act})"
            )

    out_path = Path(__file__).parent / "plt_gemma_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
