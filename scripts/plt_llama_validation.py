#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Llama 3.2 1B PLT reference encodings for Rust validation.

From-first-principles encoder oracle for the candle-mi PLT loader: loads
raw `layer_{L}.safetensors` bundles from `mntss/transcoder-Llama-3.2-1B`
via `huggingface_hub` + `safetensors.torch` directly (NO circuit-tracer),
applies the PLT encoder formula ``ReLU(W_enc @ residual + b_enc)`` in
torch on CPU, and saves top-10 activations to JSON for cross-validation
with the Rust implementation in `src/clt/mod.rs`.

Methodology mirrors plip-rs/scripts/clt_reference.py, adapted for the
PltBundle schema:

- File name: ``layer_{L}.safetensors`` (one bundle per layer) instead of
  ``W_enc_{L}.safetensors`` (CLT split-file convention).
- Tensor names: un-suffixed ``W_enc`` / ``b_enc`` (PltBundle convention)
  instead of ``W_enc_{L}`` / ``b_enc_{L}`` (CLT).
- Bundle additionally contains ``W_dec`` (rank-2 for PltBundle),
  ``W_skip`` (Llama PLT linear skip path), and ``b_dec``. The encoder
  oracle loads them for shape logging but does not use them.

The reference JSON is consumed by ``tests/validate_plt.rs`` (V3 Step 1.5).
Acceptance bar: top-10 feature indices match exactly, activation
magnitudes within abs-diff < 1e-4 (F32, CPU vs CPU).

Dependencies: ``torch``, ``safetensors``, ``huggingface_hub``.

Usage:
    python scripts/plt_llama_validation.py

Output:
    scripts/plt_llama_reference.json
"""

import json
import os
import platform
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

PLT_REPO = "mntss/transcoder-Llama-3.2-1B"
# Ends + middle of Llama 3.2 1B's 16-layer stack, mirroring plip-rs's
# [0, 12, 25] choice for Gemma 2 2B's 26-layer stack.
TEST_LAYERS = [0, 7, 15]
N_SEEDS_PER_LAYER = 3
TOP_K = 10


def main() -> None:
    # Determinism — CPU-only script so CUBLAS config is a no-op but set anyway
    # per the v0.1.9 roadmap V3 Step 1.4 spec.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)

    print(f"PLT reference generation for {PLT_REPO}")
    print(f"Test layers: {TEST_LAYERS}, seeds per layer: {N_SEEDS_PER_LAYER}")
    print(f"torch {torch.__version__} on {platform.platform()}")
    print()

    results: dict = {
        "plt_repo": PLT_REPO,
        "methodology": "from-first-principles encoder oracle (no circuit-tracer)",
        "schema": "PltBundle",
        "encoder_formula": "ReLU(W_enc @ residual + b_enc)",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "d_model": None,
        "n_features_per_layer": None,
        "test_cases": [],
    }

    for layer in TEST_LAYERS:
        # Download the per-layer bundle (safetensors cache hit if already fetched).
        bundle_path = hf_hub_download(PLT_REPO, f"layer_{layer}.safetensors")
        weights = load_file(bundle_path)

        # Un-suffixed tensor names (PltBundle schema).
        w_enc = weights["W_enc"].float()  # [n_features, d_model]
        b_enc = weights["b_enc"].float()  # [n_features]
        # Logged for completeness; encoder oracle does not use them.
        w_dec = weights["W_dec"]  # rank-2 [n_features, d_model] for PltBundle
        w_skip = weights.get("W_skip")  # [d_model, d_model] (Llama PLT only)
        b_dec = weights["b_dec"]  # [d_model]

        n_features, d_model = w_enc.shape
        assert b_enc.shape == (n_features,), f"b_enc shape {tuple(b_enc.shape)}"
        assert w_dec.shape == (n_features, d_model), (
            f"W_dec shape {tuple(w_dec.shape)} — expected rank-2 "
            f"[n_features, d_model] for PltBundle"
        )
        if w_skip is not None:
            assert w_skip.shape == (d_model, d_model), (
                f"W_skip shape {tuple(w_skip.shape)}"
            )
        assert b_dec.shape == (d_model,), f"b_dec shape {tuple(b_dec.shape)}"

        print(
            f"Layer {layer}: W_enc [{n_features}, {d_model}], "
            f"b_enc [{b_enc.shape[0]}], "
            f"W_dec [{', '.join(str(d) for d in w_dec.shape)}], "
            f"W_skip {'present' if w_skip is not None else 'absent'}"
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

            # Llama PLT encoder formula (un-suffixed W_enc, plain ReLU).
            # GemmaScope variant (v0.1.10): acts = pre_acts * (pre_acts > threshold).
            pre_acts = w_enc @ residual + b_enc
            acts = torch.relu(pre_acts)

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

            top_feat = (
                f"L{layer}:{int(top_idx[0])}" if len(top_idx) > 0 else "none"
            )
            top_act = f"{float(top_vals[0]):.4f}" if len(top_vals) > 0 else "N/A"
            print(
                f"  seed={seed:4d}: {n_active:6d} active / {n_features} features, "
                f"top={top_feat} ({top_act})"
            )

    out_path = Path(__file__).parent / "plt_llama_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(
        f"\nSaved {n_cases} test cases to {out_path} "
        f"({file_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
