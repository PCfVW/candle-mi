#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Qwen3 1.7B BlueLightAI CLT reference encodings for Rust validation.

From-first-principles encoder oracle for the candle-mi BlueLightAI CLT
loader: loads raw ``W_enc_{L}.safetensors`` files from
``bluelightai/clt-qwen3-1.7b-base-20k`` via ``huggingface_hub`` +
``safetensors.torch`` directly (NO circuit-tracer), applies the JumpReLU
encoder formula ``pre = W_enc @ residual + b_enc; acts = pre * (pre > threshold)``
in torch on CPU, and saves top-10 activations to JSON for cross-validation
with the Rust implementation in ``src/clt/mod.rs``.

Methodology mirrors ``plt_gemma_validation.py`` (GemmaScope JumpReLU
v0.1.10) for the activation formula and ``plt_llama_validation.py``
(PltBundle v0.1.9) for the file-load path, adapted for the new
``CltSplitJumpReLU`` schema:

- File layout: ``W_enc_{L}.safetensors`` + ``W_dec_{L}.safetensors`` per
  layer (mntss ``CltSplit`` convention), BF16 SafeTensors. Single-repo
  flow — no curation YAML, unlike ``GemmaScopeNpz``.
- Tensor names inside ``W_enc_{L}.safetensors`` are **layer-suffixed**
  (``W_enc_{L}``, ``b_enc_{L}``, ``b_dec_{L}``, ``threshold_{L}``),
  matching mntss ``CltSplit`` style. ``b_dec`` lives in the encoder
  file (BlueLightAI-specific; mntss ``CltSplit`` keeps ``b_dec`` with
  ``W_dec``).
- Encoder orientation: ``W_enc`` is ``[n_features, d_model] = [20480, 2048]``
  on disk — canonical, no transpose needed (contrast with ``GemmaScopeNpz``
  which stores ``[d_model, n_features]`` and requires ``.T`` on load).
- Activation: JumpReLU ``pre * (pre > threshold)`` element-wise with a
  per-feature ``threshold [n_features]`` tensor, identical to
  ``GemmaScopeNpz``.

Decoder ``W_dec_{L}.safetensors`` files are **not downloaded** by this
oracle. Their rank-3 ``[n_features, n_target_layers_L, d_model]``
cross-layer structure was verified at the format-discovery step via
``hf-fm inspect`` (Path A confirmation; see
``Rebuttal/bluelightai-loader-scope.md`` §"Format discovery results
(2026-05-22) — Path A confirmed"). Downloading all 3 test-layer decoder
files would cost ~3.4 GiB for shape-only validation; the encoder oracle
on its own exercises the JumpReLU branch the Rust loader needs.

Test layers ``{0, 13, 27}`` cover the ends + middle of Qwen3 1.7B Base's
28-layer stack, mirroring ``[0, 12, 25]`` for Gemma 2 2B (26 layers) and
``[0, 7, 15]`` for Llama 3.2 1B (16 layers). Three random seeds per
layer, deterministic via ``torch.manual_seed(seed_idx * 100 + layer)``,
total 9 test cases.

The reference JSON is consumed by ``tests/validate_clt_qwen3.rs``
(v0.1.11 Path A loader work). Acceptance bar: top-10 feature indices
match exactly, activation magnitudes within abs-diff < 1e-4 (F32, CPU vs
CPU).

Dependencies: ``torch``, ``safetensors``, ``huggingface_hub``.

Usage:
    python scripts/clt_qwen3_validation.py

Output:
    scripts/clt_qwen3_reference.json
"""

import json
import os
import platform
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

CLT_REPO = "bluelightai/clt-qwen3-1.7b-base-20k"
# Ends + middle of Qwen3 1.7B Base's 28-layer stack, mirroring
# `plt_gemma_validation.py`'s [0, 12, 25] for Gemma 2 2B (26 layers) and
# `plt_llama_validation.py`'s [0, 7, 15] for Llama 3.2 1B (16 layers).
TEST_LAYERS = [0, 13, 27]
N_SEEDS_PER_LAYER = 3
TOP_K = 10


def main() -> None:
    # Determinism — CPU-only script so CUBLAS config is a no-op but set anyway
    # per the v0.1.9 / v0.1.10 oracle template.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)

    print(f"BlueLightAI CLT reference generation for {CLT_REPO}")
    print(f"Test layers: {TEST_LAYERS}, seeds per layer: {N_SEEDS_PER_LAYER}")
    print(f"torch {torch.__version__} on {platform.platform()}")
    print()

    results: dict = {
        "clt_repo": CLT_REPO,
        "methodology": "from-first-principles encoder oracle (no circuit-tracer)",
        "schema": "CltSplitJumpReLU",
        "encoder_formula": "pre = W_enc @ residual + b_enc; acts = pre * (pre > threshold)",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "d_model": None,
        "n_features_per_layer": None,
        "test_cases": [],
    }

    for layer in TEST_LAYERS:
        # Download the encoder file (safetensors cache hit if already fetched).
        # Decoder file `W_dec_{layer}.safetensors` is intentionally NOT
        # downloaded — see module docstring.
        enc_path = hf_hub_download(CLT_REPO, f"W_enc_{layer}.safetensors")
        weights = load_file(enc_path)

        # Layer-suffixed tensor names (CltSplitJumpReLU = mntss CltSplit
        # naming convention + GemmaScope JumpReLU semantics).
        w_enc = weights[f"W_enc_{layer}"].float()
        b_enc = weights[f"b_enc_{layer}"].float()
        threshold = weights[f"threshold_{layer}"].float()
        # Logged for completeness; encoder oracle does not use them.
        # `b_dec` lives in the encoder file for BlueLightAI (not the decoder
        # file), unlike mntss CltSplit.
        b_dec = weights[f"b_dec_{layer}"]

        n_features, d_model = w_enc.shape
        assert b_enc.shape == (n_features,), f"b_enc shape {tuple(b_enc.shape)}"
        assert threshold.shape == (n_features,), (
            f"threshold shape {tuple(threshold.shape)}"
        )
        assert b_dec.shape == (d_model,), f"b_dec shape {tuple(b_dec.shape)}"
        assert f"W_skip_{layer}" not in weights, (
            "BlueLightAI CLT is a pure JumpReLU transcoder; "
            "W_skip should not be present"
        )

        print(
            f"Layer {layer}: W_enc [{n_features}, {d_model}], "
            f"b_enc [{b_enc.shape[0]}], "
            f"threshold [{threshold.shape[0]}], "
            f"b_dec [{b_dec.shape[0]}] (in encoder file), "
            f"W_skip absent"
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

            # BlueLightAI CLT encoder formula (JumpReLU, same as GemmaScope).
            # Llama PLT analog uses plain `torch.relu(pre_acts)` instead.
            pre_acts = w_enc @ residual + b_enc
            mask = (pre_acts > threshold).float()
            acts = pre_acts * mask

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

    out_path = Path(__file__).parent / "clt_qwen3_reference.json"
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
