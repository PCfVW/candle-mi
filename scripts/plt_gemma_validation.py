#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Gemma 2 2B GemmaScope PLT reference encodings for Rust validation.

From-first-principles encoder oracle for the candle-mi GemmaScope loader:
fetches `config.yaml` from `mntss/gemma-scope-transcoders` (the curation
entry-point) to discover the per-layer NPZ paths, downloads the 3 layer
NPZs from `google/gemma-scope-2b-pt-transcoders` (the actual weights repo),
applies the GemmaScope encoder formula
``pre = W_enc.T @ residual + b_enc; acts = pre * (pre > threshold)`` in
torch on CPU, and saves top-10 activations to JSON for cross-validation
with the Rust implementation in `src/clt/mod.rs`.

The methodology mirrors `plt_llama_validation.py` (V3 Step 1.4) for the
Llama PLT arm, adapted for the GemmaScopeNpz schema:

- Two-repo flow: curation YAML on mntss, weights on google/. The
  curation YAML lists 26 paths, one per layer of Gemma 2 2B.
- File format: `.npz` (NumPy archive) instead of safetensors. Each NPZ
  contains `W_enc`, `W_dec`, `b_enc`, `b_dec`, `threshold`. No `W_skip`.
- W_enc transpose: GemmaScope stores `W_enc` as
  `[d_model, n_features] = [2304, 16384]` on disk — transposed vs the
  Llama PLT convention `[n_features, d_model]`. The oracle applies `.T`
  to canonicalise the orientation, matching circuit-tracer's
  `load_gemma_scope_transcoder()` reference loader.
- JumpReLU activation: ``acts = pre * (pre > threshold)`` element-wise
  with a per-feature `threshold [n_features]` tensor, instead of plain
  `ReLU` (Llama PLT) or `ReLU` after CLT decoder skip.

Test layers `{0, 12, 25}` cover the ends and middle of Gemma 2 2B's
26-layer stack. Three random seeds per layer, deterministic via
`torch.manual_seed(seed * 100 + layer)`, total 9 test cases.

The reference JSON is consumed by `tests/validate_plt_gemma.rs`
(V3 Step 1.6 / Phase A.7). Acceptance bar: top-10 feature indices match
exactly, activation magnitudes within abs-diff < 1e-4 (F32, CPU vs CPU).

Dependencies: `torch`, `numpy`, `huggingface_hub`, `pyyaml`.

Usage:
    python scripts/plt_gemma_validation.py

Output:
    scripts/plt_gemma_reference.json
"""

import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

CURATION_REPO = "mntss/gemma-scope-transcoders"
WEIGHTS_REPO = "google/gemma-scope-2b-pt-transcoders"
# Ends + middle of Gemma 2 2B's 26-layer stack, mirroring plip-rs's [0, 12, 25]
# choice. Llama 3.2 1B used [0, 7, 15] for its 16-layer stack.
TEST_LAYERS = [0, 12, 25]
N_SEEDS_PER_LAYER = 3
TOP_K = 10


def main() -> None:
    # Determinism — CPU-only script so CUBLAS config is a no-op but set anyway
    # per the V3 Step 1.4 / 1.6 spec.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)

    print(f"GemmaScope PLT reference generation for {WEIGHTS_REPO}")
    print(f"Curation: {CURATION_REPO}/config.yaml")
    print(f"Test layers: {TEST_LAYERS}, seeds per layer: {N_SEEDS_PER_LAYER}")
    print(f"torch {torch.__version__} on {platform.platform()}")
    print()

    # Two-repo flow step 1: fetch curation YAML and parse the transcoders list.
    yaml_path = hf_hub_download(CURATION_REPO, "config.yaml")
    with open(yaml_path) as f:
        curation = yaml.safe_load(f)
    transcoders_urls: list[str] = curation["transcoders"]
    assert len(transcoders_urls) == 26, (
        f"expected 26 entries (one per Gemma 2 2B layer), got {len(transcoders_urls)}"
    )

    # Map "layer_N" → relative NPZ path inside WEIGHTS_REPO.
    layer_to_relpath: dict[int, str] = {}
    for url in transcoders_urls:
        # Strip "hf://" + WEIGHTS_REPO + "/".
        prefix = f"hf://{WEIGHTS_REPO}/"
        assert url.startswith(prefix), f"unexpected URL prefix: {url}"
        relpath = url.removeprefix(prefix)
        # First path segment is "layer_N".
        layer_id = int(relpath.split("/")[0].removeprefix("layer_"))
        layer_to_relpath[layer_id] = relpath

    results: dict = {
        "weights_repo": WEIGHTS_REPO,
        "curation_repo": CURATION_REPO,
        "methodology": "from-first-principles encoder oracle (no circuit-tracer)",
        "schema": "GemmaScopeNpz",
        "encoder_formula": "pre = W_enc.T @ residual + b_enc; acts = pre * (pre > threshold)",
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "d_model": None,
        "n_features_per_layer": None,
        "test_cases": [],
    }

    for layer in TEST_LAYERS:
        relpath = layer_to_relpath[layer]
        # Download the layer's NPZ (cache hit if already fetched).
        npz_path = hf_hub_download(WEIGHTS_REPO, relpath)
        params = np.load(npz_path)

        # GemmaScope on-disk shapes:
        #   W_enc:     [d_model, n_features]   (transposed vs PltBundle)
        #   W_dec:     [n_features, d_model]
        #   b_enc:     [n_features]
        #   b_dec:     [d_model]
        #   threshold: [n_features]
        w_enc_disk = torch.from_numpy(params["W_enc"]).float()
        b_enc = torch.from_numpy(params["b_enc"]).float()
        threshold = torch.from_numpy(params["threshold"]).float()
        # Logged for completeness; encoder oracle does not use them.
        w_dec = params["W_dec"]
        b_dec = params["b_dec"]

        d_model_disk, n_features = w_enc_disk.shape
        # Transpose to canonical [n_features, d_model] orientation, matching
        # circuit-tracer's load_gemma_scope_transcoder() and candle-mi's
        # in-memory LoadedEncoder layout.
        w_enc = w_enc_disk.T.contiguous()
        d_model = d_model_disk
        assert w_enc.shape == (n_features, d_model)
        assert b_enc.shape == (n_features,), f"b_enc shape {tuple(b_enc.shape)}"
        assert threshold.shape == (n_features,), (
            f"threshold shape {tuple(threshold.shape)}"
        )
        assert w_dec.shape == (n_features, d_model), (
            f"W_dec shape {w_dec.shape}"
        )
        assert b_dec.shape == (d_model,), f"b_dec shape {b_dec.shape}"
        assert "W_skip" not in params.files, (
            "GemmaScope is a pure JumpReLU transcoder; W_skip should not be present"
        )

        print(
            f"Layer {layer} ({relpath}): W_enc [{d_model}, {n_features}] -> "
            f"transposed to [{n_features}, {d_model}], "
            f"threshold [{threshold.shape[0]}], W_skip absent"
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

            # GemmaScope encoder formula. JumpReLU: pre * (pre > threshold)
            # element-wise. The Llama PLT analog uses plain ReLU instead.
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

    out_path = Path(__file__).parent / "plt_gemma_reference.json"
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
