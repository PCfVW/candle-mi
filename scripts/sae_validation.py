#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Validate SAE encoding against Python/SAELens reference.

Loads Gemma 2 2B, runs a forward pass capturing residual stream activations,
then encodes through a Gemma Scope SAE and saves reference outputs for
comparison with candle-mi's Rust implementation.

Requires: pip install torch transformers sae-lens

Usage:
    python scripts/sae_validation.py

Outputs:
    scripts/sae_reference.json
"""

import json
import platform
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-2-2b"
PROMPT = "The capital of France is"
HOOK_LAYER = 0  # Layer for SAE encoding
TOP_K = 10  # Number of top features to report


def print_environment():
    """Print version info for reproducibility."""
    print("=== Environment ===")
    print(f"Python:       {sys.version}")
    print(f"Platform:     {platform.platform()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"transformers: {transformers.__version__}")
    print()


def main():
    print_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Match candle-mi F32

    # --- Load model ---
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()

    # --- Tokenize ---
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    print(f"Tokens ({len(tokens)}): {tokens}")

    # --- Forward pass with hidden state capture ---
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    # Hidden states: tuple of (n_layers + 1) tensors [batch, seq, d_model]
    # Index 0 = embedding, index i = output of layer i-1
    # resid_post at layer L = hidden_states[L + 1]
    hidden_states = outputs.hidden_states
    resid_post = hidden_states[HOOK_LAYER + 1]  # [1, seq, d_model]
    print(f"resid_post shape: {resid_post.shape}")
    print(f"resid_post dtype: {resid_post.dtype}")

    # --- Load SAE via SAELens ---
    print("\nLoading SAE via SAELens...")
    try:
        from sae_lens import SAE

        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-2b-res-jb",
            sae_id=f"blocks.{HOOK_LAYER}.hook_resid_post",
            device=device,
        )
    except ImportError:
        print("ERROR: sae-lens not installed. Run: pip install sae-lens")
        sys.exit(1)

    sae.eval()
    print(f"SAE d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    print(f"SAE architecture: {sae.cfg.architecture}")
    print(f"SAE hook: {sae.cfg.hook_name}")

    # --- Encode activations ---
    with torch.no_grad():
        # SAELens encode expects [batch, seq, d_in]
        encoded = sae.encode(resid_post.float())  # [1, seq, d_sae]
        decoded = sae.decode(encoded)  # [1, seq, d_in]
        reconstructed = sae.forward(resid_post.float())  # full forward

    # --- Compute metrics ---
    mse = torch.mean((resid_post.float() - decoded) ** 2).item()
    print(f"\nReconstruction MSE: {mse:.6f}")

    # --- Per-position top features (last token) ---
    last_pos = encoded.shape[1] - 1
    last_encoded = encoded[0, last_pos]  # [d_sae]
    nonzero_mask = last_encoded > 0
    n_active = nonzero_mask.sum().item()
    print(f"Active features at last position: {n_active}")

    # Top-k features
    top_vals, top_idxs = torch.topk(last_encoded, min(TOP_K, int(n_active)))
    top_features = [
        {"index": int(idx), "value": float(val)}
        for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
    ]
    print(f"Top-{TOP_K} features: {top_features}")

    # --- Save reference ---
    # Save activations at last position for numerical comparison
    resid_last = resid_post[0, last_pos].cpu().tolist()  # [d_model]
    encoded_last = encoded[0, last_pos].cpu().tolist()  # [d_sae]
    decoded_last = decoded[0, last_pos].cpu().tolist()  # [d_model]

    # Also save first few values for spot-checking
    reference = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "hook_layer": HOOK_LAYER,
        "hook_name": f"blocks.{HOOK_LAYER}.hook_resid_post",
        "sae_release": "gemma-2b-res-jb",
        "sae_id": f"blocks.{HOOK_LAYER}.hook_resid_post",
        "d_in": sae.cfg.d_in,
        "d_sae": sae.cfg.d_sae,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "reconstruction_mse": mse,
        "n_active_last_pos": int(n_active),
        "top_features_last_pos": top_features,
        "resid_last_first10": resid_last[:10],
        "encoded_last_first10": encoded_last[:10],
        "decoded_last_first10": decoded_last[:10],
        "resid_last_norm": float(torch.norm(resid_post[0, last_pos]).item()),
        "encoded_last_norm": float(torch.norm(encoded[0, last_pos]).item()),
        "decoded_last_norm": float(torch.norm(decoded[0, last_pos]).item()),
    }

    out_path = Path(__file__).parent / "sae_reference.json"
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference to {out_path}")


if __name__ == "__main__":
    main()
