#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Validate CLT position-sweep results on Llama 3.2 1B against Python/HuggingFace reference.

Reproduces the tragos position-sweep experiment using HuggingFace
transformers + raw CLT encoder/decoder weights, for comparison with
candle-mi's Rust implementation.

Part 1 — Correlational: encode CLT features at every token position,
verify that features and magnitudes vary (position-specificity).

Part 2 — Causal (melometis Version C): inject CLT decoder vectors at
each position across all downstream layers, measure L2 logit distance
to verify planning-site concentration.

Requires: pip install torch transformers safetensors huggingface_hub

Usage:
    python scripts/clt_position_sweep_validation_llama.py

Outputs:
    scripts/clt_position_sweep_reference_llama.json
"""

import json
import platform
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
CLT_REPO = "mntss/clt-llama-3.2-1b-524k"
PROMPT = "Roses are red, violets are blue"
ENCODE_LAYER = 8
TOP_K = 10
INJECTION_STRENGTH = 5.0


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
        gpu_mem = torch.cuda.get_device_properties(0).total_mem
        print(f"GPU memory:   {gpu_mem / 1024**3:.1f} GB")
    print(f"transformers: {transformers.__version__}")
    print()


def main():
    print_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # --- Load model and tokenizer ---
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",  # match Rust manual attention
    )
    model.eval()

    # --- Tokenize (add_special_tokens=True adds BOS for Llama 3.x) ---
    token_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
    seq_len = len(token_ids)
    print(f"Prompt: '{PROMPT}' → {seq_len} tokens: {token_ids}")
    for i, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid])
        print(f"  pos {i}: '{tok_str}' (id={tid})")

    input_ids = torch.tensor([token_ids], device=device)

    # ===================================================================
    # Part 1: Correlational — encode CLT features at every position
    # ===================================================================
    print("\n" + "=" * 60)
    print("Part 1: Position sweep — correlational")
    print("=" * 60)

    # Hook to capture ResidMid at the encode layer.
    # ResidMid = hidden state after attention residual connection, before MLP.
    # In HF Llama, this is the input to post_attention_layernorm.
    resid_mid_captured = {}

    def capture_resid_mid(module, args):
        resid_mid_captured["tensor"] = args[0].detach().clone()

    layer_obj = model.model.layers[ENCODE_LAYER]
    handle = layer_obj.post_attention_layernorm.register_forward_pre_hook(
        capture_resid_mid
    )

    with torch.no_grad():
        outputs = model(input_ids)

    handle.remove()

    resid_mid = resid_mid_captured["tensor"]  # [1, seq_len, d_model]
    print(f"ResidMid shape: {list(resid_mid.shape)}")

    # Load CLT encoder for the encode layer.
    enc_path = hf_hub_download(CLT_REPO, f"W_enc_{ENCODE_LAYER}.safetensors")
    enc_weights = load_file(enc_path)
    w_enc = enc_weights[f"W_enc_{ENCODE_LAYER}"].float().to(device)
    b_enc = enc_weights[f"b_enc_{ENCODE_LAYER}"].float().to(device)
    n_features, d_model = w_enc.shape
    print(
        f"CLT encoder L{ENCODE_LAYER}: W_enc [{n_features}, {d_model}], "
        f"b_enc [{b_enc.shape[0]}]"
    )

    # Sweep: encode at every position.
    per_position_data = []
    for pos in range(seq_len):
        residual = resid_mid[0, pos].float()  # [d_model], cast to F32

        # CLT encoding: pre_acts = W_enc @ residual + b_enc, acts = ReLU(pre_acts)
        pre_acts = w_enc @ residual + b_enc
        acts = F.relu(pre_acts)

        n_active = int((acts > 0).sum())
        top_vals, top_idx = acts.topk(min(TOP_K, max(n_active, 1)))

        top_features = [(int(idx), float(val)) for idx, val in zip(top_idx, top_vals)]
        per_position_data.append(
            {
                "pos": pos,
                "token_id": token_ids[pos],
                "token": tokenizer.decode([token_ids[pos]]),
                "n_active": n_active,
                "top_features": top_features,
            }
        )

        top_str = f"L{ENCODE_LAYER}:{top_features[0][0]}" if top_features else "none"
        top_act = f"{top_features[0][1]:.4f}" if top_features else "N/A"
        tok_str = tokenizer.decode([token_ids[pos]])
        print(
            f"Pos {pos} '{tok_str}': {n_active} active features, "
            f"top: {top_str} (act={top_act})"
        )

    # Summary statistics.
    top1_features = [d["top_features"][0][0] for d in per_position_data]
    unique_top1 = set(top1_features)
    print(
        f"\nUnique top-1 features across {seq_len} positions: "
        f"{len(unique_top1)}/{seq_len}"
    )

    top1_acts = [d["top_features"][0][1] for d in per_position_data]
    act_range = max(top1_acts) - min(top1_acts)
    print(
        f"Top-1 activation range: [{min(top1_acts):.4f}, {max(top1_acts):.4f}], "
        f"spread={act_range:.4f}"
    )

    # Jaccard similarity between first and last position top-k.
    first_ids = set(f[0] for f in per_position_data[0]["top_features"] if f[1] > 0)
    last_ids = set(f[0] for f in per_position_data[-1]["top_features"] if f[1] > 0)
    intersection = len(first_ids & last_ids)
    union = len(first_ids | last_ids)
    jaccard = intersection / union if union > 0 else 1.0
    print(
        f"Jaccard(pos 0, pos {seq_len - 1}): {jaccard:.3f} "
        f"(intersection={intersection}, union={union})"
    )

    # Summary table.
    print(f"\n=== Position Sweep Summary (layer {ENCODE_LAYER}) ===")
    print(
        f"{'Pos':>3}  {'Token':<15}  {'#Active':>8}  "
        f"{'Top1 Feature':>12}  {'Top1 Act':>12}"
    )
    for d in per_position_data:
        f_str = f"L{ENCODE_LAYER}:{d['top_features'][0][0]}"
        print(
            f"{d['pos']:>3}  {d['token']:<15}  {d['n_active']:>8}  "
            f"{f_str:>12}  {d['top_features'][0][1]:>12.4f}"
        )

    # ===================================================================
    # Part 2: Causal — inject and measure L2 logit distance
    # ===================================================================
    print("\n" + "=" * 60)
    print("Part 2: Position sweep — causal (melometis Version C)")
    print("=" * 60)

    # Baseline logits at last position (F32 CPU for precise comparison).
    baseline_logits = outputs.logits[0, -1].float().cpu()  # [vocab_size]

    # Pick top-1 feature at the planning site (last position).
    last_pos_features = per_position_data[-1]["top_features"]
    chosen_feature_idx = last_pos_features[0][0]
    chosen_act = last_pos_features[0][1]
    print(
        f"Chosen feature: L{ENCODE_LAYER}:{chosen_feature_idx} "
        f"(activation={chosen_act:.4f})"
    )

    # Load CLT decoder for the encode layer.
    dec_path = hf_hub_download(CLT_REPO, f"W_dec_{ENCODE_LAYER}.safetensors")
    dec_weights = load_file(dec_path)
    w_dec = dec_weights[f"W_dec_{ENCODE_LAYER}"].float().to(device)
    print(f"CLT decoder L{ENCODE_LAYER}: W_dec {list(w_dec.shape)}")

    n_target_layers = w_dec.shape[1]
    n_total_layers = ENCODE_LAYER + n_target_layers
    print(
        f"Cached {n_target_layers} steering vectors "
        f"(layers {ENCODE_LAYER}..{n_total_layers - 1})"
    )

    # Extract decoder vectors for the chosen feature, cast to model dtype.
    # W_dec[feature_idx] has shape [n_target_layers, d_model].
    decoder_vectors = w_dec[chosen_feature_idx].to(dtype)  # [n_target_layers, d_model]

    # Sweep: inject at each position, measure L2 distance at last position.
    l2_distances = []

    print(f"\n{'Pos':>3}  {'Token':<15}  {'L2 Distance':>12}")
    print(f"{'---':>3}  {'---------------':<15}  {'------------':>12}")

    for pos in range(seq_len):
        hooks = []
        for target_offset in range(n_target_layers):
            target_layer = ENCODE_LAYER + target_offset
            steer_vec = decoder_vectors[target_offset]  # [d_model]

            def make_hook(sv, p):
                def hook_fn(module, _input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        h[0, p, :] += sv * INJECTION_STRENGTH
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        h[0, p, :] += sv * INJECTION_STRENGTH
                        return h

                return hook_fn

            h = model.model.layers[target_layer].register_forward_hook(
                make_hook(steer_vec, pos)
            )
            hooks.append(h)

        with torch.no_grad():
            injected_outputs = model(input_ids)

        for h in hooks:
            h.remove()

        injected_logits = injected_outputs.logits[0, -1].float().cpu()
        l2 = torch.sqrt(((baseline_logits - injected_logits) ** 2).sum()).item()
        l2_distances.append(l2)

        tok_str = tokenizer.decode([token_ids[pos]])
        print(f"{pos:>3}  {tok_str:<15}  {l2:>12.4f}")

    # Analysis.
    max_l2 = max(l2_distances)
    max_pos = l2_distances.index(max_l2)
    last_l2 = l2_distances[-1]

    sorted_by_l2 = sorted(enumerate(l2_distances), key=lambda x: -x[1])
    last_rank = next(
        i for i, (p, _) in enumerate(sorted_by_l2) if p == seq_len - 1
    )

    sorted_l2 = sorted(l2_distances)
    median_l2 = sorted_l2[seq_len // 2]
    concentration = last_l2 / median_l2 if median_l2 > 0 else float("inf")

    print(f"\nMax L2: {max_l2:.4f} at position {max_pos}")
    print(f"Last-position L2: {last_l2:.4f}")
    print(
        f"Last position rank: {last_rank + 1} out of {seq_len} "
        f"(top-3: {sorted_by_l2[:3]})"
    )
    print(f"Concentration ratio (last/median): {concentration:.2f}x")

    # Top-5 predictions: baseline vs best injection.
    def print_top5(label, logits_tensor):
        top_vals, top_idx = logits_tensor.topk(5)
        print(f"{label} top-5:")
        for rank, (idx, val) in enumerate(zip(top_idx, top_vals)):
            tok = tokenizer.decode([int(idx)])
            print(f"  {rank + 1}: '{tok}' (logit={float(val):.4f})")

    print_top5("Baseline", baseline_logits)

    # Re-run at the max-effect position for comparison.
    hooks = []
    for target_offset in range(n_target_layers):
        target_layer = ENCODE_LAYER + target_offset
        steer_vec = decoder_vectors[target_offset]

        def make_hook_best(sv, p):
            def hook_fn(module, _input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, p, :] += sv * INJECTION_STRENGTH
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, p, :] += sv * INJECTION_STRENGTH
                    return h

            return hook_fn

        h = model.model.layers[target_layer].register_forward_hook(
            make_hook_best(steer_vec, max_pos)
        )
        hooks.append(h)

    with torch.no_grad():
        best_outputs = model(input_ids)

    for h in hooks:
        h.remove()

    best_logits = best_outputs.logits[0, -1].float().cpu()
    print_top5(f"Injected (pos={max_pos})", best_logits)

    # ===================================================================
    # Save reference JSON
    # ===================================================================
    reference = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__,
        },
        "model_id": MODEL_ID,
        "clt_repo": CLT_REPO,
        "prompt": PROMPT,
        "token_ids": token_ids,
        "encode_layer": ENCODE_LAYER,
        "correlational": {
            "positions": per_position_data,
            "unique_top1_count": len(unique_top1),
            "jaccard_first_last": jaccard,
        },
        "causal": {
            "chosen_feature": chosen_feature_idx,
            "chosen_activation": chosen_act,
            "injection_strength": INJECTION_STRENGTH,
            "l2_distances": [
                {
                    "pos": i,
                    "token": tokenizer.decode([token_ids[i]]),
                    "l2": l2,
                }
                for i, l2 in enumerate(l2_distances)
            ],
            "max_l2": max_l2,
            "max_pos": max_pos,
            "last_l2": last_l2,
            "last_rank": last_rank + 1,
            "concentration_ratio": concentration,
        },
    }

    out_path = Path(__file__).parent / "clt_position_sweep_reference_llama.json"
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
