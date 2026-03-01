#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate RWKV-7 (Goose) reference outputs for candle-mi validation.

Self-contained: loads safetensors directly and implements the RWKV-7
forward pass in pure PyTorch (no fla/Triton dependency for inference).

Requires: pip install torch safetensors transformers flash-linear-attention

The fla import is only needed for its tokenizer registration; the actual
forward pass is implemented here to avoid Triton/Windows compatibility issues.

Usage:
    python scripts/rwkv7_validation.py

Outputs scripts/rwkv7_reference.json with top-k logits for the test prompt.

Note on LoRA activations (verified against fla source):
  - w_lora: activation='tanh'    → down → tanh → up(+bias)   ; then .sigmoid() externally
  - a_lora: activation=None      → down → up(+bias)           ; then .sigmoid() externally
  - v_lora: activation=None      → down → up(+bias)           ; then .sigmoid() externally
  - g_lora: activation='sigmoid' → down → sigmoid → up(no bias); NO external sigmoid
"""

import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Import fla to register the tokenizer with transformers
try:
    import fla  # noqa: F401
except ImportError:
    print("ERROR: flash-linear-attention not installed (needed for tokenizer).")
    print("Install: pip install flash-linear-attention")
    sys.exit(1)

from transformers import AutoTokenizer

MODEL_ID = "RWKV/RWKV7-Goose-World3-1.5B-HF"
TEST_PROMPT = "def fibonacci(n):\n    "
TOP_K = 10
NUM_GENERATE = 20


def find_model_path():
    """Find cached model in HF hub cache."""
    import os
    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    model_dir = cache_dir / f"models--{MODEL_ID.replace('/', '--')}"
    if not model_dir.exists():
        print(f"Model not found in cache: {model_dir}")
        print(f"Download with: hf-fetch-model {MODEL_ID}")
        sys.exit(1)
    refs = model_dir / "refs" / "main"
    if refs.exists():
        commit = refs.read_text().strip()
        return model_dir / "snapshots" / commit
    snapshots = list((model_dir / "snapshots").iterdir())
    return snapshots[0]


def l2_norm(x, dim=-1, eps=1e-12):
    """L2 normalize along dim."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


class LoRA:
    """LoRA block: down → middle_activation → up(+optional bias).

    The middle activation is applied BETWEEN down and up projections,
    matching fla's nn.Sequential(Linear, Activation, Linear) layout.
    """
    def __init__(self, down_w, up_w, up_b=None, middle_activation="none"):
        self.down_w = down_w
        self.up_w = up_w
        self.up_b = up_b
        self.middle_activation = middle_activation

    def __call__(self, x):
        h = x @ self.down_w.T
        if self.middle_activation == "tanh":
            h = h.tanh()
        elif self.middle_activation == "sigmoid":
            h = h.sigmoid()
        out = h @ self.up_w.T
        if self.up_b is not None:
            out = out + self.up_b
        return out


def rwkv7_forward(weights, config, token_ids_list):
    """Run RWKV-7 forward pass, return logits for the last token."""
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", 64)
    num_heads = hidden_size // head_dim
    norm_eps = config.get("norm_eps", 1e-5)

    w = weights
    device = w["model.embeddings.weight"].device
    dtype = torch.float32

    # Cast all weights to float32
    for k in w:
        w[k] = w[k].to(dtype)

    input_ids = torch.tensor([token_ids_list], dtype=torch.long, device=device)
    batch, seq_len = input_ids.shape

    # Embedding
    hidden = w["model.embeddings.weight"][input_ids]

    v_first = None

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # Pre-norm (layer 0 only)
        if layer_idx == 0:
            hidden = F.layer_norm(
                hidden, (hidden_size,),
                w[f"{prefix}.pre_norm.weight"],
                w[f"{prefix}.pre_norm.bias"],
                norm_eps
            )

        # ---- Attention ----
        normed = F.layer_norm(
            hidden, (hidden_size,),
            w[f"{prefix}.attn_norm.weight"],
            w[f"{prefix}.attn_norm.bias"],
            norm_eps
        )

        # Token shift: delta = shifted - normed
        shifted = torch.zeros_like(normed)
        shifted[:, 1:, :] = normed[:, :-1, :]
        delta = shifted - normed

        # Static lerp mixing
        ap = f"{prefix}.attn"
        xr = normed + delta * w[f"{ap}.x_r"]
        xw = normed + delta * w[f"{ap}.x_w"]
        xk = normed + delta * w[f"{ap}.x_k"]
        xv = normed + delta * w[f"{ap}.x_v"]
        xa = normed + delta * w[f"{ap}.x_a"]
        xg = normed + delta * w[f"{ap}.x_g"]

        # R, K, V projections
        r = xr @ w[f"{ap}.r_proj.weight"].T
        k = xk @ w[f"{ap}.k_proj.weight"].T
        v = xv @ w[f"{ap}.v_proj.weight"].T

        # Decay: w_lora uses tanh middle activation, then .sigmoid() externally
        w_lora = LoRA(
            w[f"{ap}.w_lora.lora.0.weight"],
            w[f"{ap}.w_lora.lora.2.weight"],
            w.get(f"{ap}.w_lora.lora.2.bias"),
            middle_activation="tanh"
        )
        decay = -0.6065306597126334 * w_lora(xw).sigmoid()

        # Value residual
        if layer_idx == 0:
            v_first = v.clone()
            v_out = v
        else:
            # v_lora uses no middle activation, then .sigmoid() externally
            v_lora = LoRA(
                w[f"{ap}.v_lora.lora.0.weight"],
                w[f"{ap}.v_lora.lora.2.weight"],
                w.get(f"{ap}.v_lora.lora.2.bias"),
                middle_activation="none"
            )
            v_mix = v_lora(xv).sigmoid()
            v_out = torch.lerp(v, v_first, v_mix)

        # Rank-1 gate: a_lora uses no middle activation, then .sigmoid() externally
        a_lora = LoRA(
            w[f"{ap}.a_lora.lora.0.weight"],
            w[f"{ap}.a_lora.lora.2.weight"],
            w.get(f"{ap}.a_lora.lora.2.bias"),
            middle_activation="none"
        )
        a = a_lora(xa).sigmoid()

        # Output gate: g_lora uses sigmoid MIDDLE activation, NO external sigmoid
        g_lora = LoRA(
            w[f"{ap}.g_lora.lora.0.weight"],
            w[f"{ap}.g_lora.lora.2.weight"],
            w.get(f"{ap}.g_lora.lora.2.bias"),
            middle_activation="sigmoid"
        )
        g = g_lora(xg)  # NO .sigmoid() — it's applied inside as middle activation

        # Key normalization
        k_scaled = k * w[f"{ap}.k_k"]
        kk = l2_norm(k_scaled.view(batch, seq_len, num_heads, head_dim), dim=-1)

        # Key modification: k = k * (1 + (a - 1) * k_a)  i.e. k.addcmul(k*(a-1), k_a)
        k_mod = k + k * (a - 1) * w[f"{ap}.k_a"]

        # Reshape for WKV
        r_4d = r.view(batch, seq_len, num_heads, head_dim)
        k_4d = k_mod.view(batch, seq_len, num_heads, head_dim)
        v_4d = v_out.view(batch, seq_len, num_heads, head_dim)
        w_4d = decay.view(batch, seq_len, num_heads, head_dim)
        a_4d = a.view(batch, seq_len, num_heads, head_dim)

        # WKV-7 recurrence (matches fla fused_recurrent_rwkv7_fwd_kernel)
        state = torch.zeros(batch, num_heads, head_dim, head_dim, dtype=dtype, device=device)
        outputs = []

        for t in range(seq_len):
            r_t = r_4d[:, t]      # [b, h, d]
            k_t = k_4d[:, t]
            v_t = v_4d[:, t]
            w_t = w_4d[:, t]
            kk_t = kk[:, t]
            a_t = a_4d[:, t]

            act_a = -kk_t
            b_t = kk_t * a_t

            # S_t = diag(exp(w)) * S + b^T @ (act_a @ S) + k^T @ v
            exp_w = w_t.exp()
            # Term 1: diag(exp(w)) * S
            term1 = state * exp_w.unsqueeze(-1)
            # Term 2: b ⊗ (act_a^T @ S)  (outer product)
            a_times_s = (act_a.unsqueeze(2) @ state)   # [b,h,1,d] @ [b,h,d,d] → [b,h,1,d]
            term2 = b_t.unsqueeze(-1) @ a_times_s      # [b,h,d,1] @ [b,h,1,d] → [b,h,d,d]
            # Term 3: k ⊗ v
            term3 = k_t.unsqueeze(-1) @ v_t.unsqueeze(2)

            state = term1 + term2 + term3

            # y_t = r^T @ S_t
            out_t = (r_t.unsqueeze(2) @ state).squeeze(2)
            outputs.append(out_t)

        out = torch.stack(outputs, dim=1)

        # GroupNorm per head (eps = head_dim * norm_eps, matching fla GroupNorm)
        gn_eps = head_dim * norm_eps
        gn_w = w[f"{ap}.g_norm.weight"]
        gn_b = w[f"{ap}.g_norm.bias"]
        out_flat = out.reshape(batch * seq_len, num_heads, head_dim)
        mean = out_flat.mean(dim=-1, keepdim=True)
        var = out_flat.var(dim=-1, unbiased=False, keepdim=True)
        out_normed = (out_flat - mean) / (var + gn_eps).sqrt()
        out_normed = out_normed.reshape(batch * seq_len, hidden_size)
        out_normed = out_normed * gn_w + gn_b
        out_gn = out_normed.reshape(batch, seq_len, hidden_size)

        # Gate output correction: (gn_out + correction) * g
        r_k = w[f"{ap}.r_k"].view(1, 1, num_heads, head_dim)
        r_4d_flat = r.view(batch, seq_len, num_heads, head_dim)
        k_mod_4d = k_mod.view(batch, seq_len, num_heads, head_dim)
        rkrk = (r_4d_flat * k_mod_4d * r_k).sum(dim=-1, keepdim=True)
        v_4d_corr = v_out.view(batch, seq_len, num_heads, head_dim)
        correction = (rkrk * v_4d_corr).view(batch, seq_len, hidden_size)

        attn_out = ((out_gn + correction) * g) @ w[f"{ap}.o_proj.weight"].T
        hidden = hidden + attn_out

        # ---- FFN ----
        normed_ffn = F.layer_norm(
            hidden, (hidden_size,),
            w[f"{prefix}.ffn_norm.weight"],
            w[f"{prefix}.ffn_norm.bias"],
            norm_eps
        )

        # Token shift for FFN
        shifted_ffn = torch.zeros_like(normed_ffn)
        shifted_ffn[:, 1:, :] = normed_ffn[:, :-1, :]
        delta_ffn = shifted_ffn - normed_ffn

        fp = f"{prefix}.ffn"
        ffn_input = normed_ffn + delta_ffn * w[f"{fp}.x_k"]
        key_out = (ffn_input @ w[f"{fp}.key.weight"].T).relu().square()
        ffn_out = key_out @ w[f"{fp}.value.weight"].T

        hidden = hidden + ffn_out

        if (layer_idx + 1) % 6 == 0:
            print(f"  Layer {layer_idx + 1}/{num_layers} done")

    # Final norm
    hidden = F.layer_norm(
        hidden, (hidden_size,),
        w["model.norm.weight"],
        w["model.norm.bias"],
        norm_eps
    )

    # LM head
    logits = hidden @ w["lm_head.weight"].T
    return logits[0, -1, :]


def main():
    model_path = find_model_path()
    print(f"Model path: {model_path}")

    with open(model_path / "config.json") as f:
        config = json.load(f)
    print(f"Model: {config.get('model_type')} -- "
          f"{config.get('hidden_size')}h, {config.get('num_hidden_layers')}L")

    print("Loading safetensors...")
    weights = load_file(str(model_path / "model.safetensors"))
    print(f"  Loaded {len(weights)} tensors")

    # Use transformers AutoTokenizer (requires fla for registration)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    token_ids = tokenizer.encode(TEST_PROMPT)
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    print(f"\nPrompt: {TEST_PROMPT!r}")
    print(f"Token IDs: {token_ids}")
    print(f"Token strings: {token_strings}")

    print("\nRunning forward pass...")
    logits = rwkv7_forward(weights, config, token_ids)

    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(logits, TOP_K)

    top_predictions = []
    print(f"\nTop {TOP_K} predictions:")
    for i in range(TOP_K):
        tid = top_indices[i].item()
        tok = tokenizer.decode([tid])
        prob = probs[tid].item()
        logit = top_values[i].item()
        top_predictions.append({
            "token_id": tid,
            "token": tok,
            "probability": prob,
            "logit": logit,
        })
        print(f"  {i+1}: id={tid} '{tok}' (logit={logit:.4f}, prob={prob:.6f})")

    top_logit_values = [p["logit"] for p in top_predictions]

    # Autoregressive generation
    print(f"\nGenerating {NUM_GENERATE} tokens...")
    all_ids = list(token_ids)
    for step in range(NUM_GENERATE):
        logits_step = rwkv7_forward(weights, config, all_ids)
        next_id = logits_step.argmax().item()
        all_ids.append(next_id)
        if (step + 1) % 5 == 0:
            print(f"  Generated {step + 1}/{NUM_GENERATE}")

    generated_token_ids = all_ids[len(token_ids):]
    generated_text = tokenizer.decode(generated_token_ids)
    print(f"\nGenerated: {generated_text!r}")

    reference = {
        "model_id": MODEL_ID,
        "test_prompt": TEST_PROMPT,
        "token_ids": token_ids,
        "token_strings": token_strings,
        "top_predictions": top_predictions,
        "top_logit_values": top_logit_values,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
    }

    out_path = Path(__file__).parent / "rwkv7_reference.json"
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
