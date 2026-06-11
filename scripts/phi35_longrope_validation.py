#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate a Phi-3.5-mini (longrope) forward-pass reference for Rust validation.

Oracle for candle-mi's **longrope** `rope_scaling` arm. Phi-3.5-mini uses the
`longrope` scheme: per-dimension `short_factor`/`long_factor` arrays divide the
RoPE inverse frequencies, plus an `attention_factor` (mscale) that scales the
cos/sin. For sequences ≤ `original_max_position_embeddings` (4096) the **short**
factors apply; beyond, the **long** factors.

Uses the **non-quantized** `microsoft/Phi-3.5-mini-instruct` (the original
checkpoint leaves `attention_factor` unset → HF derives the paper value
~1.19, so the model behaves sanely — unlike the cached unsloth bnb-4bit re-save,
which hard-codes `attention_factor: 32` and is too chaotic for exact parity).
This gives clean exact forward parity, matching the other validated families.

Computed in **F32**.  Records, alongside the forward oracle, the **ground-truth
rope parameters** the loaded model uses (`attention_scaling`, `inv_freq`) so the
Rust side can assert it reproduces them exactly.

Dependencies: torch, transformers, accelerate.

Usage:    python scripts/phi35_longrope_validation.py
Output:   scripts/phi35_longrope_forward_reference.json
"""

import json
import platform
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_REPO = "microsoft/Phi-3.5-mini-instruct"
TEST_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
]
TOP_K = 10


def main() -> None:
    torch.manual_seed(0)
    import transformers as hf_transformers

    # F32 ~15 GiB > 16 GiB VRAM once the long-context activations are added, so
    # let accelerate place/offload layers across GPU+CPU. F32 compute throughout.
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"Phi-3.5 longrope forward-pass reference for {MODEL_REPO} (F32, device_map={device_map})")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, device_map=device_map, dtype=torch.float32
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    re = model.model.rotary_emb
    attention_scaling = float(re.attention_scaling)
    inv_freq = [float(x) for x in re.inv_freq.flatten().tolist()]
    print(f"  attention_scaling={attention_scaling}, inv_freq len={len(inv_freq)}")
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, head_dim={head_dim}"
    )

    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "Phi-3.5 longrope forward-pass oracle (transformers, non-quantized, F32); "
        "validates candle-mi longrope RoPE (short + long regimes) against the original checkpoint",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
        "platform": platform.platform(),
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "head_dim": head_dim,
        "rope_theta": getattr(cfg, "rope_theta", None),
        "original_max_position_embeddings": getattr(cfg, "original_max_position_embeddings", None),
        # Ground-truth RoPE parameters the loaded model uses (short-factor regime):
        "attention_scaling": attention_scaling,
        "inv_freq": inv_freq,
        "test_cases": [],
    }

    # Build a >4096-token prompt to exercise the **long_factor** regime (the
    # short prompts above stay in the short_factor regime). Repeat a passage,
    # then trim to a target just over original_max_position_embeddings.
    orig_max = getattr(cfg, "original_max_position_embeddings", 4096)
    passage = (
        "In a distant land beyond the mountains, the seasons turned slowly and "
        "the rivers carried stories from one village to the next, year after year. "
    )
    long_ids = tokenizer(passage * 400, return_tensors="pt").input_ids[0].tolist()
    target_len = orig_max + 64  # comfortably past the short/long boundary
    long_ids = long_ids[:target_len]
    cases = [(p, None) for p in TEST_PROMPTS] + [("<long-context>", long_ids)]

    with torch.no_grad():
        for prompt, preset_ids in cases:
            if preset_ids is None:
                inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
                input_ids = inputs.input_ids
            else:
                input_ids = torch.tensor([preset_ids], device=input_device)
            tokens = input_ids[0].tolist()
            outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
            last_logits = outputs.logits[0, -1, :].float().cpu()
            top_vals, top_idx = last_logits.topk(TOP_K)
            regime = "long" if len(tokens) > orig_max else "short"
            print(
                f"  prompt={prompt!r}: {len(tokens)} tokens [{regime}], "
                f"top1=({int(top_idx[0])}, {tokenizer.decode([int(top_idx[0])])!r}, {float(top_vals[0]):.4f})"
            )
            results["test_cases"].append(
                {
                    "prompt": prompt,
                    "tokens": tokens,
                    "regime": regime,
                    "top_10": [
                        {"index": int(idx), "logit": float(val)}
                        for idx, val in zip(top_idx, top_vals, strict=False)
                    ],
                }
            )

    out_path = Path(__file__).parent / "phi35_longrope_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results['test_cases'])} test cases to {out_path}")


if __name__ == "__main__":
    main()
