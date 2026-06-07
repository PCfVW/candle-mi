#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Mistral-7B-v0.1 forward-pass reference for Rust validation.

From-first-principles forward-pass oracle for the candle-mi `Mistral`
transformer arm.  Mistral is `LLaMA`-like (GQA, SiLU, RmsNorm, separate
lm_head) plus **sliding-window attention** (window 4096 on every layer);
no soft-capping, so the default attention backend is fine.

Computed in **F32 on CPU** — the research-grade reference.  This is the
single ground-truth oracle consumed by all three Mistral tests in
``tests/validate_mistral_7b_forward.rs`` (7B F32 is ~27 GiB, larger than a
16 GiB GPU, which is why Mistral is the one family worth three tiers):

- **F32 CPU** — exact, `<1e-3` (same tier as the other families);
- **F32 GPU** — exact, `<5e-3`; F32 7B exceeds 16 GiB VRAM, so it runs via
  CUDA memory oversubscription (weights spill to host RAM) — slower but exact;
- **BF16 GPU** — fast and fully GPU-resident (~13.5 GiB); BF16 carries ~3
  significant figures, so the bar is looser (`<0.1`, observed ~5e-2).

All three were run and pass; keeping all three documents that both GPU
strategies (fast/resident and exact/oversubscribed) succeed.

Saves per prompt **(a)** top-10 next-token logits + indices and **(b)** the
post-final-norm last-token residual.

NOTE: F32 7B needs ~27 GiB RAM. Subsequent runs are cache hits.

Dependencies: ``torch``, ``transformers``, ``safetensors``.

Usage:
    python scripts/mistral_7b_validation.py

Output:
    scripts/mistral_7b_forward_reference.json

Requires ``mistralai/Mistral-7B-v0.1`` (gated; ~13.5 GiB bf16) cached.
"""

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "mistralai/Mistral-7B-v0.1"
TEST_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
]
TOP_K = 10


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    import transformers as hf_transformers

    print(f"Mistral 7B forward-pass reference generation for {MODEL_REPO}")
    print(f"  {len(TEST_PROMPTS)} prompts, top-{TOP_K} logits per prompt")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")
    print()

    print("Loading model + tokenizer in F32 on CPU (~27 GiB RAM) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, head_dim={head_dim}, "
        f"num_kv_heads={cfg.num_key_value_heads}, sliding_window={getattr(cfg, 'sliding_window', None)}"
    )
    print()

    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "from-first-principles forward-pass oracle "
        "(transformers.AutoModelForCausalLM, F32 CPU); Mistral sliding-window arm",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
        "platform": platform.platform(),
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "head_dim": head_dim,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
        "sliding_window": getattr(cfg, "sliding_window", None),
        "test_cases": [],
    }

    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            tokens = input_ids[0].tolist()

            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            last_logits = outputs.logits[0, -1, :].float()
            top_vals, top_idx = last_logits.topk(TOP_K)
            final_hidden = outputs.hidden_states[-1]
            last_residual = final_hidden[0, -1, :].float().tolist()

            top_token_str = tokenizer.decode([int(top_idx[0])])
            print(
                f"  prompt={prompt!r}: {len(tokens)} tokens, "
                f"top1=({int(top_idx[0])}, {top_token_str!r}, {float(top_vals[0]):.4f})"
            )

            results["test_cases"].append(
                {
                    "prompt": prompt,
                    "tokens": tokens,
                    "top_10": [
                        {"index": int(idx), "logit": float(val)}
                        for idx, val in zip(top_idx, top_vals, strict=False)
                    ],
                    "last_residual_f32": last_residual,
                }
            )

    out_path = Path(__file__).parent / "mistral_7b_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
