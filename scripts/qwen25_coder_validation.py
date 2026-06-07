#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Qwen2.5-Coder-3B-Instruct forward-pass reference for Rust validation.

From-first-principles forward-pass oracle for the candle-mi `Qwen2`
transformer arm.  Qwen 2/2.5 is `LLaMA`-like with one distinguishing trait:
**QKV bias** (`attention_bias: true` → bias on the Q, K, V projections, but
not `o_proj`).  No soft-capping, so the default attention backend is fine
(unlike Gemma 2, which requires eager).

Loads ``Qwen/Qwen2.5-Coder-3B-Instruct`` via HuggingFace ``transformers`` in
``F32`` on CPU, runs ``forward()`` on a small set of fixed prompts, and saves
**(a)** top-10 next-token logits + indices and
**(b)** the final-layer last-token residual (post-final-norm,
pre-LM-head) per prompt to JSON for cross-validation with the Rust
implementation in ``src/transformer/``.

The methodology mirrors ``qwen3_forward_validation.py`` /
``llama32_forward_validation.py``.  The reference JSON is consumed by
``tests/validate_qwen25_coder_forward.rs``.  Acceptance bar:

- Detected ``model_type`` is ``"qwen2"`` with QKV bias.
- ``(hidden_size, num_layers, vocab_size, head_dim)`` match the Python run.
- Per test case: top-10 logit indices match exactly, magnitudes within
  ``abs diff < 1e-3`` (`F32`, CPU vs CPU).

Dependencies: ``torch``, ``transformers``, ``safetensors``.

Usage:
    python scripts/qwen25_coder_validation.py

Output:
    scripts/qwen25_coder_forward_reference.json

Requires ``Qwen/Qwen2.5-Coder-3B-Instruct`` (~6 GiB) cached in the HF cache.
"""

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "Qwen/Qwen2.5-Coder-3B-Instruct"
# Three short completions; raw prompts (no chat template) so the comparison is
# a plain forward pass. Qwen tokenizers add no BOS by default.
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

    print(f"Qwen2.5-Coder forward-pass reference generation for {MODEL_REPO}")
    print(f"  {len(TEST_PROMPTS)} prompts, top-{TOP_K} logits per prompt")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")
    print()

    print("Loading model + tokenizer ...")
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
        f"num_kv_heads={cfg.num_key_value_heads}, attention_bias={getattr(cfg, 'attention_bias', None)}"
    )
    print()

    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "from-first-principles forward-pass oracle "
        "(transformers.AutoModelForCausalLM, F32 CPU); Qwen2 QKV-bias arm",
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
        "attention_bias": getattr(cfg, "attention_bias", None),
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

    out_path = Path(__file__).parent / "qwen25_coder_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
