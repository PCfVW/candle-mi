#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate DeepSeek-Coder-1.3B-Base forward-pass reference for Rust validation.

From-first-principles forward-pass oracle for the candle-mi `LLaMA`
transformer arm, exercising the **linear `rope_scaling`** path that
DeepSeek-Coder uses to extend its context from 4 096 to 16 384
(``rope_scaling = {"type": "linear", "factor": 4.0}``).  Loads
``deepseek-ai/deepseek-coder-1.3b-base`` via HuggingFace ``transformers``
in ``F32`` on CPU, runs ``forward()`` on a small set of fixed prompts with
deterministic seeds, and saves
**(a)** top-10 next-token logits + indices and
**(b)** the final-layer last-token residual (post-final-norm,
pre-LM-head) per prompt to JSON for cross-validation with the Rust
implementation in ``src/transformer/``.

The methodology mirrors ``qwen3_forward_validation.py`` (full forward-pass
oracle, v0.1.11) — adapted from the QK-norm Qwen3 arm to the linear
rope-scaling LLaMA arm.  ``model_type`` is ``"llama"``; the defining
feature under test is the ``rope_scaling`` block, which candle-mi must
apply to the RoPE position grid (``position / factor``) to match.

Test prompts are short completions with deterministic top-tokens.  Because
linear scaling divides *every* position by the factor (not just positions
beyond the original context window), the scaling already perturbs the
rotation at sequence positions 1, 2, 3, ... — so even short prompts
discriminate a scaled vs. unscaled RoPE implementation.

The reference JSON is consumed by ``tests/validate_deepseek_forward.rs``.
Acceptance bar:

- Detected ``model_type`` is ``"llama"`` and ``rope_scaling.type`` is
  ``"linear"`` with ``factor == 4.0``.
- ``(hidden_size, num_layers, vocab_size, head_dim)`` match the Python run.
- Per test case: top-10 logit indices match exactly, magnitudes within
  ``abs diff < 1e-3`` (`F32`, CPU vs CPU).

Dependencies: ``torch``, ``transformers``, ``safetensors``.  The repo
ships ``pytorch_model.bin`` (pickle), which ``transformers`` loads
natively.

Usage:
    python scripts/deepseek_coder_validation.py

Output:
    scripts/deepseek_coder_forward_reference.json

Will download ``deepseek-ai/deepseek-coder-1.3b-base`` (~2.5 GiB) into the
HF cache on first run.  Subsequent runs are cache hits.
"""

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "deepseek-ai/deepseek-coder-1.3b-base"
# Three short completions exercising different patterns (geographic recall,
# code continuation, library idiom) — all with deterministic top tokens.
TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "import numpy as",
]
TOP_K = 10


def main() -> None:
    # Determinism — CPU-only run so `CUBLAS_WORKSPACE_CONFIG` is a no-op
    # but set anyway per the v0.1.9 / v0.1.10 / v0.1.11 oracle template.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    import transformers as hf_transformers

    print(f"DeepSeek-Coder forward-pass reference generation for {MODEL_REPO}")
    print(f"  {len(TEST_PROMPTS)} prompts, top-{TOP_K} logits per prompt")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")
    print()

    # Load on CPU in F32 — matches candle-mi's research-grade precision
    # default ("F32 everywhere, numerically identical to Python/PyTorch").
    print("Loading model + tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    cfg = model.config
    rope_scaling = getattr(cfg, "rope_scaling", None)
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, "
        f"num_kv_heads={cfg.num_key_value_heads}"
    )
    print(f"  rope_theta={getattr(cfg, 'rope_theta', None)}, rope_scaling={rope_scaling}")
    print()

    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads

    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "from-first-principles forward-pass oracle "
        "(transformers.AutoModelForCausalLM, F32 CPU); linear rope_scaling arm",
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
        "rope_scaling": rope_scaling,
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

            # Logits: [1, seq_len, vocab_size] — take the last position.
            last_logits = outputs.logits[0, -1, :].float()
            top_vals, top_idx = last_logits.topk(TOP_K)

            # Hidden states tuple has num_layers+1 entries:
            # index 0 = embedding output, index num_layers = post-final-norm.
            final_hidden = outputs.hidden_states[-1]
            last_residual = final_hidden[0, -1, :].float().tolist()

            top_token_str = tokenizer.decode([int(top_idx[0])])
            print(
                f"  prompt={prompt!r}: {len(tokens)} tokens, "
                f"top1=({int(top_idx[0])}, {top_token_str!r}, {float(top_vals[0]):.4f})"
            )

            test_case = {
                "prompt": prompt,
                "tokens": tokens,
                "top_10": [
                    {"index": int(idx), "logit": float(val)}
                    for idx, val in zip(top_idx, top_vals, strict=False)
                ],
                "last_residual_f32": last_residual,
            }
            results["test_cases"].append(test_case)

    out_path = Path(__file__).parent / "deepseek_coder_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
