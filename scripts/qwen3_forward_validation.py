#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Qwen3-1.7B-Base forward-pass reference for Rust validation.

From-first-principles forward-pass oracle for the candle-mi `Qwen3`
transformer arm: loads ``Qwen/Qwen3-1.7B-Base`` via HuggingFace
``transformers`` in ``F32`` on CPU, runs ``forward()`` on a small set of
fixed prompts with deterministic seeds, and saves
**(a)** top-10 next-token logits + indices and
**(b)** the final-layer last-token residual (post-final-norm,
pre-LM-head) per prompt to JSON for cross-validation with the Rust
implementation in ``src/transformer/``.

The methodology mirrors ``plt_llama_validation.py`` (encoder-side
oracle, V3 Step 1.4) and ``plt_gemma_validation.py`` (JumpReLU
oracle, v0.1.10) — adapted from per-layer encoder activations to a
full forward-pass comparison.

Test prompts are short, factual completions in English with deterministic
top-tokens that exercise the `QK`-norm + `RoPE` + GQA stack end to end.

The reference JSON is consumed by ``tests/validate_qwen3_forward.rs``.
Acceptance bar:

- Detected `model_type` is ``"qwen3"`` and ``use_qk_norm == true``.
- ``(hidden_size, num_layers, vocab_size)`` match the Python run.
- Per test case: top-10 logit indices match exactly, magnitudes within
  ``abs diff < 1e-3`` (`F32`, CPU vs CPU).
- Final-token residual element-wise ``abs diff < 1e-3``.

Dependencies: ``torch``, ``transformers >= 4.55``, ``safetensors``.
``Qwen3ForCausalLM`` ships with transformers ``4.55+``; verified on
``5.1.0`` at the time this oracle was written.

Usage:
    python scripts/qwen3_forward_validation.py

Output:
    scripts/qwen3_forward_reference.json

Will download ``Qwen/Qwen3-1.7B-Base`` (~3.2 GiB) into the HF cache on
first run.  Subsequent runs are cache hits.
"""

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "Qwen/Qwen3-1.7B-Base"
# Three short, common English completions exercising different attention
# patterns (geographic recall, arithmetic, narrative continuation).
TEST_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
]
TOP_K = 10


def main() -> None:
    # Determinism — CPU-only run so `CUBLAS_WORKSPACE_CONFIG` is a no-op
    # but set anyway per the v0.1.9 / v0.1.10 oracle template.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    import transformers as hf_transformers

    print(f"Qwen3 forward-pass reference generation for {MODEL_REPO}")
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
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, head_dim={cfg.head_dim}, "
        f"num_kv_heads={cfg.num_key_value_heads}"
    )
    print()

    # Use getattr with defaults — Qwen3Config fields shift across
    # transformers releases (e.g. rope_theta moved under rope_scaling in
    # 5.x).  Only the essentials (matched by the Rust test) are required.
    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "from-first-principles forward-pass oracle "
        "(transformers.AutoModelForCausalLM, F32 CPU)",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
        "platform": platform.platform(),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "head_dim": cfg.head_dim,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
        "use_qk_norm": True,
        "test_cases": [],
    }

    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            tokens = input_ids[0].tolist()

            # Forward with hidden states so we can extract the post-final-norm
            # last-token residual (the input to lm_head).
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
            # `lm_head(hidden_states[-1][:, -1, :])` reproduces `logits[:, -1, :]`.
            final_hidden = outputs.hidden_states[-1]
            last_residual = final_hidden[0, -1, :].float().tolist()

            top_token_str = tokenizer.decode([int(top_idx[0])])
            print(
                f"  prompt='{prompt}': {len(tokens)} tokens, "
                f"top1=({int(top_idx[0])}, '{top_token_str}', {float(top_vals[0]):.4f})"
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

    out_path = Path(__file__).parent / "qwen3_forward_reference.json"
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
