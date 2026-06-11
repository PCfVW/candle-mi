#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate a bitsandbytes-NF4 forward-pass reference for Rust validation.

Oracle for candle-mi's **quantized weight-load path**: candle-mi cannot consume
quantized weights directly, so (with the `quantized` feature) it routes them
through anamnesis — `parse()` -> `remember_to_bytes(BF16)` — to dequantize NF4 to
BF16 before building the `VarBuilder`. This script produces the reference the
Rust side compares against, validating that anamnesis's NF4 dequant agrees with
bitsandbytes's.

Loads ``medmekk/Llama-3.2-1B-Instruct-bnb-nf4`` (already NF4-quantized in the
checkpoint; ``transformers`` reads its ``quantization_config`` and loads it via
bitsandbytes) on CUDA, runs ``forward()`` on fixed prompts, and saves per prompt
**(a)** the token ids (so the Rust side feeds the exact same tokens — the quant
repo ships no tokenizer, so we borrow the base Llama-3.2 tokenizer) and
**(b)** top-10 next-token logits + indices.

The forward runs in the bitsandbytes default ``bfloat16`` compute dtype, so the
logits carry ~3 significant figures; the Rust parity bar is BF16-tier
(see ``tests/validate_bnb_loading.rs``), not the ~1e-5 of full-precision
families. What it proves: candle-mi loads a real quantized checkpoint and runs
it correctly, with anamnesis's NF4 dequant matching bitsandbytes.

Dependencies: ``torch``, ``transformers``, ``bitsandbytes``, ``accelerate``.
Requires a CUDA device (bitsandbytes 4-bit needs CUDA).

Usage:
    python scripts/bnb_nf4_validation.py

Output:
    scripts/bnb_nf4_forward_reference.json
"""

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "medmekk/Llama-3.2-1B-Instruct-bnb-nf4"
# The quant repo has no tokenizer; the base Llama-3.2 tokenizer is identical.
TOKENIZER_REPO = "meta-llama/Llama-3.2-1B"
TEST_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
]
TOP_K = 10


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(0)

    import bitsandbytes as bnb
    import transformers as hf_transformers

    if not torch.cuda.is_available():
        raise SystemExit("bitsandbytes 4-bit requires CUDA; no CUDA device found")

    print(f"bitsandbytes-NF4 forward-pass reference for {MODEL_REPO}")
    print(f"  tokenizer from {TOKENIZER_REPO}")
    print(
        f"  torch {torch.__version__}, transformers {hf_transformers.__version__}, "
        f"bitsandbytes {bnb.__version__}"
    )
    print(f"  platform {platform.platform()}")
    print()

    # This checkpoint's bnb_4bit_compute_dtype is float32, so run the oracle in
    # F32 — both more faithful to the model and a tighter comparison against
    # candle-mi's F32 forward (residual = only the NF4→BF16 weight rounding
    # anamnesis applies, not an extra bf16-compute gap).
    print("Loading NF4 model (via bitsandbytes, F32 compute) + base tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        device_map="cuda",
        dtype=torch.float32,
    )
    model.eval()

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    qc = getattr(cfg, "quantization_config", None)
    quant_method = getattr(qc, "quant_method", None) if qc is not None else None
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, head_dim={head_dim}, quant_method={quant_method}"
    )
    print()

    results: dict = {
        "model_repo": MODEL_REPO,
        "tokenizer_repo": TOKENIZER_REPO,
        "methodology": "bitsandbytes-NF4 forward-pass oracle (transformers + bitsandbytes, "
        "F32 compute on CUDA); validates candle-mi's anamnesis NF4 dequant load path",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
        "bitsandbytes_version": bnb.__version__,
        "platform": platform.platform(),
        "model_type": cfg.model_type,
        "quant_method": str(quant_method),
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "head_dim": head_dim,
        "test_cases": [],
    }

    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            tokens = input_ids[0].tolist()

            outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
            last_logits = outputs.logits[0, -1, :].float().cpu()
            top_vals, top_idx = last_logits.topk(TOP_K)

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
                }
            )

    out_path = Path(__file__).parent / "bnb_nf4_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
