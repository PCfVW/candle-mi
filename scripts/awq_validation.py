#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate an AWQ forward-pass reference for Rust validation.

Oracle for candle-mi's quantized weight-load path (AWQ arm): candle-mi routes
AWQ checkpoints through anamnesis (`parse` -> `remember_to_bytes(BF16)`) to
dequantize to BF16 before building the `VarBuilder`. This script produces the
reference the Rust side compares against, validating that anamnesis's AWQ dequant
(AutoAWQ GEMM nibble interleave) agrees with the real AutoAWQ kernels.

Loads ``casperhansen/llama-3.2-1b-instruct-awq`` (4-bit AWQ) via transformers +
the ``awq`` (AutoAWQ) backend on CUDA in fp16 (AWQ's compute dtype), runs
``forward()`` on fixed prompts, and saves per prompt the token ids and top-10
next-token logits.

Bar is fp16/BF16-weight tier (see ``tests/validate_quantized_loading.rs``), not
the ~1e-5 of full-precision families. What it proves: candle-mi loads a real AWQ
checkpoint and runs it correctly (exact top-1 token).

Dependencies: ``torch``, ``transformers``, ``autoawq`` (the ``awq`` package).
Requires a CUDA device.

Usage:
    python scripts/awq_validation.py
Output:
    scripts/awq_forward_reference.json
"""

import json
import platform
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Windows consoles default to cp1252; token reprs may carry non-ASCII bytes.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_REPO = "casperhansen/llama-3.2-1b-instruct-awq"
TEST_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
]
TOP_K = 10


def main() -> None:
    torch.manual_seed(0)
    import transformers as hf_transformers

    if not torch.cuda.is_available():
        raise SystemExit("AWQ kernels require CUDA; no CUDA device found")

    print(f"AWQ forward-pass reference for {MODEL_REPO}")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        device_map="cuda",
        dtype=torch.float16,
    )
    model.eval()

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    qc = getattr(cfg, "quantization_config", None)
    quant_method = getattr(qc, "quant_method", None) if qc is not None else None
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, quant_method={quant_method}"
    )

    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "AWQ forward-pass oracle (transformers + AutoAWQ, fp16 compute on CUDA); "
        "validates candle-mi's anamnesis AWQ dequant load path",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
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
            tokens = inputs.input_ids[0].tolist()
            outputs = model(input_ids=inputs.input_ids, use_cache=False, return_dict=True)
            last_logits = outputs.logits[0, -1, :].float().cpu()
            top_vals, top_idx = last_logits.topk(TOP_K)
            print(
                f"  prompt={prompt!r}: {len(tokens)} tokens, "
                f"top1=({int(top_idx[0])}, {tokenizer.decode([int(top_idx[0])])!r}, {float(top_vals[0]):.4f})"
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

    out_path = Path(__file__).parent / "awq_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results['test_cases'])} test cases to {out_path}")


if __name__ == "__main__":
    main()
