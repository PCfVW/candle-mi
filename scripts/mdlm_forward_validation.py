#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate an MDLM masked-diffusion forward-pass reference for Rust validation.

From-first-principles fp32 oracle for the candle-mi ``MDLM`` diffusion arm.
Loads ``TheQweaker/mdlm-owt-noflash`` (a flash-attn-free, fp32 reimplementation
of ``kuleshov-group/mdlm-owt`` with **byte-identical weights**) via HuggingFace
``transformers`` on CPU, masks one word per prompt, runs ``forward()``, and
saves the top-10 **raw** logits + indices at each masked position to JSON.

Why the noflash port: the upstream ``modeling_mdlm.py`` hard-depends on
``flash-attn`` (CUDA-only) and runs its block stack under a bf16 autocast.  The
noflash port removes both — full-bidirectional ``scaled_dot_product_attention``
and an fp32 forward — so it runs anywhere and is the right numerical oracle for
candle-mi (also fp32 throughout).  The Rust side loads the *original*
``kuleshov-group/mdlm-owt`` weights, which are identical.

The reference JSON is consumed by ``tests/validate_mdlm_forward.rs``.
Acceptance bar (per test case): top-10 logit indices match exactly, magnitudes
within ``abs diff < 1e-3`` (fp32, CPU vs CPU).

Dependencies: ``torch``, ``transformers >= 5`` (for ``post_init`` finalize),
``safetensors``.  Validated on torch 2.10 / transformers 5.1 on Windows.

Usage:
    python scripts/mdlm_forward_validation.py

Output:
    scripts/mdlm_forward_reference.json
"""

import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Source of the modeling code (flash-attn-free); weights are byte-identical to
# the repo the Rust side loads.
ORACLE_REPO = "TheQweaker/mdlm-owt-noflash"
# Repo the Rust test loads (same weights, original modeling code).
WEIGHTS_REPO = "kuleshov-group/mdlm-owt"
MASK_ID = 50257  # GPT-2 has 50257 tokens (0..50256); 50257 is MDLM's [MASK].
TOP_K = 10

# Short, factual sentences; for each, the word that gets masked and predicted.
TEST_PROMPTS = [
    ("The capital of France is Paris.", " Paris"),
    ("The opposite of hot is cold.", " cold"),
    ("Water is made of hydrogen and oxygen.", " oxygen"),
]


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    import transformers as hf_transformers

    print(f"MDLM forward-pass reference generation")
    print(f"  oracle modeling: {ORACLE_REPO}  (weights == {WEIGHTS_REPO})")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForMaskedLM.from_pretrained(
        ORACLE_REPO,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    model = model.to("cpu").eval()

    cfg = model.config
    print(
        f"  hidden_dim={cfg.hidden_dim}, n_blocks={cfg.n_blocks}, "
        f"n_heads={cfg.n_heads}, vocab_size={cfg.vocab_size}, "
        f"time_conditioning={cfg.time_conditioning}"
    )
    print()

    results: dict = {
        "oracle_repo": ORACLE_REPO,
        "weights_repo": WEIGHTS_REPO,
        "methodology": "fp32 forward-pass oracle (AutoModelForMaskedLM, CPU, "
        "raw logits at masked positions)",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
        "platform": platform.platform(),
        "hidden_dim": cfg.hidden_dim,
        "n_blocks": cfg.n_blocks,
        "n_heads": cfg.n_heads,
        "vocab_size": cfg.vocab_size,
        "mask_token_id": MASK_ID,
        "test_cases": [],
    }

    with torch.no_grad():
        for prompt, target in TEST_PROMPTS:
            tokens = tokenizer(prompt)["input_ids"]
            target_id = tokenizer(target)["input_ids"][0]
            mask_position = tokens.index(target_id)

            # Replace the target token with [MASK] (the absorbing state).
            masked = list(tokens)
            masked[mask_position] = MASK_ID

            input_ids = torch.tensor([masked], dtype=torch.long)
            logits = model(input_ids=input_ids, return_dict=True).logits  # [1, L, V]

            # RAW logits at the masked position (no SUBS — pure forward parity).
            at_mask = logits[0, mask_position, :].float()
            top_vals, top_idx = at_mask.topk(TOP_K)

            top1_str = tokenizer.decode([int(top_idx[0])])
            print(
                f"  '{prompt}': mask {target!r} at pos {mask_position}, "
                f"top1=({int(top_idx[0])}, '{top1_str}', {float(top_vals[0]):.4f})"
            )

            results["test_cases"].append(
                {
                    "prompt": prompt,
                    "target": target,
                    "tokens": masked,
                    "mask_position": mask_position,
                    "top_10": [
                        {"index": int(idx), "logit": float(val)}
                        for idx, val in zip(top_idx, top_vals, strict=False)
                    ],
                }
            )

    out_path = Path(__file__).parent / "mdlm_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    print(f"\nSaved {n_cases} test cases to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
