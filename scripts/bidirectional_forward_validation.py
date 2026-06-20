#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate a bidirectional-decoder forward-pass reference for Rust validation.

fp32 oracle for candle-mi's **bidirectional** ``GenericTransformer`` path
(Stage 3), which loads decoder-style masked-diffusion LMs (Dream, a2d-qwen2, ...)
and runs the decoder *non-causally* (every position attends to every other).

Model: ``dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`` — a standard
``Qwen2.5`` layout under ``model_type`` ``"a2d-qwen2"``.  Its ``trust_remote_code``
modeling file (``modeling_qwen2.py``) is just ``Qwen2`` with two changes:

  1. a fresh **untied** ``lm_head`` (``A2DQwen2LMHeadModel.__init__``), and
  2. **bidirectional** attention (``attention_mask=None`` -> a full, non-causal
     4D mask).

Because that file ``import dllm`` (only in a trailing helper, but
``check_imports`` blocks loading without it), we instead load the weights with
**stock** ``Qwen2ForCausalLM`` and reproduce the two deltas exactly: force
``tie_word_embeddings=False`` (loads the shipped ``lm_head.weight``) and pass an
all-zeros 4D attention mask (fully bidirectional).  We assert the mask is
actually bidirectional (its position-0 logits differ from a causal run).

Why this model: the smallest exact-``Qwen2.5``-layout masked-diffusion checkpoint
(~0.5B, 1.17 GiB), so it validates the same bidirectional code path as Dream-7B
but fits 16 GB at fp32 (tight GPU parity).  We compare logits at *early*
positions, where bidirectional attention provably differs from causal.

The reference JSON is consumed by ``tests/validate_bidirectional_forward.rs``.
Acceptance bar (per position): top-10 logit indices match exactly, magnitudes
within ``abs diff < 1e-3`` (CPU vs CPU) / ``< 5e-3`` (GPU vs CPU).

Dependencies: ``torch``, ``transformers >= 5``, ``safetensors``.

Usage:
    python scripts/bidirectional_forward_validation.py

Output:
    scripts/bidirectional_forward_reference.json
"""

import json
import platform
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

REPO = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
TOP_K = 10

# Short prompts; we compare logits at first / middle / last positions.  Early
# positions exercise the bidirectional (future-attending) path that a causal
# decoder cannot reproduce.
TEST_PROMPTS = [
    "The capital of France is Paris.",
    "Water is made of hydrogen and oxygen.",
]


def bidirectional_logits(model: Qwen2ForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a fully bidirectional forward via an all-zeros 4D additive mask."""
    seq_len = input_ids.shape[1]
    # 0.0 everywhere => every query attends to every key (no causal restriction).
    mask4d = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    out = model(
        input_ids=input_ids,
        attention_mask=mask4d,
        position_ids=position_ids,
        cache_position=torch.arange(seq_len),
        use_cache=False,
        return_dict=True,
    )
    return out.logits[0].float()  # [L, V]


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    import transformers as hf_transformers

    print("Bidirectional decoder forward-pass reference generation")
    print(f"  repo: {REPO}")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(REPO)
    config = Qwen2Config.from_pretrained(REPO)
    # A2DQwen2LMHeadModel uses a fresh, untied lm_head — load the shipped
    # `lm_head.weight` rather than tying to the input embeddings.
    config.tie_word_embeddings = False
    model = Qwen2ForCausalLM.from_pretrained(
        REPO,
        config=config,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.to("cpu").eval()

    cfg = model.config

    # Head diagnostics: the effective head must differ from the embeddings
    # (untied) and candle-mi must load the same `lm_head.weight`.
    embed_w = model.get_input_embeddings().weight
    head_w = model.get_output_embeddings().weight
    head_tied = head_w.data_ptr() == embed_w.data_ptr()
    head_max_diff = float((head_w - embed_w).abs().max())

    print(
        f"  hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"heads={cfg.num_attention_heads}, kv={cfg.num_key_value_heads}, "
        f"vocab={cfg.vocab_size}"
    )
    print(
        f"  head_tied(runtime)={head_tied}, head_vs_embed_max_diff={head_max_diff:.3e} "
        f"(expect untied, > 0)"
    )

    with torch.no_grad():
        # Sanity: confirm the 4D zeros mask really is bidirectional — its
        # position-0 logits must differ from a default causal run.
        probe = torch.tensor([tokenizer(TEST_PROMPTS[0])["input_ids"]], dtype=torch.long)
        bi = bidirectional_logits(model, probe)
        causal = model(input_ids=probe, use_cache=False, return_dict=True).logits[0].float()
        pos0_diff = float((bi[0] - causal[0]).abs().max())
        print(f"  bidirectional sanity: pos-0 |bidir - causal| max = {pos0_diff:.3f} (must be > 0)")
        assert pos0_diff > 1e-2, "4D zeros mask did not enable bidirectional attention"
        print()

        results: dict = {
            "repo": REPO,
            "methodology": "fp32 bidirectional forward (stock Qwen2ForCausalLM, untied "
            "lm_head, all-zeros 4D mask, eager, CPU); raw logits at first/middle/last",
            "torch_version": torch.__version__,
            "transformers_version": hf_transformers.__version__,
            "platform": platform.platform(),
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "vocab_size": cfg.vocab_size,
            "model_type": "a2d-qwen2",
            "head_tied_runtime": head_tied,
            "head_vs_embed_max_diff": head_max_diff,
            "bidirectional_sanity_pos0_diff": pos0_diff,
            "test_cases": [],
        }

        for prompt in TEST_PROMPTS:
            tokens = tokenizer(prompt)["input_ids"]
            logits = bidirectional_logits(model, torch.tensor([tokens], dtype=torch.long))
            seq_len = logits.shape[0]

            positions = sorted({0, seq_len // 2, seq_len - 1})
            pos_dump = []
            for pos in positions:
                top_vals, top_idx = logits[pos].topk(TOP_K)
                pos_dump.append(
                    {
                        "position": pos,
                        "top_10": [
                            {"index": int(idx), "logit": float(val)}
                            for idx, val in zip(top_idx, top_vals, strict=False)
                        ],
                    }
                )

            last_top1 = tokenizer.decode([int(logits[-1].argmax())])
            print(f"  {prompt!r}: L={seq_len}, positions={positions}, last-top1={last_top1!r}")

            results["test_cases"].append(
                {"prompt": prompt, "tokens": tokens, "positions": pos_dump}
            )

    out_path = Path(__file__).parent / "bidirectional_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    print(
        f"\nSaved {n_cases} test cases to {out_path} "
        f"({out_path.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
