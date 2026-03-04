# RWKV-7 Forward Pass: Python vs Rust Comparison

**Phase 2 — RWKV Support:**
> Validate RWKV-7 (Goose) forward pass against Python reference implementation.

This document compares the output of candle-mi's Rust RWKV-7 backend
(`tests/validate_rwkv7.rs`) against a Python reference implementation
(`scripts/rwkv7_validation.py`) that implements the WKV-7 recurrence
in pure PyTorch (no Triton/fla dependency for inference).

## Environment

| | Python (reference) | Rust CPU | Rust GPU |
|---|---|---|---|
| Framework | PyTorch (pure F32) | candle 0.9 | candle 0.9 |
| Compute dtype | F32 | F32 | BF16 |
| Attention impl | Manual WKV-7 recurrence | Manual WKV-7 recurrence | Manual WKV-7 recurrence |

## Model

| | Value |
|---|---|
| Model ID | `RWKV/RWKV7-Goose-World3-1.5B-HF` |
| Architecture | RWKV-7 (Goose) |
| hidden_size | 2048 |
| num_layers | 24 |
| head_dim | 64 |
| num_heads | 32 |
| vocab_size | 65536 |

## Tokenization

All three produce identical tokenization (7 tokens, no BOS):

| Pos | Token | ID |
|:---:|-------|:---:|
| 0 | `def` | 7334 |
| 1 | ` fib` | 21676 |
| 2 | `onacci` | 41943 |
| 3 | `(` | 41 |
| 4 | `n` | 111 |
| 5 | `):` | 501 |
| 6 | `\n    ` | 28352 |

**Prompt:** `def fibonacci(n):\n    `

## Top-10 Predictions (next token after prompt)

| Rank | Token | ID | Python Logit (F32) | Rust CPU Logit (F32) | Rust GPU Logit (BF16) | CPU Rel Err | GPU Rel Err |
|:----:|-------|:---:|-------------------:|---------------------:|----------------------:|------------:|------------:|
| 1 | `if` | 1942 | 7.5585 | 7.5585 | 7.5312 | 0.00% | 0.36% |
| 2 | `a` | 98 | 5.9269 | 5.9270 | 5.8750 | 0.00% | 0.88% |
| 3 | `"""` | 5170 | 5.8649 | 5.8649 | 5.8750 | 0.00% | 0.17% |
| 4 | `#` | 36 | 4.7785 | 4.7785 | 4.7812 | 0.00% | 0.06% |
| 5 | `return` | 42178 | 4.2776 | 4.2776 | 4.3125 | 0.00% | 0.82% |
| 6 | `f` | 103 | 4.1880 | 4.1880 | 4.1875 | 0.00% | 0.01% |
| 7 | `result` | 42175 | 3.7979 | 3.7979 | 3.7500 | 0.00% | 1.26% |
| 8 | `fib` | 7560 | 3.7510 | 3.7510 | 3.7344 | 0.00% | 0.44% |
| 9 | `x` | 121 | 3.5133 | 3.5133 | — | 0.00% | — |
| 10 | `def` | 7334 | 3.4919 | 3.4919 | — | 0.00% | — |

**Note on GPU ranks 9–10:** BF16 precision causes `def` (3.5312) and `x`
(3.4844) to swap positions compared to the F32 reference (`x`=3.5133,
`def`=3.4919). The logit gap between them is only 0.02 in F32, well within
BF16 rounding. Both tokens remain in the top 10.

## Aggregate Statistics

| Metric | Python | Rust CPU | Rust GPU |
|--------|--------|----------|----------|
| Top-1 token | `if` (1942) | `if` (1942) | `if` (1942) |
| Top-1 logit | 7.5585 | 7.5585 | 7.5312 |
| Top-5 token IDs | 1942, 98, 5170, 36, 42178 | 1942, 98, 5170, 36, 42178 | 1942, 98, 5170, 36, 42178 |
| Top-1 logit diff from ref | — | < 0.001 | 0.027 |
| Top-5 ID match | — | 5/5 | 5/5 |
| Top-10 ID match | — | 10/10 | 10/10 (with rank swap at 9–10) |

## Acceptance Criteria

| Criterion | Threshold | Rust CPU | Rust GPU |
|-----------|-----------|:--------:|:--------:|
| Top-1 token = `if` (1942) | exact | pass | pass |
| Top-1 logit diff | < 1.0 (CPU) / < 2.0 (GPU) | 0.000 | 0.027 |
| Top-5 token IDs match | exact | pass | pass |

## Autoregressive Generation (Python reference)

The Python script also generates 20 tokens autoregressively to validate
coherent output:

```
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
```

This is a correct Python fibonacci implementation, confirming the model
produces coherent code completions.

## Analysis

**Rust CPU (F32) is a near-exact match** to the Python F32 reference.
All 10 logit values agree to 4 decimal places (< 0.001 absolute error).
This confirms the Rust WKV-7 recurrence, LoRA activations (tanh/sigmoid
patterns), token shift, GroupNorm, and gate correction all match the
Python reference implementation exactly.

**Rust GPU (BF16) matches within expected BF16 tolerance.** Top-1 logit
differs by 0.027 (0.36%), and all top-5 token IDs match exactly. The
only observable effect of BF16 quantization is a rank swap between
positions 9 and 10, where the F32 logit gap is only 0.02 — well within
the BF16 rounding range of ~0.03 at this magnitude.

## How to reproduce

```bash
# Generate Python reference
python scripts/rwkv7_validation.py

# Run Rust CPU test
cargo test --test validate_rwkv7 --features rwkv,rwkv-tokenizer \
    -- rwkv7_forward_cpu --nocapture

# Run Rust GPU test
cargo test --test validate_rwkv7 --features rwkv,rwkv-tokenizer \
    -- rwkv7_forward_gpu --nocapture
```
