# Oracle-suite resurrection log

The oracle/parity tests below are `#[ignore]`d — they need cached
HuggingFace models (some gated) and, for many, a CUDA GPU with ≥ 16 GiB
VRAM — so GitHub CI never runs them. This file records, **per test**, when
each last **passed** its oracle comparison locally.

**Refresh it** with [`scripts/resurrect.ps1`](scripts/resurrect.ps1):

```
scripts/resurrect.ps1          # default: all but the two slow outliers (~40-50 min)
scripts/resurrect.ps1 -Quick   # cheap ungated CPU encoder-parity smoke (~5 min)
scripts/resurrect.ps1 -Full    # + Mistral-7B CPU forward + anacrousis 28x15 (~1.5-3 h)
scripts/resurrect.ps1 -Status  # report staleness (runs nothing); -StaleDays N sets the threshold
```

"Last verified" = the last date this entry **passed** (a `⏭️ SKIP` /
`❌ FAIL` does not advance it). `never` = not yet verified on this machine.
Staleness is per-entry, so a `-Quick` run only refreshes its two rows.

- **Last run:** 2026-07-13 08:37 — tier **Default**
- **Toolchain:** rustc 1.97.0 (2d8144b78 2026-07-07)

| Test | Models | Device(s) | Last verified | Outcome |
|---|---|---|---|---|
| clt_qwen3 (encoder parity) | bluelightai/clt-qwen3-1.7b-base-20k (~240 MiB) | CPU | 2026-07-13 | ✅ PASS |
| plt_gemma (encoder parity) | google/gemma-scope-2b-pt-transcoders (~864 MiB, gated) | CPU | 2026-07-13 | ✅ PASS |
| plt_llama (encoder parity) | mntss/transcoder-Llama-3.2-1B (~16 GiB) | CPU | 2026-07-13 | ✅ PASS |
| llama32 forward | meta-llama/Llama-3.2-1B (gated) | CPU+GPU | 2026-07-13 | ✅ PASS |
| gemma2 forward | google/gemma-2-2b (gated) | CPU+GPU | 2026-07-13 | ✅ PASS |
| phi3-mini forward | microsoft/Phi-3-mini-4k-instruct | CPU+GPU | 2026-07-13 | ✅ PASS |
| mistral-7b forward | mistralai/Mistral-7B-v0.1 (gated) | CPU+GPU | 2026-07-13 | ✅ PASS |
| qwen3 forward | Qwen/Qwen3-1.7B-Base | CPU+GPU | 2026-07-13 | ✅ PASS |
| qwen2.5-coder forward | Qwen/Qwen2.5-Coder-3B-Instruct | CPU+GPU | 2026-07-13 | ✅ PASS |
| starcoder2 forward | bigcode/starcoder2-3b (gated) | CPU+GPU | 2026-07-13 | ✅ PASS |
| deepseek forward | deepseek-ai/deepseek-coder-1.3b-base | CPU+GPU | 2026-07-13 | ✅ PASS |
| longrope forward | microsoft/Phi-3.5-mini-instruct (~15 GiB) | GPU | 2026-07-13 | ✅ PASS |
| bidirectional (a2d-qwen2) | dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm (~1.2 GiB) | CPU+GPU | 2026-07-13 | ✅ PASS |
| mdlm + othello | TheQweaker/mdlm-owt-noflash; Othello fixtures (OTHELLO_MDLM_FIXTURES) | CPU+GPU | never | ⏭️ SKIP (uncached) |
| quantized (bnb/AWQ/GPTQ) | medmekk/...-bnb-nf4; casperhansen/...-awq; shuyuej/...-GPTQ | GPU | 2026-07-13 | ✅ PASS |
| clt (encode/inject/sweep) | gemma-2-2b + llama-3.2-1b + mntss CLTs (>=16 GiB VRAM) | GPU | 2026-07-13 | ✅ PASS |
| sae (encode/inject/parity) | gemma-2-2b + gemma-scope-2b-pt-res | GPU | 2026-07-13 | ✅ PASS |
| memory (VRAM probe) | (none - allocates a GPU tensor) | GPU | 2026-07-13 | ✅ PASS |
| rwkv6 + rwkv7 | RWKV v6-Finch-1B6; RWKV7-Goose-1.5B | CPU+GPU | 2026-07-13 | ✅ PASS |
| anacrousis (28x15 matrix) | meta-llama/Llama-3.2-1B (gated) | GPU | never | — never |
