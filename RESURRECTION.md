# Oracle-suite resurrection log

The oracle/parity tests below are `#[ignore]`d — they need cached
HuggingFace models (some gated) and, for many, a CUDA GPU with ≥ 16 GiB
VRAM — so GitHub CI never runs them. This file records, **per test**, when
each last **passed** its oracle comparison locally.

**Refresh it** with [`scripts/resurrect.ps1`](scripts/resurrect.ps1):

```
scripts/resurrect.ps1          # default: all but the two slow outliers (~40-50 min)
scripts/resurrect.ps1 -Quick   # cheap ungated smoke: CPU encoder-parity + the GPU patch_at guard (~6 min)
scripts/resurrect.ps1 -Full    # + Mistral-7B CPU forward + anacrousis 28x15 (~1.5-3 h)
scripts/resurrect.ps1 -Status  # report staleness (runs nothing); -StaleDays N sets the threshold
```

"Last verified" = the last date this entry **passed** (a `⏭️ SKIP` /
`❌ FAIL` does not advance it). `never` = not yet verified on this machine.
Staleness is per-entry, so a `-Quick` run only refreshes its three rows.

**Wall-clock** = end-to-end runtime of the entry on its last PASS (model
load/download + compile + run, not just the `cargo test` phase). A ⚠️ flags a
step slow enough (≥ 300 s) to suspect VRAM spill to shared memory
(e.g. `longrope`/Phi-3.5-mini at F32 overflows a 16 GiB card).

**Peak spill** = measured WDDM spill for entries sampled via `hmn spill`
(hypomnesis), as *growth* of resident shared-system memory above the benign
staging-heap baseline, paired with how long the spill lasted. This is the
real signal: during a spill NVML `used` pins near capacity and cannot show
how far over budget a run went. `none` = sampled, no spill. `—` = not
sampled. `n/a` = not measurable on this platform (Linux/macOS have no
shared-residency counter). Mark an entry `Spill = $true` in the script, or
pass `-SpillProbe` to sample every entry.

- **Last run:** 2026-09-02 12:19 — tier **partial (1 of 21: patchat)**
- **Toolchain:** rustc 1.98.0 (88d9e12ae 2026-08-18)
- **GPU:** NVIDIA GeForce RTX 5060 Ti, driver 610.88

| Test | Models | Device(s) | Last verified | Wall-clock | Peak spill | Outcome |
|---|---|---|---|---|---|---|
| clt_qwen3 (encoder parity) | bluelightai/clt-qwen3-1.7b-base-20k (~240 MiB) | CPU | 2026-09-02 | 2m05s | none | ✅ PASS |
| plt_gemma (encoder parity) | google/gemma-scope-2b-pt-transcoders (~864 MiB, gated) | CPU | 2026-09-02 | 2m00s | none | ✅ PASS |
| plt_llama (encoder parity) | mntss/transcoder-Llama-3.2-1B (~16 GiB) | CPU | 2026-08-12 | 6.4s | — | ✅ PASS |
| llama32 forward | meta-llama/Llama-3.2-1B (gated) | CPU+GPU | 2026-08-12 | 31.3s | — | ✅ PASS |
| gemma2 forward | google/gemma-2-2b (gated) | CPU+GPU | 2026-08-12 | 30.2s | — | ✅ PASS |
| phi3-mini forward | microsoft/Phi-3-mini-4k-instruct | CPU+GPU | 2026-08-12 | 33.1s | — | ✅ PASS |
| mistral-7b forward | mistralai/Mistral-7B-v0.1 (gated) | CPU+GPU | 2026-08-12 | 43.6s | — | ✅ PASS |
| qwen3 forward | Qwen/Qwen3-1.7B-Base | CPU+GPU | 2026-08-12 | 20.7s | — | ✅ PASS |
| qwen2.5-coder forward | Qwen/Qwen2.5-Coder-3B-Instruct | CPU+GPU | 2026-08-12 | 27.9s | — | ✅ PASS |
| starcoder2 forward | bigcode/starcoder2-3b (gated) | CPU+GPU | 2026-08-12 | 25.4s | — | ✅ PASS |
| deepseek forward | deepseek-ai/deepseek-coder-1.3b-base | CPU+GPU | 2026-08-12 | 9.1s | — | ✅ PASS |
| longrope forward | microsoft/Phi-3.5-mini-instruct (~15 GiB) | GPU | 2026-08-12 | 13m16s ⚠️ | 6294 MiB / 13m08s / peak ~21.8 GiB | ✅ PASS |
| bidirectional (a2d-qwen2) | dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm (~1.2 GiB) | CPU+GPU | 2026-08-12 | 1m41s | — | ✅ PASS |
| mdlm + othello | TheQweaker/mdlm-owt-noflash; Othello fixtures (OTHELLO_MDLM_FIXTURES) | CPU+GPU | 2026-08-12 | 6s | — | ✅ PASS |
| quantized (bnb/AWQ/GPTQ) | medmekk/...-bnb-nf4; casperhansen/...-awq; shuyuej/...-GPTQ | GPU | 2026-08-12 | 2m14s | — | ✅ PASS |
| clt (encode/inject/sweep) | gemma-2-2b + llama-3.2-1b + mntss CLTs (>=16 GiB VRAM) | GPU | 2026-08-12 | 3m27s | — | ✅ PASS |
| sae (encode/inject/parity) | gemma-2-2b + gemma-scope-2b-pt-res | GPU | 2026-08-12 | 2m51s | — | ✅ PASS |
| memory (VRAM probe) | (none - allocates a GPU tensor) | GPU | 2026-08-12 | 36.6s | — | ✅ PASS |
| rwkv6 + rwkv7 | RWKV v6-Finch-1B6; RWKV7-Goose-1.5B | CPU+GPU | 2026-08-12 | 3m36s | — | ✅ PASS |
| anacrousis (28x15 matrix) | meta-llama/Llama-3.2-1B (gated) | GPU | never | — | — | — never |
| patch_at (CUDA offset-view guard) | (none - allocates GPU tensors) | GPU | 2026-09-02 | 80.4s | — | ✅ PASS |
