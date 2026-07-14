# candle-mi experiments

This folder holds the **findings and data** behind candle-mi's mechanistic-
interpretability experiments. Each subfolder is one experiment with a
`findings.md` (the human-readable write-up: headline result, exact reproduce
commands, per-cell tables, caveats) plus the JSON data behind it. Large
intermediate JSONs (dense CLT vocab scans, dense-CLT census dumps) are
`.gitignore`d — they regenerate from the named example.

## The through-line: chasing the replication *floor* of Figure 13

Anthropic's [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
(§4, Figure 13) reports that when Claude 3.5 Haiku writes a rhyming couplet, it
**plans**: cross-layer-transcoder (CLT) features for the next line's rhyme word
pre-activate at the **newline** before the line is written, and steering those
features *at the newline* redirects the rhyme — while steering at the final
token does nothing.

Most of this folder asks one question: **does that hold on small open models,
and if not, where is the floor?** — the smallest model at which a
newline-localized rhyme plan appears. Our models (Gemma 2 2B, Llama 3.2 1B,
Qwen3 0.6B/1.7B) are the complete set with open CLTs on consumer hardware, so
"floor" here is empirically the smallest testable.

"Replicating Figure 13" splits into two experiments with very different force:

1. **As-is — the injection position sweep.** Suppress the natural rhyme
   features, inject an alternative, sweep the *steering position* across the
   prompt, and measure `P(inject word)`. The Figure-13 *signature* is a flat
   baseline with a single sharp spike. → the per-cell `figure13-*` folders.
   Result: the signature **reproduces** across three model families and three
   CLT scales — but the spike is at **emission**. These prompts truncate so the
   rhyme word is the *next token*: there is no line to compose, so "planning
   site" collapses onto emission.

2. **With planning — the composition horizon.** Anthropic's phenomenon *is*
   planning ahead over a line the model writes. `figure13-newline` restores
   that: truncate after the line-3 newline, let the model **compose line 4**,
   and ask whether steering at the newline shapes the rhyme (planning) or only
   emission-adjacent steering does (improvisation). Result: **emission,
   uniformly** — on all 0.6B–2B open models, including the word-level 2.5M CLT
   whose features are the closest analogue to Anthropic's "planned-word"
   features. These models sit **below the planning floor**: they improvise the
   rhyme at emission and do not plan it at the newline.

A prerequisite for trusting (2) is knowing the CLT encoder reads the right
residual. `figure13-newline/findings.md` §0 documents the **CLT-hook
reconciliation**: reconstruction proves the mntss/BlueLightAI CLT encoder was
trained on `ResidMid` (the MLP input) — the residual candle-mi already uses —
which also surfaced a latent encoder-hook mismatch in an upstream detection
step.

## Subfolders

| Folder | Experiment |
|--------|-----------|
| [`figure13-newline/`](figure13-newline/findings.md) | **The floor test.** Exp 1 newline feature census (correlational) + Exp 2 composition-horizon steering (causal, the true Figure-13 analogue with an m4 position sweep) + the CLT-hook reconciliation. |
| [`figure13-gemma-426k/`](figure13-gemma-426k/findings.md) | Figure-13 injection sweep — Gemma 2 2B × mntss 426K CLT (group-level rhyme features). |
| [`figure13-llama-524k/`](figure13-llama-524k/findings.md) | Figure-13 injection sweep — Llama 3.2 1B × mntss 524K CLT. |
| [`figure13-qwen3-0.6b-16k/`](figure13-qwen3-0.6b-16k/) · [`-0.6b-20k/`](figure13-qwen3-0.6b-20k/) · [`-1.7b-20k/`](figure13-qwen3-1.7b-20k/) | Figure-13 injection sweeps — Qwen3 0.6B/1.7B × BlueLightAI JumpReLU CLTs. |
| [`figure13-qwen3-cross-size.md`](figure13-qwen3-cross-size.md) | Qwen3 0.6B → 1.7B cross-size comparison at matched CLT width. |
| [`clt-vs-plt-planning-site/`](clt-vs-plt-planning-site/findings.md) | CLT vs PLT method-matched comparison on the rhyme planning site (Hanna & Ameisen, *Latent Planning Emerges with Scale*). |
| [`maar-replication/`](maar-replication/findings.md) | Maar et al. contrastive-activation-steering replication (*What's the plan?*). |
| [`gridworld-prolepsis/`](gridworld-prolepsis/) · [`means-ends-prolepsis/`](means-ends-prolepsis/) | Prolepsis (early irrevocable commitment) probes in planning-flavoured tasks. |

## How to read a `findings.md`

Every write-up follows the same shape: a **headline** result up top, a
**reproduce** block with the exact `cargo run` / `python` commands (set
`$env:HF_TOKEN` for gated models), per-cell **data tables**, and **caveats**.
The JSON files alongside are the raw data (`Import`-able into Mathematica); the
Figure-13 papers copy the relevant JSONs into their own `data/` directory to
drive figures.
