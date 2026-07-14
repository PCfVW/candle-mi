# Exp 4 — prompt breadth (controls-and-breadth spec)

**Date**: 2026-07-14
**Hardware**: RTX 5060 Ti 16 GB, Windows 11 (GPU, F32).
**Spec**: `controls-and-breadth-spec.md` Exp 4 (amended 2026-07-14), feeding
paper §4.1 breadth counts + Limitations boundary paragraph.
**Harness**: [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs)
(no new code — `--prompt` + `--strength-grid` already exist).
**Driver / aggregation**:
[`scripts/breadth_aggregate.py`](../../../scripts/breadth_aggregate.py).

## Registered prediction

> Spike at the final token in most prompts per cell (report k/n with exact CI).
> Failure mode: the single-prompt result was prompt-specific.

## Design

The COLM-era study validated **4 prompts per reference cell**; the current
paper's grid sweeps ran only prompt #1. We rerun the **three other prompts** per
cell through the grid harness at the cell's Table-2 best strength (**s = 25**),
holding the cell's preset **suppress + inject features and inject word fixed** —
only the prompt varies. Prompt #1 is the reference cell, taken from its committed
Table-2 grid (already at s = 25).

- **Gemma cell** (`gemma2-2b-426k`): suppress -out (L16:13725, L25:9385), inject
  `" around"` (L22:10243). 4 prompts pinned from plip-rs
  `poetry_category_steering.rs` `mode_position_sweep` `candidate_specs` (the
  authoritative record of the 136-pair Figure-13 sweep): `-out "about"` (#1),
  `-ow "so"`, `-out "shout"`, `-oo "who"`.
- **Llama cell** (`llama3.2-1b-524k`): suppress -ee (L13:30985, L9:5488,
  L14:27874, L13:32049), inject `" that"` (L14:13043). 4 prompts from the
  committed `plip-rs/corpus/llama_prompts.json` = the actual 44-pair sweep
  (`suppress_inject_sweep_llama_v2.json`): `-ee "free"` (#1), `-oo "new"`,
  `-at "sat"`, `-ore "more"`. (Authoritative set chosen by Eric, 2026-07-14,
  over the paper's differently-described "47 / -ee×2" set.)

For prompts whose natural rime differs from the cell's suppress group, the
suppress half is inert (those features are not active), so the run is a clean
test of whether **injecting the same target feature spikes P(target) at the final
token regardless of prompt** — a strictly harder generalization test than
same-group reruns.

## Result — 4/4 final-token localization in both cells

Every prompt spikes at the **final token** (the trailing-space planning site);
none is off by even one position.

### Gemma 2 2B × mntss 426K — inject `" around"`

| Prompt (natural rime) | n | spike pos | site | best P | ratio |
|---|---:|---:|---|---:|---:|
| `-out "about"` (#1, ref) | 32 | 31 | final | 0.482 | 9,974,880× |
| `-ow "so"`               | 34 | 33 | final | 0.559 | 4.66 × 10¹⁰× |
| `-out "shout"`           | 35 | 34 | final | 0.478 | 1.24 × 10⁸× |
| `-oo "who"`              | 36 | 35 | final | 0.383 | 9.31 × 10⁸× |

**Localization: 4/4 = 100%**, Clopper–Pearson 95% CI **[0.398, 1.000]**.
Best ratio median 5.3 × 10⁸×, range [9.97 × 10⁶×, 4.66 × 10¹⁰×].

### Llama 3.2 1B × mntss 524K — inject `" that"`

| Prompt (natural rime) | n | spike pos | site | best P | ratio |
|---|---:|---:|---|---:|---:|
| `-ee "free"` (#1, ref) | 31 | 30 | final | 0.852 | 806,260× |
| `-oo "new"`            | 31 | 30 | final | 0.857 | 385,411× |
| `-at "sat"`            | 33 | 32 | final | 0.750 | 1,107× |
| `-ore "more"`          | 32 | 31 | final | 0.804 | 138,683× |

**Localization: 4/4 = 100%**, Clopper–Pearson 95% CI **[0.398, 1.000]**.
Best ratio median 2.6 × 10⁵×, range [1,107×, 806,260×].

## Null-model probability (per-prompt binomial)

Under the uniform-position null (each prompt's spike equally likely at any of its
`nᵢ` positions), the probability of all four prompts landing on the final token
is `Πᵢ (1/nᵢ)`:

- Gemma: `1/(32·34·35·36) = 7.3 × 10⁻⁷`
- Llama: `1/(31·31·33·32) = 9.9 × 10⁻⁷`
- Combined (8/8 prompts): **7.2 × 10⁻¹³**

This complements the seven-cell localization null in
[`localization_null.md`](localization_null.md).

## Reading

The single-prompt Table-2 result is **not** prompt-specific: holding the
intervention fixed and swapping in the other three COLM-validated prompts
reproduces the emission-adjacent spike every time, including for prompts whose
natural rime is foreign to the injected feature. Absolute P and ratio vary with
the prompt (as expected — the transcoder and the prompt's own prior set the
ceiling), but the **position** does not. This is the breadth the paper's
Limitations paragraph previously flagged as untested.

The registered 70% / 85% per-(prompt × group) localization of Jacopin (2026) at
s = 10 is the pair-level breadth that already existed; this rerun adds
grid-validated strength (s = 25) and per-prompt resolution, and lands at 4/4 in
both cells.

## Reproduce

```powershell
# 6 sweeps (3 per cell), s=25, preset features fixed, only --prompt varies:
bash scripts/run_breadth.sh              # sets HF_TOKEN from the cached login
python scripts/breadth_aggregate.py      # -> breadth_<cell>.json + this table
```

Per-prompt raw sweeps: `_runs/breadth_<preset>_<label>.json` (6 files, full
position × strength=25 grid, prompt text verbatim). Aggregates:
[`breadth_gemma-426k.json`](breadth_gemma-426k.json),
[`breadth_llama-524k.json`](breadth_llama-524k.json).
