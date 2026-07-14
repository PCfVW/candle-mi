# Exp 3b — random-model control (the dead-salmon test)

**Date**: 2026-07-14
**Hardware**: RTX 5060 Ti 16 GB, Windows 11 (GPU, F32).
**Spec**: `controls-and-breadth-spec.md` Exp 3b, feeding paper §5 "Random
baselines" + §1 interpretability-illusion clause.
**Harness**: [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs)
`--random-init` (new); library `MIModel::from_pretrained_random_init`.
**Driver / aggregation**: [`scripts/run_random_model.sh`](../../../scripts/run_random_model.sh),
[`scripts/random_controls_aggregate.py`](../../../scripts/random_controls_aggregate.py).

## Registered prediction

> Baseline P("around") is arbitrary (random unembedding); no position-specific
> spike for the target; whatever movement exists is unstable across seeds. The
> control validates the **pipeline** (sweep machinery + metric cannot conjure a
> spike) — it does not, and cannot, replace the within-model random-feature
> control of 3a (the CLT decoder was trained against the *real* unembedding).

## Design

Gemma 2 2B built **from config with seeded Gaussian-random weights** (`N(0,
0.02)`, no trained weight values read — only the config, tokenizer, and the
safetensors *tensor-name* header, so tied embeddings stay tied). The standard
suppress + inject position sweep runs unchanged, at the cell's best strength
(**s = 25**), with the **real** mntss 426K CLT features (that mismatch is the
point — the dead-salmon standard says the pipeline must not paint the published
structure when the network under it is random). **3 seeds** (0, 1, 2), for each
of two weight variants:

- **Random init** (primary): every weight fabricated as seeded `N(0, 0.02)` via
  `MIModel::from_pretrained_random_init` — the literal form of the CFP concern.
- **Weight shuffle** (stricter): every trained tensor's elements permuted in
  place (`MIModel::from_pretrained_shuffled`), preserving each tensor's exact
  value multiset — and hence its norm and scale statistics — while destroying
  learned structure. Rules out "the effect is just the weight scales" that a
  fresh Gaussian init changes.

Both fabricate weights via a custom `VarBuilder` backend in the loader's
deterministic request order, so `seed` reproduces the model exactly.

## Result — no manufactured spike, under either control

**Random init** (fresh `N(0, 0.02)` weights):

| Seed | baseline P(`around`) | max ratio | max position | site |
|---:|---:|---:|---:|---|
| 0 | 1.12 × 10⁻⁶ | **1.73×** | 31 | final |
| 1 | 2.41 × 10⁻⁶ | **3.31×** | 31 | final |
| 2 | 1.65 × 10⁻⁵ | **3.53×** | 31 | final |

**Weight shuffle** (norm-preserving):

| Seed | baseline P(`around`) | max ratio | max position | site |
|---:|---:|---:|---:|---|
| 0 | 2.01 × 10⁻⁸ | **11.9×** | 31 | final |
| 1 | 5.39 × 10⁻¹¹ | **1.5×** | 27 | not final |
| 2 | 2.15 × 10⁻⁹ | **99.1×** | 31 | final |
| **real trained** | 4.84 × 10⁻⁸ | **9,974,880×** | 31 | final |

- **No spike under either control.** The worst ratio is **3.5×** (random init)
  and **99×** (shuffle), against **9,974,880×** on the trained model — five to
  seven orders of magnitude below. The sweep is flat; the pipeline conjures
  nothing on a network with no learned structure.
- **The stricter shuffle rules out "just the norms."** Because it preserves each
  tensor's value multiset, its baseline floors (5 × 10⁻¹¹ – 2 × 10⁻⁸) straddle
  the real model's 4.8 × 10⁻⁸ — the scale statistics are intact — yet there is
  still no spike. Norm-matched randomness does not reproduce the effect.
- **Everything is seed-unstable, as predicted.** Baselines swing ~15× (random
  init) to ~370× (shuffle: 5.4 × 10⁻¹¹ → 2.0 × 10⁻⁸), set by the random/permuted
  unembedding rather than any computation. The shuffle's max *position* is also
  unstable (seed 1 lands at token 27, not the final token) — no consistent site.
- **The residual max is a trivial emission-site artefact**, not structure:
  injecting a decoder vector at the last position perturbs that position's own
  logit slightly (the same fact seen in 3a's random controls). At 1.5–99× it
  carries no planning content.

## Reading

The random-model controls validate the **pipeline**: the suppress+inject sweep
machinery and the P(target)/ratio metric **cannot manufacture** the Figure-13
spike on a network with no trained structure — neither on fresh Gaussian weights
nor on norm-preserving shuffled weights. Combined with 3a (within-model random
features/directions stay at ≤ 7 × 10⁻⁵ absolute), the positive result survives
random-feature, random-direction, and random-model controls — the
interpretability-illusion baseline the track asks for.

As the spec notes, this control *complements* rather than replaces 3a: because
the CLT decoder was trained against the real unembedding, on a random network the
injected direction has no reason to decode to `around` in the first place, so the
flat sweep is expected; its force is that the **metric + sweep** add no spurious
localization on top.

## Reproduce

```powershell
# 3 seeds each of random-init + weight-shuffle, Gemma 426K, s=25, real CLT:
bash scripts/run_random_model.sh
python scripts/random_controls_aggregate.py   # 3b sections of the printout
```

Per-seed raw sweeps (per-position P(target), `weight_source` logged):
[`random_model_seed0.json`](random_model_seed0.json) · 1 · 2 (random init);
[`random_model_shuffle_seed0.json`](random_model_shuffle_seed0.json) · 1 · 2
(weight shuffle).
Summary: [`random_controls_summary.json`](random_controls_summary.json)
(`exp3b_model`).
