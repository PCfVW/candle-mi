# CLT vs PLT planning-site comparison — findings (Llama 3.2 1B + Gemma 2 2B)

**Experiment:** [`examples/clt_vs_plt_planning_site.rs`](../../../examples/clt_vs_plt_planning_site.rs)
**Plan:** [`PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`](../../roadmaps/PLAN-PLT-LLAMA-PLANNING-SIGNAL.md)
**Instrumentation spec:** [candle-mi v0.1.9 roadmap V3, Step 1.7](../../roadmaps/candle_mi_v019_roadmap_V3.md)
**Raw data:**
- Llama 3.2 1B — [`clt_vs_plt_llama.json`](clt_vs_plt_llama.json) (Step B) · [`clt_step_a_llama.json`](clt_step_a_llama.json) (Step A paper replication)
- Gemma 2 2B — [`clt_vs_plt_gemma2_2b.json`](clt_vs_plt_gemma2_2b.json) (Step B) · [`clt_step_a_gemma2_2b.json`](clt_step_a_gemma2_2b.json) (Step A paper replication)

**Hardware / stack:** CUDA on RTX 5060 Ti 16 GB · candle 0.9 · F32 end-to-end · single CUDA forward per sweep position

**Run command:**
```powershell
# Llama 3.2 1B (default)
cargo run --release --features clt,transformer,mmap --example clt_vs_plt_planning_site
# Gemma 2 2B (v0.1.10)
cargo run --release --features clt,sae,transformer,mmap --example clt_vs_plt_planning_site -- --family gemma
```

---

# Llama 3.2 1B — findings

## One-line answer

**Outcome C** — CLT and PLT spikes land at different positions under the method-matched top-5 decoder-projection ranking that V3 Step 1.7 prescribes. A PLT with the same ranking method detects and causally controls the planning site sharply (`ΔP = +0.986`) where a CLT with the same method does not (`ΔP ≈ 5.7×10⁻⁷`, indistinguishable from baseline noise). The Jacopin-paper CLT result (`ΔP = +0.687`, ratio 133 879× on the best prompt, from [`examples/figure13_planning_poems.rs`](../../../examples/figure13_planning_poems.rs)) is separately preserved in [`clt_step_a_llama.json`](clt_step_a_llama.json) — locked in the same harness to confirm the divergence is about the *ranking method*, not the *transcoder class*.

---

## Primary metric — causal-effect delta on the planned-token logit

Per V3 Appendix A (§589), `ΔP(planned_word)` and `Δlogit(planned_word)` at the final token, for each (transcoder, protocol):

| Arm | Protocol | ΔP | Δlogit | Spike position | Spike token |
|---|---|---|---|---|---|
| **PLT** | suppress-only (top-5 at L14) | **+0.986** | **+49.83** | 30 | `" "` (trailing) |
| **PLT** | suppress+inject (top-5 + top-1 at L14) | **+0.986** | **+48.03** | 30 | `" "` (trailing) |
| **CLT** | suppress-only (top-5 at L14) | +5.7×10⁻⁷ | +0.32 | 7 | `" tree"` |
| **CLT** | suppress+inject (top-5 + top-1 at L14) | +5.5×10⁻⁷ | +0.30 | 7 | `" tree"` |
| _Reference:_ | | | | | |
| **CLT** (Step A) | suppress+inject, Jacopin `-ee` features (cross-layer) | +0.687 | +15.84 | 30 | `" "` (trailing) |

Baseline `P("that") = 1.06×10⁻⁶`, `logit = +8.13`. The Step A CLT row uses Jacopin's hand-picked cross-layer suppress set `{(13, 30985) "he", (9, 5488) "be", (14, 27874) "ne", (13, 32049) "we"}` and the `(14, 13043)` "that" inject feature — a *hand-selected, cross-layer* set, NOT the top-5 decoder-projection-derived set Step B uses for the CLT arm.

### Outcome C mapping (V3 Appendix A)

| Signature | Outcome |
|---|---|
| PLT spike at different (layer, position) vs CLT | **C** ✓ |

The CLT "spike at pos 7" under the method-matched protocol is ΔP ≈ 5.7×10⁻⁷ — within an order of magnitude of the baseline `1.06×10⁻⁶` and far below the PLT's +0.986. So "different spike locations" is the right label, but with an unusual twist: the CLT arm has effectively *no spike at all* under this ranking method. This is neither the ΔP_PLT ≈ 0 shape (Outcome A) nor the ΔP_PLT ≈ ΔP_CLT shape (Outcome B) — the asymmetry runs in the opposite direction from what a straight reading of V3 Appendix A's "PLT is worse at detection" framing would predict.

## Secondary metrics (V3 §A.610 diagnostics)

| Metric | Value |
|---|---|
| Pearson correlation of normalised sweep profiles (suppress-only) | **r = −0.605** |
| Pearson correlation of normalised sweep profiles (suppress+inject) | **r = −0.577** |
| Decoder-projection magnitude ratio — CLT top-1 / PLT top-1 (same-layer, L14) | 0.349 / 0.508 = **0.687** |
| CLT top-1 cosine under max-over-target-layers metric | **0.608** (vs 0.349 same-layer) |
| PLT `W_skip · residual[seq_len-1]` projection onto `unembed(" that")` | **+0.541** |

**Anti-correlated sweep profiles** (r ≈ −0.6) directly match V3 Appendix A's "flat PLT profile with a sharp CLT peak → Outcome C" diagnostic, but with the arms swapped: here PLT has the sharp peak (at pos 30) and CLT's "profile" is essentially noise (max at pos 7 is 1.63×10⁻⁶, less than 2× the baseline). The anti-correlation reflects positions where CLT's noise happens to dip.

**PLT is more concentrated than CLT at L14** for `unembed(" that")` (cosine 0.508 vs 0.349). The method-matched CLT ranking at a single layer undersells the transcoder: when CLT is re-ranked across all target-layer slices, the top-1 cosine jumps to 0.608 and the top-1 feature relocates from L14:13043 to L13:7978. The CLT's best features for decoding `" that"` sit at L13, not L14.

**The PLT `W_skip` linear path contributes materially** (+0.541 projection onto `unembed(" that")` at the spike position). Per V3 §1.7 (D) this is the quantity that decides whether the apparent PLT planning signal is carried by sparse features or by the skip path. A +0.541 projection is non-negligible and should be reported as a caveat on the interpretation of PLT's suppress-only ΔP: zeroing sparse features leaves the skip path untouched, so "PLT's causal effect" is an upper bound on the sparse-feature-mediated contribution.

---

## Investigation commitment: (A)–(F) discrimination battery

V3 Appendix A commits us to investigate when Outcome C materialises. Populating from already-captured data.

### (A) Dictionary resolution mismatch — **likely not dominant**

- Llama CLT (`mntss/clt-llama-3.2-1b-524k`): 32 768 features/layer × 16 layers = 524 288 total.
- Llama PLT (`mntss/transcoder-Llama-3.2-1B`): 131 072 features/layer × 16 layers = 2 097 152 total.
- PLT has **4× more features per layer** than CLT.

If PLT's strength came entirely from finer resolution, subsampling PLT to CLT's per-layer budget (take every 4th feature) and re-ranking should halve its top-1 cosine and weaken ΔP. We have not run that ablation, but the CLT top-1 cosine at L14 (0.349) is *already* below PLT's top-5 median (0.329), so PLT's concentration advantage is at least partly independent of resolution. **Resolution is a plausible contributor, unlikely to be the whole story. Subsampling ablation is a follow-up.**

### (B) Decoder-projection ambiguity (CLT side) — **substantial contributor**

CLT's rank-3 `W_dec[feature, target_offset, :]` requires picking a target-layer slice for the projection. We captured both:

- **Same-layer** metric (what the suppress set used): top-1 is `L14:13043` at cosine 0.349.
- **Max-over-target-layers**: top-1 is `L13:7978` at cosine **0.608** — roughly 75% higher than same-layer, and relocated one layer earlier.

**If we had run the suppress causal test against the max-over-target-layers top-5, the CLT spike would likely have materialised.** This was predicted by V3 Appendix A §B and the captured instrumentation now confirms it. The CLT's "best decoders for `" that"`" are at L13 (feeding L14 via the cross-layer decoder), not at L14 itself. The method-matched-at-L14 suppress set misses them.

This is the most mechanically resolvable explanation on the data we have. A rerun with the max-over-target-layers top-5 suppress set is the obvious follow-up experiment — it would either recover the CLT spike (confirming (B) as the dominant mechanism) or leave a residual gap (elevating (C) and (D) instead).

### (C) Cross-layer binding — **consistent with data**

Per-layer max of each top-5 feature's post-ReLU activation, across all 16 layers on the prompt:

CLT top-5:
- `L14:13043` — fires only at L14 (peak 75.2, zero elsewhere).
- `L8:13602` — fires only at L8 (peak 50.9, zero elsewhere).
- `L10:26284`, `L12:28751` — do not fire at all on this prompt.
- `L0:30018` — barely fires at L0 (peak 1.6).

PLT top-5:
- `L14:97352`, `L14:96121`, `L14:53461` — all fire only at L14 (peaks 3.0, 15.3, 8.4).
- `L14:10018`, `L14:116146` — do not fire at all on this prompt.

Two observations:

1. **Features that fire, fire only at their attached layer** for both arms. There is no multi-layer activation smear — the `encode_pre_activation` histograms bear this out (both L13/L15 neighbour distributions are quieter than L14). On this prompt, the "cross-layer binding" mechanism that V3 Appendix A posits for Outcome C ("CLT feature activates monotonically from layer X onward, PLT peaks sharply at layer Y ≠ X") does *not* manifest — both transcoders' features are layer-local.
2. **Several top-5 features don't fire at all.** Decoder-cosine ranking picks features that *could* point at `" that"` structurally, but some of them are never activated on this specific prompt. For a suppression test this is harmless (zeroing a zero has no effect), but it means the effective suppress set is smaller than 5 for both arms — the CLT arm's "method-matched suppress" may effectively reduce to one or two live features.

**(C) does not explain the Llama 3.2 1B outcome on this prompt.** The Jacopin cross-layer `-ee` features (which DO carry the Step A ΔP = +0.687 signal) would need separate trace inspection — they are not in the top-5 by decoder-projection at L14.

### (D) Activation-function regime — **observed differences, but difficult to attribute**

32-bin pre-activation histograms at L14 (single-position slice × all features flattened across 31 positions):

| Arm | pre-act range | strictly-positive fraction | 1% / 50% / 99% quantiles |
|---|---|---|---|
| CLT L14 | [−543, +445] | 99.8% | 12.9 / 12.9 / 43.7 |
| PLT L14 | [−219, +109] | 98.1% | −3.5 / 6.8 / 6.8 |

CLT's pre-activation distribution is substantially wider (range ≈ 1000 vs PLT's ≈ 330) and more strongly positively biased (99.8% positive vs 98.1%). PLT has a tighter, more symmetric pre-activation distribution — consistent with its pure-ReLU regime vs CLT's ReLU on a wider residual norm.

The PLT `W_skip · x` contribution (+0.541 projection onto `unembed(" that")` at the spike position) is the directly causal component: it says that even without any sparse feature, the PLT's linear skip path already contributes ~0.54 in logit units toward "that" at L14 on this prompt. This is a chunk of the PLT's total +49.8 Δlogit — small in magnitude but non-zero. **(D) flags an interpretation caveat rather than a resolution:** suppress-only intervention in the PLT leaves the skip path untouched, so the reported PLT ΔP is an upper bound on the sparse-feature contribution. A full `W_skip`-aware intervention (zero sparse features AND zero the skip projection onto this direction) would isolate the sparse-feature share.

### (E) Feature-granularity mismatch — **decoder vectors exported, qualitative inspection deferred**

Top-20 decoder vectors for each arm are serialized in the output JSON (`arms.{clt,plt}.top_20_decoder_vectors_same_layer`, each a `[20 × 2048]` F32 array). Qualitative semantic inspection — matching the decoder direction against `unembed(token)` for a vocabulary sweep, then clustering — is labour-intensive and deferred. We can report that the CLT top-5 spans multiple layers (0, 8, 10, 12, 14) while the PLT top-5 is all at L14 (by construction). The CLT feature inventory at L14 alone is 32 768 vs PLT's 131 072; PLT has ~4× more features to distribute the "that"-aligned decoder directions across.

### (F) Training-objective / training-data divergence — **flagged, not resolved**

The two transcoders come from separate training runs with different hyperparameters; reverse-engineering their training divergences is out of scope without access to the training code and data. Flagged for completeness.

---

## Interpretation

The Step B outcome is not the simple "PLT weak vs CLT strong" story V3 Appendix A's framing anticipated — the asymmetry runs in the opposite direction *when both arms use the same decoder-projection-at-a-single-layer ranking method*. Three readings, all supported by the instrumentation:

1. **The CLT arm's weakness is a ranking-method artefact, not a CLT transcoder limitation.** Discrimination (B) shows the CLT's best decoders for `" that"` are at L13, not L14. The method-matched-at-L14 ranking misses them. The Jacopin paper result (Step A, ΔP = +0.687) recovers CLT's signal by using cross-layer hand-picked features that span layers 9–14. A max-over-target-layers rerun should mechanically resolve this.
2. **The PLT arm's strength is partly mediated by the linear `W_skip` path** (+0.541 projection). Sparse-feature-only intervention leaves it untouched; PLT's ΔP = +0.986 is an upper bound on the sparse-feature contribution. Full `W_skip`-aware intervention would tighten this.
3. **On the primary metric as V3 Step 1.7 defines it** (top-5 decoder-projection at the spike layer, suppress at spike position, measure ΔP(planned_word)), **the PLT detects and controls the planning signal on Llama 3.2 1B** — a positive finding that extends Hanna & Ameisen's scale-emergence claim to 1B (their lower bound was nascent signal at 4B on Qwen-3).

The Stage 1 decision-checkpoint options are therefore:

- **Treat Step B as a clean Outcome C** and move to Gemma (v0.1.10) to get a second data point. Current read: PLT detects on Llama 3.2 1B; does it detect on Gemma 2 2B?
- **Run the (B) follow-up first** (CLT with max-over-target-layers top-5 suppress) to settle whether the asymmetry is ranking-method or transcoder-class. Low cost (one extra sweep); high information value.

Recommendation: run the (B) follow-up before committing Stage 2 scope.

---

## Caveats

- **Single prompt.** Results are on one rhyming-couplet prompt (the Llama `-ee` preset). V3 originally envisioned two prompts; we ran one. A two-prompt rerun would reduce the risk that Outcome C reflects a prompt-specific feature-ranking accident.
- **`W_skip` intervention not isolated.** Sparse-feature suppression leaves the skip path untouched; the reported PLT ΔP is an upper bound on the sparse-feature-mediated causal effect.
- **Candle 0.9 stack-drift.** The Step A CLT number reproduces at `0.687` on this stack vs plip-rs §Q2's `0.777` — documented in [`memory/project_llama_planning_reference.md`](../../../../../../.claude/projects/c--Users-Eric-JACOPIN-Documents-Code-Source-candle-mi/memory/project_llama_planning_reference.md). The PLT ΔP = +0.986 has no prior published number to drift from; its absolute magnitude may shift on other F32 runtimes.
- **Top-k sensitivity not tested.** We used top-5 per V3 Step 1.7. Whether ΔP scales smoothly with k or saturates quickly is not known from this run.
- **No Qwen-3 or Gemma data.** Llama 3.2 1B is one point on the CLT×PLT coverage matrix. Gemma 2 2B lands in v0.1.10; Qwen-3 is conditional (V3 Stage 2).

---

## Follow-ups (prioritised)

1. **Rerun CLT arm with max-over-target-layers top-5 suppress set** — directly tests discrimination (B). Expected to recover ΔP_CLT ≳ 0.3 if (B) is dominant. Low cost, decisive.
2. **Gemma 2 2B Llama-analogous experiment** — [V3 Step 1.6](../../roadmaps/candle_mi_v019_roadmap_V3.md), ships in v0.1.10 with the GemmaScopeNpz loader. Gives a second data point on the Outcome C signature.
3. **Two-prompt extension** on Llama — add a second rhyming-couplet prompt to reduce single-prompt risk.
4. **`W_skip`-aware intervention** — full PLT suppression that zeroes both sparse features AND the skip-projection onto `unembed(" that")`. Isolates the sparse-feature share of ΔP_PLT.
5. **PLT subsampling ablation** — subsample PLT's 131 072 features to CLT's 32 768 per-layer budget and re-rank. Mechanical resolution of (A).
6. **Qualitative top-20 decoder inspection** — identify what semantic content each top-20 CLT/PLT feature fires on (V3 Appendix A §E). Labour-intensive; deferred.

Items 1, 2, 3 are natural v0.1.9/v0.1.10 scope. Items 4, 5, 6 are post-release.

---

## Follow-up 1 results — CLT with max-over-target-layers top-5 suppress

Ran 2026-04-19 alongside the primary Step B sweeps (three extra CLT position sweeps, ~30 s added to total runtime). Same invocation, same prompt, same device. Raw data under `arms.clt.max_over_target_follow_up` in [`clt_vs_plt_llama.json`](clt_vs_plt_llama.json).

Suppress set (CLT max-over-target top-5 at L14): `{L13:7978, L13:6484, L13:29869, L10:157, L10:18549}` — all cross-layer, none at L14.

| Arm / protocol | ΔP | Δlogit | Spike pos |
|---|---|---|---|
| **CLT** suppress-only (max-over-target top-5) | **+0.871** | **+32.19** | 30 |
| **CLT** suppress+inject (max-over-target top-5 + max-over-target top-1 inject `L13:7978`) | **+0.901** | **+29.39** | 30 |
| **CLT** suppress+inject (max-over-target top-5 + same-layer top-1 inject `L14:13043`, held constant) | **+0.917** | **+36.19** | 30 |
| *Reference — primary Step B* | | | |
| CLT suppress-only (same-layer top-5, at L14) | +5.7×10⁻⁷ | +0.32 | 7 |
| PLT suppress-only (top-5 at L14) | +0.986 | +49.83 | 30 |

### Revised outcome: Outcome B, CLT/PLT ratio ≈ 0.93

Under a **method-matched ranking that respects the CLT's cross-layer decoder structure** (max-over-target-layers), the CLT arm recovers a strong causal spike at position 30 — the same position the PLT and the Step A Jacopin result both find. The ratio `ΔP_CLT / ΔP_PLT` is **0.871/0.986 = 0.88** (suppress-only) to **0.917/0.986 = 0.93** (best suppress+inject), well within the "comparable" band of V3 Appendix A's Outcome B ("_PLT is adequate for both detection and intervention. CLTs add cross-layer routing resolution but are not necessary for detection._"). Reclassification:

- **Primary Step B result: Outcome C** (different spike positions under the default same-layer ranking) — retained as a finding about ranking-method choice, not about transcoder-class limitation.
- **Follow-up 1 result: Outcome B** — the method-matched-per-transcoder-capabilities comparison. CLTs and PLTs both detect and control the Llama 3.2 1B planning site; the observed asymmetry disappears once the CLT's ranking metric respects its cross-layer decoder topology.

### Interpretation

1. **Discrimination (B) is confirmed as the dominant driver of the primary Step B asymmetry.** The CLT's best `" that"`-aligned features are at L13 (cosine 0.608 under max-over-target) and L10 (cosine ≈ 0.565), not at L14 (cosine 0.349 under same-layer). Same-layer ranking at L14 systematically missed them — a methodological artefact, not a CLT weakness.

2. **The ranking-method question is more general than this experiment.** V3 Step 1.7's "top-5 decoder-projection at the spike layer" prescription is neutral on which decoder-slice metric to use. For cross-layer transcoders, same-layer undersells; for per-layer transcoders (PltBundle, GemmaScopeNpz) the choice is trivial (only one slice by construction). A one-liner addition to the V3 methodology section — "use `max_{l' ≥ source_layer} cosine(W_dec[feature, l', :], unembed(word))` for CltSplit, `cosine(W_dec[feature, :], unembed(word))` for PltBundle/GemmaScopeNpz" — makes the comparison fully apples-to-apples without manual per-experiment tuning.

3. **Same-layer top-1 inject beats max-over-target top-1 inject** (0.917 vs 0.901). The dedicated same-layer "that" feature (L14:13043) is a more effective counterfactual than the max-over-target top-1 (L13:7978 — better aligned structurally but writes to L14 via a decoder projection rather than *being* the L14 representation). Minor, but suggests a natural asymmetry: for **suppression** the max-over-target ranking wins; for **injection** the same-layer ranking wins. Inject at the target layer, suppress wherever the decoder projects from.

4. **The outcome-label pair (C primary, B with method-matched ranking) is itself an informative result.** A paper reporting the Llama 3.2 1B CLT-vs-PLT comparison would need to pick one; the honest answer is both, with the caveat spelled out. Hanna & Ameisen's scale-emergence claim for PLTs (nascent signal at 4B on Qwen-3) is *consistent* with PLTs detecting on Llama 3.2 1B under Outcome B, but the (B)-discrimination attribution means their lower bound does not need revising upward — PLT on Llama 3.2 1B matches a well-ranked CLT, so the "PLT planning emerges with scale" story is not directly challenged by our single-model result.

### Stage 1 decision (revised)

Given Outcome B with `ΔP_PLT ≈ ΔP_CLT` (ratio 0.93 at best) on Llama 3.2 1B:

- **V3 stop condition (§444)** says: "_Outcome B with ΔP_PLT ≈ ΔP_CLT (both detection and intervention match) and Hanna & Ameisen's existing 0.6B–14B data already suffices. Write up Stage 1 as a short paper; defer Stage 2._"
- **Our Llama data meets the Outcome B match-criterion** under method-matched ranking. What it does **not** meet is the second clause's premise — a rigorous re-read of Hanna & Ameisen (2604.12493v1) shows their Qwen-3 rhyming result is _nascent even at 14B_ ("overall, these results suggest a lack of strong backward planning"; local planning features in a small minority of couplets). Their rhyming coverage does not "suffice" in the sense §444 assumed; the Stage-1 / Stage-2 trade-off needs to be revisited once Gemma is in.
- **Gemma 2 2B (v0.1.10) is the next data point** — second same-methodology observation on a different model family. Turns one into a pair, strengthens any public claim about the Llama finding.
- **V3 Stage 2 (Qwen-3 scale sweep)** — previously flagged as "redundant given H&A coverage" in an earlier draft of this section. Withdrawn. H&A's Qwen-3 rhyming result uses a fundamentally different feature-identification method (pattern-matched top-activating tokens) and intervention protocol (multiplicative ×7/×−3) from ours (decoder-projection cosine + additive ±10). Our Step B follow-up already demonstrated that ranking-method choice alone can swing ΔP by seven orders of magnitude on the same transcoder and prompt — so it is at least plausible that the H&A Qwen-3 nascency is partly methodology-driven rather than scale-limited. A matched-methodology Qwen-3 run is the only way to disentangle this, and it is scientifically interesting regardless of direction.

Recommendation (revised): ship v0.1.9 with this finding, proceed to Gemma 2 2B (v0.1.10). **Decide on V3 Stage 2 after Gemma lands**, with the methodology-confound hypothesis as explicit motivation rather than as something to overrule.

### Updated discrimination-battery summary

| Finding | Status |
|---|---|
| (A) Dictionary resolution mismatch | Plausible contributor; subsampling ablation still a follow-up but no longer load-bearing. |
| (B) CLT decoder-projection ambiguity | **Confirmed dominant**. Max-over-target ranking recovers CLT spike (ΔP_CLT = +0.871 to +0.917, up from 5.7×10⁻⁷). |
| (C) Cross-layer binding | Does not manifest on this prompt (top-5 features that fire are all layer-local). |
| (D) Activation-function regime | PLT `W_skip · x` projection = +0.541 is non-negligible; suppress-only PLT ΔP still an upper bound on sparse-feature contribution. Isolated by follow-up 4. |
| (E) Feature-granularity mismatch | Decoder vectors exported; qualitative inspection deferred. |
| (F) Training-objective divergence | Out of scope. |

---

# Gemma 2 2B — findings

**Transcoders:** CLT `mntss/clt-gemma-2-2b-426k` (16 384 features × 26 layers, `CltSplit` schema, `BF16`); PLT `mntss/gemma-scope-transcoders` curation entry-point routing to `google/gemma-scope-2b-pt-transcoders` weights (16 384 features × 26 layers, `GemmaScopeNpz` schema, `JumpReLU` with per-feature threshold, no `W_skip`, F32 NPZ on disk).

**Prompt** (figure13 `gemma2-2b-426k` preset): `"The stars were twinkling in the night, / The lanterns cast a golden light. / She wandered in the dark about, / And found a hidden passage"` — `"-out"` → `" around"` rhyming-couplet pair.

**Step A (Jacopin replication, hand-picked features):** suppress `(L16:13725)` + `(L25:9385)` ("about", "out"), inject `(L22:10243)` ("around"), strength 10. Result: **`P(" around") = 0.4567`** at the trailing-space spike (position 31, after "passage"), within ±0.01 of the pinned reference (`GEMMA.reference_max_prob = 0.457`, measured 2026-05-01 via figure13). Step A confirms the planning signal is in the model and is causally controllable when the suppress/inject features are chosen via the original Jacopin protocol.

## What "detection" means here

The Step B "no detection" verdict below is a claim about the **V3 Step 1.7 method-matched ranking**, not about whether the planning signal exists in Gemma 2 2B. Two distinct protocols inhabit this experiment. **Step A (the Jacopin paper protocol)** uses hand-picked suppress + inject features chosen by domain expertise on the rhyme-cluster geometry — figure13 reproduces this on both Llama 3.2 1B and Gemma 2 2B; the v0.1.10 candle-mi run reproduced it again at `P(" around") = 0.4567` (within ±0.01 of the pinned reference). **Step B (the V3 Step 1.7 method-matched ranking)** replaces the hand-picking with an automatic procedure: take the top-5 features by `cosine(decoder_row(feature, target_layer), unembed(inject_word))` at the inject feature's source layer, then run the same suppress / suppress+inject sweep. The point of Step B is to score CLT and PLT *by the same scoring procedure*, so the comparison across transcoder classes is fair.

When Step B's `ΔP` is at noise level, the model has not lost its planning behaviour — Step A is the ground truth that the planning signal exists and is causally controllable, and Step A is reproducible on this build. What a Step B null tells us is that **the automatic top-5 decoder-projection ranking does not recover the same features the hand-picked Jacopin protocol does**. On Llama 3.2 1B, swapping the CLT's same-layer ranking for max-over-target ranking (which respects the cross-layer decoder topology) recovered the signal (Outcome B, ratio 0.93). On Gemma 2 2B that same fix does *not* recover it — the Step B null is robust across both same-layer and max-over-target rankings, on both transcoder classes. So the Gemma Step B result is a finding about the **method-matched protocol's sensitivity at this model scale and architecture**, not a claim that Gemma 2 2B fails to plan rhymes (it manifestly does, per Step A's `P(" around") = 0.4567`).

## One-line answer

**Outcome A-mirror — neither arm detects under the V3 Step 1.7 method-matched ranking.** Both the GemmaScope PLT and the mntss CLT, ranked by top-5 decoder-projection-cosine onto `unembed(" around")` at L22 (the hand-picked inject feature's layer), produce ΔP values at baseline noise:

| Arm | Protocol | `top_k_target_layer` | Suppress set (top-5) | ΔP | Δlogit | Spike pos |
|---|---|---|---|---|---|---|
| **PLT** (GemmaScope) | suppress-only | L22 | `{L22:1804, L22:2111, L22:16225, L22:11833, L22:6010}` | **+4.46×10⁻¹¹** | +0.001 | 20 (` in`) |
| **PLT** (GemmaScope) | suppress+inject | L22 | + inject `L22:1804` | **+4.41×10⁻¹¹** | +0.001 | 20 (` in`) |
| **CLT** (same-layer top-5 at L22) | suppress-only | L22 | `{L19:962, L20:13957, L21:6897, L20:14424, L7:8254}` | **+1.75×10⁻⁷** | -3.07 | 31 (` `) |
| **CLT** (same-layer top-5 at L22) | suppress+inject | L22 | + inject `L19:962` | **+1.07×10⁻⁹** | +0.04 | 20 (` in`) |
| **CLT** (max-over-target top-5) | suppress-only | L22 | `{L24:8075, L23:5179, L21:11103, L23:2282, L19:962}` | **−1.69×10⁻⁹** | −0.034 | 20 (` in`) |
| **CLT** (max-over-target top-5) | suppress+inject (ranked) | L22 | + inject `L24:8075` | **−1.63×10⁻⁹** | −0.033 | 20 (` in`) |
| **CLT** (max-over-target top-5) | suppress+inject (same-layer inject) | L22 | + inject `L19:962` (same-layer top-1) | **−1.08×10⁻⁹** | −0.024 | 20 (` in`) |
| *Reference* | | | | | | |
| Step A (Jacopin features) | suppress+inject | — | `{L16:13725, L25:9385}` + inject `L22:10243` | **+0.4567** | (huge) | 31 (` `) |
| Baseline `P(" around")` | — | — | — | 4.84×10⁻⁸ | — | — |

All Step B ΔPs are within an order of magnitude of the baseline 4.84×10⁻⁸ — the apparent "spike" at position 20 (` in`, mid-prompt) is at probability ~5×10⁻⁸, indistinguishable from no-intervention noise. Position 31 (the structural planning site that Step A controls cleanly) registers as the top spike for only one arm/protocol combination (CLT suppress-only same-layer), and even there ΔP = +1.75×10⁻⁷ — three to four orders of magnitude below Step A's +0.4567.

## Outcome label

**Degenerate Outcome B (both arms ≈ 0)** — `ΔP_PLT ≈ ΔP_CLT` matches Outcome B's signature, but with both at noise level rather than at a strong shared peak. The honest reading is **"neither method-matched protocol detects the planning signal that Step A demonstrably can control"**, which V3 Appendix A's four outcomes (A/B/C/D) don't cleanly capture — the closest cell is "Outcome A inverted": Outcome A is "PLT misses, CLT detects strongly"; here PLT misses *and* CLT also misses (under both same-layer and max-over-target rankings).

## Comparison to Llama 3.2 1B

The two-model picture is now genuinely informative:

| Model | CLT same-layer | CLT max-over-target | PLT same-layer | Step A (Jacopin) |
|---|---|---|---|---|
| **Llama 3.2 1B** | +5.7×10⁻⁷ (≈0) | **+0.871** | **+0.986** | +0.687 |
| **Gemma 2 2B** | +1.75×10⁻⁷ (≈0) | −1.69×10⁻⁹ (≈0) | +4.46×10⁻¹¹ (≈0) | +0.4567 |

Three contrasts that didn't appear in the Llama-only writeup:

1. **The (B) discrimination resolution does not transfer.** On Llama, switching CLT from same-layer to max-over-target ranking recovered ΔP from 5.7×10⁻⁷ → +0.871 (six-orders-of-magnitude improvement). On Gemma, the same switch leaves CLT at noise (−1.69×10⁻⁹). So discrimination (B) is *not* the dominant story on Gemma — the CLT's max-over-target features (`L24:8075`, `L23:5179`, …) are simply not the features that mediate the planning signal here.
2. **GemmaScope's `W_skip` is structurally absent.** On Llama, the PLT `W_skip · x` projection (+0.541) was a non-negligible contributor to the apparent PLT signal — sparse-feature suppression was an upper bound. On Gemma, GemmaScope is a pure JumpReLU transcoder with no `W_skip`, so the suppress-only ΔP *is* the full sparse-feature-mediated effect. The number is +4.46×10⁻¹¹ — meaning the top-5 decoder-aligned sparse features genuinely don't move `P(" around")`.
3. **PLT cosines are weaker on Gemma.** Top-5 at L22 max cosine = 0.50 (Gemma) vs ~0.65 (Llama). The decoder-projection ranking method finds weaker `unembed(" around")`-aligned features at the inject-feature's layer.

## Implications for Stage 1 decision

Combining the two models:

- **Llama 3.2 1B (revised, with max-over-target follow-up):** Outcome B, ratio 0.93 — both transcoders detect when the CLT ranking respects its cross-layer decoder topology.
- **Gemma 2 2B:** degenerate Outcome B (both ≈ 0) — neither method-matched ranking detects, even though Step A confirms the planning signal is present and controllable via the Jacopin features.

The Stage 1 stop condition (V3 § Decision checkpoint) reads: "_Outcome B with ΔP_PLT ≈ ΔP_CLT (both detection and intervention match) and Hanna & Ameisen's coverage of the same task genuinely suffices._" Gemma's degenerate-B result **fails the "match" clause** in spirit — both arms at zero is not the kind of "match" the stop condition envisaged. So:

- The Llama-only Outcome B (ratio 0.93) is no longer the whole story.
- The Gemma double-null is itself a notable finding: **the V3 Step 1.7 method-matched ranking has lower sensitivity than the original Jacopin paper protocol on this model**, possibly losing the planning signal entirely on the larger of our two test models.
- This **strengthens the case for proceeding to Stage 2** (Qwen-3 scale sweep at 0.6B/1.7B/4B) — the Llama-Gemma pair already shows that the method-matched ranking's behaviour varies wildly across models within a 1.7-2× parameter span, and Hanna & Ameisen's "PLT planning emerges with scale" claim deserves a matched-methodology test.

Specifically, the V3 § Decision checkpoint stop condition says the run should fire on Outcome B with `ΔP_PLT ≈ ΔP_CLT` *and* the H&A coverage being sufficient. Gemma's null-B doesn't trigger the stop, and the H&A "suffices" premise was already shown to be unsupported for rhyming. **Both clauses now point toward continuing to Stage 2** — but Stage 2 is conditional on user direction; this section reports the data, not the decision.

## Caveats specific to Gemma

- **Single prompt.** Same prompt as figure13's `gemma2-2b-426k` preset; no second-prompt cross-check.
- **Hookpoint difference.** GemmaScope's encoder reads from `MlpPre` (post-`LN2`); CLT reads from `ResidMid`. The harness captures both at every layer (`PltInputHook::MlpPre` for GemmaScope, `ResidMid` for the CLT — see [`examples/clt_vs_plt_planning_site.rs`](../../../examples/clt_vs_plt_planning_site.rs) `FamilyPreset.plt_input_hook`). Per-arm residuals confirmed correct in Step A's reproduction of `P(" around") = 0.4567`.
- **`top_k_target_layer = 22`** (the inject feature's layer per the Jacopin protocol). Whether a different target-layer choice would reveal different decoder-projection features is not tested here — the V3 Step 1.7 prescription pins it to the inject-feature's source layer.
- **Discrimination battery (A)–(F) not yet populated for Gemma.** The instrumentation (top-20 decoder vectors, all-layer activation traces, pre-activation histograms at L22 ± 1, both CLT decoder-slice metrics in parallel) is captured in [`clt_vs_plt_gemma2_2b.json`](clt_vs_plt_gemma2_2b.json); a full (A)–(F) analysis paralleling the Llama section is a v0.1.11+ follow-up.

## Discrimination battery — Gemma (preliminary, from already-captured data)

| Finding | Status |
|---|---|
| (A) Dictionary resolution mismatch | Both arms 16 384 features/layer × 26 layers — **identical** dictionary sizes, no resolution gap to attribute. |
| (B) CLT decoder-projection ambiguity | **Does not transfer from Llama.** Max-over-target ranking on Gemma leaves CLT at noise (−1.7×10⁻⁹), unlike Llama where it recovered the spike. |
| (C) Cross-layer binding | Top-5 features under both rankings are layer-local; not investigated in detail. |
| (D) Activation-function regime | GemmaScope is `JumpReLU` with per-feature threshold, no `W_skip`. The +4.5×10⁻¹¹ PLT result *is* the full sparse-feature contribution (no upper-bound caveat as on Llama). |
| (E) Feature-granularity mismatch | Top-5 same-layer PLT features at L22 (cosines 0.50, 0.40, 0.38, 0.17, 0.15) — qualitative inspection deferred. |
| (F) Training-objective divergence | Out of scope. |
