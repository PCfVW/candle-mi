# CLT vs PLT planning-site comparison on Llama 3.2 1B — findings

**Experiment:** [`examples/clt_vs_plt_planning_site.rs`](../../../examples/clt_vs_plt_planning_site.rs)  
**Plan:** [`PLAN-PLT-LLAMA-PLANNING-SIGNAL.md`](../../roadmaps/PLAN-PLT-LLAMA-PLANNING-SIGNAL.md)  
**Instrumentation spec:** [candle-mi v0.1.9 roadmap V3, Step 1.7](../../roadmaps/candle_mi_v019_roadmap_V3.md)  
**Raw data:** [`clt_vs_plt_llama.json`](clt_vs_plt_llama.json) (Step B) · [`clt_step_a_llama.json`](clt_step_a_llama.json) (Step A paper replication)  
**Hardware / stack:** CUDA on RTX 5060 Ti 16 GB · candle 0.9 · F32 end-to-end · single CUDA forward per sweep position

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
