# candle-mi v0.1.9 — PLT Support, CLT vs PLT Comparison & Latent Planning Replication
**Target:** (1) Run the CLT vs PLT controlled comparison on Llama 3.2 1B + Gemma 2 2B; (2) Replicate Hanna & Ameisen (2026), *"Latent Planning Emerges with Scale"* [arXiv:2604.12493](https://arxiv.org/abs/2604.12493)
**Status baseline:** v0.1.8, Phase 6 partially complete. CLT pipeline validated on Gemma 2 2B + Llama 3.2 1B. No Qwen-3 support yet.
**Hardware constraint:** RTX 5060 Ti, 16 GB VRAM. Models not fitting on this card are out of scope.

---

## Transcoder inventory

| Model | CLTs (candle-mi validated) | PLTs (newly confirmed) |
|---|---|---|
| Llama 3.2 1B | [`mntss/clt-llama-3.2-1b-524k`](https://huggingface.co/mntss/clt-llama-3.2-1b-524k) | [`mntss/transcoder-Llama-3.2-1B`](https://huggingface.co/mntss/transcoder-Llama-3.2-1B) |
| Gemma 2 2B | [`mntss/clt-gemma-2-2b-426k`](https://huggingface.co/mntss/clt-gemma-2-2b-426k) · [`mntss/clt-gemma-2-2b-2.5M`](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M) | [`mntss/gemma-scope-transcoders`](https://huggingface.co/mntss/gemma-scope-transcoders) (originally `google/gemma-scope-2b-pt-transcoders`) |
| Qwen3-0.6B | — | [`mwhanna/qwen3-0.6b-transcoders-lowl0`](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0) |
| Qwen3-1.7B | — | [`mwhanna/qwen3-1.7b-transcoders-lowl0`](https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0) |
| Qwen3-4B | — | [`mwhanna/qwen3-4b-transcoders`](https://huggingface.co/mwhanna/qwen3-4b-transcoders) |
| Qwen3-8B | — | [`mwhanna/qwen3-8b-transcoders`](https://huggingface.co/mwhanna/qwen3-8b-transcoders) |
| Qwen3-14B | — | [`mwhanna/qwen3-14b-transcoders-lowl0`](https://huggingface.co/mwhanna/qwen3-14b-transcoders-lowl0) ❌ OOM |

All PLTs are from the [`mntss/per-layer-transcoders`](https://huggingface.co/collections/mntss/per-layer-transcoders) collection. All CLTs are from the [`mntss/cross-layer-transcoders`](https://huggingface.co/collections/mntss/cross-layer-transcoders) collection.

---

## Priority reordering

The discovery that PLTs exist for **both Llama 3.2 1B and Gemma 2 2B** changes the roadmap materially. The scientifically most valuable experiment — a controlled CLT vs PLT comparison on the same models, same prompts, same hardware — requires **no Qwen-3 support at all**. It can run the moment PLT loading and injection land in candle-mi.

This comparison directly tests the central tension in the two papers: Jacopin's Q1 result claims PLTs (and all residual-stream methods) miss the planning site; Hanna & Ameisen use PLTs and find interpretable planning features. No existing paper has run both transcoder types on the same prompt.

New sequence: **PLT support → CLT vs PLT comparison → Qwen-3 backend → latent planning scale sweep.**

---

## Step 1 — PLT Support in `src/clt/mod.rs`

PLTs are architecturally simpler than CLTs: one transcoder per MLP layer, injecting only into its own layer's output.

**`W_dec` shape is the distinguishing signal:**
- CLT: `W_dec_{l}` has shape `[n_features, n_out_layers, d_model]` (rank-3)
- PLT: `W_dec_{l}` has shape `[n_features, d_model]` (rank-2)

Shape detection happens at load time in `open()`, when `W_dec_0` is downloaded for dimension inspection (already downloaded alongside `W_enc_0` in the current implementation). No API breakage.

**Changes to `src/clt/mod.rs`:**
- Add `pub enum TranscoderKind { CrossLayer, PerLayer }` and `kind: TranscoderKind` field to `CltConfig`
- In `open()`: after loading `W_dec_0`, check tensor rank → set `kind`
- In `cache_steering_vectors_all_downstream()`: branch on `kind`
  - `CrossLayer`: existing behaviour (`n_target_layers = n_layers - source_layer`, iterating all offsets)
  - `PerLayer`: `n_target_layers = 1`, `target_layer = source_layer`, `target_offset = 0` — single column extraction
- `prepare_hook_injection()` and `inject()`: **no changes needed** — they already work correctly for a single `ResidPost(layer)` entry

**Gemma 2 2B PLT caveat:** GemmaScope transcoders reconstruct the MLP output *after the post-MLP RMSNorm*, not before. Gemma 2's 4-norm architecture means the PLT decoder vector lives in a different space than the CLT decoder. Verify the correct injection hook point (`MlpOut` vs `ResidPost`) by checking how circuit-tracer injects GemmaScope PLTs before writing the validation test.

**Validation:** load `mntss/transcoder-Llama-3.2-1B`, compare top-10 active features on 3 prompts vs Python circuit-tracer reference. Llama 3.2 has no extra norms, so it's the cleaner first target.

---

## Step 2 — CLT vs PLT Controlled Comparison

This is the novel scientific contribution enabled by Step 1. Everything else (models, prompts, infrastructure) already exists in candle-mi.

**Experiment: `clt_vs_plt_planning_site.rs`**

For each of the two rhyming-couplet prompts used in `figure13_planning_poems`:

1. **CLT run** (existing): load `mntss/clt-llama-3.2-1b-524k`, encode residual at each position, identify the planning-site spike, record: spike layer, spike magnitude, top-5 active features, position-sweep profile
2. **PLT run** (new): load `mntss/transcoder-Llama-3.2-1B`, same encode loop, same recording
3. **Compare**: does the PLT encoder fire on the same layer? On the same position? With comparable magnitude? Or does it miss the spike entirely (confirming Q1)?

Run identically on Gemma 2 2B (CLT: `mntss/clt-gemma-2-2b-426k`, PLT: `mntss/gemma-scope-transcoders`) once the Gemma injection point is confirmed.

**Output:** JSON per model × transcoder type, compatible with deloson format. Tabular summary of spike layer, spike position, and top feature overlap (Jaccard) between CLT and PLT activations at the planning site.

This is the experiment that either confirms or refutes the implicit methodological disagreement between the two papers.

---

## Step 3 — Qwen-3 Transformer Backend

Only needed for the scale-sweep replication. Qwen-3 is Qwen-2 plus per-head QK LayerNorm.

**Changes to `src/transformer/`:**
- Add `qk_norm: bool` + `qk_norm_eps: f64` to `TransformerConfig`
- Add optional `q_norm` / `k_norm` `RmsNorm` fields to the attention module, applied after RoPE: `Q = q_norm(Q)`, `K = k_norm(K)`
- Add `parse_qwen3()` reading `model_type = "qwen3"`; weight names follow LLaMA-style convention (`model.layers.{i}.self_attn.q_norm.weight`)

**VRAM budget:**

| Model | VRAM (est.) | Mode |
|---|---|---|
| Qwen3-0.6B | ~3 GB | ✅ F32 |
| Qwen3-1.7B | ~7 GB | ✅ F32 |
| Qwen3-4B | ~16 GB | ✅ F32 (tight) |
| Qwen3-8B | ~16 GB | ✅ BF16 only |
| Qwen3-14B | ~28 GB BF16 | ❌ out of scope |

**Validation:** 5 prompts × top-10 logits vs Python HF, abs < 1e-4 (F32), on Qwen3-0.6B-Base and Qwen3-1.7B-Base.

---

## Step 4 — Latent Planning Scale Sweep (Hanna & Ameisen replication)

### 4a — Article Prediction (`latent_planning_article.rs`)

Prompt: `"Someone who handles financial records is →"`

Protocol:
1. Forward pass; encode `ResidMid` at article-position token via PLT encoder for each layer
2. Score active features by dot product with `unembed("an") - unembed("a")`
3. **Causal test:** suppress top planning features; measure shift in `logit("an") - logit("a")`
4. Log per model size: planning feature count, depth of top feature, causal effect magnitude

Run across Qwen3-0.6B, 1.7B, 4B (F32), 8B (BF16).

### 4b — Rhyming Couplets (`latent_planning_rhyme.rs`)

Reuse `figure13_planning_poems` position-sweep structure with PLT injection on Qwen3-1.7B. Bridges your arXiv prolepsis paper (CLTs, Gemma 2 + Llama 3.2) with the Hanna & Ameisen result (PLTs, Qwen-3).

---

## Validation & Output

| Artefact | Format |
|---|---|
| `scripts/plt_llama_validation.py` | Python circuit-tracer reference for Llama 3.2 1B PLT |
| `scripts/plt_llama_reference.json` | Reference feature activations |
| `scripts/plt_vs_clt_comparison.md` | Side-by-side CLT vs PLT planning-site results |
| `tests/validate_plt.rs` | `#[ignore]`, CUDA required |
| JSON results | Deloson-compatible format |

---

## Sequencing & Tags

```
TranscoderKind enum + rank-2 detection in open()          → commit
cache/inject for PLT (n_target_layers=1)                   → commit
validate_plt on Llama 3.2 1B vs Python reference           → commit  PUSH
clt_vs_plt_planning_site.rs (Llama 3.2 1B)                → commit
Gemma 2 2B PLT injection point confirmed → add Gemma run  → commit  PUSH
parse_qwen3() + QK LayerNorm                               → validate → commit
latent_planning_article.rs  (0.6B–4B F32)                 → commit
latent_planning_rhyme.rs    (1.7B F32)                     → commit
8B at BF16                                                 → commit  PUSH → tag v0.1.9
```

**Note:** 14B excluded (28 GB BF16). The CLT vs PLT comparison on Llama 3.2 1B and Gemma 2 2B is the highest-priority result and requires no new model backend — it lands after Step 1 alone.

---

## Appendix — The CLT vs PLT Methodological Disagreement

### What the disagreement is

Jacopin's Q1 result is a strong negative claim: planning is **invisible to six residual-stream methods**, and CLTs are **necessary** to detect it. The word "necessary" is doing significant work. It implies that any analysis tool operating locally — reading from the residual stream at one layer and projecting to interpretable components at that same layer — will fail to detect the planning-site spike.

PLTs are exactly this kind of tool. They encode from the residual stream at layer *l* and decode to the MLP output at layer *l* only. They are local by construction.

Hanna & Ameisen use PLTs on Qwen-3 and find interpretable planning features with measurable causal effects. If the necessity claim is correct, they should not have been able to find them — or at least not find features with genuine causal power over the planned token.

The disagreement, made explicit: **is cross-layer feature sharing necessary to detect latent planning, or is per-layer decomposition sufficient?**

### Why it is currently unresolvable from the two papers alone

The two papers use different models, different tasks, and different transcoder types simultaneously — there is no controlled variable. Any of the following could explain the apparent tension without there being a genuine methodological disagreement:

- The **task** could matter (article prediction is structurally simpler than rhyming couplets)
- The **model architecture** could matter (Qwen-3 vs Gemma 2 / Llama 3.2)
- Jacopin's six residual-stream methods might not have included PLTs at all (PLTs are learned sparse decompositions, not raw residual projections — they may sit in a different category)
- The planning representation could genuinely differ across model families

The controlled experiment — same model, same prompt, both transcoder types — is the only way to isolate the variable. candle-mi v0.1.9 is the first setup that makes this possible.

### Primary diagnostic metric

Before examining any of the four outcomes below, a single theory-neutral number determines which regime the experiment is in: the **Jaccard overlap between the CLT and PLT top-k active features at the planning-site position**.

Jaccard is theory-neutral because it requires no prior commitment to which transcoder is "right" — it simply measures whether the two dictionaries agree on which features are active at the critical position. Feature indices are not directly comparable across CLT and PLT (the two dictionaries are trained independently and may find different bases), so overlap is measured at the level of **decoder projection onto the unembedding direction of the planned token**: rank the CLT features and PLT features separately by their dot product with `unembed(planned_word)`, take the top-k from each, and compute Jaccard on the resulting index sets within each transcoder's own namespace — then compare the rank orderings of their decoder projections as a secondary check.

- **Jaccard high** (≥ 0.3): both dictionaries are finding the same underlying structure, expressed in different bases. The planning representation is detectable at the per-layer level; cross-layer sharing adds resolution but is not necessary for detection. → Outcome B or D.
- **Jaccard near zero** (< 0.05): the two transcoders are decomposing the computation differently. Either PLT misses the planning site entirely, or it finds a genuinely different (possibly shallower) representation. → Outcome A or C.
- **Jaccard intermediate**: PLT finds something at the planning position but not the same features as CLT. Requires examining the position-sweep profiles to distinguish Outcome C from Outcome D.

This metric should be computed and logged first, before running any causal intervention. It is cheap (no additional forward passes required beyond the encoding step) and determines the interpretation of everything that follows.

### Outcome A — PLT misses the planning-site spike on Llama 3.2 1B

This confirms the strong reading of Q1. The planning representation is genuinely cross-layer: the signal is distributed across multiple layers' feature spaces in a way that only the shared CLT dictionary can capture. A PLT at layer *l* sees a projection of the planning state but cannot resolve it into a clean interpretable feature because the feature's causal influence flows through multiple downstream MLP outputs.

**What this means for Hanna & Ameisen:** their findings are valid for article prediction, but that task may be structurally easier — the planning signal more locally concentrated, because deciding "an" vs "a" is a shallower computation than maintaining a rhyme target across many generation steps. PLTs suffice for shallow planning; CLTs are necessary for deep planning. The two papers are then measuring different phenomena on the same continuum, not contradicting each other.

**What this means for the field:** CLTs are strictly necessary for detecting planning in tasks requiring multi-step commitment. PLT-based circuit analyses may systematically miss the most interesting cross-layer dynamics.

### Outcome B — PLT finds the spike too, same layer and position

This refutes the necessity claim, at least for this phenomenon. The planning representation is detectable at the single-layer level, and the Jaccard overlap between CLT and PLT top activations at the planning site is substantial.

This would mean Jacopin's six methods failed not because planning is inherently cross-layer, but because those methods were not learned sparse decompositions. The distinction that matters is not CLT vs PLT but **learned sparse features vs direct residual projections**. Both transcoders succeed; raw residual-stream methods fail. Q1 would need to be reframed: "planning is invisible to raw residual-stream analysis; learned transcoder features are sufficient."

**What this means for Hanna & Ameisen:** their methodology is fully vindicated. PLTs are as good as CLTs for detecting planning features; CLTs add value only in explaining cross-layer *propagation*, not in *detecting* the signal.

**What this means for the field:** PLTs, being cheaper to train, are sufficient for planning detection across a wide range of tasks. The additional complexity of CLTs is justified only for understanding cross-layer circuits, not for finding features in the first place.

### Outcome C — PLT finds a spike, but at a different layer or with a flatter position profile

This is the most scientifically interesting outcome. PLTs detect something at the planning position, but the peak layer differs from the CLT result, or the position sweep shows a flatter profile without the sharp spike that characterises the CLT analysis.

This would suggest CLTs and PLTs decompose the planning representation into genuinely different basis functions. The CLT's cross-layer feature captures the planning state in its most concentrated form; the PLT finds a local projection of the same underlying structure that is noisier and less specific. Both are looking at the same phenomenon, but the CLT has a sharper lens.

**What this means for the field:** PLTs are a valid tool but systematically underestimate the sharpness and localisation of planning phenomena. Hanna & Ameisen's findings are real but potentially understating the clarity of the mechanism. The planning site exists; the question is how crisply different tools can resolve it.

### Outcome D — Both find the spike, but causal suppression diverges

Detection and intervention are separable. Even if PLTs identify the planning features correctly, suppressing them via PLT injection may be less effective at redirecting the rhyme — manifesting as a smaller probability redirect and a less clean position-sweep profile compared to CLT-based suppression.

This aligns with the quantitative result from Ameisen et al.'s original circuit-tracing paper, which found CLTs outperform PLTs on all metrics including causal faithfulness. It would confirm that PLTs are adequate for *finding* features but inadequate for *controlling* model behaviour through those features — a practically important distinction for any safety-relevant application of these methods.

### Why this matters beyond the two papers

The underlying question is whether mechanistic interpretability results are methodology-dependent in a fundamental way. If CLTs and PLTs give the same detection results and differ only in intervention fidelity, the field can use whichever is cheaper to train. If they give different detection results, a large body of PLT-based MI work may be systematically missing phenomena that only CLTs can see — and CLT-based work may be over-attributing causal structure to cross-layer interactions that are locally representable.

The candle-mi v0.1.9 experiment provides the first direct empirical answer to that question, on models where both transcoder types are publicly available and where validated ground truth already exists from the CLT side.
