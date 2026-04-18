# candle-mi v0.1.9 Roadmap V2 — PLT Support & CLT vs PLT Comparison (staged)

**Target:** Add PLT (Per-Layer Transcoder) support to candle-mi, then run the first controlled CLT vs PLT comparison on the same rhyming-couplet prompts used in `figure13_planning_poems`. Releases land incrementally:
- **`v0.1.9`** — Stage 1 Llama 3.2 1B arm (PLT support + Llama CLT-vs-PLT experiment).
- **`v0.1.10`** — Gemma 2 2B arm (gated on the anamnesis GemmaScope `.bin` deliverable — Appendix C).
- **`v0.1.11`** — conditional Stage 2: Qwen-3 scale sweep (Hanna & Ameisen, [arXiv:2604.12493](https://arxiv.org/abs/2604.12493)) gated on the Stage 1 decision checkpoint.

**Baseline:** v0.1.8. CLT pipeline validated on Gemma 2 2B + Llama 3.2 1B via `examples/figure13_planning_poems.rs`. No PLT support, no Qwen-3 backend.

**Hardware:** RTX 5060 Ti, 16 GB VRAM. Models that do not fit at this card's practical VRAM budget are out of scope.

**What changed from V1:**
1. **Staged with a decision gate.** Stage 1 (PLT + CLT-vs-PLT comparison) is a self-contained, publishable deliverable. Stage 2 (Qwen-3 + scale sweep) is committed to *only after* Stage 1 data justifies it.
2. **Primary metric replaced.** V1 proposed Jaccard between CLT and PLT top-k features; feature indices are not comparable across independently-trained dictionaries, so Jaccard is either trivially zero or ill-defined. V2 uses a **causal-effect delta on the planned-token logit** — measured in the base model's output space, intrinsically comparable across transcoders.
3. **VRAM budget trimmed.** Qwen3-4B is BF16-only (4B × 4 bytes = 16 GB weights alone under F32, leaves no room for activations + transcoder). Qwen3-8B is BF16 and must be validated for fit before any coding.
4. **Explicit stop conditions** and risk register.
5. **Pinned file paths and commit sequence** so the roadmap can bootstrap a session directly.
6. **Pre-flight model inventory** added — all missing models listed with `hf-fm` commands.
7. **Schema-level branching, not rank-level.** Remote inspection via `hf-fm inspect` revealed the PLT repos do **not** use the CLT-split `W_enc_{l}.safetensors` / `W_dec_{l}.safetensors` layout. The real branch point is schema (file naming, tensor naming, extra tensors like `W_skip`), not W_dec rank. Step 1.1 reframed accordingly.
8. **anamnesis cross-crate dependency.** GemmaScope's custom `.bin` format is not in any anamnesis phase yet. The Gemma PLT arm now depends on a new anamnesis deliverable (Appendix C). Stage 1 Llama arm proceeds independently.

---

## Tooling convention — dogfood `hf-fetch-model`

All HuggingFace inspection and download operations in this roadmap go through [`hf-fetch-model`](https://crates.io/crates/hf-fetch-model) (pinned in `Cargo.toml` as `hf-fetch-model = "0.9"`). candle-mi ships this crate, so every fetch in this work dogfoods it.

- **CLI (`hf-fm`)** — interactive inspection and one-shot downloads:
  - `hf-fm list-files <repo>` — list remote files without downloading.
  - `hf-fm inspect <repo_id> [filename]` — tensor names, shapes, dtypes via HTTP range reads (no download). Use this to confirm a repo's schema (file naming + tensor set) by hand before writing auto-detection code.
  - `hf-fm <repo> --preset safetensors` — download weights + config + tokenizer.
  - `hf-fm download-file <repo> <filename>` — single-file fetch (e.g., `W_dec_0.safetensors`).
  - `hf-fm status <repo>` / `hf-fm du` — cache state.
- **Library (`hf_fetch_model::*`)** — programmatic use in `src/` and tests. Already used in [`src/clt/mod.rs:263`](../../src/clt/mod.rs#L263) for `open()`. New code MUST NOT reach past this crate to `hf-hub` directly — the whole point is to keep the dependency surface consistent with the rest of candle-mi and to surface rough edges in `hf-fetch-model` through real use.

If a workflow needs a pattern `hf-fetch-model` doesn't yet support, file an issue / PR on that crate rather than working around it.

---

## Pre-flight — model inventory

Checked against the local HF cache (`~/.cache/huggingface/hub/`) on 2026-04-18. Run `hf-fm status <repo>` to re-check any specific repo.

### Stage 1

**Already cached** ✅
- `meta-llama/Llama-3.2-1B`
- `google/gemma-2-2b`
- `mntss/clt-llama-3.2-1b-524k`
- `mntss/clt-gemma-2-2b-426k` (alt: `mntss/clt-gemma-2-2b-2.5M`)

**Must download before Step 1.4** ❌
```bash
hf-fm mntss/transcoder-Llama-3.2-1B
hf-fm mntss/gemma-scope-transcoders
```

⚠ The cached `google/gemma-scope-2b-pt-res` is the **residual-stream SAE** variant, *not* the MLP transcoder PLT. It is **not** a substitute for `mntss/gemma-scope-transcoders`.

**Layout already inspected (2026-04-18, via HTTP-range header reads — no full downloads):**

`mntss/transcoder-Llama-3.2-1B`:
- 16 × `layer_{l}.safetensors` at repo root (1.01 GiB each, BF16)
- Each bundle contains: `W_enc [131072, 2048]`, `W_dec [131072, 2048]`, `W_skip [2048, 2048]`, `b_enc [131072]`, `b_dec [2048]`
- **Un-suffixed tensor names** (`W_enc`, not `W_enc_0`) — one file is self-contained per layer
- **Extra `W_skip` tensor** — MLP-transcoder linear skip path; orthogonal to encode/inject/suppress (see Appendix A), ignored for the CLT-vs-PLT experiment

`mntss/gemma-scope-transcoders`:
- **No safetensors at all.** 26 × `features/layer_{l}.bin` (~140 MiB each, custom GemmaScope binary format)
- Blocks Gemma PLT arm until anamnesis ships a `.bin` reader — see Appendix C

`mwhanna/qwen3-*-transcoders*` (spot-checked on `qwen3-0.6b-transcoders-lowl0`):
- Same bundle layout as `mntss/transcoder-Llama-3.2-1B` — 28 × `layer_{l}.safetensors` (640 MiB each) + irrelevant `features/layer_{l}.bin` metadata
- **Implication:** one `MntssPltBundle` schema handles Stage 1 Llama **and** all Stage 2 Qwen3 repos. Zero extra loader work for Stage 2.

**If you need to re-inspect anything (zero downloads):**
```bash
hf-fm list-files <repo>
hf-fm inspect <repo> [<filename>]   # safetensors only; HTTP Range requests for headers
```

### Stage 2 (conditional — only after the go/no-go gate)

**Base models** ❌
```bash
hf-fm Qwen/Qwen3-0.6B-Base
hf-fm Qwen/Qwen3-1.7B-Base
hf-fm Qwen/Qwen3-4B
hf-fm Qwen/Qwen3-8B              # only if Step 2.2 fit-test passes
```

⚠ Cached `Qwen/Qwen3-1.7B-FP8` and `Qwen/Qwen3-4B-Instruct-2507-FP8` are FP8-quantised and unusable for F32/BF16 MI analysis. We need the unquantised masters.

**PLTs** ❌
```bash
hf-fm mwhanna/qwen3-0.6b-transcoders-lowl0
hf-fm mwhanna/qwen3-1.7b-transcoders-lowl0
hf-fm mwhanna/qwen3-4b-transcoders
hf-fm mwhanna/qwen3-8b-transcoders    # only if 8B fit-test passes
```

### Disk-space note

Run `hf-fm du` before bulk-downloading. Qwen3 bases at F32 totals ~25 GB; the four `mwhanna` PLTs likely add another ~5–15 GB depending on `n_features`. Don't blindly fetch Stage 2 assets until the Stage 1 gate clears.

---

## Scope

**In:**
- `TranscoderSchema` abstraction (`CltSplit`, `PltBundle`, `GemmaScopeBin`) in `src/clt/mod.rs` — permanent library capability.
- Filename-based auto-detection at `open()` time.
- CLT vs PLT controlled comparison on **Llama 3.2 1B** using the two rhyming-couplet prompts already in `figure13_planning_poems`. The Gemma 2 2B arm is a follow-up release (`v0.1.10`) gated on the anamnesis GemmaScope deliverable — see Appendix C.
- Conditional Stage 2: Qwen-3 transformer backend with per-head QK LayerNorm; article-prediction + rhyme replication at 0.6B/1.7B/4B scales.

**Out:**
- Qwen3-14B (~28 GB BF16, OOM).
- Any F32 run at 4B+.
- Production-quality PLT training (we only load pretrained PLTs).

---

## Stage 1 — PLT support + CLT vs PLT comparison

Stage 1 is the high-value, low-risk deliverable. It yields a permanent library capability (PLT loading/injection) and a novel experimental result (first same-model same-prompt CLT-vs-PLT comparison on Llama 3.2 1B).

**Release plan under the publish workflow** (see [`.github/workflows/publish.yml`](../../.github/workflows/publish.yml) — hyphenated tags do NOT publish to crates.io):

| Tag | Publishes? | Contains |
|---|---|---|
| `v0.1.9-plt` (optional) | ❌ milestone only | git-level checkpoint after Stage 1 commits land, before final polish |
| **`v0.1.9`** | ✅ **crates.io release** | Stage 1 Llama arm complete: PLT support + Llama CLT-vs-PLT experiment + README update |
| `v0.1.10` | ✅ crates.io release | Gemma arm follow-up (when anamnesis GemmaScope deliverable lands — Appendix C) |
| `v0.1.11` | ✅ crates.io release | Stage 2: Qwen-3 backend + latent-planning scale sweep (conditional on decision gate) |

### Step 1.1 — `TranscoderSchema` enum + filename-based detection

**File:** [`src/clt/mod.rs`](../../src/clt/mod.rs)

The branch point is the on-disk layout (file naming + tensor naming + tensor set), not just `W_dec` rank — see the pre-flight findings. Detection happens on the repo file listing that `open()` already fetches, so **no extra download is needed** (revision from earlier draft that assumed a `W_dec_0` probe).

**Changes:**

1. Add at module scope, near `CltConfig` ([`src/clt/mod.rs:167`](../../src/clt/mod.rs#L167)):
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   #[non_exhaustive]
   pub enum TranscoderSchema {
       /// Cross-Layer Transcoder (`mntss/clt-*`): two files per layer,
       /// `W_enc_{l}.safetensors` + `W_dec_{l}.safetensors`, layer-suffixed tensor
       /// names (`W_enc_{l}`, `W_dec_{l}`). `W_dec` is rank-3
       /// `[n_features, n_target_layers, d_model]` — writes to multiple downstream
       /// layers.
       CltSplit,
       /// Per-Layer Transcoder bundle (`mntss/transcoder-*`, `mwhanna/qwen3-*-transcoders*`):
       /// one file per layer, `layer_{l}.safetensors`, un-suffixed tensor names
       /// (`W_enc`, `W_dec`, `W_skip`, `b_enc`, `b_dec`). `W_dec` is rank-2
       /// `[n_features, d_model]` — writes only to layer `l`. The `W_skip` linear
       /// path is loaded but not used by encode/inject/suppress (see Appendix A).
       PltBundle,
       /// GemmaScope `.bin` format (`mntss/gemma-scope-transcoders`). Not loadable
       /// directly by candle-mi — requires anamnesis `.bin` reader (see Appendix C).
       GemmaScopeBin,
   }

   impl TranscoderSchema {
       /// MI semantics: does this transcoder inject into its own layer only,
       /// or into multiple downstream layers?
       #[must_use]
       pub fn is_cross_layer(self) -> bool { matches!(self, Self::CltSplit) }
   }
   ```
2. Add `pub schema: TranscoderSchema` field to `CltConfig`.
3. In `open()` ([`src/clt/mod.rs:263`](../../src/clt/mod.rs#L263)), classify the repo from the already-fetched `repo_files` listing (currently at [mod.rs:285](../../src/clt/mod.rs#L285)) **before** any download:
   - Any file matching `^W_enc_\d+\.safetensors$` → `CltSplit`
   - Any file matching `^layer_\d+\.safetensors$` at repo root → `PltBundle`
   - `config.yaml` present plus only `features/layer_\d+\.bin` → `GemmaScopeBin` → return `MIError::Config("GemmaScope .bin format not yet supported — pending anamnesis support, see Appendix C")`
   - Otherwise → `MIError::Config("unrecognised transcoder repo layout")`
4. For `n_layers`:
   - `CltSplit`: count `W_enc_\d+\.safetensors` files (existing logic).
   - `PltBundle`: count `layer_\d+\.safetensors` files at repo root.
5. For dimension detection, branch the first-file download:
   - `CltSplit`: download `W_enc_0.safetensors`, read tensor `W_enc_0`, shape `[n_features, d_model]` (existing logic, unchanged).
   - `PltBundle`: download `layer_0.safetensors`, read tensor `W_enc` (un-suffixed), same shape interpretation. File is ~1 GiB (vs ~75 MiB for the CLT encoder), so this is a one-time cost at `open()` — acceptable.
6. Update the `info!` log in `open()` to include the detected schema.

**Rename consideration:** leaving the struct name `CrossLayerTranscoder` is fine for now; rename to `Transcoder` in a separate breaking-change commit if desired. Do **not** bundle the rename into this step — it churns call sites across the whole crate. The type now loads both CLT and PLT; the name becomes cosmetically inaccurate but functionally correct.

### Step 1.2 — Schema-aware loaders for encoder and decoder

**File:** [`src/clt/mod.rs`](../../src/clt/mod.rs)

Two independent branches per schema:

**(a) Encoder loading** — `load_encoder()` ([`src/clt/mod.rs:449`](../../src/clt/mod.rs#L449)):
- `CltSplit`: download `W_enc_{layer}.safetensors`, extract tensors `W_enc_{layer}` + `b_enc_{layer}` (existing logic).
- `PltBundle`: download `layer_{layer}.safetensors`, extract tensors `W_enc` + `b_enc` (un-suffixed). File is larger (~1 GiB) because it also contains `W_dec`, `W_skip`, `b_dec` — those tensors are loaded lazily by the decoder path.

**(b) Decoder access** — every site that currently indexes `w_dec.i((idx, target_offset))?`:

| Function | Line | `CltSplit` (current) | `PltBundle` |
|---|---|---|---|
| `decoder_vector` | ~610 | open `W_dec_{l}.safetensors`, read `W_dec_{l}`, index `(index, target_offset)` | open `layer_{l}.safetensors`, read `W_dec`, index `index` (target_offset must equal source layer) |
| `cache_steering_vectors` | ~687 | per-target-layer caching | single `(feature, source_layer)` entry |
| `cache_steering_vectors_all_downstream` | [`805`](../../src/clt/mod.rs#L805) | `n_target_layers = n_layers - source_layer`, loop `target_offset` | `n_target_layers = 1`, no offset loop |
| `score_features_by_decoder_projection` | ~1132 | rank-3 indexing | rank-2 indexing, single target |
| `score_features_by_decoder_projection_batch` | ~1268 | same | same |
| `extract_decoder_vectors` | ~1419 | same | same |
| `build_attribution_graph` / `_batch` | ~1516 / ~1546 | same | same |

**Implementation note:** concentrate the branch in two private helpers:

```rust
fn decoder_file_and_tensor_name(schema: TranscoderSchema, layer: usize) -> (String, String) {
    match schema {
        TranscoderSchema::CltSplit  => (format!("W_dec_{layer}.safetensors"), format!("W_dec_{layer}")),
        TranscoderSchema::PltBundle => (format!("layer_{layer}.safetensors"), "W_dec".into()),
        TranscoderSchema::GemmaScopeBin => unreachable!("rejected at open()"),
    }
}

fn decoder_row(w_dec: &Tensor, index: usize, target_offset: usize, schema: TranscoderSchema) -> Result<Tensor> {
    match schema {
        TranscoderSchema::CltSplit  => w_dec.i((index, target_offset)),
        TranscoderSchema::PltBundle => { debug_assert_eq!(target_offset, 0); w_dec.i(index) },
        TranscoderSchema::GemmaScopeBin => unreachable!("rejected at open()"),
    }
}
```

Call these helpers from every site above. Do not scatter match arms across ~7 functions.

**No changes required** in `prepare_hook_injection` ([`src/clt/mod.rs:942`](../../src/clt/mod.rs#L942)) or `inject` ([`src/clt/mod.rs:1037`](../../src/clt/mod.rs#L1037)) — both iterate per `(feature, target_layer)` cache key, which is schema-agnostic once caching works.

**`W_skip` tensor** in `PltBundle` files is present in memory after `load_encoder()` (if bundled load is adopted) but is **not wired into any MI operation** in this step. It exists for completeness; adding skip-aware reconstruction would be a Stage 2 polish item only if needed.

### Step 1.3 — Unit tests for both schemas

**File:** [`src/clt/mod.rs`](../../src/clt/mod.rs) (tests module)

1. Existing CLT tests remain — rename assertions to check `config.schema == TranscoderSchema::CltSplit`.
2. Add a `PltBundle` fixture: write `layer_{0..2}.safetensors` each containing `W_enc`, `W_dec` (rank-2), `W_skip`, `b_enc`, `b_dec`.
3. Open the fake PLT repo, assert `config.schema == TranscoderSchema::PltBundle`, assert `W_skip` is reachable, and run `cache_steering_vectors_all_downstream` — confirm the cache contains exactly `(feature, source_layer)` entries and no downstream ones.
4. Negative test: synthesise a GemmaScope-like layout (only `features/layer_*.bin`) and assert `open()` returns the expected `MIError::Config` with the "pending anamnesis" text.

### Step 1.4 — Python reference for Llama 3.2 1B PLT

**Files:** `scripts/plt_llama_validation.py`, `scripts/plt_llama_reference.json`

Use circuit-tracer (Python) to establish ground truth:
1. Load `mntss/transcoder-Llama-3.2-1B` via circuit-tracer.
2. Run 3 fixed prompts (reuse two rhyming-couplet prompts from `figure13_planning_poems` + one simple factual prompt).
3. For each (layer, token position), dump top-10 active PLT features (feature index + activation magnitude) to JSON.
4. Script must be deterministic: seeded, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:16:8`.

Commit both the script and the JSON reference. JSON is the frozen oracle.

### Step 1.5 — Rust validation test for Llama 3.2 1B PLT

**File:** `tests/validate_plt.rs`

`#[ignore]` (CUDA required; large download).

1. Load `mntss/transcoder-Llama-3.2-1B` via `CrossLayerTranscoder::open()`; assert `config.schema == TranscoderSchema::PltBundle`.
2. Same 3 prompts as the Python reference.
3. For each (layer, position), encode the residual and capture top-10 features.
4. Compare against `scripts/plt_llama_reference.json`:
   - Top-10 feature indices: **exact match** (set equality).
   - Activation magnitudes: abs diff < 1e-4 (F32, CUDA).

**Commit point:** all previous steps land cleanly → push → optional milestone tag `v0.1.9-plt-llama` (hyphenated — does not publish to crates.io; pure git-level checkpoint for "Llama PLT validated").

### Step 1.6 — Gemma 2 2B PLT arm (deferred, depends on anamnesis)

**Status:** **parked as a `v0.1.10` follow-up release**, not a blocker for `v0.1.9`. Two independent obstacles, both outside candle-mi:

**Obstacle 1 — file format.** `mntss/gemma-scope-transcoders` ships only `features/layer_{l}.bin` (custom GemmaScope binary format). candle-mi is safetensors-only and cannot load these directly. Unblocking requires a `.bin` reader in [anamnesis](../../../anamnesis), which is not currently on its roadmap. See **Appendix C** for the proposed anamnesis deliverable and success criteria.

**Obstacle 2 — injection point.** Once the weights load, there is still a Gemma-specific semantic question: GemmaScope PLTs reconstruct MLP output *after* the post-MLP RMSNorm, and Gemma 2's 4-norm architecture means the decoder vector lives in a different space than the CLT decoder. Circuit-tracer's Python source is the reference; the candidate targets are an existing `ResidPost(l)` / `MlpOut(l)` or a new `ResidPostMlpNorm(l)` `HookPoint` variant.

**When anamnesis support lands:**
1. Add `TranscoderSchema::GemmaScopeBin` handling to `open()` — replaces the current `MIError::Config` return with a real load path via `anamnesis::parse_gemmascope_bin()` (or whatever anamnesis names it).
2. Read circuit-tracer's Python GemmaScope injection code. Identify which residual-stream slot it writes to.
3. If that slot maps to an existing [`HookPoint`](../../src/hooks.rs) variant, use it directly. If not, add `HookPoint::ResidPostMlpNorm(l)` (prefer this over inverse-norm math — no numerical round-trip).
4. Write `scripts/plt_gemma_validation.py` + `scripts/plt_gemma_reference.json` (mirror of Step 1.4).
5. Add `tests/validate_plt_gemma.rs` (mirror of Step 1.5).
6. Extend `examples/clt_vs_plt_planning_site.rs` to run the Gemma arm.

**Ship plan:** `v0.1.9` ships Llama-only. When the anamnesis deliverable (Appendix C) lands, `v0.1.10` adds the Gemma arm as a patch release (purely additive — new `TranscoderSchema::GemmaScopeBin` path, no breaking changes). The Gemma result extends but does not gate the publication of the Llama CLT-vs-PLT comparison.

### Step 1.7 — CLT vs PLT controlled experiment

**Files:** `examples/clt_vs_plt_planning_site.rs`, `outputs/clt_vs_plt_{model}.json`, `docs/clt_vs_plt_findings.md`

For each (model, prompt) in `{Llama 3.2 1B, Gemma 2 2B}` × `{two rhyming-couplet prompts from figure13_planning_poems}`:

1. **CLT arm** (reuses existing code paths):
   - Load `mntss/clt-llama-3.2-1b-524k` (Llama) or `mntss/clt-gemma-2-2b-426k` (Gemma).
   - Encode residual-mid at every (layer, position).
   - Rank features at each (layer, position) by dot product of decoder vector with `unembed(planned_word)`.
   - Record: spike (layer, position), top-5 planning-aligned features, full position-sweep profile (max planning-alignment per position, for the spike layer).
2. **PLT arm** (new): identical protocol on `mntss/transcoder-Llama-3.2-1B` or `mntss/gemma-scope-transcoders`.
3. **Causal test (both arms):**
   - Suppress (zero out) the top-5 planning-aligned features at the spike position.
   - Measure `ΔP(planned_word)` and `Δlogit(planned_word)` at the final token position.
   - This is the **primary comparison metric** (see Appendix A).
4. **Output:** `outputs/clt_vs_plt_{model}.json` in deloson-compatible format. Summary markdown `docs/clt_vs_plt_findings.md` with one table per model and narrative interpretation mapped to the four outcomes in Appendix A.

**Stage 1 deliverable:** merged PR, version bump to `0.1.9` in `Cargo.toml` + `Cargo.lock`, `CHANGELOG.md` entry under `[0.1.9]`, tag **`v0.1.9`** (publishes to crates.io). README Paper replications table gains a Hanna & Ameisen row (with "Llama arm complete; Gemma arm in v0.1.10; scale-sweep TBD" note).

---

## Decision checkpoint (between stages)

Inspect Stage 1 results against Appendix A outcomes. Proceed to Stage 2 only if at least one of:

- **Outcome B** on Llama 3.2 1B: PLT detection and causal effect comparable to CLT. Scale-sweep then directly extends Hanna & Ameisen on aligned methodology — coherent.
- **Outcome C**: PLT spike at different layer or flatter profile. Scale-sweep becomes a depth-vs-capacity ablation — interesting.
- **Outcome A**: PLT misses the spike on small models. Justifies asking "at what scale does PLT start detecting planning?" — sweep answers this directly.

**Stop condition:** Outcome B *with* ΔP_PLT ≈ ΔP_CLT (both detection and intervention match) and Hanna & Ameisen's existing 0.6B–14B data already suffices. Write up Stage 1 as a short paper; defer Stage 2.

**Review gate:** write `docs/stage1_decision.md` with the data, the chosen outcome label, and a one-line go/no-go.

---

## Stage 2 (conditional) — Qwen-3 backend + latent planning scale sweep

### Step 2.1 — Qwen-3 transformer backend

**Files:** [`src/config.rs`](../../src/config.rs), [`src/transformer/attention.rs`](../../src/transformer/attention.rs), [`src/transformer/mod.rs`](../../src/transformer/mod.rs)

Qwen-3 = Qwen-2 + per-head QK LayerNorm (RmsNorm variant) applied **after RoPE**.

1. Add to `TransformerConfig` ([`src/config.rs:243`](../../src/config.rs#L243)): `pub qk_norm: bool` (default `false`), `pub qk_norm_eps: f64` (default `1e-6`).
2. Add optional `q_norm: Option<RmsNorm>`, `k_norm: Option<RmsNorm>` to the attention module.
   - Weight names: `model.layers.{i}.self_attn.q_norm.weight`, `.k_norm.weight`.
   - Shape: `[head_dim]` (per-head, not per-hidden).
3. Apply after RoPE in [`src/transformer/attention.rs:227-228`](../../src/transformer/attention.rs#L227-L228):
   ```rust
   let q = rope.apply(&q, 0)?;
   let k = rope.apply(&k, 0)?;
   let q = if let Some(qn) = &self.q_norm { qn.forward(&q)? } else { q };
   let k = if let Some(kn) = &self.k_norm { kn.forward(&k)? } else { k };
   ```
4. Add `parse_qwen3()` in `src/config.rs`, dispatched from the `match model_type` at [`src/config.rs:335`](../../src/config.rs#L335). Base on `parse_qwen2()`; set `qk_norm: true`, read `rms_norm_eps` for `qk_norm_eps`.
5. Add `"qwen3"` to `SUPPORTED_MODEL_TYPES`.

**Validation:** `tests/validate_qwen3.rs` (`#[ignore]`, CUDA). 5 prompts × top-10 logits vs Python HF Transformers. Abs tolerance 1e-4 at F32. Target: Qwen3-0.6B-Base (F32) and Qwen3-1.7B-Base (F32).

### Step 2.2 — Revised VRAM budget

| Model | Weights F32 | Weights BF16 | Mode on 16 GB | Note |
|---|---|---|---|---|
| Qwen3-0.6B | ~2.4 GB | ~1.2 GB | ✅ F32 | |
| Qwen3-1.7B | ~6.8 GB | ~3.4 GB | ✅ F32 | |
| Qwen3-4B | ~16 GB | ~8 GB | ✅ BF16 only | F32 infeasible with activations + transcoder. |
| Qwen3-8B | ~32 GB | ~16 GB | ⚠ BF16, must validate fit | Weights alone are at budget; transcoder adds 1–2 GB. Validate empirically before coding examples. |
| Qwen3-14B | — | ~28 GB | ❌ OOM | Out of scope. |

Reserve 4 GB headroom for activations + transcoder cache + tokenizer + CUDA workspace. Realistic sweep ceiling: **Qwen3-4B BF16**. Qwen3-8B is a stretch — fit-test before committing.

### Step 2.3 — Article prediction replication

**File:** `examples/latent_planning_article.rs`

Prompt: `"Someone who handles financial records is"` (no trailing space — the article is the *next* token).

1. Forward pass. PLT-encode `ResidMid` at the article-token position for every layer.
2. Rank features by dot product with `unembed("an") - unembed("a")`.
3. Causal test: suppress top-k planning features, measure `Δ[logit("an") - logit("a")]`.
4. Run across Qwen3-0.6B, 1.7B (F32), 4B (BF16). Optionally 8B (BF16) if fit-test passes.
5. Output: `outputs/latent_planning_article_{model}.json`, summary `outputs/latent_planning_article_summary.md`.

### Step 2.4 — Rhyming couplets on Qwen-3

**File:** `examples/latent_planning_rhyme.rs`

Reuse `figure13_planning_poems` position-sweep structure with PLT injection on Qwen3-1.7B (primary). If the 1.7B result is clean, extend to Qwen3-4B BF16. This bridges the prolepsis paper (Gemma / Llama, CLTs) and Hanna & Ameisen (Qwen-3, PLTs) on the same rhyming task.

### Stage 2 deliverable

Merged PR, version bump to `0.1.11` in `Cargo.toml` + `Cargo.lock` (patch — Qwen-3 backend is purely additive: new optional `TransformerConfig` fields with defaults, new `Option<RmsNorm>` attention fields, new `parse_qwen3()` dispatch), `CHANGELOG.md` entry under `[0.1.11]`, tag **`v0.1.11`** (publishes to crates.io). README Paper replications table updated with Hanna & Ameisen row pointing to both `latent_planning_article.rs` and `latent_planning_rhyme.rs`.

---

## Validation matrix

| Artefact | Step | Release | Acceptance |
|---|---|---|---|
| `TranscoderSchema` enum, `CltConfig.schema` | 1.1 | `v0.1.9` | Compiles; doc comments present; `#[non_exhaustive]` enum. |
| Schema-aware `decoder_row()` + `decoder_file_and_tensor_name()` helpers | 1.2 | `v0.1.9` | All `W_dec_` call sites routed through helpers. |
| `PltBundle` + `CltSplit` fixture unit tests | 1.3 | `v0.1.9` | Both schemas pass; GemmaScope negative test passes. |
| `scripts/plt_llama_validation.py` + JSON | 1.4 | `v0.1.9` | Deterministic Python run committed. |
| `tests/validate_plt.rs` (Llama) | 1.5 | `v0.1.9` | CUDA test passes: top-10 IDs match, abs diff < 1e-4. |
| `examples/clt_vs_plt_planning_site.rs` (Llama arm) | 1.7 | `v0.1.9` | Runs end-to-end on Llama 3.2 1B. |
| `docs/clt_vs_plt_findings.md` | 1.7 | `v0.1.9` | Outcome label assigned (A/B/C/D) per Appendix A. |
| `tests/validate_plt_gemma.rs` | 1.6 | `v0.1.10` | Deferred — lands with anamnesis GemmaScope deliverable. |
| Gemma arm in `clt_vs_plt_planning_site.rs` | 1.6 | `v0.1.10` | Extends the Llama arm once 1.6 unblocks. |
| `docs/stage1_decision.md` | gate | — | Go/no-go for Stage 2 recorded (not a published artefact). |
| `tests/validate_qwen3.rs` | 2.1 | `v0.1.11` | Top-10 logits match HF, abs diff < 1e-4 at F32, 0.6B + 1.7B. |
| `examples/latent_planning_article.rs` | 2.3 | `v0.1.11` | Runs across ≥3 model sizes; `Δ[logit("an") - logit("a")]` table populated. |
| `examples/latent_planning_rhyme.rs` | 2.4 | `v0.1.11` | Runs on Qwen3-1.7B. |

All steps must also pass the CLAUDE.md pre-commit gate: `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`, **plus** `cargo clippy --features transformer -- -W clippy::pedantic` and `cargo clippy --features rwkv -- -W clippy::pedantic` (both backends separately), `cargo test`, and `CHANGELOG.md` update.

---

## Commit & tag sequence

**Stage 1 — release `v0.1.9` (publishes):**
1. `feat(clt): add TranscoderSchema enum and filename-based auto-detection at open()`
2. `refactor(clt): route W_dec access through decoder_row and decoder_file_and_tensor_name helpers`
3. `feat(clt): PltBundle branch in encoder loader and cache/score helpers`
4. `test(clt): PltBundle and CltSplit fixtures, GemmaScopeBin negative test`
5. `feat(scripts): add plt_llama_validation.py reference` — PUSH
6. `test(clt): validate PLT on Llama 3.2 1B against Python reference` — PUSH (optional milestone tag `v0.1.9-plt-llama` — no publish)
7. `feat(examples): clt_vs_plt_planning_site comparison (Llama arm)`
8. `docs: CLT vs PLT findings + README Paper replications row`
9. `chore: bump version to 0.1.9 and update CHANGELOG` (Cargo.toml **and** Cargo.lock committed together per CLAUDE.md) — PUSH, wait for green CI, `cargo publish --dry-run`, then tag **`v0.1.9`** (publishes)

**Gemma follow-up — release `v0.1.10` (publishes, requires anamnesis GemmaScope):**
10. (anamnesis) `gemmascope` feature lands in an anamnesis release
11. `feat(clt): GemmaScopeBin schema load path via anamnesis`
12. `feat(hooks): add ResidPostMlpNorm variant if needed for Gemma PLT injection`
13. `feat(scripts): add plt_gemma_validation.py reference`
14. `test(clt): validate PLT on Gemma 2 2B against Python reference`
15. `feat(examples): extend clt_vs_plt_planning_site with Gemma arm`
16. `chore: bump version to 0.1.10 and update CHANGELOG` — PUSH, green CI, dry-run, tag **`v0.1.10`** (publishes)

**Stage 2 (conditional on decision gate) — release `v0.1.11` (publishes):**
17. `feat(transformer): qk_norm post-RoPE for Qwen-3 family`
18. `feat(config): parse_qwen3 and SUPPORTED_MODEL_TYPES entry`
19. `test(transformer): Qwen-3 parity vs Python HF (0.6B, 1.7B)`
20. `feat(examples): latent_planning_article across 0.6B/1.7B/4B`
21. `feat(examples): latent_planning_rhyme on Qwen3-1.7B`
22. `chore: bump version to 0.1.11 and update CHANGELOG` — PUSH, green CI, dry-run, tag **`v0.1.11`** (publishes)

> **Tag convention reminder:** any tag matching `v*-*` (hyphenated) is a git-level milestone only and does NOT trigger `publish.yml`. Only clean `vMAJOR.MINOR.PATCH` tags publish to crates.io. See [`.github/workflows/publish.yml`](../../.github/workflows/publish.yml) for the trigger rules.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| anamnesis GemmaScope `.bin` investigation reveals a format harder than pickle/NPY (orbax, msgpack, proprietary). | Medium | `v0.1.10` delayed indefinitely. | Time-box the anamnesis investigation; if stuck, fall back to `amn convert --from gemmascope` (Appendix C.4) via a Python one-shot conversion script, then load via `PltBundle`. |
| Gemma PLT injection point requires new `HookPoint` variant. | Medium | `v0.1.10` needs a minor `src/hooks.rs` addition. | Adding a `HookPoint` variant is mechanical and `#[non_exhaustive]` already covers forward compatibility. Not a real blocker. |
| Qwen3-8B BF16 doesn't fit with transcoder + activations. | Medium | Stage 2 ceiling is 4B. | Empirical fit-test before coding Step 2.3 examples. |
| Outcome B with `ΔP_PLT ≈ ΔP_CLT` after Stage 1 — Stage 2 scientifically redundant. | Medium | `v0.1.11` deferred. | Outcome B is a real finding on its own; publish `v0.1.9` (and `v0.1.10` if Gemma confirms) and stop. |
| Rust encode result diverges from Python reference in Step 1.5. | Low | `v0.1.9` blocked. | Likely a missed `W_dec` call site — audit via the `decoder_row()` / `decoder_file_and_tensor_name()` helpers. Less likely: bias or dtype handling in the PltBundle loader. |
| Decoder-projection ranking unstable across the two rhyming prompts. | Low | Primary metric noisy. | Average across both prompts + one factual-prompt control; report std. |
| Llama PLT `W_skip` turns out to matter for faithful reconstruction (affects causal-suppression delta). | Low | Interpretation of ΔP needs caveat. | Step 1.2 notes `W_skip` is loaded but unused. If Step 1.7 causal deltas look off, test with `W_skip` included in reconstruction and compare. |

---

## Appendix A — Methodology (revised)

### Why V1's Jaccard metric is ill-defined

V1 proposed "Jaccard overlap between CLT and PLT top-k active features at the planning-site position" as the theory-neutral primary diagnostic. This is **not well-posed**: feature indices in independently-trained CLT and PLT dictionaries are unrelated. Taking top-k from each dictionary's own namespace and intersecting returns zero unless indices happen to collide — they will not. V1 hedged with "rank both on decoder-projection onto `unembed(planned_word)`, then Jaccard within each namespace", but that produces two independent rank lists, not a genuine overlap measure.

### Revised primary metric: causal-effect delta on the planned-token logit

Both transcoders produce **interventions in the base model's output space**. That space is intrinsically comparable across transcoders. Protocol:

1. In each transcoder, rank features at the spike (layer, position) by dot product of decoder vector with `unembed(planned_word)`.
2. Take the top-5 planning-aligned features from each transcoder.
3. In two independent runs, suppress (zero out) each set at the spike position.
4. Measure `ΔP(planned_word)` and `Δlogit(planned_word)` at the generated token.
5. Compare `ΔP_CLT` vs `ΔP_PLT` directly.

Both deltas live in the base model's vocabulary distribution — no namespace mismatch, no feature-index ambiguity.

### Interpretation bands (calibrate on Llama first)

| Signature | Outcome | Interpretation |
|---|---|---|
| `ΔP_PLT ≈ ΔP_CLT`, same spike (layer, position) | **B** | PLT is adequate for both detection and intervention. CLTs add cross-layer routing resolution but are not necessary for detection. |
| `ΔP_PLT ≈ 0`, CLT has strong `ΔP_CLT` | **A** | PLT misses the planning site entirely. Jacopin's necessity claim confirmed. |
| `0 < ΔP_PLT < ΔP_CLT`, same (layer, position) | **D** | PLT detects the features but cannot control them as cleanly. Detection/intervention separable. Matches Ameisen et al.'s original CLT-vs-PLT faithfulness finding. |
| PLT spike at different (layer, position) vs CLT | **C** | Different decomposition. Report both peaks; CLT likely has sharper lens, PLT finds a local projection of the same structure. |

### Secondary metrics (diagnostic)

- **Spike layer / position** (integer coordinates): direct equality check.
- **Position-sweep profile correlation**: Pearson correlation between CLT and PLT normalised profiles over token positions, at the spike layer. A flat PLT profile with a sharp CLT peak → Outcome C.
- **Decoder-projection magnitude ratio**: `max_{CLT features} ⟨dec, unembed(w)⟩ / max_{PLT features} ⟨dec, unembed(w)⟩` — strength of alignment of the best feature in each dictionary. Diagnostic of dictionary concentration, not overlap.

### Why this framing is cleaner than V1

- **Primary metric is measured in the base model's space.** Jaccard-across-namespaces replaced by a quantity with unambiguous meaning.
- **Aligns with Hanna & Ameisen's own methodology** — they report causal effect on the planned token, directly reproducible.
- **Matches Ameisen et al. (original circuit-tracing)** — they already compare CLT vs PLT on causal faithfulness; we extend that comparison to the rhyme task.
- **Decides the outcome from a single table** — one row per (model, prompt) with ΔP_CLT and ΔP_PLT side by side.

---

## Appendix B — Transcoder inventory

| Model | CLT (candle-mi validated) | PLT (HF repo) |
|---|---|---|
| Llama 3.2 1B | [`mntss/clt-llama-3.2-1b-524k`](https://huggingface.co/mntss/clt-llama-3.2-1b-524k) | [`mntss/transcoder-Llama-3.2-1B`](https://huggingface.co/mntss/transcoder-Llama-3.2-1B) |
| Gemma 2 2B | [`mntss/clt-gemma-2-2b-426k`](https://huggingface.co/mntss/clt-gemma-2-2b-426k) · [`mntss/clt-gemma-2-2b-2.5M`](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M) | [`mntss/gemma-scope-transcoders`](https://huggingface.co/mntss/gemma-scope-transcoders) |
| Qwen3-0.6B | — | [`mwhanna/qwen3-0.6b-transcoders-lowl0`](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0) |
| Qwen3-1.7B | — | [`mwhanna/qwen3-1.7b-transcoders-lowl0`](https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0) |
| Qwen3-4B | — | [`mwhanna/qwen3-4b-transcoders`](https://huggingface.co/mwhanna/qwen3-4b-transcoders) |
| Qwen3-8B | — | [`mwhanna/qwen3-8b-transcoders`](https://huggingface.co/mwhanna/qwen3-8b-transcoders) (fit-test first) |
| Qwen3-14B | — | ❌ out of scope |

All PLTs from the [`mntss/per-layer-transcoders`](https://huggingface.co/collections/mntss/per-layer-transcoders) collection; all CLTs from [`mntss/cross-layer-transcoders`](https://huggingface.co/collections/mntss/cross-layer-transcoders).

---

## Appendix C — anamnesis requirements for candle-mi v0.1.9

candle-mi and [anamnesis](../../../anamnesis) are co-developed. anamnesis is the framework-agnostic *"parse any format, recover any precision"* crate ([`anamnesis/ROADMAP.md`](../../../anamnesis/ROADMAP.md)); candle-mi consumes its output. This appendix enumerates what anamnesis must deliver — and what it already does — for v0.1.9 to ship.

### C.1 — Already delivered by anamnesis (no new work)

| Capability | anamnesis phase | Covers candle-mi repo |
|---|---|---|
| Safetensors parsing | Phase 1 (v0.1.0) — via `safetensors` crate, already used by candle-mi's CLT loader | `mntss/clt-*`, `mntss/transcoder-Llama-3.2-1B`, all `mwhanna/qwen3-*-transcoders*` |
| NPZ parsing | Phase 3 (v0.3.0) | `google/gemma-scope-2b-pt-res` *if* we later use residual-stream SAEs (not in this roadmap) |
| PyTorch `.pth` parsing | Phase 3.5 (v0.3.1) | Not used by any v0.1.9 PLT repo — kept for completeness |

**Implication:** Stage 1 Llama arm (`mntss/transcoder-Llama-3.2-1B`) and all of Stage 2 (`mwhanna/qwen3-*-transcoders*`) load through plain safetensors — **zero new anamnesis work required**. The loader extensions live entirely in candle-mi's `TranscoderSchema::PltBundle` path (Step 1.1).

### C.2 — Crucial new deliverable: GemmaScope `.bin` parser

**Purpose:** unblock the Gemma PLT arm of Stage 1 (Step 1.6) and, by extension, any future candle-mi work that touches GemmaScope-derived transcoder repos.

**Target repo:** `mntss/gemma-scope-transcoders` — 26 × `features/layer_{l}.bin` (~140 MiB each) + `config.yaml` (2.5 KiB) + `features/index.json.gz` (1.64 MiB). No safetensors.

**Investigation prerequisite (one-off, before designing the kernel):**
1. Download `config.yaml` — expected to describe dtype, d_model, n_features, activation function (ReLU / JumpReLU / TopK), possibly tensor layout.
2. Download `features/index.json.gz` — expected to describe tensor names, shapes, and byte offsets within each `layer_{l}.bin`.
3. Hex-dump the first ~1 KiB of one `layer_0.bin` — check for magic bytes:
   - `\x80\x02` / `\x80\x04` → PyTorch pickle (reuse Phase 3.5 pickle VM)
   - `\x93NUMPY` → NPY (reuse Phase 3 parser)
   - `PK\x03\x04` → ZIP (NPZ or .pth variant)
   - no magic → raw tensor blob indexed by `index.json`
4. Cross-check against circuit-tracer's Python GemmaScope loader to confirm the decoding path.

**Proposed anamnesis deliverable (new phase slot, e.g. Phase 3.75 or folded into Phase 4.5 follow-up):**

- **Feature flag:** `gemmascope` (opt-in, same pattern as `npz`, `pth`, `gguf`).
- **Module:** `src/parse/gemmascope.rs` — decode `config.yaml` + `index.json.gz` + one or more `layer_*.bin` files per repo.
- **Public type:** `ParsedGemmaScope` with the same shape as `ParsedNpz` / `ParsedGguf` — a map from tensor name to `{ shape: Vec<usize>, dtype: GsDtype, data: Cow<'_, [u8]> }`. No candle dependency; bytes stay raw.
- **Expected tensor set per layer:** `W_enc`, `W_dec`, `W_skip` (if present), `b_enc`, `b_dec` — matching the bundle schema PLTs use elsewhere. Tensor names should be un-suffixed; the layer index is implicit in the file (same convention as `PltBundle` safetensors).
- **Dtype handling:** if GemmaScope stores BF16 natively, no dequantisation needed — `parse` alone suffices (the Phase 3 NPZ case). If it stores FP32 + per-block scales, route through `remember/` (the Phase 1 FP8 case). The investigation above determines which.
- **Public API:**
  ```rust
  pub fn parse_gemmascope(repo_dir: &Path) -> Result<ParsedGemmaScope>;
  impl ParsedGemmaScope {
      pub fn n_layers(&self) -> usize;
      pub fn layer(&self, l: usize) -> Option<&GemmaScopeLayer>;
      pub fn to_safetensors_bytes(&self, l: usize) -> Result<Vec<u8>>;  // optional convenience
  }
  ```

**Acceptance criteria for the anamnesis deliverable:**
1. Loads one layer from `mntss/gemma-scope-transcoders` end-to-end, returning the expected tensors.
2. Output byte-exact or abs-diff < 1e-6 against circuit-tracer's Python loader on a hand-picked layer.
3. Integrates into candle-mi's `TranscoderSchema::GemmaScopeBin` path with **no temp files** and no Python shim.
4. Feature-gated; users who don't enable `gemmascope` pay zero compile-time or runtime cost.
5. Unit tests + cross-validation fixture following the anamnesis Phase 1/3/4 pattern.

**Scope boundary (what anamnesis must NOT do):**
- Must not depend on candle — bytes out, no tensor conversion.
- Must not interpret MI semantics (encoder/decoder roles, rank-2 vs rank-3). Those are candle-mi's responsibility.
- Must not require Python at runtime.

### C.3 — Qwen3 auto-config extension (cross-crate — NOT blocking v0.1.9)

Already planned in [`anamnesis/ROADMAP.md`](../../../anamnesis/ROADMAP.md) lines 533–537 as a Phase 1 spillover. The implementation lives in **candle-mi**, not anamnesis (it's a candle-mi `src/config.rs` / `src/transformer/attention.rs` change — see Step 2.1 of this roadmap). Listed here only to confirm that anamnesis already documents the dependency direction: no duplication.

### C.4 — Nice-to-have (post-v0.1.9)

- **`amn convert --from gemmascope <repo>`** CLI — one-shot conversion of a GemmaScope repo to a bundle of standard safetensors files. Would let candle-mi's existing `PltBundle` loader handle the converted repo with **no `GemmaScopeBin` schema branch at all**, collapsing Step 1.6 into "run the converter, then use PltBundle". Pragmatic if the reader investigation shows the `.bin` format is stable. Appropriate Phase 5 (Lethe) or Phase 6 (Format Conversion Matrix) work in anamnesis.
- **`hf-fm --convert gemmascope`** — matches the `hf-fetch-model` dogfooding pattern at `anamnesis/ROADMAP.md:152`, where `hf-fm --dequantize bf16` already calls `anamnesis::parse + remember`. Would be the end-to-end `hf-fm <repo> --convert gemmascope → safetensors` download+convert pipeline.

### C.5 — Minimum path to ship `v0.1.9`

1. **anamnesis does nothing new.** Ship `v0.1.9` (Stage 1 Llama arm) on plain safetensors via existing anamnesis capabilities. Gemma arm waits. Optional milestone tag `v0.1.9-plt-llama` (no publish) can mark the intermediate Llama-validated checkpoint during development.
2. **Then** (independent cadence) anamnesis lands C.2. candle-mi follows up with Step 1.6 and ships `v0.1.10` (patch release — adds the Gemma arm).

This ordering keeps the Llama release unblocked while giving anamnesis a clean, independently-specified deliverable.
