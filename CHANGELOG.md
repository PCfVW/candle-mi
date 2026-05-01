# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.10] - 2026-05-01

### Added (Phase B — Gemma arm of `clt_vs_plt_planning_site`)

- **`--family {llama,gemma}` CLI flag** on `examples/clt_vs_plt_planning_site.rs`,
  backed by a new `FamilyPreset` struct that centralises every model-,
  transcoder-, prompt-, and sanity-gate constant. Two shipped presets:
  `LLAMA` (the original Jacopin replication target — unchanged behaviour
  by default) and `GEMMA` (`mntss/clt-gemma-2-2b-426k` +
  `mntss/gemma-scope-transcoders` curation entry-point routing to
  `google/gemma-scope-2b-pt-transcoders` weights). `GEMMA.reference_max_prob = 0.457`,
  measured 2026-05-01 via `figure13_planning_poems --preset gemma2-2b-426k`
  on this candle 0.9 stack (cf. plip-rs's reported `0.483`, ~5% drift).
- **Dual-hookpoint capture** in Step B's harness. `PltInputHook` enum
  (`ResidMid` for Llama `PltBundle`, `MlpPre` for `GemmaScopeNpz`)
  resolves to the concrete `HookPoint` per family. The Gemma run captures
  both `ResidMid` (for the CLT) and `MlpPre` (for the GemmaScope PLT) at
  every layer, so each transcoder is fed its native input. Llama keeps
  its single-hook path (no change) when both arms share `ResidMid`.
- **`plt_has_w_skip` capability bit** on `FamilyPreset`. Llama PLT (`true`)
  retains its `W_skip · x` projection at the spike position; GemmaScope
  (`false`, pure `JumpReLU` transcoder, no skip path) emits `null` for
  that field — preserving cross-family JSON schema compatibility.
- **Per-family output filenames** (`clt_step_a_{family}.json`,
  `clt_vs_plt_{family}.json`) and per-family default repos (CLI
  overrides via `--model` / `--clt-repo` / `--plt-repo` still work).
- **Gemma 2 2B Step A reproduction** committed at
  `docs/experiments/clt-vs-plt-planning-site/clt_step_a_gemma2_2b.json` —
  hand-picked Jacopin features `{(L16:13725), (L25:9385)}` + inject
  `(L22:10243)`, `P(" around") = 0.4567` at trailing-space spike (pos 31).
- **Gemma 2 2B Step B run** committed at `clt_vs_plt_gemma2_2b.json`
  with the full V3 Step 1.7 instrumentation (top-20 features per arm,
  all-layer activation traces, pre-activation histograms at L22 ± 1,
  both CLT decoder-slice metrics in parallel, GemmaScope-side `W_skip`
  projection emitted as `null`). Runtime 12.5 min on the 5060 Ti.
- **`docs/experiments/clt-vs-plt-planning-site/findings.md` Gemma section**
  with a "What 'detection' means here" disambiguation block (Step A
  paper-protocol vs Step B method-matched protocol), the Gemma headline
  result table (degenerate Outcome B — both arms ≈ 0 under both
  same-layer and max-over-target rankings), a Llama-vs-Gemma contrast
  table, and the (A)–(F) discrimination battery preliminary status.
  The Llama analysis is now under `# Llama 3.2 1B — findings` and is
  unchanged in content.

### Fixed (Phase B)

- **`GemmaScope` decoder access** (`src/clt/mod.rs`) — Phase A's deferral
  surfaced as `HeaderTooLarge` failures when the PLT arm of
  `clt_vs_plt_planning_site --family gemma` reached
  `score_features_by_decoder_projection` on a `GemmaScopeNpz` transcoder
  (the call read the `.npz` file via `SafeTensors::deserialize` →
  immediate parse failure). New `load_decoder_w_dec(schema, path, layer)`
  free function dispatches to either safetensors deserialisation
  (`CltSplit`, `PltBundle`) or `crate::sae::npz::load_npz_selective`
  (`GemmaScopeNpz`, requires the `sae` feature). All six decoder-load
  sites — `decoder_vector`, `cache_steering_vectors`,
  `cache_steering_vectors_all_downstream`,
  `score_features_by_decoder_projection`,
  `score_features_by_decoder_projection_batch`, `extract_decoder_vectors`
  — collapse from a 5-line read+deserialize+name-lookup+view block to a
  one-line helper call. Diagnostic `info!` byte-size lines compute from
  the returned `Tensor` via `elem_count() * dtype().size_in_bytes()`.

### Added

- **`GemmaScope` PLT loader** (`src/clt/gemmascope.rs`, `src/clt/mod.rs`) —
  realises v0.1.9's deferred [`TranscoderSchema::GemmaScopeNpz`] arm:
  - New `pub` module `clt::gemmascope` with `parse_gemmascope_config()`
    hand-rolled `YAML` parser (no `serde_yaml` dependency) and the
    crate-private `GEMMASCOPE_WEIGHTS_REPO` constant pointing to
    `google/gemma-scope-2b-pt-transcoders` (the actual NPZ weights repo).
  - Two-repo flow inside `CrossLayerTranscoder::open()`: caller passes
    `mntss/gemma-scope-transcoders` (curation), open() fetches
    `config.yaml` from there, parses the `transcoders:` list, and routes
    NPZ fetches to the `google/*` weights repo.
  - NPZ encoder loader handles the `W_enc [d_model, n_features]`
    on-disk transpose to canonical `[n_features, d_model]` orientation
    (matching `circuit-tracer`'s `load_gemma_scope_transcoder()` reference)
    and loads the per-feature `threshold` tensor for `JumpReLU` gating.
  - `encode()` branches on schema: `CltSplit`/`PltBundle` keep plain
    `ReLU`; `GemmaScopeNpz` applies `pre * (pre > threshold)` element-wise.
  - `LoadedEncoder.threshold: Option<Tensor>` field — `None` for
    non-`JumpReLU` schemas.
  - The whole `GemmaScope` path is gated behind the `sae` feature
    (NPZ parsing requires `anamnesis/npz`); without `sae`, `open()`
    surfaces a clear `MIError::Config` explaining the feature gate.
- **`scripts/plt_gemma_validation.py`** + **`scripts/plt_gemma_reference.json`** —
  from-first-principles encoder oracle for `google/gemma-scope-2b-pt-transcoders`.
  Loads NPZ files directly via `huggingface_hub` + `numpy` (no
  `circuit-tracer` involvement), applies
  `pre = W_enc.T @ residual + b_enc; acts = pre * (pre > threshold)` in
  torch on CPU, dumps top-10 feature indices + activations for 9 test
  cases (3 seeds × layers `{0, 12, 25}`). Methodology mirrors
  `plt_llama_validation.py` (V3 Step 1.4) for the Llama PLT arm.
- **`tests/validate_plt_gemma.rs`** — `#[ignore]` integration test
  (CPU; requires ~864 MiB of cached NPZs) asserting candle-mi's
  `GemmaScope` encoder reproduces the Python oracle's top-10 feature
  indices exactly with abs-diff < 1e-4 on activation magnitudes.
  Validated on Gemma 2 2B `width_16k`: 9/9 cases pass with max abs-diff
  4.20e-5. Gated by `required-features = ["clt", "sae", "transformer"]`.

### Changed

- **`src/sae/npz.rs` visibility** — promoted from private `mod npz`
  to `pub(crate) mod npz` so the CLT `GemmaScope` loader can reuse
  the existing `NPZ → candle Tensor` bridge instead of duplicating
  `F32`/`F64` conversion logic.
- **`CrossLayerTranscoder::open()` `GemmaScopeNpz` branch** — replaces
  the v0.1.9 deferral early-return with cfg-branched dispatch:
  `#[cfg(feature = "sae")]` calls `Self::open_gemmascope()`;
  `#[cfg(not(feature = "sae"))]` returns an `MIError::Config`
  instructing the caller to enable `sae`.
- **`download_repo()` helper** (`src/clt/mod.rs`) — new private method
  that routes lazy downloads to the right `HuggingFace` repo per
  schema. For `CltSplit` / `PltBundle` this is `self.repo_id`; for
  `GemmaScopeNpz` it is `GEMMASCOPE_WEIGHTS_REPO`. Fixes the bug
  where layer-N≥1 NPZ fetches incorrectly targeted the `mntss/*`
  curation repo (caught by `validate_plt_gemma` on layer 12).
- **Bump `anamnesis` dependency** from `0.4.1` to `0.4.2`. v0.4.2 closes
  Phase 4.5 of anamnesis (full GGUF block-quant coverage — 22 of 22
  kernels, MXFP4 added) and ships a CLI feature-gate UX fix; neither
  affects candle-mi's `sae` (npz) or `stoicheia` (pth) feature paths
  — the bump is a "stay-current" validation against the new release.
  All 191 candle-mi tests pass against `anamnesis 0.4.2` with the
  `sae,stoicheia` feature set.
- **Bump `anamnesis` dependency** from `0.4.2` to `0.4.3`. v0.4.3 ships
  Phase 4.7 — `inspect_npz_from_reader<R: Read + Seek>`, the
  reader-generic NPZ inspector that resolves the library-side half of
  Phase A's algorithmic finding 4 (the candle-mi v0.1.10 GemmaScope
  `open()` flow is explicitly credited as the dogfooding cycle that
  drove the API). The remaining piece — an HTTP-range `Read + Seek`
  adapter for HF files — is downstream work in `hf-fetch-model`. The
  `open_gemmascope` `TODO` is updated to point at the new API and to
  document the call-site refactor pattern. v0.4.3 also ships an
  mmap-based always-on `parse()` (~3236× faster on a 11.6 GiB
  safetensors shard) and an `n_elements` overflow saturation fix;
  neither affects candle-mi's CLT path. All candle-mi tests pass
  against `anamnesis 0.4.3` with the `sae,stoicheia` feature set.
- **Bump `hf-fetch-model` dependency** from `0.9.7` to `0.9.8` via
  `cargo update`. v0.9.8 adds download durability features
  (per-file timeout, automatic resume on retry); no breaking changes.

### Removed

- **`GEMMASCOPE_DEFERRAL_ERR` constant** (`src/clt/mod.rs`) — superseded
  by the actual `GemmaScope` loader. The v0.1.9 deferral test
  (`gemmascope_deferral_error_message_is_informative`) is also removed.

## [0.1.9] - 2026-04-19

### Added

- **`TranscoderSchema` enum** (`src/clt/mod.rs`) — three-variant
  `#[non_exhaustive]` enum (`CltSplit`, `PltBundle`, `GemmaScopeNpz`)
  classifying transcoder repositories by on-disk layout. Auto-detected at
  [`CrossLayerTranscoder::open`] time from the repo file listing, before
  any weight downloads. `is_cross_layer()` and `is_jump_relu()` accessors.
- **`CltConfig` schema fields** — `schema: TranscoderSchema` and
  `gemmascope_npz_paths: Vec<String>` exposed on the auto-detected config.
  The `PltBundle` variant covers `mntss/transcoder-*` and
  `mwhanna/qwen3-*-transcoders*` per-layer bundles; `GemmaScopeNpz` detection
  is wired but loading is intentionally deferred to a follow-up release
  (returns a clear error pointing to roadmap Step 1.6).

### Changed

- **CLT `open()`** now branches on detected schema for both layer counting
  and first-file dimension probing. `CltSplit` keeps reading
  `W_enc_0.safetensors` unchanged; `PltBundle` reads the un-suffixed `W_enc`
  tensor from `layer_0.safetensors`.
- **CLT decoder access routed through schema-aware helpers.**
  `decoder_file_and_tensor_name`, `decoder_row`, and `decoder_layer_slice`
  concentrate per-schema branching in three private free functions. All
  seven `W_dec` access sites (`decoder_vector`, `cache_steering_vectors`,
  `cache_steering_vectors_all_downstream`, `score_features_by_decoder_projection`,
  `score_features_by_decoder_projection_batch`, `extract_decoder_vectors`,
  `ensure_decoder_path`) use the helpers. `CltSplit` behaviour is unchanged
  — refactor only, no external API break. Prepares the decoder side for
  the `PltBundle` encoder wiring that lands alongside.
- **CLT encoder access routed through schema-aware helpers.**
  New `encoder_file_and_tensor_names` helper returns `(filename, W_enc name,
  b_enc name)` per schema. `ensure_encoder_path`, `load_encoder`, and
  `open()`'s first-layer dimension probe all use it. For non-`CltSplit`
  schemas, `ensure_decoder_path` now delegates to `ensure_encoder_path` —
  encoder and decoder share the same bundle file, so the path cache is
  unified instead of double-tracked.
- **`classify_transcoder_schema` extracted as a pure function.** The schema
  detection logic previously inlined in `open()` is now a pure `&[&str] ->
  Result<TranscoderSchema>` helper. `open()` becomes three lines of
  collect + call + log; the logic is independently unit-testable.

### Added (continued)

- **`scripts/plt_llama_validation.py`** + **`scripts/plt_llama_reference.json`** —
  from-first-principles Python encoder oracle for
  `mntss/transcoder-Llama-3.2-1B`. Loads `layer_{L}.safetensors` bundles
  directly via `huggingface_hub` + `safetensors.torch` (no circuit-tracer),
  applies `ReLU(W_enc @ residual + b_enc)` in torch on CPU, dumps top-10
  feature indices + activations for 9 test cases (3 seeds × layers {0, 7, 15}).
  Mirrors plip-rs's `scripts/clt_reference.py` methodology that achieved
  90/90 top-10 CLT parity at max relative error 1.2×10⁻⁶. `scripts/README.md`
  gains a "PLT — Llama 3.2 1B (v0.1.9)" section documenting the pair.
  Consumed by the Rust parity test in V3 Step 1.5 (`tests/validate_plt.rs`).
- **`fetch_config_builder()`** public helper (`src/download.rs`, re-exported
  as `candle_mi::fetch_config_builder`) returns a pre-configured
  `hf_fetch_model::FetchConfigBuilder` with `.token_from_env()` applied, so
  every `hf-fetch-model` call site reads `HF_TOKEN` uniformly. See the
  matching entry under _Fixed_ below for the regression this closes.
- **`examples/clt_vs_plt_planning_site.rs`** — shared harness for the
  Hanna & Ameisen CLT-vs-PLT planning-site comparison on Llama 3.2 1B
  (PLAN-PLT-LLAMA-PLANNING-SIGNAL.md, Step A). `--schema clt` reproduces the
  `figure13_planning_poems.rs` Llama `-ee` preset on CUDA and additionally
  records decoder-projection top-5 features aligned with `unembed("that")`
  at the inject layer plus raw logits alongside probabilities; outputs
  `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json`. Built-in
  sanity gate: soft-warns if `max P("that")` drifts more than `1e-2` from
  the candle-mi reference `0.687`, hard-fails below `0.50`. `--schema plt`
  is a Step-B stub that exits with a pointer to the plan. CUDA-or-bust:
  errors out if the device selector falls back to CPU.
- **`CrossLayerTranscoder::encode_pre_activation`** — returns the dense
  `W_enc @ x + b_enc` pre-activation tensor **before** the `ReLU`/`JumpReLU`
  sparsifier. Step B uses this to histogram encoder pre-activations at the
  spike layer and its two neighbours (V3 Step 1.7 (D) activation-regime
  discrimination). `encode()`'s sparse path now routes through the same
  internal workhorse so the invariant `encode == relu ∘ encode_pre_activation`
  holds by construction. Unit test confirms the invariant on a synthetic
  `PltBundle` fixture.
- **`CrossLayerTranscoder::load_skip_matrix`** — loads the `W_skip` matrix
  `[d_model, d_model]` from a `PltBundle` layer (e.g.
  `mntss/transcoder-Llama-3.2-1B`) as dense `F32` on the requested device.
  Step B uses it to project `W_skip · x` at the spike position onto the
  unembedding direction, decomposing the apparent PLT planning signal into
  sparse-feature and linear-skip contributions (V3 Step 1.7). Explicitly
  errors with `MIError::Config` for `CltSplit` and `GemmaScopeNpz` schemas
  (no skip path defined). Unit tests cover round-trip values on a synthetic
  `PltBundle` and the negative path on `CltSplit`.
- **`examples/clt_vs_plt_planning_site.rs` Step B harness** — `--schema both`
  (new default) runs the full V3 Step 1.7 CLT-vs-PLT comparison on Llama
  3.2 1B. Four position sweeps per invocation (2 arms × 2 protocols:
  suppress-only top-5 + suppress+inject using decoder-projection-derived
  features), full instrumentation payload serialized to
  `docs/experiments/clt-vs-plt-planning-site/clt_vs_plt_llama.json`:
  top-20 decoder-projection rankings per arm, CLT's max-over-target-layers
  second-metric ranking (slice ambiguity control), top-20 decoder vectors,
  20×n_layers×seq_len all-layer activation traces, 32-bin pre-activation
  histograms at the spike layer and its two neighbours, PLT `W_skip · x`
  projection at the spike position. `--schema clt` preserves Step A's
  Jacopin replication path unchanged; its output path moves from
  `clt_vs_plt_llama.json` to `clt_step_a_llama.json` to keep the two
  experiments separate on disk. Total runtime ~4 min on CUDA. First
  empirical numbers on Llama 3.2 1B: PLT suppress-only ΔP = +0.986
  (Δlogit = +49.8) at position 30, method-matched CLT ΔP = +5.7×10⁻⁷ at
  position 7.
- **`docs/experiments/clt-vs-plt-planning-site/findings.md`** — Step C
  write-up mapping the Step B data onto V3 Appendix A. Outcome label:
  **C** (PLT and CLT spike at different positions under the
  method-matched top-5 decoder-projection ranking). Primary-metric table
  with Step A paper-replication row for cross-check; secondary metrics
  (Pearson sweep-profile correlation r ≈ −0.6, decoder-projection
  magnitude ratios, PLT `W_skip · x` projection = +0.541); and the
  (A)–(F) discrimination battery populated from the already-captured
  instrumentation. Key finding under (B): CLT's best decoders for
  `" that"` sit at L13, not L14 — cosine jumps from 0.349 (same-layer)
  to 0.608 (max-over-target-layers). Highest-priority follow-up: rerun
  the CLT arm with max-over-target-layers top-5 suppress to test whether
  the arm asymmetry is a ranking-method artefact rather than a
  transcoder-class limitation.
- **README Paper replications table** — new row for Hanna & Ameisen,
  *Latent Planning Emerges with Scale* (arXiv 2604.12493, ICLR 2026),
  pointing at
  [`docs/experiments/clt-vs-plt-planning-site/findings.md`](docs/experiments/clt-vs-plt-planning-site/findings.md).
  Summary: both transcoder classes detect the Llama 3.2 1B rhyming-couplet
  planning site at comparable ΔP when each is ranked via a method that
  respects its decoder topology (same-layer for PLT, max-over-target for
  CLT). Llama arm complete; Gemma arm scoped for v0.1.10; Qwen-3 scale
  sweep TBD.
- **Follow-up 1** (same example, added to `--schema both` default run):
  three extra CLT position sweeps using the max-over-target-layers top-5
  as the suppress set (suppress-only; suppress+inject with max-over-target
  top-1 inject; suppress+inject with same-layer top-1 inject held
  constant). Serialized under `arms.clt.max_over_target_follow_up` —
  `None` for PLT (PltBundle has only one decoder slice by construction).
  Result: CLT ΔP recovers to **+0.871** (suppress-only) through **+0.917**
  (suppress+inject with same-layer inject held constant), spike at
  position 30 matching PLT and the Step A Jacopin reference. CLT/PLT
  ratio 0.88–0.93 → outcome reclassifies from **C → B** under the
  method-matched-per-transcoder-capabilities comparison. Confirms
  discrimination (B) as dominant. `findings.md` updated with the
  "Follow-up 1 results" section and a revised Stage 1 decision
  (proceed to Gemma 2 2B in v0.1.10; defer V3 Stage 2 unless Gemma
  surfaces something new). Runtime ~30 s added to Step B total (~4.5 min
  total on CUDA).

### Changed

- **`score_features_by_decoder_projection` + batch variant** now skip source
  layers that cannot decode to the requested target layer on per-layer
  schemas (`PltBundle`, `GemmaScopeNpz`). Previously any call with
  `target_layer != source_layer` on a `PltBundle` transcoder errored with
  `per-layer schema PltBundle only writes to its own layer` when
  `decoder_layer_slice` rejected the non-zero `target_offset`. Now the
  outer loop consults the existing `schema.is_cross_layer()` and skips
  incompatible source layers cleanly. `CltSplit` behaviour is unchanged.
  Caught while wiring Step B's `--schema both` path against
  `mntss/transcoder-Llama-3.2-1B`.

### Tests

- **Schema classification suite** — seven unit tests covering `CltSplit`,
  `PltBundle`, `GemmaScopeNpz` (both mntss-metadata and google-direct NPZ
  layouts), unrecognised layout, empty listing, the CltSplit-over-PltBundle
  precedence rule, and the deferral-error-message content.
- **Schema-aware helper suite** — ten unit tests for
  `encoder_file_and_tensor_names`, `decoder_file_and_tensor_name`,
  `decoder_row` (rank-3 indexing for CltSplit, rank-2 for PltBundle,
  rejection of non-zero target_offset), and `decoder_layer_slice`.
- **PltBundle round-trip** — `create_synthetic_plt_bundle` helper writes
  a fake `layer_N.safetensors` with all five un-suffixed tensors
  (`W_enc`/`W_dec`/`W_skip`/`b_enc`/`b_dec`), and a regression test
  verifies `cache_steering_vectors_all_downstream` produces exactly one
  cache entry per feature (not n_layers), catching the pre-aa23c90 bug.
- **CltSplit companion regression** — parallel test confirms CltSplit
  still caches `n_layers - source_layer` entries.
- **`tests/validate_plt.rs`** — integration test that loads
  `mntss/transcoder-Llama-3.2-1B` via `CrossLayerTranscoder::open()`,
  asserts detected schema is `PltBundle`, then for each of the 9 test
  cases in `scripts/plt_llama_reference.json` (3 seeds × layers
  {0, 7, 15}) reconstructs the oracle's residual vector, runs the Rust
  encoder, and verifies: active-feature count matches, top-10 indices
  match exactly, top-10 activation abs-diff < 1e-4.
  **Parity confirmed:** max abs-diff across all 90 top-10 comparisons
  is **1.34×10⁻⁵** (well under the 1e-4 bar). `#[ignore]`-gated;
  requires the PLT (~16 GiB) cached. Runs on CPU to match the Python
  oracle bit-for-bit.

### Fixed

- **`hf-fetch-model` 0.9.6 API alignment** — `list_repo_files_with_metadata`
  now receives the required `&reqwest::Client` via the re-exported
  `hf_fetch_model::build_client` helper. The `clt` feature path would not
  compile against 0.9.6 before this fix; the CI per-backend clippy matrix
  (transformer, rwkv) did not cover `clt` and so missed the regression.
- **`HF_TOKEN` auth across all download call sites.** `hf-fetch-model` 0.9.x
  no longer auto-reads `HF_TOKEN` from the default `FetchConfig::builder()` —
  callers must opt in via `.token_from_env()`. Every candle-mi call site
  (`MIModel::from_pretrained`, `CrossLayerTranscoder::open`,
  `Sae::from_npz_hf` / `Sae::from_pretrained`, the
  `download_model{,_blocking}` helpers, the `auto_config_dogfood`,
  `recurrent_feedback`, and `clt_vs_plt_planning_site` examples, and the
  `validate_models` Mistral harness) now routes through the new
  `candle_mi::fetch_config_builder()` helper, unblocking gated models
  (Llama, Mistral, Gemma, Qwen). `MIModel::from_pretrained` also switches
  from `download_files_blocking` to
  `download_files_with_config_blocking(..., &fetch_config)` so the token
  actually propagates. The raw `build_client(None)` call in the CLT schema
  probe gains a matching inline `HF_TOKEN` read.

## [0.1.8] - 2026-04-13

### Added

- **Stoicheia backends** (`src/stoicheia/`) — two `MIBackend` implementations
  for ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo) tiny
  models (8–1,408 parameters), behind the `stoicheia` feature flag:
  - `StoicheiaRnn` — single-layer ReLU RNN for continuous tasks (2nd argmax,
    argmedian, median), with per-timestep hook points via `HookPoint::Custom`
  - `StoicheiaTransformer` — attention-only transformer for discrete tasks
    (longest cycle), with standard `HookPoint` variants (Embed, AttnScores,
    AttnPattern, ResidPre/Post)
  - `StoicheiaConfig::from_task()` — config constructor with task→architecture
    mapping matching AlgZoo's Python registry
  - Ground-truth task functions (`tasks::second_argmax`, `argmedian`, `median`,
    `longest_cycle`) for model validation
  - Cross-validation tests: RNN and transformer outputs match Python reference
    to 1e-4 (RNN) and 1e-2 (transformer) tolerance
  - `stoicheia_inference` example with CLI for running any AlgZoo model
- **Stoicheia MI tooling — Phase B** (`src/stoicheia/`) — six analysis modules
  for exhaustive mechanistic understanding of AlgZoo ReLU RNNs:
  - `fast` — raw f32 forward-pass kernel bypassing candle tensor overhead
    (18–25× faster on tiny models); `RnnWeights` shared weight container,
    `forward_fast`, `forward_fast_ablated`, `forward_fast_traced`, `accuracy`
  - `standardize` — weight rescaling so `|W_ih[j]| = 1`, exact equivalence
    transformation following the AlgZoo blog methodology
  - `piecewise` — ReLU activation region enumeration; `ActivationPattern`
    (320-bit compact vector), `classify_regions`, `region_linear_map`
  - `ablation` — single-neuron and pairwise zero-ablation with interaction
    scores detecting functional redundancy
  - `probing` — neuron functional classification via structured inputs;
    `NeuronRole` enum (RunningMax, MaxIncrement, LeaveOneOutMax, etc.)
  - `surprise` — ARC's information-theoretic metric; `MechanisticEstimator`
    trait, `OracleEstimator`, `SurpriseReport`
  - `stoicheia_analysis` example — full Phase B pipeline CLI
  - Integration test (`stoicheia_analysis`) exercising all six modules on
    the M₂,₂ fixture
- **Agnostic weight loading** — `StoicheiaRnn::load()` and
  `StoicheiaTransformer::load()` now accept `.safetensors`, `.pth`, or
  `.pkl` files. Format is detected from the file extension; `.pth`/`.pkl`
  files are converted in memory via anamnesis' pickle VM (no manual
  preprocessing step). The `stoicheia` feature now pulls in anamnesis
  with the `pth` feature gate.
- **`hf-fetch-model` dependency** relaxed from exact version pin to semver
  range `"0.9"` — `cargo update -p hf-fetch-model` picks up patches without
  cross-repo workflow automation

### Changed

- **CONVENTIONS.md refactored** from rule-type grouping to trigger-based
  grouping — rules organized by "when writing X, check Y" with a trigger
  checklist at the top. Same rules, different organization optimized for
  LLM-assisted development. Previous version archived in
  `docs/conventions/CONVENTIONS-v1-reference-based.md`.
- **`anamnesis` dependency** bumped from 0.3.0 to 0.4.1:
  - v0.3.1 added `.pth` pickle parsing (minimal VM, security allowlist)
  - v0.4.0 added GGUF support
  - v0.4.1 added `pth_to_safetensors_bytes()` for in-memory conversion
    (candle-mi dogfooding feedback)
  - Per-feature activation: `stoicheia` activates `anamnesis/pth`;
    `sae` activates `anamnesis/npz`

### Fixed

- **Stale example counts in documentation** — `README.md`, `ROADMAP.md`, and
  `examples/STYLE_GUIDE.md` referenced "19 examples" or "21 examples" while the
  actual count (verified against both `examples/*.rs` and the `[[example]]`
  entries in `Cargo.toml`) is 22. Updated all five occurrences.
- **`ROADMAP.md` example inventory out of sync** — the file-structure tree in
  §6 listed only 15 of the 22 examples, and the topical lists in §6 and the
  Phase 5 task entry omitted the same seven. Added `counterfact_patching`,
  `factual_routing`, `steering_convergence`, `attention_routing`,
  `correction_test`, `clt_probe`, and `stoicheia_inference` to the tree (in
  logical groupings) and extended the topical lists to mention CLT feature
  probing, prolepsis correction tests, recurrent CLT feedback, and
  AlgZoo/Stoicheia inference.

## [0.1.7] - 2026-03-30

### Added

- **`clt_probe` example** — inspect CLT feature activations at any token position
  across all encoder layers; includes `--decoder-search` mode for finding suppress
  candidates by decoder projection
- **`correction_test` example** — test whether downstream layers can reverse a
  prolepsis commitment by injecting contradictory features at late layers
  (referenced in COLM 2026 submission, Appendix G)
- **N=4 Llama attention routing results** — 4 prompts across 3 rhyme groups with
  validated features from `rhyme_pairs_llama.json`; updated Mathematica plots

### Changed

- **Llama `figure13_planning_poems` preset** — replaced with -ee group suppress
  features (`L13:30985`, `L9:5488`, `L14:27874`, `L13:32049`) and -ee prompt;
  strength 15 → 10. All features traceable to systematic decoder-projection
  vocabulary scan.
- **Attention routing plots** — regenerated from N=4 data; cross-model comparison
  now shows all 4 Llama curves alongside Gemma

### Fixed

- **Hallucinated suppress feature L5:19894 removed** — the Llama figure13 preset
  used a fabricated CLT feature ID introduced during a context continuation.
  Replaced with legitimate features from `rhyme_pairs_llama.json`. See
  `docs/dogfooding-feedbacks/` for the full correction report.

### Removed

- Old single-prompt Llama routing data superseded by N=4 validated results

## [0.1.6] - 2026-03-25

### Added

- **`MIModel::forward_text()`** — text-in, MI-out: combines encode + tensor
  creation + forward in one call, returning `TextForwardResult` with both
  `HookCache` and `EncodingWithOffsets` for position-aware analysis
- **`TextForwardResult`** struct — bundles hook cache with token offset mapping;
  provides shortcuts for `output()`, `require()`, `tokens()`, `seq_len()`
- **`EncodingWithOffsets::label_spans()`** — classify tokens by named byte-range
  spans (e.g., subject, relation) with automatic `_final` suffix on last token
  per span; replaces ad-hoc token classification in examples
- **`counterfact_patching` example** — replicates the Transluce activation
  patching protocol (Li et al., 2025, arXiv:2511.08579) on Llama 3.2 1B:
  contiguous layer-block patching with CounterFact forced-choice prompts,
  JSON output compatible with causal tracing heatmaps; first example to use
  the new `forward_text` + `label_spans` API
- **`factual_routing` example** — measures attention routing changes during
  CounterFact patching; identifies L15:H8 as the dominant factual routing
  head on Llama 3.2 1B (zero overlap with planning routing head L13:H14);
  establishes **prolepsis** — early irrevocable commitment via task-specific
  late-layer attention routing — as a structural motif across tasks, models,
  and scales
- **`examples/STYLE_GUIDE.md`** — codifies example conventions: `forward_text`
  for token positions, `--no-runtime` flag, memory reporting, JSON output
  with timing, `clap` CLI pattern, and new-example checklist
- **Attention routing cross-model results** — Llama 3.2 1B planning routing
  (524K CLT, L13:H14 dominant) with cross-model comparison plots against
  Gemma 2 2B (426K CLT, L21:H5 dominant)

### Changed

- **`attention_patterns` example** refactored to use `encode_with_offsets()` —
  token strings come directly from the encoding instead of per-token `decode()`
  loop (7 lines → 1 line)

### Fixed

- **`memory.rs`** — collapsed nested `if let` / `if` block for clippy 1.94
  `collapsible_if` lint

## [0.1.5] - 2026-03-24

### Added

- **`design/add-at-positions.md`** — design document for `Intervention::AddAtPositions`,
  a position-specific sparse injection variant inspired by
  [PR #1](https://github.com/PCfVW/candle-mi/pull/1) and the
  [K-BERT](https://arxiv.org/abs/1909.07606) injection paradigm

### Changed

- **NPZ parsing migrated from internal implementation to `anamnesis` v0.3.0** —
  4.9x faster (84 ms vs 413 ms on 302 MB Gemma Scope file), broader dtype
  support (F16, BF16, F32, F64, integers, Bool), big-endian handling
- **MSRV bumped from 1.87 to 1.88** — required by `libloading 0.9.0` dependency;
  CI workflow updated accordingly
- **`hf-fetch-model` dependency** bumped from 0.8.1 to 0.9.0
- **Collapsed nested `if`/`if let` blocks** into `let` chains in `config.rs`,
  `hooks.rs`, and `transformer/mod.rs` — required by `clippy::collapsible_if`
  in Rust 1.94

### Fixed

- **Unused variable warnings** in `steering_convergence` when compiled without
  `clt` feature — `args` and `device` parameters are only used in CLT mode;
  silenced with `let _ = (args, device)` guard

### Removed

- **Internal NPZ/NPY parser** (`src/sae/npz.rs`, ~365 lines) — replaced by
  `anamnesis` dependency behind the `sae` feature gate
- **Direct `zip` crate dependency** — ZIP extraction now handled internally by
  `anamnesis`

## [0.1.4] - 2026-03-19

### Added

- **`attention_routing` example** — measures how CLT suppress+inject changes
  attention patterns from the output position to the planning site; uses the
  exact Figure 13 API (`prepare_hook_injection` with
  `cache_steering_vectors_all_downstream`); identifies specific attention heads
  involved in rhyme planning (L21:H5 dominant, H5 family across layers 17-25);
  includes strength sweep revealing a soft attractor boundary (gradual
  saturation at ~15× strength); supports `--suppress` flags for full
  suppress+inject paradigm; fills a specific gap identified by Anthropic:
  *"attention head routing is invisible to our current approach"*
- **`attention_routing` results and plots** (`examples/results/attention_routing/`) —
  JSON output for 426K and 2.5M CLTs, Mathematica plotting script with
  strength sweep curves, top-10 routing head bar charts (both CLTs), and
  linear extrapolation showing saturation onset; README with pedagogical
  explanation and detailed comparison with Anthropic's "Planning in Poems"
- **CLT decoder vector steering mode** in `steering_convergence` — `--clt`,
  `--feature`, `--decoder-layer` flags for using CLT decoder vectors as
  steering direction instead of contrastive subtraction; per-layer decoder
  extraction; multi-layer simultaneous injection matching Figure 13 paradigm;
  diagnostic output showing residual diff at inject vs output positions
- **`steering_convergence` example** — inject contrastive steering vectors at
  each layer, measure cosine similarity to natural activations, identify
  absorption boundaries; convergence matrix, strength sweep, batch mode
  (`--batch-file`) for 20 rhyme groups, `--inject-position` with `auto`
  mode for planning site detection
- **`steering_convergence` results** (`examples/results/steering_convergence/`) —
  JSON output for Llama 3.2 1B and Gemma 2 2B, batch results for 20 rhyme
  groups, Mathematica plots; key findings: factual recall has a hard attractor
  boundary at ~1.2× contrastive distance, rhyme planning is invisible to
  last-token residual stream perturbation
- **`figure13_planning_poems` chart and explanation** (`examples/README.md`) —
  `gemma_log.png` with pedagogical walkthrough

### Changed

- **`ROADMAP.md` consistency pass** — updated to reflect v0.1.3 project state
- **examples `README.md` overhaul** — added output sections for
  `steering_convergence` and `attention_routing`, run commands for
  `attention_routing` (3 variants), prerequisites, consistency pass across
  all 13 output sections and 17 table entries

## [0.1.3] - 2026-03-16

### Added

- **`sync_and_trim_gpu` public API** (`src/memory.rs`) — synchronizes the CUDA
  device and trims the stream-ordered memory pool (`cuMemPoolTrimTo`) to release
  unused reserved VRAM back to the device; exported from `candle_mi` for use by
  examples and downstream crates
- **VRAM-aware `max_tokens` auto-tuning** in `character_count_helix` — measures
  free VRAM after model load and selects a safe chunk size (1024 on 16 GB cards)
  to prevent OOM from cuBLAS workspace accumulation across hundreds of forward
  passes; prints `Auto-tuned max_tokens: N` when the value is lowered
- **Explicit GPU tensor cleanup** in `character_count_helix` — drops all GPU
  tensors (`cache`, `input`, residuals) and calls `sync_and_trim_gpu` after each
  chunk to bound VRAM usage; keeps memory flat at ~+20 MB above model load
  across entire sweeps
- **Multi-layer `--sweep N` and `--sweep all`** in `character_count_helix` —
  `--sweep` (bare) still sweeps 1 layer; `--sweep 5` sweeps the next 5 layers
  in one run; `--sweep all` sweeps all remaining layers (may be overnight run on consumer hardware);
  `--sweep 0` exits immediately with a message; progress is saved to JSON
  after each layer so interrupted runs resume cleanly
- **Rotating helix GIF** — `L12_helix_rotating.gif` checked into
  `examples/results/character_count_helix/plots/`, embedded in both
  `examples/README.md` and the experiment `README.md`; generated from
  30-chapter Dickens corpus (1.58M tokens, 98.5% top-6 variance at layer 12)
- **Experiment README** (`examples/results/character_count_helix/README.md`) —
  documents the full experiment setup, key findings across all 26 layers,
  reproduction commands, and references
- **Paper replications table** — added Anthropic's "When Models Manipulate
  Manifolds" (2025) to the main `README.md`
- **Full causal trace heatmap** in `activation_patching` — extends the
  subject-position sweep to a full layer × token position grid (Meng et al.
  Figure 1e); prints a text heatmap table and writes structured JSON with
  `--output` for Mathematica plotting; adds the paper's original "Space Needle
  → Seattle" prompt alongside the existing "France → Paris"
- **Mathematica plotting script** for activation patching
  (`examples/results/activation_patching/causal_trace_plot.wl`) — generates
  the causal trace heatmap (tokens on Y-axis, layers on X-axis) and a
  subject-position recovery curve

### Changed

- **`dxgi-debug` feature renamed to `memory-debug`** — now covers both raw DXGI
  query output and per-chunk VRAM measurements; all references updated in
  `Cargo.toml`, `src/memory.rs`, `examples/character_count_helix.rs`,
  `examples/README.md`, and `CHANGELOG.md`

### Fixed

- **Compile error without `memory` feature** — `sync_and_trim_gpu` was called
  unconditionally in `character_count_helix` but only imported under
  `#[cfg(feature = "memory")]`; added matching `cfg` guard on the call site
- **Missing `#[must_use]` on `vram_qualifier()`** (`src/memory.rs`) — pure
  accessor was missing the annotation required by CONVENTIONS.md Rule 17

## [0.1.2] - 2026-03-14

### Added

- **Per-process VRAM via DXGI on Windows** (`src/memory.rs`) — new primary
  VRAM measurement path using `IDXGIAdapter3::QueryVideoMemoryInfo` (DXGI 1.4,
  Windows 10+); returns true per-process GPU memory under WDDM, where NVML
  returns `NOT_AVAILABLE` because the Windows kernel manages GPU memory, not
  the NVIDIA driver; added `windows` crate (v0.62) as an optional dependency
  behind `features = ["memory"]`; three-tier fallback chain: DXGI (Windows
  per-process) → NVML (Linux per-process) → `nvidia-smi` (device-wide)
- **GPU adapter name** — `MemorySnapshot::gpu_name` field captures the adapter
  description from DXGI (e.g., `NVIDIA GeForce RTX 5060 Ti`);
  `MemoryReport::print_before_after` appends it to the VRAM line for
  multi-GPU identification
- **`memory-debug` feature** (implies `memory`, replaces `dxgi-debug`) — prints
  raw DXGI query results (adapter name, dedicated VRAM, current usage, budget)
  and per-chunk VRAM measurements to stderr for diagnosing GPU memory issues
- **`--sweep` mode** for `character_count_helix` — one-layer-per-invocation
  PCA analysis with auto-resume from JSON output file; repeated runs walk
  through layers 0, 1, 2, ... automatically
- **Chunking for long sequences** in `character_count_helix` — splits token
  sequences exceeding `--max-tokens` into independent chunks for forward
  passes instead of truncating, preventing OOM on long texts (e.g., Dickens
  chapters on 16 GB VRAM)
- **Wall-clock completion time** in `character_count_helix` sweep mode —
  prints UTC finish time and total elapsed duration

### Fixed

- **NVML VRAM reporting garbage values** — `nvmlDeviceGetComputeRunningProcesses`
  returns `u64::MAX` (`0xFFFF_FFFF_FFFF_FFFF` = `NVML_VALUE_NOT_AVAILABLE`) for
  `usedGpuMemory` on all Windows WDDM systems; this sentinel was passed through
  as a real byte count, producing `17592186044416 MB` in output; now detected
  and triggers fallback to DXGI (per-process) or `nvidia-smi` (device-wide)
- **NVML struct alignment** — `NvmlProcessInfo` doc comment corrected to
  reference `nvmlProcessInfo_v2_t` (24 bytes), matching the struct layout used
  by `nvmlDeviceGetComputeRunningProcesses_v3` (the `_v3` suffix is a function
  version, not a struct version)

### Changed

- **VRAM measurement strategy** — documentation updated throughout
  `src/memory.rs` to reflect the three-tier DXGI → NVML → `nvidia-smi`
  approach; platform support table now shows DXGI for Windows per-process,
  NVML for Linux per-process
- **`GpuMemoryResult` type alias** — extracted complex return tuple into a
  named type for readability
- **`examples/README.md`** — added `memory` and `memory-debug` feature examples,
  Dickens `--text-dir` sweep command, and prerequisites section for the
  `memory` feature explaining the DXGI/NVML/WDDM story

## [0.1.1] - 2026-03-12

### Added

- **Recurrent feedback depth** — `RecurrentSpec::depth` field generalizes
  recurrent re-execution from hardcoded 2 passes to configurable N passes;
  updated `forward_recurrent()` and `recurrent_feedback` example accordingly

### Changed

- **README overhaul** — added supported model families table, hardware
  statement, RWKV callout, "See it in action" section with logit lens and
  CLT flagship examples, hook point definition, Quick Start with hooks,
  Paper Replications table, Design Philosophy section with "not an
  inference engine" positioning, and measured GPU/CPU timing
- **BACKENDS.md** — added "What failure looks like" subsection with three
  runnable auto-config commands (success, weight mismatch, unsupported arch)
- **examples/README.md** — updated `figure13_planning_poems` prerequisites
  to document automatic model/CLT download, sizes, and `HF_TOKEN` requirement
- **VRAM measurement upgraded to per-process** (`src/memory.rs`) — replaced
  `nvidia-smi` subprocess with direct NVML FFI via `libloading`; dynamically
  loads `nvml.dll` (Windows) or `libnvidia-ml.so.1` (Linux) at runtime and
  calls `nvmlDeviceGetComputeRunningProcesses` to get true per-process GPU
  memory; falls back to `nvidia-smi` (device-wide) if NVML is unavailable;
  new `MemorySnapshot::vram_per_process` field indicates measurement quality;
  `MemoryReport::print_delta` and `print_before_after` now append
  `[per-process]` or `[device-wide]` qualifier; added `libloading` as an
  optional dependency behind `features = ["memory"]`; zero new crate
  dependencies when the feature is off; no changes to the public API surface
  (all examples work without modification)

### Fixed

- **Broken intra-doc links in `clt` module** — added `crate::` prefix to
  `HookSpec`, `HookPoint::ResidPost`, and `Intervention::Add` doc links
  in `src/clt/mod.rs` that failed under `--no-default-features` builds
- **docs.rs build** — added `[package.metadata.docs.rs]` to `Cargo.toml`
  with `no-default-features = true` and all CPU-safe features enabled;
  the docs.rs sandbox lacks the CUDA toolkit (`nvcc`), so the default
  `cuda` feature caused `cudarc` build script failures; docs will build
  correctly on the next crates.io publish

## [0.1.0] - 2026-03-11

### Added

- **PCA utility** (`src/util/pca.rs`) — `pca_top_k()` computes the top principal
  components via power iteration with deflation on the kernel matrix; pure candle
  tensor ops (runs transparently on CPU or GPU with zero host-device transfers);
  returns `PcaResult` with components, eigenvalues, and explained variance ratios
- **Character count helix example** (`character_count_helix.rs`) — replicates the
  core finding from [Gurnee et al. (2025)](https://transformer-circuits.pub/2025/linebreaks/index.html)
  "When Models Manipulate Manifolds" (Transformer Circuits); wraps prose at 14 widths, captures `ResidPost`,
  averages residual vectors by character count, and runs PCA;
  demonstrates `pca_top_k`, `HookPoint::ResidPost`, `encode_with_offsets`, and
  full-sequence activation capture; `--scan-layers` for lightweight variance scan across layer ranges,
  `--pca-layers` for full PCA + cosine similarity + JSON on selected layers,
  `--text-dir` for multi-file batches, `--max-tokens` (default 4096) to prevent OOM on long sequences,
  `--text` for custom prose input, `--output` for structured JSON export;
  per-text progress with timing, memory reporting via `--features memory`;
  bundled with 10 Dickens chapters (~29K words) for large-scale experiments;
  companion Mathematica plotting script for 3D helix, cosine heatmap, and variance bar chart
- **Memory reporting API** (`src/memory.rs`) — `MemorySnapshot` and
  `MemoryReport` types for measuring RAM and VRAM consumption; RAM via
  Windows FFI (`K32GetProcessMemoryInfo`, per-process, exact) or Linux
  `/proc/self/status` (`VmRSS`, per-process, exact); VRAM via `nvidia-smi`
  subprocess (device-wide); gated behind `features = ["memory"]` which
  relaxes `forbid(unsafe_code)` to `deny(unsafe_code)` for one Windows FFI
  call; `MIError::Memory` variant for measurement failures
- **Autoregressive text generation example** (`generate.rs`) — greedy
  decoding (temperature 0) with full-sequence recompute at each step (no KV
  cache — all activations available for MI analysis); demonstrates
  `sample_token`, `GenerationResult`, `HookSpec`; CLI model selection or
  all-cached-models discovery; timing and estimated weight size reporting
- **Logit lens example** (`logit_lens.rs`) — captures `ResidPost` at every
  layer, projects to vocabulary via `project_to_vocab`, builds
  `LogitLensAnalysis` with per-layer top-k predictions; demonstrates
  `first_appearance()` for convergence tracking; Clap CLI with `--output`
  for structured JSON export; tested on Llama 3.2 1B ("Paris" at layer 11),
  Gemma 2 2B ("Paris" at layer 25, rank 8), and StarCoder2 3B (BPE subword
  "Par" dominates from layer 22); golden JSON results in
  `examples/results/logit_lens/`
- **Attention knockout example** (`attention_knockout.rs`) — knocks out a
  single attention edge (last → first token) across all heads at a middle
  layer; baseline vs ablated forward passes with `KnockoutSpec`,
  `create_knockout_mask`, and `Intervention::Knockout`; prints KL divergence,
  logit diff, and top-10 changed tokens; Clap CLI with `--output` for
  structured JSON export; tested on Llama 3.2 1B (Paris 39.3% → 26.0%,
  KL=0.056), Gemma 2 2B (Paris 3.9% → 6.7%, inverted), StarCoder2 3B
  (code model, "Par" dominates); golden JSON in
  `examples/results/attention_knockout/`
- **Cross-model result tables** in `examples/README.md` — documented logit
  lens convergence and attention knockout effects across 3 model families
- **Auto-config for unknown model families** — `from_hf_config_auto()`
  automatically infers `TransformerConfig` from any HuggingFace `config.json`,
  with a compatibility check that verifies weight tensor names match
  `GenericTransformer` expectations before loading; validated against all 7
  known model families (produces identical configs to manual parsers);
  `auto_config_dogfood` example demonstrates success and failure cases
- **Actionable auto-config error diagnostics** — when `check_auto_compatibility()`
  fails for non-standard models, error messages now show which tensors *were*
  found per category (embedding, norm, attention, MLP) and detect known naming
  conventions (GPT-2, Falcon, BLOOM, GPT-NeoX/Pythia) with architecture-specific
  guidance; unknown naming conventions show the first 5 tensor names as a
  diagnostic aid
- **Figure 13 planning-in-poems example** (`figure13_planning_poems`) —
  replicates Anthropic's Figure 13 (suppress + inject position sweep) with
  three presets: `llama3.2-1b-524k` (Llama 3.2 1B, P("that")=0.98),
  `gemma2-2b-426k` (Gemma 2 2B, P("around")=0.457), and `gemma2-2b-2.5m`
  (Gemma 2 2B 2.5M word-level CLT, P("can")=0.425); includes Mathematica
  plotting script and CLT landscape documentation
- **Download progress bars** — switched from tracing log lines to `indicatif`
  progress bars showing bytes, throughput, and ETA (via `hf-fetch-model` 0.7.1)
- **Steering dose-response example** (`steering_dose_response.rs`) —
  calibrates steering interventions and builds dose-response curves;
  demonstrates `SteeringCalibration`, `DoseResponseCurve`, `SteeringSpec`,
  `SteeringResult`, `apply_steering`, `measure_attention_to_targets`,
  `DOSE_LEVELS`, and `Intervention::Replace`; sweeps 6 dose levels with
  KL divergence and logit diff tracking; tested on Llama 3.2 1B, Gemma 2 2B,
  StarCoder2 3B
- **Attention patterns example** (`attention_patterns.rs`) — captures
  per-head attention patterns at every layer via `AttentionCache`; demonstrates
  `attention_from_position`, `attention_to_position`, and
  `top_attended_positions`; identifies the BOS sink pattern and peak
  last→first attention layer; tested on Llama 3.2 1B, Gemma 2 2B,
  StarCoder2 3B
- **Opt-in memory reporting** in all 7 high-impact examples — RAM + VRAM
  before/after model load via `MemorySnapshot` and `MemoryReport`, gated
  behind `#[cfg(feature = "memory")]`
- `extract_token_prob()` — extract a single token's probability from logits
  (softmax over last position)
- `HookSpec::extend()` — merge two hook specs (used to combine suppress +
  inject interventions)
- `MITokenizer::find_token_id()` — look up a token ID by word string
- `MITokenizer::decode_token()` — decode a single token ID back to string
- `MITokenizer::encode_with_offsets()` and `encode_raw_with_offsets()` — encode
  text with character offset mapping, returning `EncodingWithOffsets` for
  character-to-token position lookups; RWKV backend returns an error (offset
  mapping not supported)
- **Activation patching example** (`activation_patching.rs`) — causal tracing
  via position-specific activation patching (Meng et al., "Locating and Editing
  Factual Associations in GPT", NeurIPS 2022); clean vs. corrupted prompt
  ("France" → "Poland"/"Canada"), restore subject token residual at each layer,
  measure recovery; demonstrates `FullActivationCache`, `Intervention::Replace`,
  `Intervention::Add`, `HookPoint::Embed`; tested on Llama 3.2 1B, Gemma 2 2B,
  StarCoder2 3B
- **Token positions example** (`token_positions.rs`) — character-to-token
  mapping with `EncodingWithOffsets` and `convert_positions`; pure utility
  example (no GPU, no `transformer` feature); demonstrates `char_to_token`,
  `char_range_to_tokens`, `token_to_char_range`, `tokens_with_offsets`, and
  exact vs. fuzzy batch conversion; tested on Llama 3.2 1B, Gemma 2 2B,
  StarCoder2 3B
- **RWKV inference example** (`rwkv_inference.rs`) — RWKV linear RNN inference
  with RWKV-specific hook capture (`RwkvState`, `RwkvDecay`, `ResidPost`) and
  state knockout via `StateKnockoutSpec`; supports both RWKV-6 (Finch) and
  RWKV-7 (Goose); auto-discovers cached RWKV models; RWKV-6 requires
  `rwkv-tokenizer` feature for the RWKV World tokenizer fallback
- **Recurrent feedback example** (`recurrent_feedback.rs`) — anacrousis /
  recurrent passes for rhyme completion; loads `GenericTransformer` directly
  (not via `MIModel`) to access `forward_recurrent()` and `generate_recurrent()`;
  15 couplets with rhyme direction computed from averaged L2-normalised
  embedding vectors; Clap CLI with `--sustained`, `--strength`, `--loop-start`,
  `--loop-end`, `--max-couplets`, `--output` options; `--output` for structured
  JSON export with per-couplet results; opt-in memory reporting via
  `#[cfg(feature = "memory")]`; golden JSON results in
  `examples/results/recurrent_feedback/` (prefill L8–15 s=2.0: 11/15,
  sustained L14–15 s=1.0: 9/15); Mathematica plotting script in
  `examples/figure13/recurrent_feedback_plot.wl`; reference: Taufeeque et al.,
  arXiv:2407.15421, 2024
- Rust 2024 edition badge in `README.md`
- **`HOOKS.md`** — comprehensive hook point reference documenting all 14
  transformer and 7 RWKV hook points with tensor shapes, `TransformerLens`
  string equivalents, all 5 `Intervention` types (Replace, Add, Knockout,
  Scale, Zero), RWKV state interventions (`StateKnockoutSpec`,
  `StateSteeringSpec`), zero-overhead guarantee, and 5 worked examples
  (capture, logit lens, knockout, activation patching, RWKV state ablation)
- **`BACKENDS.md`** — step-by-step guide to adding new model architectures:
  three paths (auto-config for standard HF transformers, config parser for
  known families with quirks, custom `MIBackend` for non-transformer
  architectures); `TransformerConfig` axes reference, existing parser
  templates, hook integration checklist, weight naming conventions, and
  testing checklist
- **Crate-level documentation** (`src/lib.rs`) — expanded from minimal
  stub to full reference: feature flags table, quick start with real
  tokenization, activation capture, intervention (knockout), logit lens
  walkthrough, fast downloads (async + sync), and links to `HOOKS.md`,
  `BACKENDS.md`, and examples
- **`README.md` documentation table** — links to API docs, `HOOKS.md`,
  `BACKENDS.md`, examples, `CHANGELOG.md`, and `ROADMAP.md`
- **Cross-references** — `design/hook-system.md` and
  `design/intervention-api.md` now link to `HOOKS.md`; `examples/README.md`
  has a table of contents with clickable links and see-also references
- **`README.md` rewrite** — pedagogical structure: "What is this?" section
  explaining mechanistic interpretability, "Why Rust?" motivation (consumer GPU,
  memory/runtime bottlenecks, candle), MI techniques table with links to example
  output, quick start code block, auto-config screenshot, supported models table
  distinguishing model families from validated models, complete feature flags
  table, clickable table of contents, license links, development credits
- **Feature flag documentation** — added `rwkv-tokenizer` and `probing` to
  feature tables in both `README.md` and `src/lib.rs` crate-level docs

### Changed

- **Version bump to v0.1.0** — first minor release
- **Networked tests isolated** — `fast_download` integration tests marked
  `#[ignore]` to prevent transient HuggingFace Hub outages from blocking CI
  or publish workflows; run manually with `cargo test --test fast_download -- --ignored`
- **Rustdoc link fixes** — fixed 10 broken intra-doc links: feature-gated items
  (`clt::CltFeatureId`, `sae::SaeFeatureId`) replaced with plain text,
  cross-module references (`MIError::Model`, `MIError::Intervention`) given
  explicit `crate::` paths
- **CONVENTIONS.md intra-doc link safety** — new subsection under Doc-Comment
  Rules documenting two patterns: plain text for feature-gated items, explicit
  `crate::` paths for cross-module links

- **CONVENTIONS.md `// SAFETY:` policy** — updated from "not expected" to a
  feature-gated policy table; `mmap` and `memory` features each have
  documented accepted unsafe scopes; three requirements: dedicated module,
  `// SAFETY:` comments, `#[cfg(feature)]` gating
- **`lib.rs` unsafe code policy** — `cfg_attr` lines now cover both `mmap`
  and `memory` features: `forbid(unsafe_code)` by default, `deny(unsafe_code)`
  when either feature is enabled
- **Public API surface audit** — tightened visibility (`pub` → `pub(crate)`)
  across all modules; added missing `#[must_use]` annotations on all pure
  public functions and methods (two rounds: `70649e9`, `8595a61`, `2eedecf`)

### Fixed

- **`project_to_vocab` now applies final layer norm** — the logit lens
  projection was missing the final norm (`RmsNorm`/`LayerNorm`) before the
  unembedding matrix, producing near-random predictions from intermediate
  layers; both transformer and RWKV backends now apply the model's final norm
  before projection, matching the standard logit lens technique
  (nostalgebraist, 2020) and TransformerLens convention
- **Attention knockout NaN** — full-row knockout (`from_position`) caused NaN
  in softmax (all attention weights become -inf after causal mask); changed
  to single-edge knockout (`edge(last, 0)`) which preserves valid attention
  for other positions
- Adapted to `hf-fetch-model` 0.7.2 `DownloadOutcome` API — added
  `.into_inner()` calls across `clt/mod.rs` (4 sites), `sae/mod.rs`
  (3 sites), `download.rs` (1 site), and `auto_config_dogfood.rs` (1 site)
- `Display` formatting for error messages in `auto_config_dogfood` example
- **Logit lens probability formatting** — adaptive precision via
  `format_probability()`: ≥1% shows 1 decimal, ≥0.01% shows 3 decimals,
  <0.01% uses scientific notation; applied to both `print_summary` and
  `print_detailed` output
- **`--output` parent directories** — `logit_lens`, `attention_knockout`,
  `figure13_planning_poems`, and `recurrent_feedback` now auto-create parent
  directories via `create_dir_all` before writing JSON output
- **Sharded model error message** — `buffered_var_builder` now reports the
  number of shard files and shows both library (`features = ["mmap"]`) and
  example (`--features mmap`) remediation paths
- **`figure13_planning_poems` clippy fixes** — replaced `Vec` indexing in
  `parse_feature` with `split_once` (eliminates `indexing_slicing` errors);
  inlined format args; split 248-line `run()` into `select_preset`,
  `run_experiment`, `sweep_positions`, `print_sweep_summary`, and
  `write_sweep_output`
- **`attention_knockout` refactoring** — extracted `write_knockout_json` to
  bring `run_knockout` under clippy's 100-line threshold; removed file-level
  `allow(too_many_lines)`

## [0.0.5] - 2026-03-06

### Added

- **Sparse Autoencoder (SAE) support** — `SparseAutoencoder` struct with
  `SaeConfig`, `SaeFeatureId`, `SaeArchitecture`, `NormalizeActivations`, and
  `TopKStrategy` types; loading from SAELens-format safetensors + `cfg.json`
  or from Gemma Scope NPZ archives; three architecture variants: ReLU,
  JumpReLU (learned threshold per feature), and TopK (keep only k largest
  activations with auto-detected CPU/GPU dual-path)
- **NPZ/NPY parser** (`src/sae/npz.rs`) — from-scratch NumPy archive parser
  using the `zip` crate; supports NPY format v1/v2, float32/float64 dtypes
  (promoted to F32), C-order arrays; `load_npz()` returns a HashMap of candle
  Tensors; designed for future extraction to `hf-fetch-model` crate
- **SAE NPZ loading** — `from_npz()` and `from_pretrained_npz()` methods
  load SAE weights from Google Gemma Scope NPZ files
  (`google/gemma-scope-2b-pt-res`); config inferred from tensor shapes;
  architecture auto-detected (threshold present → JumpReLU, else ReLU);
  downloads via `hf-fetch-model`
- **SAE encoding and decoding** — `encode()` for batched dense encoding,
  `encode_sparse()` for single-position sparse features sorted by magnitude,
  `decode()` for reconstruction, `reconstruct()` and `reconstruction_error()`
  for round-trip analysis; `encode_with_strategy()` for explicit TopK
  strategy override
- **SAE feature injection** — `decoder_vector()` to extract individual
  feature steering directions, `prepare_hook_injection()` to build
  `HookSpec` entries for additive interventions at the SAE's hook point
- **Generic `SparseActivations<F: FeatureId>`** — refactored from CLT-only
  to a generic sparse representation shared between CLT and SAE; `FeatureId`
  marker trait implemented by both `CltFeatureId` and `SaeFeatureId`
- Python validation script (`scripts/sae_validation.py`) using direct NPZ
  loading (no SAELens dependency); integration tests (`tests/validate_sae.rs`)
  with 4 test cases: config detection, encode/decode/sparse, injection, and
  Python reference comparison; `quick_start_sae` example

### Fixed

- Mask cache now uses `DeviceLocation` as key instead of a collapsed device
  type ID, making it correct for multi-GPU / multi-Metal processes
- All 13 transformer hook points now support both capture and intervention
  (`ResidPre`, `AttnQ`, `AttnK`, `AttnV`, `AttnOut`, `ResidMid`, `MlpPre`,
  `MlpPost`, `MlpOut`, `FinalNorm` were previously capture-only)
- `sample_with_temperature()` now returns `MIError::Model("empty logits")`
  on empty input, matching `argmax()` behaviour (previously returned
  `u32::MAX` as an invalid token ID)
- `tests/fast_download.rs` now documents its non-hermetic, network-dependent
  nature so CI failures are easier to triage
- `ROADMAP.md` status line updated to v0.0.4 / Phase 3 complete; three
  implemented items (anacrousis, anacrousis validation, `scripts/README.md`)
  marked as done

## [0.0.4] - 2026-03-05

### Added

- **Recurrent feedback (anacrousis)** — `RecurrentPassSpec` and
  `RecurrentFeedbackEntry` types for re-running transformer commitment layers
  with directional feedback injection; `forward_recurrent()` for single-pass
  feedback, `generate_recurrent()` for autoregressive generation with per-step
  feedback; `embedding_vector()` on `MIBackend` for computing feedback
  directions from token embeddings; validated on Gemma 2 2B rhyme-completion
  task (baseline 9/15 → best 11/15 with unembed layers 8-15, scale 2.0)
- **Cross-Layer Transcoder (CLT) support** — `CrossLayerTranscoder`
  struct with `CltConfig`, `CltFeatureId`, and `SparseActivations` types;
  loading encoder/decoder weight pairs from HuggingFace repos (e.g.
  `mntss/clt-gemma-2-2b-426k`); `encode()` for full sparse activations,
  `top_k()` for the k strongest features at any layer
- **CLT feature injection** — `cache_steering_vectors_all_downstream()` to
  pre-compute per-layer decoder vectors, `prepare_hook_injection()` to build
  `HookSpec` entries for multi-layer causal interventions; reproduces
  Anthropic's cross-layer steering methodology
- **Melometis position-sweep validation tests** — correlational (encode at
  every token position, verify position-specificity) and causal (inject at
  every position, measure L2 logit distance) tests reproducing Anthropic's
  "Planning in Poems" Figure 13 result in Rust
- **Tragos position-sweep validation** (Llama 3.2 1B) — second independent
  replication on `mntss/clt-llama-3.2-1b-524k` (16 layers, 2048 d_model,
  32768 features/layer); config detection, encoding at 5 layers, injection
  (L2=77.9), correlational sweep (8/11 unique top-1, Jaccard=0.000), causal
  sweep (last position #1, concentration 24.85x); confirms the planning-site
  concentration phenomenon generalises across architectures
- **CLT attribution graph construction** — `AttributionEdge` and
  `AttributionGraph` types for circuit analysis; `score_features_by_decoder_projection()`
  scores all features by decoder-direction dot product or cosine similarity;
  batch variant `score_features_by_decoder_projection_batch()` loads each
  decoder file once for all directions; `extract_decoder_vectors()` for bulk
  decoder extraction (OOM-safe); `build_attribution_graph()` and
  `build_attribution_graph_batch()` convenience methods; graph pruning via
  `top_k()` and `threshold()` methods
- Python validation scripts (`scripts/clt_position_sweep_validation.py`,
  `scripts/clt_position_sweep_validation_llama.py`) and comparison documents
  (`scripts/clt_position_sweep_comparison.md`,
  `scripts/rwkv7_validation_comparison.md`) for cross-implementation
  reproducibility

### Fixed

- `SparseActivations` now derives `Debug` and `Clone` for consistency with
  other public types
- `Intervention::Add` now applies at `ResidPost` hook point with automatic
  dtype coercion (F32 steering vectors applied to BF16/F32 hidden states)

### Changed

- **Default GPU dtype changed from BF16 to F32** — research-grade precision
  matching Python/PyTorch exactly; RWKV-7 GPU logit error dropped from 0.027
  (0.36%) under BF16 to 0.000002 under F32; all validation tests updated
  accordingly; models up to ~7B fit in 16GB VRAM at F32
- Transformer attention mask dtype now derived from embedding weights instead
  of being hardcoded, ensuring consistency regardless of chosen precision
- CLT validation tests now document **16 GiB VRAM minimum** — F32 precision
  plus CUDA memory pool retention pushes peak usage near the limit when
  running the full Gemma 2 2B + Llama 3.2 1B suite sequentially

## [0.0.3] - 2026-03-01

### Added

- **RWKV-6 (Finch) backend** — `RwkvConfig` with V6/V7 version dispatch,
  `GenericRwkv` struct implementing `MIBackend`, WKV-5/6 recurrence kernel,
  `TimeMixV6`/`ChannelMixV6` blocks, `RwkvState` and `RwkvDecay` hook
  points for mechanistic interpretability of recurrent state dynamics
- **RWKV-7 (Goose) backend** — WKV-7 kernel with generalized delta rule
  (`S_t = diag(exp(w)) * S + b^T(a @ S) + k^T v`), `TimeMixV7`/`ChannelMixV7`
  blocks, `LoraBlock` with tanh/sigmoid/identity middle activations, value
  residual mixing across layers, gate output correction, L2-norm key
  normalization, and plain squared-ReLU FFN (no receptance gate)
- `hf-fetch-model` integration for parallel multi-connection model downloads,
  replacing `hf-hub` v0.4 as the sole download backend; `from_pretrained()`
  and `resolve_safetensors_paths()` now use `hf-fetch-model` directly
- `download_model()` (async) and `download_model_blocking()` convenience
  functions that populate the standard HF cache
- `SUPPORTED_MODEL_TYPES` const for runtime model-type discovery
- `quick_start_transformer` and `fast_download` examples
- Python validation scripts (`scripts/rwkv6_validation.py`,
  `scripts/rwkv7_validation.py`) for reproducible reference output generation
- **RWKV effective attention** — `RwkvEffectiveAttn` hook point for both
  V6 and V7, deriving attention-like matrices from the WKV recurrence:
  - V6: prefix-sum of log-decay for efficient cumulative decay products,
    then ReLU + L1 normalisation (`O(seq² × d × heads)`)
  - V7: backward propagation of a linear functional through diag+rank-1
    state transitions (`l = l ⊙ exp(w) + (l · b) * act_a`), same asymptotic cost
- **RWKV state knockout + steering** — `HookSpec::set_state_knockout()` and
  `set_state_steering()` wiring the existing `StateKnockoutSpec`/`StateSteeringSpec`
  types into the WKV loops; knockout skips kv write (`state = decay * state`),
  steering scales it (`state = scale * kv + decay * state`); layer-targeted
  via `LayerSpec`, O(1) position lookup via `HashSet`
- `MIModel::from_pretrained("RWKV/RWKV7-Goose-World3-1.5B-HF")` integration
  test validating the full one-line loading path for RWKV-7 models
- Integration tests for RWKV-6 (against plip-rs reference) and RWKV-7
  (against fla/flash-linear-attention reference), CPU F32 + GPU F32
  (BF16 variant retained as regression test)
- RWKV clippy and test steps in CI publish workflow
- VRAM budget table and `config.json` field reference in rustdoc
- `MIError::Download` variant for download failures

### Fixed

- RWKV-7 `g_lora` sigmoid placement: sigmoid is the **middle** activation
  (between down and up projections), not applied after the full LoRA output;
  `down(x) -> sigmoid -> up` vs the incorrect `down(x) -> up -> sigmoid`
- Serialized GPU integration tests with `serial_test` to prevent CUDA OOM
  when running multiple model tests concurrently
- Pre-existing `cargo doc` link warnings resolved
- CI `no-default-features` build: gated `apply_intervention` with `#[cfg]`
  to eliminate dead-code error when no backend feature is enabled
- CI workflow: added RWKV build/clippy/test steps (matching publish.yml);
  integration tests gated by `required-features` in `Cargo.toml`
- `hf-fetch-model` dependency changed from local path to crates.io v0.5
- `HookSpec::is_empty()` now accounts for `state_knockout` and
  `state_steering` specs (previously only checked captures/interventions)
- Stale documentation updated: RWKV-7 status changed from "planned" to
  implemented, `MIModel` doc corrected re: `from_pretrained` availability
- Removed dead `layer_idx` field from `TimeMixV7` and simplified
  `v_for_first` return path (no behavioural change)

### Changed

- Dropped `hf-hub` v0.4 dependency; all HuggingFace file resolution now
  goes through `hf-fetch-model` (parallel chunked downloads by default)
- `#[must_use]` policy applied across public API (Rule 17)
- Phase 1 audit remediation (code quality, documentation, consistency)

## [0.0.2] - 2026-02-25

### Added

- **Generic Transformer backend** — one config-driven forward pass covering
  7 model families: LLaMA, Qwen2, Gemma, Gemma 2, Phi-3, StarCoder2, Mistral
- `TransformerConfig` with ~12 configuration axes parsed from HuggingFace
  `config.json` (norm type, activation, QKV layout, MLP layout, bias
  granularity, embedding scale, soft-capping, sliding window, etc.)
- Config parsers for `llama`, `qwen2`, `gemma`, `gemma2`, `phi3`,
  `starcoder2`, `mistral` — adding a new model family requires only a
  ~30-line parser function
- `GenericTransformer` struct implementing `MIBackend` with hook points
  at all 14 TransformerLens-equivalent locations (Embed, ResidPre, AttnQ/K/V,
  AttnScores, AttnPattern, AttnOut, ResidMid, MlpPre/Post, MlpOut,
  ResidPost, FinalNorm)
- Multi-head attention supporting GQA/MHA/MQA, separate and fused QKV
  projections, optional soft-capping, and sliding window (global,
  per-layer, or alternating)
- MLP variants: gated separate (LLaMA/Qwen/Gemma), gated fused (Phi-3),
  and plain (StarCoder2)
- Normalization: RmsNorm, LayerNorm, GemmaRmsNorm (weight + 1)
- RoPE via `candle_nn::rotary_emb::rope()` with pre-computed cos/sin cache
- `MIModel::from_pretrained(model_id)` for HuggingFace model loading
  with automatic config detection and sharded safetensors support
- `mmap` feature gate: `#![forbid(unsafe_code)]` by default, opt-in
  memory-mapped weight loading for 7B+ models (`features = ["mmap"]`)
- `Activation::GeluApprox` for PyTorch tanh-approximated GELU
  (`gelu_pytorch_tanh`)
- `AttentionCache` for per-layer attention pattern storage
- Integration tests validating all 7 model families on CPU (F32) and
  GPU (BF16) against Python HuggingFace reference outputs
- Hook overhead benchmark: +11.5% on GPU with full capture (194 hook
  points on LLaMA 3.2 1B), within noise on CPU

### Fixed

- Tokenizer `encode()` now adds special tokens (BOS) by default,
  matching HuggingFace convention; added `encode_raw()` for MI analyses
  needing raw tokenization
- StarCoder2 config now reads `norm_type` from `config.json` (LayerNorm,
  not RmsNorm) and uses `GeluApprox` activation

### Changed

- Clarified that plip-rs is a frozen predecessor project (v1.4.0) in
  `MIBackend` trait documentation

## [0.0.1] - 2026-02-23

### Added

- `MIError` typed error hierarchy with `thiserror` (`#[non_exhaustive]`)
- `MIBackend` trait and `MIModel` wrapper for dynamic dispatch over model backends
- `HookSpec`, `HookCache`, and `HookPoint` for activation capture and intervention
- `KVCache` and `ActivationCache` for inference state management
- `KnockoutSpec`, `SteeringSpec`, `StateKnockoutSpec`, `StateSteeringSpec` for interpretability interventions
- `CltInjectionSpec` for CLT feature injection (behind `clt` feature flag)
- `LogitLensAnalysis` and `SteeringCalibration` with dose-response curves
- `MITokenizer` enum supporting `HuggingFace` and RWKV World tokenizers
- Causal mask and generation mask utilities
- Token-to-character position mapping
- CI workflow (fmt, clippy pedantic, tests, feature-flag hygiene)
- Tag-triggered publish workflow with `workflow_dispatch` fallback

[Unreleased]: https://github.com/PCfVW/candle-mi/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/PCfVW/candle-mi/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/PCfVW/candle-mi/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/PCfVW/candle-mi/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/PCfVW/candle-mi/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/PCfVW/candle-mi/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/PCfVW/candle-mi/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/PCfVW/candle-mi/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/PCfVW/candle-mi/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PCfVW/candle-mi/compare/v0.0.5-phase4...v0.1.0
[0.0.5-phase4]: https://github.com/PCfVW/candle-mi/compare/v0.0.4-phase3...v0.0.5-phase4
[0.0.4]: https://github.com/PCfVW/candle-mi/compare/v0.0.3...v0.0.4-phase3
[0.0.3]: https://github.com/PCfVW/candle-mi/releases/tag/v0.0.3
[0.0.2-phase1]: https://github.com/PCfVW/candle-mi/releases/tag/v0.0.2-phase1
[0.0.1]: https://github.com/PCfVW/candle-mi/releases/tag/v0.0.1
