# TransluceAI Assessment for candle-mi

**Date:** March 25, 2026 | **Context:** Evaluating cooperation potential between Transluce's open-source MI research and candle-mi

---

## Organization Overview

[Transluce](https://github.com/TransluceAI) is an independent research lab building open, scalable technology for understanding AI systems and steering them in the public interest. Led by Jacob Steinhardt (UC Berkeley) and Sarah Schwettmann.

10 public repositories. 3 are relevant to candle-mi.

---

## Highly Relevant Repositories

### 1. circuits (20 stars)

**Paper:** "Language Model Circuits are Sparse in the Neuron Basis"

**Repository:** https://github.com/TransluceAI/circuits

**What it provides:**

| Component | File | What it does |
|---|---|---|
| Gradient-based node attribution | `circuits/core/attribution.py` | Identifies which neurons matter for a given output |
| Gradient utilities | `circuits/core/grad.py` | Core gradient computation for attribution |
| JVP edge tracing | `circuits/core/jvp.py` | Traces information flow between layers via Jacobian-vector products |
| Circuit data structure | `circuits/core/circuit.py` | Circuit representation and operations |
| Feature scoring | `circuits/analysis/score_features.py` | Scores features against hypotheses |
| Feature clustering | `circuits/analysis/cluster.py` | Clusters neurons/features in discovered circuits |
| Circuit-informed steering | `circuits/analysis/steer.py` | Steers using discovered circuits, not raw directions |
| NAP evaluation | `circuits/evals/nap.py` | Node Attribution Patching scoring |
| ENAP evaluation | `circuits/evals/enap.py` | Edge Node Attribution Patching scoring |
| SAE dictionary loading | `circuits/utils/dictionary_loading_utils.py` | Loads SAE dictionaries for feature-level circuit analysis |
| SVA benchmark data | `data/feature_circuits/` | Subject-verb agreement test cases |

**Relevance to candle-mi:** candle-mi captures activations via hooks but has no mechanism to *rank* which activations matter for a prediction. Gradient attribution fills this gap. JVP edge tracing is more principled than activation patching for circuit discovery. NAP provides a standardized benchmark metric for candle-mi's existing patching pipeline.

### 2. observatory (232 stars)

**Flagship toolkit.**

**Repository:** https://github.com/TransluceAI/observatory

**What it provides:**

| Sub-project | Path | What it does |
|---|---|---|
| Activation exemplars | `lib/activations/` | Computes per-neuron activations and finds max-activation exemplars at scale |
| Neuron descriptions | `lib/explanations/` | LLM-based auto-labeling of features |
| LatentQA steering | (top-level) | Trains decoders to read/steer latent representations |
| Monitor UI | (top-level) | Web observability interface for internal computations |
| Neuron database | `lib/neurondb/` | Postgres-backed neuron database with schemas and filters |

**Relevance to candle-mi:** The activation exemplar pipeline (`lib/activations/`) is a higher-level analysis layer that sits on top of activation capture -- exactly what `HookCache` produces. Neuron description generation is a downstream consumer of this data. LatentQA is a more sophisticated steering approach than direct activation addition.

### 3. introspective-interp (21 stars)

**Paper:** "Training Language Models To Explain Their Own Computations" ([arXiv:2511.08579](https://arxiv.org/abs/2511.08579))

**Repository:** https://github.com/TransluceAI/introspective-interp

**What it provides:**

| Component | HuggingFace repo | What it actually is |
|---|---|---|
| Activation patching explainer (Llama) | `Transluce/act_patch_llama3.1_8b_llama3.1_8b` | **LoRA adapter** (PEFT, rank 128, 3.21 GB) fine-tuned on Llama-3.1-8B to *predict* patching effects |
| Activation patching explainer (Qwen) | `Transluce/act_patch_qwen3_8b_qwen3_8b` | **LoRA adapter** (PEFT, rank 128, 3.62 GB) fine-tuned on Qwen3-8B to *predict* patching effects |
| Feature description explainer | `Transluce/features_explain_llama3.1_8b_llama3.1_8b_instruct` | LoRA adapter for SAE feature description |
| Feature description simulator | `Transluce/features_explain_llama3.1_8b_simulator` | Full fine-tuned Llama-3.1-8B-Instruct (14.96 GB, 4 shards) that scores feature explanations |
| SAE feature descriptions | `Transluce/features_explain_llama3.1_8b_llama3_8b` | LoRA adapter variant |

**CORRECTION (March 25, 2026):** The HuggingFace *model* repos named `act_patch_*` are **LoRA adapters**, not datasets. However, Transluce **also publishes actual activation patching datasets** on HuggingFace's *datasets* namespace (which `hf-fm search` cannot query -- see [hf-fm dogfooding report](hf-fm-dogfooding-transluce-session.md)):

- **`Transluce/act_patch_llama_3.1_8b_counterfact`** -- 126k rows (Parquet, MIT license)
- **`Transluce/act_patch_qwen3_8b_counterfact`** -- 135k rows (Parquet, MIT license)

**Dataset schema per row:**

| Field | Type | Description |
|---|---|---|
| `layer` | Int32 array (8-9 elements) | Layer indices where patching was applied |
| `input_tokens` | Text array | Tokenized input sequence |
| `original_continuation` | Text array | Model output before patching |
| `ablated_continuation` | Text array | Model output after patching |
| `is_different` | Boolean | Whether the intervention changed the output |
| `patch_position.orig_pos` | Int64 | Token position in the original prompt |
| `patch_position.counterfact_pos` | Int64 | Token position in the counterfactual prompt |
| `patch_position.intervention_vector` | Float32 array | The actual activation difference vector |
| `counterfactual_text` | Text | The counterfactual prompt |
| `gt_original_target` | Text | Ground truth original answer |
| `gt_counterfactual_target` | Text | Ground truth counterfactual answer |

These are genuine (layer, position, effect) tuples -- exactly the ground truth needed for validation.

**Additional Transluce datasets on HuggingFace (10 total):**
- `input_ablation_llama_3.1_8b_instruct_mmlu_hint` (14k rows)
- `input_ablation_qwen3_8b_mmlu_hint` (14k rows)
- `PRISM-gender-Llama-3.1-8B-Instruct` (5k rows)
- `SelfDescribe-Llama-3.1-{8B,70B}-Instruct` (2.6k rows each)
- `SynthSys-Llama-3.1-{8B,70B}-Instruct` (131k-158k rows)

**Implications for candle-mi:**
- The patching datasets are directly usable for validation -- real ground-truth data, not model predictions.
- **Blocker:** `hf-fm` v0.9.0 cannot search or download HuggingFace datasets. Workaround: manual download from the HuggingFace web UI or direct HTTP fetch of the Parquet files.
- The LoRA adapter models require LoRA merging -- candle-mi does not have LoRA support.
- The simulator model (`features_explain_llama3.1_8b_simulator`) is a full 15 GB fine-tuned Llama-3.1-8B-Instruct (not a LoRA) -- candle-mi could load it directly once Llama-3.1-8B-Instruct auto-config is added.

---

## Not Relevant

| Repository | Stars | Why not relevant |
|---|---|---|
| docent | 87 | Agent analytics platform, not model internals |
| jailbreaking-frontier-models | 25 | Red-teaming, not MI |
| inspect_ai | 8 | Fork of UK Gov evaluation framework |
| inspect_evals | 1 | Fork of eval collection |
| tau2-bench | 0 | Conversational agent benchmarks |
| claude-code-plugins | 2 | Infrastructure |
| .github | 18 | Organization profile |

---

## Cooperation Opportunities

### What we can actually do today

Both `circuits` (pure algorithms) and the patching datasets (Parquet on HuggingFace) are actionable. The datasets require a manual download until `hf-fm` v0.10.0 adds dataset support.

### 1. Gradient-Based Attribution (from `circuits`) -- RECOMMENDED FIRST

**What:** Port the gradient attribution algorithm from `circuits/core/attribution.py` to Rust/candle.

**Why this is the strongest capability upgrade:**
- candle-mi captures activations but doesn't rank them. Attribution answers "which of these 194 hook points actually drove this prediction?"
- Fills the gap between "capture everything" (current) and "know what matters" (what researchers need).
- Enables circuit discovery -- the core use case of `circuits`.
- The `circuits` repo is pure algorithms (Python + PyTorch). No LoRA, no adapters, no external datasets required. Everything needed is in the code.

**Technical considerations:**
- Requires backward pass / gradient computation. candle supports `backward()` on tensors, but candle-mi's forward pass is not currently set up for gradient tracking.
- The `circuits` implementation uses PyTorch autograd. Porting to candle means working with `candle_core::Var` and `.backward()`.
- JVP (Jacobian-vector product) is more advanced -- candle may not have native JVP support, requiring manual implementation.
- Design decision: should attribution be a new module (`src/interp/attribution.rs`) or an extension of the hook system?

**Steps:**
1. Read and understand `circuits/core/attribution.py` and `grad.py` in detail.
2. Audit candle's gradient/backward support for the operations needed (attention scores, MLP activations, residual stream).
3. Design the Rust API -- likely `AttributionResult` struct with per-hook-point importance scores.
4. Implement on a single model (Llama 3.2 1B) first, validate against `circuits` Python output on the same prompt.
5. Add NAP evaluation metric from `circuits/evals/nap.py` as a benchmark.
6. Use SVA benchmark data from `circuits/data/feature_circuits/` for validation.

### 2. Activation Patching Validation against Transluce Datasets

**What:** Validate candle-mi's activation patching pipeline against Transluce's 126k-row CounterFact ground-truth dataset.

**Why this is now viable:**
- The datasets exist as Parquet files on HuggingFace (`Transluce/act_patch_llama_3.1_8b_counterfact`). Each row contains layer indices, token positions, intervention vectors, and before/after continuations.
- candle-mi already has `Intervention::Replace` and the `activation_patching` example.
- Publishable result: "candle-mi reproduces Transluce's activation patching ground truth on consumer hardware."

**Blockers (all have workarounds):**
- `hf-fm` cannot download HuggingFace datasets yet. **Workaround:** download Parquet files manually from the HuggingFace web UI or via direct HTTP.
- Datasets target 8B models (Llama-3.1-8B, Qwen3-8B). At F32, 8B = ~32 GB -- exceeds 16 GB VRAM. **Workaround:** run at BF16, use anamnesis FP8 path, or validate on a subset using CPU.
- Alternative: reproduce the patching protocol on Llama 3.2 1B (already validated) using CounterFact prompts, producing independent results comparable to Transluce's methodology.

### 3. Activation Exemplar Computation (from `observatory`)

**What:** Port `observatory/lib/activations/` exemplar computation to Rust.

**Why:** Enables at-scale neuron/feature profiling -- "for each SAE feature, which input tokens maximally activate it?" This is the data layer that feeds neuron description, feature visualization, and interpretability dashboards.

**Depends on:** SAE pipeline maturity in candle-mi (already functional for Gemma Scope).

### 4. What we CANNOT do yet

| Opportunity | Blocker |
|---|---|
| Validate against Transluce's patching ground truth | Dataset exists (126k rows Parquet) but `hf-fm` can't download HF datasets; manual download works |
| Use their LoRA explainer models | candle-mi has no LoRA support |
| Use the simulator model directly | Requires Llama-3.1-8B-Instruct support (not yet in candle-mi auto-config) |
| Download CounterFact dataset via `hf-fm` | `hf-fm` v0.9.0 cannot search/download HuggingFace datasets |

---

## Strategic Value

The technical relationship created by porting algorithms from `circuits` and producing independent results on shared benchmarks has value beyond improving candle-mi:

1. **Reproducibility on consumer hardware** is a differentiator. Most MI research assumes cloud GPUs. Demonstrating the same results on an RTX 5060 Ti makes the work accessible and notable.
2. **Code visibility.** A PR against `circuits` or a blog post showing candle-mi reproducing their results creates direct visibility with Steinhardt and Schwettmann.
3. **Rust MI ecosystem.** candle-mi is (to our knowledge) the only Rust MI toolkit with hook points, activation patching, and SAE support. Adding gradient attribution would make it the first Rust crate with circuit discovery capabilities.

---

## Recommended Execution Order

1. **Gradient-based attribution** from `circuits` (actionable now, high value, self-contained)
2. **NAP evaluation metric** (follows naturally from attribution, uses SVA data bundled in `circuits`)
3. **CounterFact patching on Llama 3.2 1B** (independent validation, manual dataset download until hf-fm v0.10.0)
4. **Activation exemplar computation** from `observatory` (longer-term, depends on SAE maturity)
