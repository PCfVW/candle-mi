# Implementation Plan: Stoicheia MI Tooling — candle-mi Phase B

> **ekthesis** (ἔκθεσις) — "setting out, exposition." In Euclid's *Elements*,
> the ekthesis names the parts of a figure and restates the proposition in
> terms of those parts. Phase B names the parts of an AlgZoo model — which
> neurons fire, what regions they carve, how many bits of surprise remain —
> and restates the model's behaviour in terms of those parts.

- [Jacob Hilton](https://www.alignment.org/author/jacob/), [AlgZoo](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/): uninterpreted models with fewer than 1,500 parameters, January 26th, 2026.

- Alignment Research Center, [AlgZoo Github page](https://github.com/alignment-research-center/alg-zoo).

---

## Goal

Add six MI analysis modules to the stoicheia subsystem, enabling exhaustive
mechanistic understanding of AlgZoo's tiny ReLU RNNs. These tools implement
the methodology from the AlgZoo blog post: weight standardization, piecewise-
linear region enumeration, neuron ablation, functional probing, and surprise
accounting — all accelerated by a fast-path RNN kernel that bypasses candle's
per-tensor overhead.

**Scope**: MI analysis of ReLU RNNs (2nd argmax, argmedian, median tasks).
No transformer MI tools (Phase 7/8 of the main roadmap), no training loop,
no gradient computation.

**Deliverable**: A researcher loads an AlgZoo RNN, standardizes its weights,
enumerates its linear regions, ablates and probes each neuron, and runs
surprise accounting — all in under a second on CPU for M₁₆,₁₀.

---

## Motivation: The M₁₆,₁₀ Challenge

The AlgZoo blog poses a concrete research challenge:

> Design a method for mechanistically estimating the accuracy of M₁₆,₁₀
> (432 parameters, 95.3% accuracy on 2nd argmax) that matches the
> performance of random sampling in terms of mean squared error versus
> compute.

The blog's partial analysis of M₁₆,₁₀ identifies 5 of 16 neurons as
isolated subcircuits (running-max and leave-one-out-max features), but the
remaining 11 neurons are densely connected and unexplained. Phase B's tools
are designed to systematically attack this challenge:

| Phase B tool | Role in the M₁₆,₁₀ challenge |
|---|---|
| Fast-path kernel | Enables millions of forward passes for perturbation and sweep analyses |
| Weight standardization | Normalizes W^ih to {±1}, revealing the sign structure of each neuron |
| Piecewise-linear enumeration | Maps the decision boundary; counts and classifies active regions |
| Neuron ablation | Identifies which of the 11 unexplained neurons are critical |
| Neuron probing | Classifies each neuron's function (running max, comparator, etc.) |
| Surprise accounting | Scores any mechanistic estimate against the MSE-vs-compute metric |

---

## Architecture: Why Raw f32, Not Tensor

Phase A benchmarks (10K samples, release mode, CPU) revealed that candle's
per-tensor-operation overhead dominates on tiny models:

| Model | Params | Rust (candle) | Python (PyTorch) | Ratio |
|-------|--------|---------------|------------------|-------|
| M₂,₂ | 10 | 86ms | 3.4ms | 25× |
| M₁₆,₁₀ | 432 | 111ms | 6.1ms | 18× |
| Transformer h4n4 | 176 | 110ms | 13ms | 8.5× |

The bottleneck is **fixed per-operation overhead** (shape validation, device
dispatch, storage allocation), not compute. PyTorch's `nn.RNN` fuses the
entire timestep loop into a single C++ kernel; candle executes 4 separate
tensor ops per timestep.

Phase B's solution: a **dual-path architecture**. The candle-based `forward()`
(Phase A) stays for MI analysis with hooks. A new raw-f32 fast path operates
on `&[f32]` slices — no `Tensor` objects, no heap allocation per timestep.
All six MI modules use the fast path internally for bulk operations.

---

## The Six Modules

### Module 1: `fast.rs` — Fast-Path RNN Kernel

The foundation. All other modules depend on it for bulk forward passes.

#### `RnnWeights` — Shared Weight Container

```rust
/// Raw f32 weight storage for an AlgZoo RNN.
///
/// Extracted from `Tensor` once at load time, then used by the fast-path
/// kernel and all analysis modules without further candle overhead.
pub struct RnnWeights {
    /// Input-to-hidden weights, row-major `[H, 1]`.
    pub weight_ih: Vec<f32>,
    /// Hidden-to-hidden weights, row-major `[H, H]`.
    pub weight_hh: Vec<f32>,
    /// Output projection weights, row-major `[output_size, H]`.
    pub weight_oh: Vec<f32>,
    /// Hidden size.
    pub hidden_size: usize,
    /// Output size.
    pub output_size: usize,
}

impl RnnWeights {
    /// Extract raw f32 weights from a loaded `StoicheiaRnn`.
    ///
    /// Copies each weight tensor to a contiguous `Vec<f32>` on CPU.
    /// This is a one-time cost; all subsequent fast-path operations
    /// use these slices directly.
    ///
    /// # Errors
    ///
    /// Returns `MIError::Model` if tensor extraction fails.
    pub fn from_model(model: &StoicheiaRnn) -> Result<Self>;

    /// Create from explicit weight vectors (for standardized or
    /// handcrafted models).
    #[must_use]
    pub fn new(
        weight_ih: Vec<f32>,
        weight_hh: Vec<f32>,
        weight_oh: Vec<f32>,
        hidden_size: usize,
        output_size: usize,
    ) -> Self;
}
```

#### Forward Pass Functions

```rust
/// Run a batch of RNN forward passes on raw f32 slices.
///
/// No `Tensor` objects, no heap allocation per timestep. The inner
/// loop is structured for auto-vectorization: contiguous slices,
/// no branches in the hot path, hoisted invariants.
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
/// - `outputs`: row-major `[n_inputs, output_size]`
pub fn forward_fast(
    weights: &RnnWeights,
    inputs: &[f32],
    outputs: &mut [f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
);

/// Run forward passes with neuron ablation.
///
/// Same as `forward_fast`, but neurons marked `true` in `ablated`
/// have their hidden state forced to zero after each ReLU step.
///
/// # Shapes
/// - `ablated`: `[hidden_size]`, `true` = zero this neuron
pub fn forward_fast_ablated(
    weights: &RnnWeights,
    inputs: &[f32],
    outputs: &mut [f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
    ablated: &[bool],
);

/// Run a single forward pass, returning the full activation trace.
///
/// Records pre-activation values at every (timestep, neuron) pair.
/// Used by piecewise-linear analysis to determine which neurons fire.
///
/// # Shapes
/// - `input`: `[seq_len]` (single input)
/// - `pre_activations`: row-major `[seq_len, hidden_size]` (output)
/// - `output`: `[output_size]` (output)
pub fn forward_fast_traced(
    weights: &RnnWeights,
    input: &[f32],
    pre_activations: &mut [f32],
    output: &mut [f32],
    config: &StoicheiaConfig,
);

/// Compute model accuracy on a batch of inputs.
///
/// Runs `forward_fast`, takes argmax of each output row, compares
/// with targets.
///
/// # Returns
///
/// Fraction of correct predictions (0.0 to 1.0).
pub fn accuracy(
    weights: &RnnWeights,
    inputs: &[f32],
    targets: &[u32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> f32;
```

#### Inner Loop Design

```
For each input i in 0..n_inputs:
    hidden = [0.0; H]                            // stack-allocated
    for t in 0..seq_len:
        x_t = inputs[i * seq_len + t]
        for j in 0..H:
            acc = x_t * W_ih[j]
            for k in 0..H:
                acc += W_hh[j * H + k] * hidden[k]
            pre_act[j] = acc
        for j in 0..H:
            hidden[j] = pre_act[j].max(0.0)       // ReLU
    for o in 0..output_size:
        acc = 0.0
        for j in 0..H:
            acc += W_oh[o * H + j] * hidden[j]
        outputs[i * output_size + o] = acc
```

**SIMD auto-vectorization strategy** (following anamnesis patterns):
- Hidden state: contiguous `[f32; 32]` buffer on the stack (covers all
  AlgZoo hidden sizes: 2, 4, 6, 8, 10, 12, 16, 20, 24, 32)
- H ≤ 4: entire hidden state fits in one SSE register; matmul fully unrolls
- H ≤ 8: one AVX2 register; compiler vectorizes multiply-adds
- H = 16: two AVX2 registers; inner loop processes 8 elements per iteration
- H = 32: four AVX2 registers
- No explicit `unsafe` / intrinsics: `#![forbid(unsafe_code)]` maintained
- Verify with `cargo-show-asm` that the inner loop vectorizes

**Expected performance**: near PyTorch speed or faster. PyTorch's `nn.RNN`
still routes through Python→C++ dispatch; a raw Rust loop with auto-
vectorized SIMD has zero dispatch overhead. The anamnesis dequantization
kernels achieve 2.7–54× over PyTorch CPU on similar "tight loop over f32
slices" workloads.

---

### Module 2: `standardize.rs` — Weight Standardization

From the AlgZoo blog: for M₂,₂, after rescaling each neuron so that
`|W_ih[j]| = 1`, the sign pattern and coefficient magnitudes directly
reveal the algorithm. The standardization is an exact equivalence
transformation with scale factor `s_j = |W_ih[j]|` per neuron:

- `W_ih[j]` → `W_ih[j] / s_j` = ±1 (input sensitivity sign)
- `W_hh[j,k]` → `W_hh[j,k] * s_k / s_j` (rescale both source and target)
- `W_oh[o,j]` → `W_oh[o,j] * s_j` (compensate output for scaled hidden state)

The product of scaling factors cancels exactly: `h_j` is divided by `s_j`
inside the RNN, then multiplied by `s_j` on the way to the output.

```rust
/// Result of standardizing an RNN's weights.
///
/// The standardized model is input-output equivalent to the original:
/// `forward(standardized) == forward(original)` for all inputs.
pub struct StandardizedRnn {
    /// Per-neuron scale factors applied (`scales[j] = |W_ih_orig[j]|`).
    /// `W_ih[j] = W_ih_orig[j] / scales[j]` → ±1,
    /// `W_hh[j,k] = W_hh_orig[j,k] * scales[k] / scales[j]`,
    /// `W_oh[o,j] = W_oh_orig[o,j] * scales[j]`.
    pub scales: Vec<f32>,
    /// Standardized input-to-hidden weights, row-major `[H, 1]`.
    /// Each entry is ±1 (to floating-point precision).
    pub weight_ih: Vec<f32>,
    /// Standardized hidden-to-hidden weights, row-major `[H, H]`.
    pub weight_hh: Vec<f32>,
    /// Standardized output weights, row-major `[output_size, H]`.
    pub weight_oh: Vec<f32>,
    /// Hidden size.
    pub hidden_size: usize,
    /// Output size.
    pub output_size: usize,
}

/// Standardize an RNN so that each neuron's input weight satisfies
/// `|W_ih[j]| = 1`.
///
/// This preserves the input-output map exactly: the product of
/// scaling factors cancels across each neuron. After standardization,
/// the sign of `W_ih[j]` directly indicates whether neuron `j` is
/// excited (+1) or inhibited (−1) by its input.
///
/// # Errors
///
/// Returns `MIError::Model` if weight extraction fails.
/// Returns `MIError::Config` if any `W_ih[j]` is exactly zero
/// (degenerate neuron — no input sensitivity).
pub fn standardize_rnn(model: &StoicheiaRnn) -> Result<StandardizedRnn>;

/// Maximum deviation of standardized `W_ih` from {+1, −1}.
///
/// A value near 0.0 means the model was already near-standardized.
/// A value of 0.0 exactly means perfect standardization (always the
/// case unless limited by f32 precision).
#[must_use]
pub fn standardization_quality(std_rnn: &StandardizedRnn) -> f32;
```

**Sign table for M₂,₂** (from the blog):

After standardization, the weight matrices have a specific sign structure:

| Matrix | Entry | Sign |
|--------|-------|------|
| W^ih | `[0]` | +1 |
| W^ih | `[1]` | −1 |
| W^hh | `[0,0]` | − (magnitude < 1) |
| W^hh | `[0,1]` | + (magnitude ≥ 1) |
| W^hh | `[1,0]` | + (magnitude ≥ 1) |
| W^hh | `[1,1]` | − (magnitude < 1) |
| W^oh | `[0,0]` | + |
| W^oh | `[0,1]` | − |
| W^oh | `[1,0]` | − |
| W^oh | `[1,1]` | + |

This sign pattern directly implies the model's decision boundary geometry.
Phase B's standardization module exposes this structure programmatically.

**`StandardizedRnn` → `RnnWeights` conversion**: `StandardizedRnn` can be
converted to `RnnWeights` for use with the fast-path kernel, enabling
analysis on the standardized model directly.

---

### Module 3: `piecewise.rs` — Piecewise-Linear Region Enumeration

A single-layer ReLU RNN with H neurons over T timesteps has at most
2^(H·T) linear regions in input space. Each region is defined by an
**activation pattern**: which neurons fire (pre-activation ≥ 0) at
which timesteps. Within each region, the RNN reduces to a single
affine map `output = A · input + b`.

**Feasibility by model size**:

| Model | H | T | H·T | Max regions | Exhaustive? |
|-------|---|---|-----|-------------|-------------|
| M₂,₂ | 2 | 2 | 4 | 16 | Yes |
| M₄,₃ | 4 | 3 | 12 | 4,096 | Yes |
| M₈,₅ | 8 | 5 | 40 | ~10¹² | No (sample) |
| M₁₆,₁₀ | 16 | 10 | 160 | ~10⁴⁸ | No (sample) |

#### `ActivationPattern` — Compact Bit Vector

```rust
/// The activation pattern of an RNN over a sequence.
///
/// For each timestep `t` and neuron `j`, records whether
/// `pre_activation[t][j] >= 0` (active) or `< 0` (inactive).
/// Bit `t * H + j` in the internal representation corresponds
/// to neuron `j` at timestep `t`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActivationPattern {
    /// Compact bit storage. Uses `[u64; 5]` for up to 320 bits,
    /// covering all AlgZoo RNN configurations (max H=32, T=10 = 320).
    bits: [u64; 5],
    /// Number of neurons (H).
    hidden_size: usize,
    /// Number of timesteps (T).
    seq_len: usize,
}

impl ActivationPattern {
    /// Create from a pre-activation trace.
    ///
    /// # Shapes
    /// - `pre_activations`: row-major `[seq_len, hidden_size]`
    pub fn from_pre_activations(
        pre_acts: &[f32],
        hidden_size: usize,
        seq_len: usize,
    ) -> Self;

    /// Whether neuron `j` is active (pre-act ≥ 0) at timestep `t`.
    #[must_use]
    pub const fn is_active(&self, t: usize, j: usize) -> bool;

    /// Number of active neurons across all timesteps.
    #[must_use]
    pub fn count_active(&self) -> u32;

    /// Total neuron-timestep slots (H · T).
    #[must_use]
    pub const fn total_slots(&self) -> usize;

    /// Per-timestep active neuron count.
    ///
    /// Returns a vector of length `seq_len`, where entry `t` is the
    /// number of active neurons at timestep `t`.
    #[must_use]
    pub fn active_per_timestep(&self) -> Vec<u32>;
}
```

#### Region Classification and Affine Maps

```rust
/// Information about a single linear region.
pub struct RegionInfo {
    /// Number of inputs that fell into this region.
    pub count: usize,
    /// A representative input from this region.
    pub representative: Vec<f32>,
    /// The activation pattern defining this region.
    pub pattern: ActivationPattern,
}

/// Result of classifying a batch of inputs into linear regions.
pub struct RegionMap {
    /// Distinct activation patterns observed, with counts and
    /// representatives. Sorted by count descending (most populated
    /// region first).
    pub regions: Vec<RegionInfo>,
    /// Total number of inputs classified.
    pub total_inputs: usize,
}

/// Classify inputs into their piecewise-linear regions.
///
/// Runs `forward_fast_traced` on each input, records the activation
/// pattern, and groups inputs by pattern.
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
pub fn classify_regions(
    weights: &RnnWeights,
    inputs: &[f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> RegionMap;

/// Compute the affine map for a given activation pattern.
///
/// For a fixed activation pattern, the RNN reduces to a sequence of
/// linear operations: at each timestep, the ReLU gate is either
/// identity (neuron active) or zero (neuron inactive). Composing
/// these gives a single affine map `output = A · input + b`.
///
/// Since the RNN has no biases (`bias=False` in AlgZoo), the map is
/// purely linear: `b = 0`, so output = A · input.
///
/// # Returns
///
/// Matrix `A` of shape `[output_size, seq_len]`, row-major. Since
/// `bias=False`, the affine offset is always zero.
pub fn region_linear_map(
    weights: &RnnWeights,
    pattern: &ActivationPattern,
    config: &StoicheiaConfig,
) -> Vec<f32>;
```

**Derivation of the linear map**: for a fixed activation pattern, define
the diagonal gate matrix `G_t` ∈ R^{H×H} where `G_t[j,j] = 1` if neuron
`j` is active at timestep `t`, else `0`. Then:

```
h_1 = G_0 · W_ih · x_0
h_2 = G_1 · (W_ih · x_1 + W_hh · h_1)
    = G_1 · W_ih · x_1 + G_1 · W_hh · G_0 · W_ih · x_0
...
h_T = sum_{t=0..T-1} M(T-1, t+1) · G_t · W_ih · x_t

where M(b, a) = G_b · W_hh · G_{b-1} · W_hh · ... · G_a · W_hh
             (product from left to right in decreasing index order)
      M(b, a) = I  when a > b  (empty product)

output = W_oh · h_T
       = sum_t  W_oh · M(T-1, t+1) · G_t · W_ih · x_t
       = A · input
```

where column `t` of `A` is `W_oh · M(T-1, t+1) · G_t · W_ih`.

This is computed analytically in `region_linear_map()` by composing the
matrices right-to-left.

---

### Module 4: `ablation.rs` — Exhaustive Neuron Ablation

Zero each neuron's hidden state at all timesteps, measure the accuracy
change. Identifies critical vs. redundant neurons.

```rust
/// Result of ablating a single neuron.
pub struct NeuronAblationResult {
    /// Neuron index (0-indexed).
    pub neuron: usize,
    /// Accuracy with this neuron zeroed (0.0 to 1.0).
    pub ablated_accuracy: f32,
    /// Accuracy change relative to baseline (negative = important).
    pub accuracy_delta: f32,
}

/// Result of a full single-neuron ablation sweep.
pub struct AblationSweep {
    /// Baseline accuracy (no ablation).
    pub baseline_accuracy: f32,
    /// Per-neuron results, sorted by `accuracy_delta` ascending
    /// (most damaging ablation first).
    pub results: Vec<NeuronAblationResult>,
    /// Number of test inputs used.
    pub n_inputs: usize,
}

/// Run exhaustive single-neuron ablation on an RNN.
///
/// For each of the H neurons, zeros that neuron's hidden state at
/// every timestep and measures accuracy on `inputs`.
///
/// Uses `forward_fast_ablated` internally. For M₁₆,₁₀ with 10K
/// inputs: 16 × 10K = 160K forward passes (~10ms with fast path).
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
/// - `targets`: `[n_inputs]`, ground-truth output class indices
pub fn ablate_neurons(
    weights: &RnnWeights,
    inputs: &[f32],
    targets: &[u32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> AblationSweep;
```

#### Pairwise Ablation

```rust
/// Result of ablating a pair of neurons simultaneously.
pub struct PairAblationResult {
    /// First neuron index.
    pub neuron_a: usize,
    /// Second neuron index.
    pub neuron_b: usize,
    /// Accuracy with both neurons zeroed.
    pub ablated_accuracy: f32,
    /// Accuracy change relative to baseline.
    pub accuracy_delta: f32,
    /// Interaction score: `pair_delta - (delta_a + delta_b)`.
    /// Negative = super-additive damage (redundancy — each neuron
    /// compensates for the other, but removing both is catastrophic).
    /// Near zero = independent (additive damage).
    /// Positive = sub-additive damage (ceiling effect — each neuron
    /// is independently important, but combined damage saturates).
    pub interaction_score: f32,
}

/// Run pairwise neuron ablation.
///
/// For each pair (i, j), zeros both neurons and measures accuracy.
/// The interaction score reveals functional redundancy or synergy.
///
/// For M₁₆,₁₀: C(16,2) = 120 sweeps × 10K inputs = 1.2M forward passes.
pub fn ablate_neuron_pairs(
    weights: &RnnWeights,
    inputs: &[f32],
    targets: &[u32],
    n_inputs: usize,
    config: &StoicheiaConfig,
    single_results: &AblationSweep,
) -> Vec<PairAblationResult>;
```

**Interaction score**: `interaction = pair_delta - (delta_a + delta_b)`.

- **Negative** (super-additive): ablating either alone has little effect
  (the other compensates), but ablating both is catastrophic. Indicates
  **functional redundancy** — both neurons implement the same function.
- **Near zero** (additive): combined damage ≈ sum of individual damages.
  Indicates **independence** — neurons serve distinct functions.
- **Positive** (sub-additive): each neuron is important individually, but
  combined damage is less than expected (accuracy floor / saturation).

The negative (redundancy) pattern is expected in M₁₆,₁₀'s densely connected
neurons, which the blog's analysis was unable to disentangle.

---

### Module 5: `probing.rs` — Neuron Functional Classification

Run structured inputs through the RNN and classify each neuron's response
pattern by correlating activations with known reference signals.

```rust
/// Functional role identified for a neuron.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronRole {
    /// Tracks the running maximum: h_t ≈ max(0, x_0, ..., x_{t-1}).
    RunningMax,
    /// Tracks the running minimum: h_t ≈ min(x_0, ..., x_{t-1}).
    RunningMin,
    /// Tracks max increment: h_t ≈ max(x_0,...,x_{t-1}) - max(x_0,...,x_{t-2}).
    MaxIncrement,
    /// Tracks a leave-one-out maximum: h_T ≈ max(x \ x_i) for some i.
    LeaveOneOutMax,
    /// Compares two specific positions: h_T ≈ sign(x_i - x_j).
    Comparator,
    /// Tracks recent input: h_t ≈ x_{t-1} or h_t ≈ -x_{t-1}.
    RecentInput,
    /// No clear functional role identified (correlation below threshold).
    Unknown,
}
```

From the AlgZoo blog's analysis of M₁₆,₁₀:

| Neuron | Identified role | Blog description |
|--------|----------------|------------------|
| 2 | `RunningMax` | ≈ max(0, x_0, ..., x_{t-2}) |
| 4 | `MaxIncrement` | ≈ max(0,...,x_{t-1}) − max(0,...,x_{t-2}) |
| 6 | `LeaveOneOutMax` | ≈ max(0, x_0,...,x_{t-3}, x_{t-1}) − x_{t-1} |
| 7 | `LeaveOneOutMax` | ≈ max(0, x_0,...,x_{t-1}) − x_{t-1} |
| 1 | `LeaveOneOutMax` | ≈ max(0, x_0,...,x_{t-4},x_{t-2},x_{t-1}) − x_{t-1} |

```rust
/// Result of probing a single neuron.
pub struct NeuronProbeResult {
    /// Neuron index.
    pub neuron: usize,
    /// Best-matching functional role.
    pub role: NeuronRole,
    /// Pearson correlation between neuron activation and the best
    /// reference signal (at the final timestep).
    pub correlation: f32,
    /// Per-timestep correlations with the best reference signal.
    /// Length = `seq_len`.
    pub temporal_correlations: Vec<f32>,
}

/// Probe results for all neurons in a model.
pub struct ProbeReport {
    /// Per-neuron probe results, ordered by neuron index.
    pub neurons: Vec<NeuronProbeResult>,
    /// Model configuration.
    pub config: StoicheiaConfig,
    /// Number of probe inputs used.
    pub n_probes: usize,
}

/// Run functional probes on all neurons of an RNN.
///
/// For each probe type, generates `n_probes` random inputs, runs
/// `forward_fast_traced` to capture per-timestep activations, and
/// correlates each neuron's trace with the probe's reference signal.
/// The best-matching probe (highest absolute correlation) determines
/// the neuron's role.
///
/// # Probe types
///
/// | Probe | Reference signal | Detects |
/// |-------|------------------|---------|
/// | Running max | `cummax(x[0..t])` | `RunningMax` |
/// | Running min | `cummin(x[0..t])` | `RunningMin` |
/// | Max increment | `cummax(x[0..t]) - cummax(x[0..t-1])` | `MaxIncrement` |
/// | Leave-one-out max | `max(x) - x_i` for each i | `LeaveOneOutMax` |
/// | Recent input | `x_{t-1}` and `-x_{t-1}` | `RecentInput` |
/// | Comparator | `sign(x_i - x_j)` for all (i,j) pairs | `Comparator` |
///
/// # Arguments
///
/// * `n_probes` — number of random inputs per probe type. Default: 1000.
///   Each forward pass captures all H neurons, so the total is
///   `n_probes × (number of probe types)` forward passes.
pub fn probe_neurons(
    weights: &RnnWeights,
    config: &StoicheiaConfig,
    n_probes: usize,
) -> ProbeReport;
```

**Correlation threshold**: a neuron is classified as `Unknown` if the best
correlation is below 0.8. This threshold is chosen to be conservative —
a correlation of 0.8 means the reference signal explains 64% of the
variance in the neuron's activation.

---

### Module 6: `surprise.rs` — Surprise Accounting

Implements ARC's information-theoretic metric for evaluating mechanistic
understanding. Two complementary evaluation methods from the blog:

1. **MSE vs. compute**: how close does the mechanistic estimate get to the
   model's actual accuracy, as a function of compute?
2. **Surprise accounting**: how many bits of surprise remain? Full
   understanding ≈ total surprise matches bits of optimization.

#### The `MechanisticEstimator` Trait

```rust
/// A mechanistic estimator: predicts the model's output class without
/// running the model, based on mechanistic understanding of its internals.
///
/// Implementors encode their understanding of the model as a prediction
/// function. The `OracleEstimator` provides the upper bound (perfect
/// task understanding). Users implement this trait with increasingly
/// refined mechanistic explanations.
// TRAIT_OBJECT: user-provided mechanistic understanding
pub trait MechanisticEstimator {
    /// Predict the model's output class for a single input.
    ///
    /// # Shapes
    /// - `input`: `[seq_len]` — one input sequence
    /// - returns: predicted output class index
    fn predict(&self, input: &[f32]) -> u32;

    /// Human-readable description of this estimator's methodology.
    fn description(&self) -> &str;
}
```

#### Built-in Estimators

```rust
/// Oracle estimator: uses the ground-truth task function.
///
/// This is the upper bound — if the model perfectly implements the
/// task, the oracle estimator achieves zero surprise residual.
pub struct OracleEstimator {
    task: StoicheiaTask,
    seq_len: usize,
}

impl OracleEstimator {
    #[must_use]
    pub const fn new(task: StoicheiaTask, seq_len: usize) -> Self;
}

impl MechanisticEstimator for OracleEstimator { ... }

/// Piecewise-linear estimator: decomposes the model into per-region
/// affine maps.
///
/// For each input, runs `forward_fast_traced` to determine the
/// activation pattern (which neurons fire at which timesteps),
/// computes the region's affine map analytically via
/// `region_linear_map`, and takes argmax of the result.
///
/// This estimator always agrees with the model (it IS the model,
/// decomposed into named parts). The value is not in prediction
/// but in **surprise accounting**: the bits needed to describe the
/// region structure and coefficient magnitudes constitute the
/// "surprise of the explanation" from the blog.
///
/// This mirrors the blog's analysis of M₂,₂: classify the input's
/// linear region, verify that coefficients imply correct ordering,
/// count the bits of surprise for each verification step.
pub struct PiecewiseEstimator {
    weights: RnnWeights,
    config: StoicheiaConfig,
}

impl PiecewiseEstimator {
    pub fn new(weights: RnnWeights, config: StoicheiaConfig) -> Self;
}

impl MechanisticEstimator for PiecewiseEstimator { ... }
```

#### Surprise Report

```rust
/// Result of a surprise accounting measurement.
pub struct SurpriseReport {
    /// Model accuracy on random inputs (0.0 to 1.0).
    pub model_accuracy: f32,
    /// Mechanistic estimate accuracy.
    pub estimate_accuracy: f32,
    /// Mean squared error between model and estimate per-input
    /// predictions (0 = perfect agreement, 1 = complete disagreement).
    pub mse: f32,
    /// Chance accuracy: 1/output_size (uniform random baseline).
    pub chance_accuracy: f32,
    /// Number of model parameters.
    pub param_count: usize,
    /// Number of test samples used.
    pub n_samples: usize,
    /// Per-input agreement: fraction where model and estimate
    /// predict the same class.
    pub agreement_rate: f32,
    /// Description of the estimator used.
    pub estimator_description: String,
}

/// Run surprise accounting: compare model accuracy vs. mechanistic
/// estimate on random inputs.
///
/// Generates `n_samples` random inputs (fixed seed for reproducibility),
/// runs both the model (via fast path) and the mechanistic estimator,
/// and computes agreement metrics.
///
/// # The blog's suggested evaluation
///
/// > A cheap way to measure mean squared error is to add noise to
/// > the model's weights (enough to significantly alter the model's
/// > accuracy) and check the squared error of the method on average
/// > over the choice of noisy model.
///
/// For this, call `surprise_accounting_noisy()` which perturbs the
/// weights and averages MSE over multiple noise realizations.
pub fn surprise_accounting(
    weights: &RnnWeights,
    estimator: &dyn MechanisticEstimator,
    config: &StoicheiaConfig,
    n_samples: usize,
) -> SurpriseReport;

/// Run surprise accounting with weight perturbation.
///
/// For each of `n_noise` random perturbations, adds Gaussian noise
/// (scaled by `noise_scale`) to the model's weights, measures MSE
/// between the perturbed model and the estimator, and returns the
/// average MSE across perturbations.
///
/// This implements the blog's suggested evaluation method:
/// "add noise to the model's weights (enough to significantly alter
/// the model's accuracy) and check the squared error of the method
/// on average over the choice of noisy model."
pub fn surprise_accounting_noisy(
    weights: &RnnWeights,
    estimator: &dyn MechanisticEstimator,
    config: &StoicheiaConfig,
    n_samples: usize,
    n_noise: usize,
    noise_scale: f32,
) -> SurpriseReport;
```

**Design note on weight-dependent estimators**: the `MechanisticEstimator`
trait takes only `&[f32]` input, not weights. This means the estimator is
a *fixed* explanation applied to all perturbations. For the `OracleEstimator`
(weight-independent), this is correct. For a `PiecewiseEstimator` (which
holds its own copy of weights), `surprise_accounting_noisy` tests whether
the *original model's* piecewise structure still explains the *perturbed*
model — exactly the blog's intended evaluation: "a good method should work
well on average over random seeds." If future work requires re-applying the
estimation *procedure* to each perturbed model (not just the fixed estimate),
a `MechanisticEstimationMethod` trait taking `&RnnWeights` as additional
input would be needed. This is deferred to avoid over-engineering Phase B.

---

## Changes to Existing Files

### `src/stoicheia/mod.rs`

Add weight accessors to `StoicheiaRnn` so that `RnnWeights::from_model()`
can extract the raw f32 data:

```rust
impl StoicheiaRnn {
    /// Access the input-to-hidden weight tensor.
    #[must_use]
    pub fn weight_ih(&self) -> &Tensor { &self.weight_ih }

    /// Access the hidden-to-hidden weight tensor.
    #[must_use]
    pub fn weight_hh(&self) -> &Tensor { &self.weight_hh }

    /// Access the output projection weight tensor.
    #[must_use]
    pub fn weight_oh(&self) -> &Tensor { &self.weight_oh }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &StoicheiaConfig { &self.config }
}
```

Add module declarations:

```rust
pub mod ablation;
pub mod fast;
pub mod piecewise;
pub mod probing;
pub mod standardize;
pub mod surprise;
```

### `src/lib.rs`

Add re-exports under `#[cfg(feature = "stoicheia")]`:

```rust
#[cfg(feature = "stoicheia")]
pub use stoicheia::{
    // Phase A (existing)
    StoicheiaArch, StoicheiaConfig, StoicheiaOutput, StoicheiaRnn,
    StoicheiaTask, StoicheiaTransformer,
    // Phase B (new)
    fast::RnnWeights,
    standardize::StandardizedRnn,
    ablation::AblationSweep,
    piecewise::{ActivationPattern, RegionMap},
    probing::{NeuronRole, ProbeReport},
    surprise::{MechanisticEstimator, SurpriseReport},
};
```

### `Cargo.toml`

No new dependencies. Add new example and test entries:

```toml
[[example]]
name = "stoicheia_analysis"
required-features = ["stoicheia"]

[[test]]
name = "stoicheia_analysis"
required-features = ["stoicheia"]
```

---

## Composition Pipeline

```
StoicheiaRnn::load()                     (Phase A)
      │
      ├── RnnWeights::from_model()       (fast.rs)
      │        │
      │        ├── forward_fast()        ─── accuracy()
      │        │                              │
      │        ├── forward_fast_ablated() ─── ablate_neurons()
      │        │                              ablate_neuron_pairs()
      │        │
      │        ├── forward_fast_traced() ─── classify_regions()
      │        │                              probe_neurons()
      │        │
      │        └── accuracy()            ─── surprise_accounting()
      │                                      surprise_accounting_noisy()
      │
      └── standardize_rnn()              (standardize.rs)
               │
               └── StandardizedRnn
                    │
                    ├── → RnnWeights     (for analysis on standardized form)
                    │     └── classify_regions() → region_linear_map()
                    │                                    │
                    │                              PiecewiseEstimator
                    │                                    │
                    │                              surprise_accounting()
                    │
                    └── standardization_quality()
                         sign/magnitude analysis
```

---

## Example: Full M₂,₂ Analysis

```rust
// examples/stoicheia_analysis.rs
use candle_core::Device;
use candle_mi::stoicheia::{
    ablation, fast, piecewise, probing, standardize, surprise,
    StoicheiaConfig, StoicheiaRnn, StoicheiaTask,
};
use rand::Rng;

fn main() -> candle_mi::Result<()> {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), "rnn_2_2.safetensors", &Device::Cpu)?;

    // 1. Extract raw weights
    let weights = fast::RnnWeights::from_model(&model)?;

    // 2. Generate test inputs (standard Gaussian, fixed seed)
    let n = 10_000;
    let mut rng = rand::thread_rng();
    let inputs: Vec<f32> = (0..n * config.seq_len)
        .map(|_| rng.gen_range(-3.0_f32..3.0))  // approximate; use rand_distr for true Gaussian
        .collect();
    let targets: Vec<u32> = /* compute second_argmax on inputs */;

    // 3. Baseline accuracy
    let baseline = fast::accuracy(&weights, &inputs, &targets, n, &config);
    println!("Baseline accuracy: {baseline:.4}");

    // 4. Standardize weights
    let std_rnn = standardize::standardize_rnn(&model)?;
    let quality = standardize::standardization_quality(&std_rnn);
    println!("Standardization quality: max deviation {quality:.6}");

    // 5. Print sign table
    for j in 0..config.hidden_size {
        let sign = if std_rnn.weight_ih[j] > 0.0 { "+" } else { "-" };
        println!("  Neuron {j}: W_ih sign = {sign}");
    }

    // 6. Ablation sweep
    let sweep = ablation::ablate_neurons(&weights, &inputs, &targets, n, &config);
    println!("\nAblation results:");
    for r in &sweep.results {
        println!("  Neuron {}: accuracy {:.4} (delta {:+.4})",
            r.neuron, r.ablated_accuracy, r.accuracy_delta);
    }

    // 7. Neuron probing
    let probes = probing::probe_neurons(&weights, &config, 1000);
    println!("\nNeuron roles:");
    for p in &probes.neurons {
        println!("  Neuron {}: {:?} (corr {:.3})", p.neuron, p.role, p.correlation);
    }

    // 8. Region enumeration
    let regions = piecewise::classify_regions(&weights, &inputs, n, &config);
    println!("\nPiecewise-linear regions: {}", regions.regions.len());
    for (i, r) in regions.regions.iter().enumerate().take(5) {
        println!("  Region {i}: {} inputs, {} active neurons",
            r.count, r.pattern.count_active());
    }

    // 9. Surprise accounting
    let oracle = surprise::OracleEstimator::new(StoicheiaTask::SecondArgmax, config.seq_len);
    let report = surprise::surprise_accounting(&weights, &oracle, &config, n);
    println!("\nSurprise accounting (oracle estimator):");
    println!("  Model accuracy:    {:.4}", report.model_accuracy);
    println!("  Estimate accuracy: {:.4}", report.estimate_accuracy);
    println!("  Agreement rate:    {:.4}", report.agreement_rate);
    println!("  MSE:               {:.6}", report.mse);

    Ok(())
}
```

---

## Testing Strategy

### Unit Tests (in each module's `#[cfg(test)]` block)

| Module | Test | Description |
|--------|------|-------------|
| `fast.rs` | `fast_matches_candle` | Compare `forward_fast` output to `StoicheiaRnn::forward()` on M₂,₂ fixture (1e-6 tolerance) |
| `fast.rs` | `fast_traced_captures_pre_activations` | Verify pre-activation signs match hook captures from Phase A's `rnn.hook_pre_activation.{t}` |
| `fast.rs` | `accuracy_trivial` | Accuracy on inputs where all predict one class should be 1.0 |
| `standardize.rs` | `standardize_preserves_output` | Standardize M₂,₂ fixture, verify `forward_fast` output unchanged (1e-5 tolerance) |
| `standardize.rs` | `standardized_wih_near_one` | After standardization, `|W_ih[j]|` should be 1.0 (1e-6 tolerance) |
| `standardize.rs` | `degenerate_neuron_errors` | If W_ih[j] = 0, standardization should return an error |
| `piecewise.rs` | `activation_pattern_roundtrip` | Create pattern, query `is_active`, verify consistency |
| `piecewise.rs` | `m2_2_region_count` | M₂,₂ should have ≤ 16 regions on 10K Gaussian inputs |
| `piecewise.rs` | `linear_map_matches_forward` | For a given pattern, the linear map should reproduce `forward_fast` output exactly |
| `ablation.rs` | `no_ablation_matches_baseline` | Ablation with empty mask should equal baseline accuracy |
| `ablation.rs` | `full_ablation_near_chance` | Ablating all neurons should give near-chance accuracy |
| `probing.rs` | `running_max_detection` | Construct a model with known running-max neuron; verify probe detects it |
| `surprise.rs` | `oracle_matches_task` | Oracle estimator should agree with `tasks::second_argmax` |
| `surprise.rs` | `perfect_model_high_agreement` | If model ≈ oracle, agreement rate should be high |

### Integration Test

```
tests/stoicheia_analysis.rs
```

Full pipeline test on M₂,₂ fixture:
1. Load model → extract `RnnWeights`
2. Verify `fast_matches_candle` (correctness gate)
3. Standardize → verify output preserved
4. Classify regions → verify count is reasonable
5. Ablate → verify baseline matches
6. Probe → verify at least one neuron has correlation > 0.8
7. Surprise accounting → verify oracle agreement > 0.9

Uses the existing `tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors`
fixture (10 parameters, committed in Phase A). No new fixtures needed.

---

## Module Structure

```
src/stoicheia/
  mod.rs           — Phase A + weight accessors + module declarations
  config.rs        — Phase A (unchanged)
  tasks.rs         — Phase A (unchanged)
  fast.rs          — NEW: RnnWeights, forward_fast, accuracy
  standardize.rs   — NEW: standardize_rnn, StandardizedRnn
  piecewise.rs     — NEW: ActivationPattern, classify_regions, region_linear_map
  ablation.rs      — NEW: ablate_neurons, ablate_neuron_pairs
  probing.rs       — NEW: NeuronRole, probe_neurons, ProbeReport
  surprise.rs      — NEW: MechanisticEstimator, surprise_accounting
```

---

## File Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `src/stoicheia/fast.rs` | **Create** | 200–250 |
| `src/stoicheia/standardize.rs` | **Create** | 120–150 |
| `src/stoicheia/piecewise.rs` | **Create** | 250–320 |
| `src/stoicheia/ablation.rs` | **Create** | 180–220 |
| `src/stoicheia/probing.rs` | **Create** | 250–300 |
| `src/stoicheia/surprise.rs` | **Create** | 150–200 |
| `src/stoicheia/mod.rs` | Edit | +30 (accessors + module decls) |
| `src/lib.rs` | Edit | +10 (re-exports) |
| `examples/stoicheia_analysis.rs` | **Create** | 120–160 |
| `tests/stoicheia_analysis.rs` | **Create** | 150–200 |
| `CHANGELOG.md` | Edit | +15 |

**Total new Rust code**: ~1,270–1,640 lines + ~150–200 lines tests.

---

## Implementation Order

1. **`fast.rs`** — Foundation: `RnnWeights`, `forward_fast`, `accuracy`. Test: `fast_matches_candle`.
2. **`standardize.rs`** — Weight normalization. Requires weight accessors on `StoicheiaRnn`. Test: `standardize_preserves_output`.
3. **`piecewise.rs`** — Region enumeration. Depends on `fast.rs` (`forward_fast_traced`). Test: `m2_2_region_count`.
4. **`ablation.rs`** — Neuron ablation. Depends on `fast.rs` (`forward_fast_ablated`). Test: `no_ablation_matches_baseline`.
5. **`probing.rs`** — Functional classification. Depends on `fast.rs` (`forward_fast_traced`). Test: `running_max_detection`.
6. **`surprise.rs`** — Surprise accounting. Depends on `fast.rs` (`accuracy`). Test: `oracle_matches_task`.
7. **Module wiring** — `mod.rs` module decls, `lib.rs` re-exports, `Cargo.toml` entries.
8. **Integration test** — `tests/stoicheia_analysis.rs`: full pipeline on M₂,₂ fixture.
9. **Example** — `examples/stoicheia_analysis.rs`: CLI-driven analysis.
10. **`CHANGELOG.md`** — Phase B entries.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RNN-only for Phase B? | Yes | Blog methodology (standardization, piecewise-linear) is ReLU-RNN-specific. Transformer MI (QK/OV circuits) belongs to main roadmap Phase 7/8. |
| Fast path on `&[f32]` not `Tensor`? | Yes | Eliminates candle's 18–25× overhead on tiny models. All MI tools need bulk forward passes. |
| `RnnWeights` shared by all modules? | Yes | Extract from `Tensor` once, reuse everywhere. One-time cost. |
| `ActivationPattern` storage? | `[u64; 5]` (320 bits) | Covers all AlgZoo RNN configs (max H=32, T=10 = 320). No heap allocation. `Hash` + `Eq` derivable. |
| `region_linear_map` returns Vec, not Tensor? | Yes | Pure linear algebra on raw f32. Models are tiny; no BLAS needed. |
| `MechanisticEstimator` as trait? | Yes | User-extensible: researchers encode their understanding as an estimator. Oracle and piecewise built-in. |
| Probing limited to 2nd-argmax domain? | Initially yes | Blog focuses on 2nd-argmax RNNs. `#[non_exhaustive]` on `NeuronRole` allows extension to other tasks. |
| Pairwise ablation included? | Yes | Detects functional redundancy, central to understanding M₁₆,₁₀'s dense connectivity. |
| New Cargo dependencies? | None | Correlation, small matrix multiply, random input generation all inline. `rand` already in deps. |
| `#![forbid(unsafe_code)]` maintained? | Yes | Auto-vectorization only. Consistent with crate policy. |
| Hidden state buffer? | `[f32; 32]` on stack | Covers all AlgZoo hidden sizes. No heap allocation in the hot path. |
| Standardization on zero W_ih? | Error | A neuron with zero input weight is degenerate (no input sensitivity). Standardization is undefined. |

---

## Scope Boundary: What Phase B Does NOT Include

| Out of scope | Why | Where it belongs |
|---|---|---|
| Transformer MI tools | Different methodology (attention circuits, not piecewise-linear) | Main roadmap Phase 7/8 |
| Training loop | Pre-trained weights are the artifact; AlgZoo trains in Python | AlgZoo Python codebase |
| Training dynamics | Requires checkpoint series not yet available in safetensors | Future Phase C |
| Gradient computation | No autograd in candle-core for this use case | Not needed for forward-pass MI |
| Visualization | Plots of regions, ablation heatmaps, etc. | Deloson (separate crate) |
| GCS download automation | One-time manual step documented in Phase A | Phase A prerequisites |
| Handcrafted model construction | Research output, not library code | User's research notebooks |

---

## Connection to Prolepsis Research

Phase B's tools focus on phenomena meaningful for tiny/shallow models:

| Phenomenon | Requires depth? | Phase B tool |
|---|---|---|
| Neuron feature specialization | No (1 layer suffices) | `probing.rs` |
| Piecewise-linear decision boundaries | No (ReLU creates regions at any depth) | `piecewise.rs` |
| Functional redundancy between neurons | No (emerges from optimization) | `ablation.rs` (pairwise) |
| Subcircuit isolation | No (weight sparsity, not depth) | `standardize.rs` + `probing.rs` |
| Prolepsis (early irrevocable commitment) | Yes (>16 layers) | NOT in scope — requires depth |
| Routing heads | Yes (multi-layer attention) | NOT in scope |

The user's prolepsis research and ARC's AlgZoo attack "minimum model for
full understanding" from complementary axes: ARC minimizes *parameters*
(8–1,400), the user minimizes *depth* (16 vs 26 layers). Phase B tools
answer the parameter axis; the main roadmap's transformer tools answer the
depth axis. The open research question — whether prolepsis can emerge in
shallow models — remains for future investigation.
