# Implementation Plan: Stoicheia (AlgZoo Backends) — candle-mi Phase A

> **stoicheia** (στοιχεῖα) — "elements, first principles." Euclid's *Elements*
> built all of geometry from irreducible foundations; ARC's AlgZoo builds MI
> from irreducibly small models. The candle-mi module is named `stoicheia`.



- [Jacob Hilton](https://www.alignment.org/author/jacob/), [AlgZoo](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/): uninterpreted models with fewer than 1,500 parameters,  January 26th, 2026.

- Alignment Research Center , [AlgZoo Github page](https://github.com/alignment-research-center/alg-zoo).

---

## Goal

Add two `MIBackend` implementations for AlgZoo's tiny models: a single-layer
ReLU RNN and an attention-only transformer. These load pre-trained weights
(converted to safetensors via anamnesis) and run forward passes with full hook
support, enabling Phase B's exhaustive MI tooling.

**Scope**: Load weights, run inference, capture activations via hooks.
No training loop, no GCS download automation, no MI analysis tools (Phase B).

---

## Architecture Fit

AlgZoo models are **not language models** — they operate on continuous floats
(RNN) or small-range integers (transformer), not token vocabularies. The
`MIBackend` trait can accommodate this with minor semantic stretching:

| `MIBackend` method | RNN interpretation | Transformer interpretation |
|----|----|----|
| `num_layers()` | 1 (always single-layer) | `len(attns)` (1 or 2) |
| `hidden_size()` | RNN hidden dimension `H` | Embedding dimension `H` |
| `vocab_size()` | `output_size` (seq_len or 1) | `output_size` (seq_len or 1) |
| `num_heads()` | 1 | `num_heads` (typically 1) |
| `forward(input, hooks)` | Input is `[batch, seq]` **floats** | Input is `[batch, seq]` **integers** |
| `project_to_vocab(hidden)` | `linear.weight @ hidden` | `unembed.weight @ hidden` |

Output shape: both architectures produce output from the **final position only**,
shaped `[batch, 1, output_size]` (unsqueeze to match the `[batch, seq, vocab]`
convention). `output_size` is `seq_len` for distribution tasks, `1` for scalar tasks.

`MIModel::from_pretrained()` is **not extended** — AlgZoo models aren't on
HuggingFace and have no `config.json`. Instead, each backend has its own
`load()` constructor that takes an explicit config and a safetensors path.

---

## The Two Architectures

### 1. Single-Layer ReLU RNN

From AlgZoo's `architectures.py`:

```python
class OneLayerRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        self.rnn = nn.RNN(input_size=1, hidden_size=H, nonlinearity="relu",
                          bias=False, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, init_state=None):
        output, final_state = self.rnn(x[..., None], init_state)
        return self.linear(final_state.squeeze(0))
```

**Forward pass in Rust** (no `nn.RNN` needed — manual cell):

```
For each timestep t in 0..seq_len:
    pre_act_t = W_ih @ x_t + W_hh @ h_{t-1}     // [batch, H]
    h_t = relu(pre_act_t)                         // [batch, H]
output = W_oh @ h_final                           // [batch, output_size]
```

Where `x_t` is scalar (input_size=1), so `W_ih @ x_t` is really `x_t * W_ih`.

**State_dict keys** (safetensors tensor names after anamnesis conversion):

| Key | Shape | Description |
|-----|-------|-------------|
| `rnn.weight_ih_l0` | `[H, 1]` | Input-to-hidden weights |
| `rnn.weight_hh_l0` | `[H, H]` | Hidden-to-hidden weights |
| `linear.weight` | `[output_size, H]` | Output projection |

**Parameter count**: `H*(1 + H + output_size)`. For M₁₆,₁₀: 16*(1+16+10) = 432.

### 2. Attention-Only Transformer

From AlgZoo's `architectures.py`:

```python
class AttentionOnlyTransformer(nn.Module):
    def __init__(self, hidden_size, output_size, seq_len, input_range, n_layers=2):
        self.embed = nn.Embedding(input_range, hidden_size)
        self.pos_embed = nn.Embedding(seq_len, hidden_size)
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, 1, bias=False, batch_first=True)
            for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, init_state=None):
        x = self.embed(x) + self.pos_embed(positions)
        for attn in self.attns:
            x = x + attn(x, x, x)[0]       # residual, NO causal mask
        return self.unembed(x[:, -1])       # last position only
```

**Key properties**:
- **No MLP blocks** — attention only
- **No layer normalization** — raw residual additions
- **No causal mask** — full bidirectional attention
- **Single attention head** (default)
- **Output from last position only**

**State_dict keys**:

| Key | Shape | Description |
|-----|-------|-------------|
| `embed.weight` | `[input_range, H]` | Token embedding |
| `pos_embed.weight` | `[seq_len, H]` | Positional embedding |
| `attns.{i}.in_proj_weight` | `[3*H, H]` | Packed Q, K, V projection |
| `attns.{i}.out_proj.weight` | `[H, H]` | Attention output projection |
| `unembed.weight` | `[output_size, H]` | Final unembedding |

**PyTorch packs Q/K/V** into a single `in_proj_weight` tensor of shape
`[3*H, H]`. To extract: `Q = in_proj[0:H]`, `K = in_proj[H:2H]`,
`V = in_proj[2H:3H]`. Use `narrow(0, ...)` in candle.

---

## Hook Points

### RNN Hooks

Use `HookPoint::Custom(String)` for Phase A (avoids changing core `hooks.rs`).
Phase B may promote frequently-used ones to proper enum variants (following
the RWKV precedent of `RwkvState`, `RwkvDecay`, `RwkvEffectiveAttn`).

| Hook (Custom string) | Location | Shape | MI use |
|----|----|----|-----|
| `"rnn.hook_pre_activation.{t}"` | Before ReLU at timestep `t` | `[batch, H]` | Decision boundary analysis |
| `"rnn.hook_hidden.{t}"` | Hidden state after timestep `t` | `[batch, H]` | Feature accumulation tracking |
| `"rnn.hook_final_state"` | Final hidden state `h_n` | `[batch, H]` | What the model "remembers" |
| `"rnn.hook_output"` | After output projection | `[batch, output_size]` | Logit analysis |

The per-timestep hooks (`{t}`) are critical for Phase B's neuron-level analysis.
To avoid capturing all timesteps by default, the backend checks each
`Custom("rnn.hook_hidden.{t}")` individually — only captured timesteps allocate.

### Transformer Hooks

Reuse existing `HookPoint` variants — they fit perfectly:

| Hook | Location | Shape |
|------|----------|-------|
| `Embed` | After token + positional embedding | `[batch, seq, H]` |
| `ResidPre(i)` | Before attention layer `i` | `[batch, seq, H]` |
| `AttnScores(i)` | Pre-softmax attention weights | `[batch, 1, seq, seq]` |
| `AttnPattern(i)` | Post-softmax attention weights | `[batch, 1, seq, seq]` |
| `AttnOut(i)` | Attention output | `[batch, seq, H]` |
| `ResidPost(i)` | After residual add | `[batch, seq, H]` |

No `MlpPre/Post/Out` (no MLP), no `FinalNorm` (no normalization), no
`ResidMid` (no MLP means ResidPost = after attention).

---

## Config

```rust
/// Task type for AlgZoo models.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoicheiaTask {
    /// Find the position of the second-largest number.
    SecondArgmax,
    /// Find the position of the median number.
    Argmedian,
    /// Output the median value.
    Median,
    /// Count the longest cycle in a permutation.
    LongestCycle,
}

/// Architecture type for AlgZoo models.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoicheiaArch {
    /// Single-layer ReLU RNN (continuous input → distribution or scalar output).
    Rnn,
    /// Attention-only transformer (discrete input → distribution or scalar output).
    Transformer,
}

/// Output type for AlgZoo models.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoicheiaOutput {
    /// Output is a distribution over positions (cross-entropy loss).
    Distribution,
    /// Output is a single scalar (MSE loss).
    Scalar,
}

/// Configuration for an AlgZoo model.
pub struct StoicheiaConfig {
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Input sequence length.
    pub seq_len: usize,
    /// Task type.
    pub task: StoicheiaTask,
    /// Architecture type (inferred from task).
    pub arch: StoicheiaArch,
    /// Output type (inferred from task).
    pub output: StoicheiaOutput,
    /// Number of attention layers (transformer only; always 1 for RNN).
    pub num_layers: usize,
    /// Number of attention heads (transformer only).
    pub num_heads: usize,
    /// Input range for discrete tasks (transformer only; = seq_len).
    pub input_range: usize,
}
```

**Task → Architecture mapping** (from AlgZoo's `tasks.py`):

| Task | Input | Output | Architecture |
|------|-------|--------|-------------|
| `SecondArgmax` | continuous | distribution (seq_len classes) | RNN |
| `Argmedian` | continuous | distribution (seq_len classes) | RNN |
| `Median` | continuous | scalar (1 output) | RNN |
| `LongestCycle` | discrete | distribution (seq_len classes) | Transformer |

`StoicheiaConfig` should provide a `from_task(task, hidden_size, seq_len)` constructor
that fills in defaults (arch, output, num_layers, num_heads, input_range) based on
the task, matching AlgZoo's Python registry.

---

## Module Structure

```
src/stoicheia/
  mod.rs       — StoicheiaRnn, StoicheiaTransformer structs + MIBackend impls + load()
  config.rs    — StoicheiaConfig, StoicheiaTask, StoicheiaArch, StoicheiaOutput enums
  tasks.rs     — Task functions for validation (second_argmax, argmedian, etc.)
```

Feature-gated behind `stoicheia` in `Cargo.toml`. No new dependencies required —
candle-core and candle-nn already provide everything needed (Tensor, Linear,
Embedding, VarBuilder, matmul, relu, softmax).

---

## Weight Loading

AlgZoo weights start as `.pth` on GCS. The workflow is:

```
1. Download .pth from GCS             (manual, one-time)
2. amn remember model.pth --to safetensors   (anamnesis)
3. StoicheiaRnn::load(config, path)       (candle-mi)
```

### Load Constructors

```rust
impl StoicheiaRnn {
    /// Load an AlgZoo RNN from a safetensors file.
    ///
    /// The safetensors file must contain:
    /// - `rnn.weight_ih_l0`: `[hidden_size, 1]`
    /// - `rnn.weight_hh_l0`: `[hidden_size, hidden_size]`
    /// - `linear.weight`: `[output_size, hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns `MIError::Model` if weights are missing or have wrong shapes.
    pub fn load(
        config: StoicheiaConfig,
        safetensors_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> { ... }
}

impl StoicheiaTransformer {
    /// Load an AlgZoo attention-only transformer from a safetensors file.
    ///
    /// # Errors
    ///
    /// Returns `MIError::Model` if weights are missing or have wrong shapes.
    pub fn load(
        config: StoicheiaConfig,
        safetensors_path: impl AsRef<Path>,
        device: &Device,
    ) -> Result<Self> { ... }
}
```

Use `VarBuilder::from_mmaped_safetensors` (with `mmap` feature) or
`VarBuilder::from_buffered_safetensors` (without) — same pattern as
`GenericTransformer::load()`.

---

## Task Functions (for Validation)

Pure functions that compute ground truth, used to verify model predictions:

```rust
/// Compute the position of the second-largest value in each row.
///
/// # Shapes
/// - `input`: `[batch, seq_len]` (continuous floats)
/// - returns: `[batch]` (position indices)
pub fn second_argmax(input: &Tensor) -> Result<Tensor> { ... }

/// Compute the position of the median value in each row.
pub fn argmedian(input: &Tensor) -> Result<Tensor> { ... }

/// Compute the median value of each row.
pub fn median(input: &Tensor) -> Result<Tensor> { ... }

/// Compute the longest cycle length in each row's permutation.
///
/// # Shapes
/// - `input`: `[batch, seq_len]` (integers in 0..seq_len)
/// - returns: `[batch]` (cycle lengths)
pub fn longest_cycle(input: &Tensor) -> Result<Tensor> { ... }
```

These are straightforward tensor operations: `arg_sort`, `gather`, loops for
cycle detection. Used in tests and examples, not in the forward pass.

---

## Testing Strategy

### Unit Tests (in `src/stoicheia/mod.rs`)

- `rnn_forward_shape` — verify output shape `[batch, 1, output_size]`
- `transformer_forward_shape` — same
- `rnn_hook_capture` — capture `Custom("rnn.hook_hidden.{t}")`, verify shape
- `transformer_hook_capture` — capture `AttnPattern(0)`, verify shape
- `rnn_zero_overhead` — empty HookSpec adds no allocations
- `config_from_task` — verify task → arch/output mapping

### Integration Tests

Cross-validate against Python reference outputs:

1. Generate reference data with Python:
   ```python
   model = zoo.zoo_2nd_argmax(hidden_size=2, seq_len=2)
   x = torch.tensor([[0.5, -0.3]])
   output = model(x)
   # Save x and output to JSON
   ```

2. In Rust, load the same safetensors weights, run the same input, compare
   output tensors to 6 decimal places (F32 precision).

### Fixture Files

```
tests/fixtures/stoicheia/
  rnn_2_2.safetensors         — M₂,₂ weights (10 parameters)
  rnn_2_2_reference.json      — input + expected output
  transformer_4_4.safetensors — small transformer weights
  transformer_4_4_reference.json
```

These are tiny (<1 KB each). Generate with a Python script, commit to repo.

---

## Example

```rust
// examples/stoicheia_inference.rs
use candle_mi::{StoicheiaConfig, StoicheiaRnn, StoicheiaTask, HookPoint, HookSpec, MIBackend};

fn main() -> candle_mi::Result<()> {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 16, 10);
    let model = StoicheiaRnn::load(config, "models/rnn_16_10.safetensors", &Device::Cpu)?;

    // Generate random input (continuous floats)
    let input = Tensor::randn(0.0_f32, 1.0, (1, 10), &Device::Cpu)?;

    // Forward with hook capture
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::Custom("rnn.hook_hidden.9".into()));  // final timestep
    hooks.capture(HookPoint::Custom("rnn.hook_final_state".into()));

    let cache = model.forward(&input, &hooks)?;
    let output = cache.output();  // [1, 1, 10] — distribution over 10 positions

    // Check accuracy against ground truth
    let target = candle_mi::stoicheia::tasks::second_argmax(&input)?;
    let predicted = output.squeeze(1)?.argmax(1)?;
    println!("Predicted: {:?}, Target: {:?}", predicted, target);

    Ok(())
}
```

---

## File Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `src/stoicheia/mod.rs` | **Create** | 350–450 |
| `src/stoicheia/config.rs` | **Create** | 100–130 |
| `src/stoicheia/tasks.rs` | **Create** | 80–100 |
| `src/lib.rs` | Edit | +8 (feature-gated exports) |
| `Cargo.toml` | Edit | +1 (feature flag) |
| `examples/stoicheia_inference.rs` | **Create** | 60–80 |
| `tests/stoicheia_cross_validation.rs` | **Create** | 80–120 |
| `tests/fixtures/stoicheia/` | **Create** | (binaries + JSON, <5 KB) |
| `tests/fixtures/stoicheia/generate_algzoo_fixtures.py` | **Create** | 40–50 |
| `CHANGELOG.md` | Edit | +8 |

**Total new Rust code**: ~590–760 lines + ~80–120 lines tests.

---

## Implementation Order

1. **`src/stoicheia/config.rs`** — enums and `StoicheiaConfig::from_task()`
2. **`src/stoicheia/mod.rs`** — `StoicheiaRnn` struct, weights, forward pass (no hooks yet)
3. **Fixture generation** — Python script → safetensors + reference JSON
4. **RNN cross-validation test** — verify output matches Python
5. **RNN hooks** — add Custom hook points to forward pass
6. **`StoicheiaTransformer`** — struct, weights, forward pass with hooks
7. **Transformer cross-validation test**
8. **`src/stoicheia/tasks.rs`** — ground truth functions
9. **`examples/stoicheia_inference.rs`** — runnable example
10. **Module wiring** — `lib.rs`, `Cargo.toml` feature flag
11. **CHANGELOG.md**

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Implement `MIBackend`? | Yes | Enables `HookSpec`/`HookCache` + future `MIModel` wrapping |
| `from_pretrained()` support? | No | AlgZoo models have no HF config; use explicit `load()` |
| Hook points | `Custom(String)` for RNN; existing variants for transformer | Avoids touching core `hooks.rs`; Phase B may add dedicated variants |
| Output shape | `[batch, 1, output_size]` | Matches `[batch, seq, vocab]` convention (seq=1 for final-position-only) |
| Feature gate | `stoicheia` (no extra deps) | Clean separation; compiles to nothing when disabled |
| CPU only? | CPU default, GPU works if available | Models are tiny — no GPU needed, but `Device` is configurable |
| Task functions | Separate `tasks.rs` | Needed for validation now, exhaustive MI later (Phase B) |
| Training loop | Out of scope (Phase A) | Pre-trained weights are the primary artifact |

---

## Prerequisite: AlgZoo Weight Conversion

### Step 1 — Clone the AlgZoo repository

```powershell
cd C:\Users\Eric JACOPIN\Documents\Code\Source
git clone https://github.com/alignment-research-center/alg-zoo.git
```

This gives us the Python source (architectures, tasks, training, loading) for
generating reference outputs and understanding weight layouts.

### Step 2 — Download ALL pre-trained weights from GCS

AlgZoo ships 400+ tiny `.pth` files on Google Cloud Storage. Each is under
10 KB, so the entire zoo is a few MB. Download everything:

```powershell
# pip install blobfile alg-zoo
python -c "
import blobfile as bf
import os

# Files are flat in the zoo directory, named:
#   {task}_{hidden_size}_{seq_len}_{n_seqs}_{seed}.pth
base = 'gs://arc-ml-public/alg/zoo'
os.makedirs('algzoo_weights', exist_ok=True)

for path in bf.listdir(base):
    if path.endswith('.pth'):
        name = os.path.basename(path)
        local = f'algzoo_weights/{name}'
        if not os.path.exists(local):
            with bf.BlobFile(path, 'rb') as src:
                with open(local, 'wb') as dst:
                    dst.write(src.read())
            print(f'  {local}')
"
```

### Step 3 — Bulk-convert ALL weights with anamnesis

Convert every `.pth` to `.safetensors` into a separate sibling directory, keeping
originals untouched. The conversion script is saved alongside the output for
reproducibility.

**Directory layout:**

```
C:\Users\Eric JACOPIN\Documents\Data\
  algzoo_weights\         ← originals (.pth + .jsonl), never modified
  algzoo_safetensors\     ← converted .safetensors + conversion script
    convert_all.py        ← the script that produced these files
    *.safetensors         ← one per .pth file, same base name
```

**Conversion script** (`algzoo_safetensors/convert_all.py`):

```python
"""Bulk-convert AlgZoo .pth weights to .safetensors via anamnesis CLI."""
import subprocess, os, sys

src = r"C:\Users\Eric JACOPIN\Documents\Data\algzoo_weights"
dst = r"C:\Users\Eric JACOPIN\Documents\Data\algzoo_safetensors"
os.makedirs(dst, exist_ok=True)

pth_files = sorted(f for f in os.listdir(src) if f.endswith(".pth"))
total = len(pth_files)
converted = 0
failed = []

for i, name in enumerate(pth_files, 1):
    st_name = name.replace(".pth", ".safetensors")
    st_path = os.path.join(dst, st_name)
    if os.path.exists(st_path):
        converted += 1
        continue
    pth_path = os.path.join(src, name)
    result = subprocess.run(
        ["amn", "remember", pth_path, "--to", "safetensors", "--output", st_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        converted += 1
    else:
        failed.append((name, result.stderr.strip()))
    if i % 500 == 0:
        print(f"  {i}/{total} processed...")

print(f"Done: {converted}/{total} converted")
if failed:
    print(f"FAILED ({len(failed)}):")
    for name, err in failed:
        print(f"  {name}: {err}")
```

```powershell
# Run the conversion
python algzoo_safetensors\convert_all.py

# Verify: count .pth vs .safetensors (should match)
$pth = (Get-ChildItem $env:USERPROFILE\Documents\Data\algzoo_weights -Filter *.pth).Count
$st  = (Get-ChildItem $env:USERPROFILE\Documents\Data\algzoo_safetensors -Filter *.safetensors).Count
Write-Host "Converted $st / $pth files"
```

**Why convert all?** Two benefits:
1. **anamnesis stress-test** — exercises the pickle VM on 6,960 real `.pth` files
   with varying hidden sizes (2–32), sequence lengths (2–10), and both RNN and
   transformer architectures. Any pickle edge case will surface here.
2. **candle-mi test corpus** — the full zoo enables systematic accuracy sweeps
   across all `(task, hidden_size, seq_len)` configurations, not just spot checks.

### Step 4 — Generate Python reference outputs

Use the cloned AlgZoo repo (`C:\Users\Eric JACOPIN\Documents\Code\Source\alg-zoo`)
to produce reference inputs + outputs for cross-validation. The reference script
is saved alongside the output for reproducibility.

**Directory layout:**

```
C:\Users\Eric JACOPIN\Documents\Data\
  algzoo_weights\         ← originals (.pth + .jsonl)
  algzoo_safetensors\     ← converted .safetensors + convert_all.py
  algzoo_reference\       ← Python reference outputs + generation script
    generate_reference.py ← the script that produced these files
    ref_*.json            ← input + expected output per model config
```

**Reference script** (`algzoo_reference/generate_reference.py`):

```python
"""Generate Python reference outputs for candle-mi cross-validation.

Loads AlgZoo models via the alg_zoo Python package, runs inference on
fixed-seed inputs, and saves input + output tensors as JSON. These are
compared against candle-mi's forward pass on the same .safetensors weights.

Usage:
    pip install -e C:/Users/Eric JACOPIN/Documents/Code/Source/alg-zoo
    python generate_reference.py
"""
import torch, json, os

# alg_zoo must be installed: pip install -e <path-to-alg-zoo>
from alg_zoo import zoo

DST = r"C:\Users\Eric JACOPIN\Documents\Data\algzoo_reference"
os.makedirs(DST, exist_ok=True)

# Reference models covering both architectures, all 4 tasks,
# distribution + scalar output, and a range of sizes
configs = [
    # (task_name, loader_fn, hidden_size, seq_len)
    ("2nd_argmax", zoo.zoo_2nd_argmax, 2, 2),      # RNN, 10 params (M_2_2)
    ("2nd_argmax", zoo.zoo_2nd_argmax, 4, 3),       # RNN, 32 params (M_4_3)
    ("2nd_argmax", zoo.zoo_2nd_argmax, 16, 10),     # RNN, 432 params (M_16_10, blog)
    ("argmedian",  zoo.zoo_argmedian,  4, 3),        # RNN, different task
    ("argmedian",  zoo.zoo_argmedian,  8, 5),        # RNN, medium
    ("median",     zoo.zoo_median,     4, 3),        # RNN, scalar output
    ("median",     zoo.zoo_median,     8, 5),        # RNN, scalar, medium
    ("longest_cycle", zoo.zoo_longest_cycle, 4, 4),  # Transformer, small
    ("longest_cycle", zoo.zoo_longest_cycle, 6, 4),  # Transformer, medium
]

torch.manual_seed(42)
for task_name, loader, h, n in configs:
    model = loader(hidden_size=h, seq_len=n)
    if task_name == "longest_cycle":
        x = torch.randint(0, n, (4, n))
    else:
        x = torch.randn(4, n)
    with torch.no_grad():
        output = model(x)
    ref = {
        "task": task_name,
        "hidden_size": h,
        "seq_len": n,
        "input": x.tolist(),
        "output": output.tolist(),
    }
    fname = f"ref_{task_name}_h{h}_n{n}.json"
    path = os.path.join(DST, fname)
    with open(path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"  {fname}  ({output.shape})")

print(f"\nDone: {len(configs)} reference files in {DST}")
```

```powershell
# Install alg-zoo package (editable, from cloned repo)
pip install -e "C:\Users\Eric JACOPIN\Documents\Code\Source\alg-zoo"

# Generate reference outputs
python "C:\Users\Eric JACOPIN\Documents\Data\algzoo_reference\generate_reference.py"
```

The JSON files record fixed-seed inputs and Python's exact outputs. For
cross-validation, candle-mi loads the same `.safetensors` weights, runs the
same inputs, and compares output tensors to 6 decimal places (F32 precision).

A subset of these JSON files + the corresponding `.safetensors` weights are
committed as test fixtures in `tests/fixtures/stoicheia/` (kept small — 3–5
configs, under 10 KB total).

---

## What Phase B Will Build On Top

Phase A provides the foundation. Phase B adds the MI tooling:

- **Exhaustive ablation**: zero each neuron, measure accuracy change (uses hooks)
- **Neuron probing**: structured inputs → classify neuron function (uses forward)
- **Piecewise-linear enumeration**: map ReLU activation regions (uses pre-activation hooks)
- **Surprise accounting**: ARC's information-theoretic metric (uses tasks + forward)
- **Training dynamics**: load checkpoint series, track feature emergence (uses load)

All of these depend on Phase A's `forward()` + hooks + task functions.

### Fast-path RNN kernel (SIMD, no candle tensors)

Phase A benchmarks (10K samples, release, CPU) revealed that candle's
per-tensor-operation overhead dominates on tiny models:

| Model | Params | Rust (candle) | Python (`PyTorch`) | Ratio |
|-------|--------|--------------|-------------------|-------|
| M₂,₂ | 10 | 86ms | 3.4ms | 25x |
| M₁₆,₁₀ | 432 | 111ms | 6.1ms | 18x |
| Transformer h4n4 | 176 | 110ms | 13ms | 8.5x |

The ratio improves with model size (25x→18x→8.5x) confirming the bottleneck
is **fixed per-operation overhead** (shape validation, device dispatch, storage
allocation), not compute. `PyTorch`'s `nn.RNN` fuses the entire timestep loop
into a single C++ kernel; candle executes 4 separate tensor ops per timestep.

**Phase B should add a raw-`f32` fast path** for bulk forward passes without
hooks (surprise accounting, perturbation sweeps). Design:

1. **Dual-path forward**: `forward()` (candle tensors, hooks) stays for MI
   analysis. A new `forward_fast(&[f32], &mut [f32])` operates on raw slices
   — no `Tensor` objects, no heap allocation per timestep.

2. **SIMD-friendly loop structure** (following anamnesis patterns):
   - Extract weights as contiguous `&[f32]` slices at load time
   - Process batch in `chunks_exact(4)` or `chunks_exact(8)` for AVX2
   - RNN cell body: `h_new = relu(x_t * W_ih + W_hh @ h_prev)` as scalar
     ops that auto-vectorize (no branches, contiguous slices, hoisted invariants)
   - Verify with `cargo-show-asm` that the inner loop vectorizes
   - `#![forbid(unsafe_code)]` — auto-vectorization only, no explicit intrinsics

3. **AlgZoo hidden sizes are small enough for register-level tricks**:
   - H=2: entire hidden state = 2 floats, fits in a single SIMD lane
   - H=4: 4 floats = one SSE register
   - H=8: 8 floats = one AVX2 register
   - H=16: 2 AVX2 registers
   - H=32: 4 AVX2 registers
   - For H ≤ 8, the "matmul" `W_hh @ h` can be unrolled entirely — no loop,
     just explicit multiply-adds. The compiler will vectorize this.

4. **Expected performance**: near `PyTorch` speed or faster. `PyTorch`'s
   `nn.RNN` still goes through Python→C++ dispatch; a raw Rust loop with
   auto-vectorized SIMD has no dispatch overhead at all. The anamnesis
   dequantization kernels achieve 2.7–54x over `PyTorch` CPU on similar
   "tight loop over `f32` slices" workloads.

5. **Use cases in Phase B**:
   - **Surprise accounting**: perturb weights (add noise), measure accuracy
     change. Thousands of forward passes, no hooks needed. The fast path
     makes this seconds instead of minutes on M₁₆,₁₀.
   - **Piecewise-linear enumeration**: systematic input sweeps to map ReLU
     activation regions. Millions of forward passes, pure classification.
   - **Accuracy sweeps**: evaluate accuracy across all `(hidden_size, seq_len)`
     configs in the zoo. 6,960 models × 10K samples each.

6. **Scope boundary**: the fast path is **inference only** — no hooks, no
   interventions, no gradient. MI analysis that needs activation capture
   uses the candle path. The two paths share the same weight storage and
   `StoicheiaConfig`; only the forward computation differs.
