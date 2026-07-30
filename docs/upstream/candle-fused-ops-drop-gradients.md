# Fused ops in `candle-nn` silently return non-differentiable tensors

**Title for the issue:** Fused ops in `candle-nn` silently return non-differentiable tensors (`softmax_last_dim`, `rms_norm`, `layer_norm`, `rope*`, `sdpa`)

**Labels to suggest:** `bug`, `documentation`

## Summary

Seven public functions in `candle-nn` return a tensor that has no entry in the
autograd graph. If such a tensor lies on the path from a parameter to the loss,
`backward()` still returns `Ok`, and the gradient for that parameter is simply
absent from the returned `GradStore`.

Nothing warns you: no error, no panic, no log line, and none of the seven
functions carries a doc comment mentioning it.

We hit this while training a small model with `candle-nn`. The training loop ran,
the loss decreased, and it took a while to discover that **1 of 29 parameters**
was actually being updated. The loss was falling because the one parameter that
did receive a gradient was the output head, which on its own can learn the
marginal token distribution. It plateaued there, which looked like an ordinary
optimization problem rather than a bug.

## Workaround available today

Putting this first, since it is what a reader arriving from a search engine
needs. For five of the seven, a differentiable equivalent already exists in
`candle-nn` and is a drop-in substitution:

| if you use this | and you need gradients, use this instead |
|---|---|
| `ops::softmax_last_dim(xs)` | `ops::softmax(xs, D::Minus1)` |
| `ops::rms_norm(xs, alpha, eps)` | `ops::rms_norm_slow(xs, alpha, eps)` |
| `ops::layer_norm(xs, alpha, beta, eps)` | `ops::layer_norm_slow(xs, alpha, beta, eps)` |
| `rotary_emb::rope_i(xs, cos, sin)` | `rotary_emb::rope_i_slow(xs, cos, sin)` |
| `rotary_emb::rope(xs, cos, sin)` | `rotary_emb::rope_slow(xs, cos, sin)` |
| `rotary_emb::rope_thd(xs, cos, sin)` | no equivalent exists, compose by hand |
| `ops::sdpa(...)` | no equivalent exists, compose by hand |

If your model is used for both inference and training, note that you only need
the substitution on the training path. `Tensor::track_op()` tells you which one
you are on, and the last section of this issue shows how.

## Why this is easy to miss

There are three separate layers of silence, and they compound.

The first is that `backward()` cannot fail here. It walks the graph backwards
from the loss. When it reaches one of these tensors, that tensor has no recorded
operation, so as far as the traversal is concerned it is a *leaf*: an input,
something with nothing behind it. A leaf is a perfectly ordinary thing to find,
so there is no basis on which to raise an error.

The second is that the gradient's absence is not an error either. `GradStore` is
a map, and a missing parameter is just a key that is not there. Both optimizers
in `candle-nn` skip such parameters without complaint: `SGD` does
`if let Some(grad) = grads.get(var)` at `src/optim.rs:60`, and `AdamW` does
`if let Some(g) = grads.get(theta)` at `src/optim.rs:165`. That is the right
behaviour for an optimizer, but it means the information is discarded at the last
point where it could have been noticed.

The third is that the loss usually still goes down. Whichever parameters remain
connected keep learning, so the run produces a plausible curve. This is what
makes the failure expensive. It does not look like a bug, it looks like a model
that needs tuning.

## Reproducer

Verified as a standalone project with exactly these two dependencies:

```toml
[dependencies]
candle-core = "0.11.0"
candle-nn = "0.11.0"
```

```rust
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    println!("{:<34} {:>10} {:>12}", "path", "backward()", "got grad?");

    // 1. rms_norm: `alpha` is a trainable parameter of the op itself.
    let alpha = vb.get(8, "alpha")?;
    let x = Tensor::randn(0f32, 1f32, (2, 8), &dev)?;
    let g = candle_nn::ops::rms_norm(&x, &alpha, 1e-5)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "ops::rms_norm (fused)", "Ok", g.get(&alpha).is_some());
    let g = candle_nn::ops::rms_norm_slow(&x, &alpha, 1e-5)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "ops::rms_norm_slow", "Ok", g.get(&alpha).is_some());

    // 2. layer_norm: `alpha` and `beta` are trainable parameters.
    let beta = vb.get(8, "beta")?;
    let g = candle_nn::ops::layer_norm(&x, &alpha, &beta, 1e-5)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "ops::layer_norm (fused)", "Ok", g.get(&beta).is_some());
    let g = candle_nn::ops::layer_norm_slow(&x, &alpha, &beta, 1e-5)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "ops::layer_norm_slow", "Ok", g.get(&beta).is_some());

    // 3. softmax_last_dim: no parameters of its own, but in a transformer it
    //    sits between q_proj/k_proj and the loss, so gradients must pass through.
    let w = vb.get((2, 8), "w")?;
    let g = candle_nn::ops::softmax_last_dim(&w)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "ops::softmax_last_dim (fused)", "Ok", g.get(&w).is_some());
    let g = candle_nn::ops::softmax(&w, 1)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "ops::softmax (composed)", "Ok", g.get(&w).is_some());

    // 4. rope: likewise parameter-free, but it sits on the q_proj/k_proj path.
    let q = vb.get((1, 1, 4, 8), "q")?;
    let cos = Tensor::ones((4, 4), DType::F32, &dev)?;
    let sin = Tensor::zeros((4, 4), DType::F32, &dev)?;
    let g = candle_nn::rotary_emb::rope(&q, &cos, &sin)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "rotary_emb::rope (fused)", "Ok", g.get(&q).is_some());
    let g = candle_nn::rotary_emb::rope_slow(&q, &cos, &sin)?.sum_all()?.backward()?;
    println!("{:<34} {:>10} {:>12}", "rotary_emb::rope_slow", "Ok", g.get(&q).is_some());
    Ok(())
}
```

Output, CPU, `candle-nn` 0.11.0:

```
path                               backward()    got grad?
ops::rms_norm (fused)                      Ok        false
ops::rms_norm_slow                         Ok         true
ops::layer_norm (fused)                    Ok        false
ops::layer_norm_slow                       Ok         true
ops::softmax_last_dim (fused)              Ok        false
ops::softmax (composed)                    Ok         true
rotary_emb::rope (fused)                   Ok        false
rotary_emb::rope_slow                      Ok         true
```

Every row returns `Ok`. Only the `got grad?` column distinguishes a working path
from a broken one, and nothing in the API surfaces that column.

## The mechanism

All seven functions end in a call to `apply_op1_no_bwd`, `apply_op2_no_bwd` or
`apply_op3_no_bwd`. In `candle-core` 0.11.0, `src/custom_op.rs:156`:

```rust
/// Applies a unary custom op without backward support
pub fn apply_op1_no_bwd<C: CustomOp1>(&self, c: &C) -> Result<Self> {
    let (storage, shape) = self.storage().apply_op1(self.layout(), c)?;
    Ok(from_storage(storage, shape, BackpropOp::none(), false))
}
```

`BackpropOp::none()` is what makes the result a leaf. It records no producing
operation, so the graph has no edge from this tensor back to its inputs. The
`false` is the `is_variable` argument (`src/tensor.rs:159`), so the result is not
treated as a parameter either. It is a tensor with nothing in front of it and
nothing behind it.

Worth stressing: **`candle-core` is not at fault here.** These three functions
are named `_no_bwd`, and each carries a doc comment saying "without backward
support". They do exactly what they advertise.

The gap is entirely in `candle-nn`'s public wrappers. Those are the ones a user
calls, they do not carry `_no_bwd` in their names, and they have no doc comments
at all.

## The affected functions

In `candle-nn` 0.11.0:

| function | defined at | `no_bwd` call at |
|---|---|---|
| `ops::softmax_last_dim` | `src/ops.rs:437` | `src/ops.rs:438` |
| `ops::rms_norm` | `src/ops.rs:674` | `src/ops.rs:684` |
| `ops::layer_norm` | `src/ops.rs:932` | `src/ops.rs:944` |
| `ops::sdpa` | `src/ops.rs:1308` | `src/ops.rs:1317` |
| `rotary_emb::rope_i` | `src/rotary_emb.rs:262` | `src/rotary_emb.rs:287` |
| `rotary_emb::rope` | `src/rotary_emb.rs:555` | `src/rotary_emb.rs:580` |
| `rotary_emb::rope_thd` | `src/rotary_emb.rs:830` | `src/rotary_emb.rs:855` |

None of these seven has a doc comment. We checked:

```
$ grep -rn "///.*\(gradient\|backward\|backprop\|differentiab\)" src/ops.rs src/rotary_emb.rs
(no output)
```

This is long-standing rather than a recent regression. All seven sites are
present in 0.9.2 as well, at `ops.rs:430`, `645`, `901`, `1266` and
`rotary_emb.rs:266`, `537`, `808`.

A note on the scope of our evidence. We measured the first three plus `rope`
directly, in the reproducer above. `sdpa`, `rope_i` and `rope_thd` we have **not**
run. They are included because they use the identical `apply_op*_no_bwd`
mechanism, which is visible in the source at the lines given. We would rather
flag them than quietly leave them out, but they are inference from the mechanism
rather than measurement, and we did not want to blur the difference.

## `candle-nn` already ships most of the differentiable twins

This is the part that suggests a cheap fix. For five of the seven, a composed and
differentiable equivalent already exists in the same file:

| fused (no gradient) | differentiable twin |
|---|---|
| `ops::rms_norm` (`src/ops.rs:674`) | `ops::rms_norm_slow` (`src/ops.rs:661`) |
| `ops::layer_norm` (`src/ops.rs:932`) | `ops::layer_norm_slow` (`src/ops.rs:912`) |
| `ops::softmax_last_dim` (`src/ops.rs:437`) | `ops::softmax` (`src/ops.rs:22`) |
| `rotary_emb::rope_i` (`src/rotary_emb.rs:262`) | `rotary_emb::rope_i_slow` (`src/rotary_emb.rs:290`) |
| `rotary_emb::rope` (`src/rotary_emb.rs:555`) | `rotary_emb::rope_slow` (`src/rotary_emb.rs:590`) |

So the knowledge is already in the codebase. What is missing is any signal, at
the point of use, that the choice between them matters for something other than
speed. The `_slow` suffix reads as a performance note, and a user reasonably
picks the fast one.

Two have no differentiable alternative at all: `rope_thd` and `sdpa`.

## Why documenting alone is not quite enough

Documentation is worth doing immediately and would have saved us, so please
consider it regardless of anything else here. But it leaves a sharp edge.

Consider a library that builds a transformer meant for both inference and
training. With docs alone it must either pick one path at construction time or
thread a flag through every layer. The fused kernels are the right choice for
inference, the composed ones the only correct choice for training, and the author
has to build that dispatch themselves.

A clean signal for it already exists in `candle-core`, at `src/tensor.rs:592`:

```rust
pub fn track_op(&self) -> bool {
    self.is_variable || self.op.is_some()
}
```

That is exactly the condition which decides whether a gradient will be needed, so
the fused and composed paths can be selected per call with no user-visible flag.

## Suggested fixes, cheapest first

**1. Document the seven functions, and cross-reference the twins.** One line
each: this returns a tensor with no backward, use `<twin>` if you need gradients.
Nearly free, and it turns a silent failure into a discoverable one.

**2. Dispatch on `track_op()` inside `candle-nn`.** Take the fused kernel when
the input is not tracked, the composed form when it is. Correct by default, no
API change, no cost to inference.

We did exactly this downstream, and can offer one measurement. On our workload
the composed forward ran at **0.70x the time of the fused one**, meaning the
composed path was *faster*. We had expected a penalty and found none. That is a
single workload on a single GPU and we would not generalize from it, but it does
suggest the performance objection to this option is worth measuring before it is
assumed.

**3. Implement `CustomOp::bwd()` for each kernel.** The best end state: the fused
kernel, a correct gradient, and one node on the tape instead of several.
Considerably more work than the other two, and neither of them blocks it.

If a documentation PR along the lines of option 1 would be welcome, we are glad
to put one up; just say so and we will. We have the option-2 dispatch working
downstream already and can share that code either way.

## Environment

- `candle-core` and `candle-nn` 0.11.0, with the same seven sites present in 0.9.2
- The reproducer is CPU-only and needs no GPU
- Developed on Windows 11, but nothing here is platform-specific: the behaviour
  is in graph construction, not in any backend kernel
