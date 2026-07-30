# Add state accessors to `AdamW` so a training run can be resumed

**Title for the PR:** `candle-nn`: add `AdamW::step_t`, `set_step_t` and `moments` so training runs can be checkpointed

## What this adds

Three methods on `candle_nn::AdamW`. About fifteen lines, no behaviour change, and
no change to any existing signature:

```rust
impl AdamW {
    /// The number of steps taken so far.
    pub fn step_t(&self) -> usize {
        self.step_t
    }

    /// Overwrite the step counter, e.g. when resuming from a checkpoint.
    pub fn set_step_t(&mut self, step_t: usize) {
        self.step_t = step_t
    }

    /// Iterate over `(parameter, first_moment, second_moment)` for each tracked
    /// variable, in the order the optimizer holds them.
    pub fn moments(&self) -> impl Iterator<Item = (&Var, &Var, &Var)> + '_ {
        self.vars
            .iter()
            .map(|v| (&v.var, &v.first_moment, &v.second_moment))
    }
}
```

## Why

`VarMap::save` and `VarMap::load` serialize model weights. They do not serialize
optimizer state, and with `AdamW` there is currently no way to reach it.

That state is not incidental. Adam keeps two running averages per parameter, and
divides them by a bias-correction factor that depends on how many steps have been
taken, `1 - beta^t`. On the first step that factor is far from 1, which is
deliberate: it compensates for moments that started at zero and have not yet
warmed up.

So if you stop a run and start a new process, the moments reset to zero and `t`
resets to 1. Adam then applies a full warm-up correction to a model that is
already thousands of steps in. The optimizer behaves as though training had just
begun.

For a single uninterrupted run this never comes up. It comes up as soon as a run
is staged. Ours is a 40-epoch training that takes about 21 hours and runs in
roughly 5-hour stages, so it crosses a process boundary three times. Each
boundary was leaving a discontinuity in the trajectory, which matters especially
when the analysis reads a quantity off consecutive checkpoints, because then the
artefact lands exactly where the measurement is taken.

## The state is already there, it is just not reachable

`AdamW` holds everything needed:

```rust
#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}
```

`VarAdamW` is declared without `pub` at `candle-nn/src/optim.rs:104`, and `step_t`
is a private field. `AdamW`'s entire public surface today is `new_lr`, `params`
and `set_params`, plus the `Optimizer` trait methods.

This reads more like an accessor nobody needed yet than a deliberate
encapsulation boundary, because `SGD`, in the same file at `src/optim.rs:73`,
already exposes its variables:

```rust
impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }
    ...
}
```

`src/optim.rs` is also stable. It is 201 lines and byte-identical between 0.9.2
and 0.11.0, with `diff` of the two returning nothing. So this is settled code
rather than something in flux where an added method would be in the way.

## One detail that keeps the patch this small

The three accessors are read-only, yet they suffice to *restore* state as well as
save it. That is because `Var` has interior mutability: `Var::set` takes `&self`,
not `&mut self` (`candle-core/src/variable.rs:130`). `AdamW::step` already relies
on this. Inside a `for var in self.vars.iter()` loop over immutable references, it
calls `m.set(&next_m)?`.

So a restore needs no `&mut` and no per-moment setter:

```rust
for ((_, m, v), (saved_m, saved_v)) in opt.moments().zip(saved.iter()) {
    m.set(saved_m)?;
    v.set(saved_v)?;
}
```

Only the step counter needs a real setter, since `usize` has no such trick.

## Evidence that it works

Two tests added to `candle-nn/tests/optim.rs`. The first checkpoints an optimizer
mid-run, restores into a *fresh* `AdamW` as a new process would, and compares
against an uninterrupted run:

```
uninterrupted 6 steps  : [0.99394083, 1.9938803, 2.9938202]
checkpointed 3 + 3     : [0.99394083, 1.9938803, 2.9938202]
```

Bit-identical.

The second is a control, so the first cannot pass vacuously. It performs the same
resume *without* restoring the moments:

```
resumed WITHOUT moments: [0.9939403, 1.99388, 2.9938202]
```

Different, as it must be. In fairness the difference here is small, because the
toy objective (`loss = sum(w^2)`) converges almost immediately and leaves the
moments little history to carry. On a real run the gap is far larger. We mention
it so that the small numbers above are not read as evidence that the underlying
problem is small.

Full run of the existing test target, with candle's own optimizer tests included
to show that nothing regressed:

```
running 6 tests
test resume_accessors::resumed_run_matches_uninterrupted_run ... ok
test resume_accessors::losing_the_moments_is_visible ... ok
test sgd_optim ... ok
test adamw_linear_regression_varmap ... ok
test adamw_linear_regression ... ok
test sgd_linear_regression ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## One caveat worth documenting

`AdamW::new` filters the variables it is given:

```rust
let vars = vars.into_iter().filter(|var| var.dtype().is_float()).map(...)
```

So `moments()` can yield fewer entries than the `Vec<Var>` originally passed in,
if any of them were non-float. Anyone saving by position should iterate
`moments()` on both the save and the restore side, rather than zipping it against
`varmap.all_vars()`, which would silently desynchronize.

Happy to add that as a doc note on `moments()` if you would like it in the patch.

## Alternatives considered

**A `state()` and `load_state()` pair returning a serializable struct.** More
convenient, but it puts a serialization format into `candle-nn`'s API and invites
questions about versioning that format. The accessors leave the choice to the
caller, who already has `safetensors` to hand.

**Making `VarAdamW` public.** That exposes a field layout which then cannot
change. The iterator returns plain `&Var` triples and keeps the struct private.

## Notes

- Purely additive. No existing signature changes, so it is a minor-version item.
- Verified with `cargo check --lib` and `cargo test --test optim` against
  `candle-nn` 0.11.0.
- Context: this would replace a transcribed copy of `AdamW` that we maintain
  downstream, with the update rule taken verbatim from `candle-nn` 0.11.0 and held
  to candle's own trajectory by a parity test at `< 1e-6` over 1, 2, 17 and 120
  steps. It has driven three real resume cycles. If these accessors land we delete
  the copy and call stock `AdamW`. There is no urgency on our side, since the copy
  works; the point of the PR is that the next person should not have to write one.
