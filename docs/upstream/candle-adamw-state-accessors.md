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

Two tests added to `candle-nn/tests/optim.rs`, both driving `loss = sum(w^2)`
from `w = [1, 2, 3]` with `ParamsAdamW::default()`.

`adamw_resume_matches_uninterrupted_run` runs six steps straight, then runs
three, checkpoints `step_t` and `moments()`, restores both into a **fresh**
`AdamW` as a new process would, and runs the remaining three:

```
uninterrupted 6 steps   : [0.99394083, 1.9938803, 2.9938202]
checkpointed 3 + 3      : [0.99394083, 1.9938803, 2.9938202]
```

Bit-identical, hence `assert_eq!` rather than a tolerance.

`adamw_losing_the_moments_is_visible` is the control, so the first test cannot
pass vacuously. It performs the same resume and restores the step counter, but
**not** the moments, which isolates the moments as the variable:

```
resumed WITHOUT moments : [0.99480873, 1.9947485, 2.9946885]
```

Roughly 8.7e-4 away from the uninterrupted run, on a convex toy objective that
converges in a handful of steps and therefore gives the moments very little
history to carry. A real staged run diverges considerably further; the point
here is only that the loss of state is detectable at all, so the resume test is
testing something.

Full run of the existing test target, with candle's own optimizer tests included
to show that nothing regressed:

```
running 6 tests
test adamw_losing_the_moments_is_visible ... ok
test adamw_resume_matches_uninterrupted_run ... ok
test sgd_optim ... ok
test adamw_linear_regression_varmap ... ok
test adamw_linear_regression ... ok
test sgd_linear_regression ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
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
- Verified against current `main`: `cargo test -p candle-nn --test optim` passes
  6/6 including the four pre-existing optimizer tests, `cargo fmt` is clean, and
  `cargo clippy -p candle-nn --tests -- -D warnings` is clean.
- `candle-nn/src/optim.rs` is byte-identical across 0.9.2, 0.11.0 and current
  `main` (201 lines, `diff` returns nothing between any pair), so this is settled
  API rather than something in flux.
- **Checked for overlap before filing**, since two limits on one path is worse
  than none: no open issue or PR touches optimizer state. The nearest items are
  #695 (optimizer trait), #685 (scheduler) and #402 (parameter grouping), all
  closed in 2023 and none about saving or restoring state.
- Context: this would replace a transcribed copy of `AdamW` that we now **ship in
  a published crate**: [candle-mi
  0.1.21](https://crates.io/crates/candle-mi), behind a default-off `training`
  feature, as
  [`optim::AdamW`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/v0.1.21/src/optim.rs).
  The update rule is taken verbatim from `candle-nn` 0.11.0 with attribution and
  the same licence; only the state's ownership changes. It is held to candle's own
  trajectory by
  [`tests/validate_optim_parity.rs`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/v0.1.21/tests/validate_optim_parity.rs)
  at `< 1e-6` over 1, 2, 17 and 120 steps, with a resume test and a power control
  showing that silently dropping the moments *would* be visible. Those run in CI,
  not behind an `#[ignore]`. It has driven three real resume cycles on a 40-epoch
  run.

  That the copy is now published rather than private is the honest argument for
  this patch: a second implementation of candle's own update rule is on crates.io
  because fifteen lines of accessor do not exist upstream, and anyone else staging
  a training run will write a third. If these accessors land we delete ours and
  call stock `AdamW`. There is no urgency on our side, since the copy works.

## Thread record

**2026-08-01.** Filed as [candle#3819](https://github.com/huggingface/candle/pull/3819),
branch `adamw-state-accessors`, commit `9e2f1311` on base `6e823a43`.

**2026-08-31.** Both workflow runs queued at filing time,
[30696257295](https://github.com/huggingface/candle/actions/runs/30696257295)
and [30696257317](https://github.com/huggingface/candle/actions/runs/30696257317),
expired unapproved after exactly thirty days and were stamped
`conclusion: failure` with zero jobs executed. Nothing had compiled and nothing
had run, so the red X carried no signal about the patch, but it reads on the PR
page as a broken change.

Rebased `9e2f1311` onto `638a819a` (then current `main`), giving `67ee82d4`.
`git patch-id --stable` is `e7fabbe3` before and after, and neither
`candle-nn/src/optim.rs` nor `candle-nn/tests/optim.rs` was touched by the twenty
intervening commits, so the replay was mechanical and the patch is still the same
additive +80/-0.

Re-verified on the rebased tree at rustc 1.98.0, mirroring the lanes in
`rust-ci.yml`: `cargo fmt --all -- --check` clean, `cargo clippy --workspace
--tests --examples --benches -- -D warnings` clean, `cargo test -p candle-nn`
13 targets with 60 passed and 0 failed, and the `optim` target 6/6 including
candle's four pre-existing optimizer tests. The rustc 1.98.0 check mattered:
stable had moved on since filing, and a new lint on the `impl Iterator`
signature would have been the plausible regression. None fired.

Force-pushed and commented, posted as
[issuecomment-5479119668](https://github.com/huggingface/candle/pull/3819#issuecomment-5479119668).
The comment states only what the runs actually show, offers to narrow the surface
to `step_t` / `set_step_t` if `moments()` is contentious, and does not speculate
about which GitHub setting imposed the gate.

The push queued two fresh runs, both `action_required`, so the PR now awaits one
maintainer click rather than displaying an expired failure.

### The lesson, which generalises to every fork PR we file

A fork pull request's workflow run needs a maintainer to approve it, and a fork
cannot approve its own. If nobody clicks within thirty days, GitHub expires the
run and stamps it `failure`. The PR then advertises a broken patch to exactly the
reviewers whose attention it is competing for, and the author is never notified
that the cause was an unclicked button rather than a real defect.

Given that upstream reacts in weeks rather than days, this is not an edge case
for us, it is the default outcome. Worth checking the approval state of every
open fork PR periodically, keyed on the run's `created_at` plus thirty days,
rather than waiting for the red X to appear.
