# `Replace` is whole-tensor only, so activation patching has to splice by hand

> **Status: IMPLEMENTED** (2026-09-02), unreleased. Shipped as
> `Intervention::PatchAt { position, value }`, accepting `[hidden]`,
> `[1, 1, hidden]` and `[batch, 1, hidden]` values, written with
> `Tensor::slice_scatter` rather than the three-way `cat` sketched below (one
> call, on device, and gradient-tracked). Rejection is `MIError::Intervention`,
> as asked.
>
> **The dim-1 hazard is closed the preferred way.** `apply_intervention` now
> takes the `HookPoint`, and the policy is public as
> `HookPoint::accepts_positional_patch`, so a caller can ask before building a
> spec instead of finding out from a failed forward. A table test pins the
> answer for every variant against `one_of_each_variant()`, so a new
> `HookPoint` has to be given an explicit answer rather than inheriting one.
>
> **All four in-repo copies are gone**, and with them the extra forward each
> example ran only to store the recipient's own activations: the recipient pass
> now runs with no captures at all in three of the four, and keeps only its
> `AttnPattern` captures in `factual_routing`.
>
> **Deliberately not done:** key/value patching and positional ranges, both as
> recorded under "Not asked for" below. The K/V restriction is an explicit
> documented error, never a silent no-op, and
> [`design/patch-at-position.md`](../../design/patch-at-position.md) records the
> pre-broadcast constraint so a later head-indexed variant does not have to
> rediscover it.

**Date:** September 2, 2026
**Source:** askesis `canvas` leg, registering Measurement `G` substeps 4 to 6, the cross-size
activation-patching probe (`reference/canvas/docs/open-measurements.md`)
**Affected area:** `src/hooks.rs` (`Intervention`), `src/steering/contrastive.rs`
(`position_delta`), and the four `examples/` copies listed below
**Severity:** Ergonomics. **No defect, nothing blocked.** The probe is implementable today.
This is one missing variant that would remove hand-written tensor surgery from the one place a
probe can least afford it.

---

## The use

`G` asks whether a computation the model performs at `n` = 6 is **absent** at `n` = 7 or
**present and overridden**. The instrument is activation patching: take the residual stream of a
solved `n` = 6 decode at one grid position, write it into the failing `n` = 7 decode at the SAME
absolute position (the grid is fixed-offset, which is what makes two sizes comparable at all),
re-run the remaining layers, and read whether the correct token overtakes the emitted one.

So the operation wanted is: **at hook point `ResidPost(L)`, replace position `p` and leave every
other position untouched.**

## What the API offers

`Intervention` (`src/hooks.rs:212-234`) has five variants. Two are relevant:

- `Replace(Tensor)`, documented as *"Replace the tensor entirely with a provided value."*
  Whole-tensor, no position. `apply_intervention` implements it as `Ok(replacement.clone())`: the
  incoming activation is discarded unread, and its shape is never checked against the
  replacement's.
- `Add(Tensor)`, whose doc already anticipates exactly this need, in the other direction:

  > To fire at a single sequence position (zero elsewhere), build the broadcast payload with
  > `steering::position_delta` (requires a backend feature).

`steering::position_delta` (`src/steering/contrastive.rs:293`) builds a `[1, seq_len, hidden]`
payload that is a direction at one position and zero elsewhere. So **`Add` has a positional story
and `Replace` does not.** The asymmetry is the whole of this report.

## What we write instead

Patching one position becomes: capture the recipient's own `ResidPost(L)`, `narrow` the donor's,
splice the two along the sequence axis, and hand the reassembled `[1, seq_len, hidden]` back as
`Replace`. Roughly

```rust
// Wanted:
//   hooks.intervene(
//       HookPoint::ResidPost(l),
//       Intervention::PatchAt { position: p, value: donor_row },
//   );
// Written: capture the recipient, then rebuild the whole tensor around one row.
let recipient = cache.require(&HookPoint::ResidPost(l))?;        // [1, seq_len, hidden]
let before  = recipient.narrow(1, 0, p)?;
let after   = recipient.narrow(1, p + 1, seq_len - p - 1)?;
let donor   = donor_resid.narrow(1, p, 1)?;                      // the row being injected
let spliced = Tensor::cat(&[&before, &donor, &after], 1)?;
hooks.intervene(HookPoint::ResidPost(l), Intervention::Replace(spliced));
```

Three costs, in increasing order of seriousness:

1. **It needs a prior capture of the recipient**, which is one extra forward pass and, more to the
   point, a `FullActivationCache` of every layer's residual held resident for the duration of the
   sweep. The forward is the cheap half: it is amortized over the whole layer-by-position grid,
   not paid per patch. The VRAM is the part that binds at 7B.
2. **The splice is an off-by-one waiting to happen**, and the dangerous slip is not the one that
   crashes. Getting the *length* wrong (`seq_len - p` for `after`) changes the total sequence
   length and dies at the next matmul. Getting the *offset* wrong (`narrow(1, p, seq_len - p - 1)`)
   keeps the length correct and shifts every row past the patch site: right shape, wrong content,
   and since `Replace` validates nothing, no error anywhere. The failure mode is a *plausible
   figure* rather than a crash.
3. **Every consumer writes it again.** This is not a forecast. It has already happened four times
   inside this repository:

   | Example | Helper |
   |---|---|
   | `examples/activation_patching.rs:732` | `patch_position` |
   | `examples/contrastive_patch.rs:158` | `patch_position` |
   | `examples/counterfact_patching.rs:213` | `replace_position` |
   | `examples/factual_routing.rs:255` | `replace_position` |

   Two names, four private copies, and none of them is the `cat` splice above. All four build a
   host-side mask instead: allocate `vec![0.0_f32; seq_len * hidden]`, set the chosen row to `1.0`,
   then `base * (1 - mask) + src * mask`, reallocated on every call inside the sweep. So the
   crate's own de facto answer to "how do I patch one position" is a third technique, written down
   nowhere, and a new consumer reaching for the obvious `narrow` plus `cat` is not even converging
   on it. Patching is not a niche operation: it is the standard causal instrument in this
   literature, and `candle-mi` is an interpretability crate.

## Ask

A positional variant, so the intervention says what it means:

```rust
/// Replace a single sequence position, leaving the rest of the activation untouched.
PatchAt { position: usize, value: Tensor },   // value: [hidden] or [1, 1, hidden]
```

`apply_intervention` (`src/hooks.rs:253`) can implement it in one call rather than the three-way
`cat` above. candle 0.11 has `Tensor::slice_scatter(&self, src, dim, start)`
(`candle-core/src/tensor.rs:1723`), which writes `src` into `self` at `start` along `dim` and
bounds-checks as it goes. The out-of-range check belongs there too, and the house error for it is
`MIError::Intervention` ("intervention validation or application error", `src/error.rs:22`) rather
than the `MIError::Config` that `position_delta` returns: `Intervention` is what
`src/interp/intervention.rs` raises throughout, and what `src/backend.rs:62` already documents a
backend as returning for an invalid intervention. `Intervention` is `#[non_exhaustive]`, so the
variant is additive and stays inside the `0.1.x` compatibility promise.

### One thing the variant has to settle first

`apply_intervention` never sees the `HookPoint`. Its signature is `(tensor: &Tensor, intervention:
&Intervention)`, and it is called from 17 sites where dim 1 does not mean the same thing:

- `Embed`, `ResidPre` / `ResidMid` / `ResidPost`, and the MLP points: `[batch, seq_len, hidden]`,
  so dim 1 is the sequence.
- `AttnQ`: `[batch, n_heads, seq_len, head_dim]` (reshape then transpose,
  `src/transformer/attention.rs:225-233`).
- `AttnK` / `AttnV`: `[batch, n_kv_heads, seq_len, head_dim]`, **not** `n_heads`. Both reshape
  with `num_kv_heads` (`src/transformer/attention.rs`, lines 229 and 232), and `repeat_kv` does
  not run until lines 269-270 of that file, which is after the `AttnK` / `AttnV` intervention
  blocks at lines 251-262.
- `AttnScores` / `AttnPattern`: `[batch, n_heads, seq_len, seq_len]`.

A `PatchAt` that writes at dim 1 therefore patches **a head index** rather than a position at those
five attention hook points, and does so *silently* whenever that index happens to be in range. That
is precisely the failure the "Not asked for" section below is guarding against, arriving by the
shape axis instead of the range axis. A doc line saying "dim 1 is assumed to be the sequence" does
not close it, because nothing enforces it.

The GQA split makes this worse rather than better. The silent-success window is `p < n_heads` at
`AttnQ` / `AttnScores` / `AttnPattern` but the tighter `p < n_kv_heads` at `AttnK` / `AttnV`. On
Llama-3.2-1B that is 32 against 8, so a single bad `p` can fail loudly at one attention hook and
pass quietly at another in the same layer of the same model.

Two ways out. Preferred: pass the `HookPoint` into `apply_intervention` and reject `PatchAt`
anywhere the sequence is not dim 1. That is a signature change, but `apply_intervention` is
`pub(crate)` with all 17 call sites inside this crate, so no consumer sees it, and the rejection
becomes a property of the crate rather than a comment. The alternative, having the variant carry
its own axis, pushes the choice onto the caller, which is one more degree of freedom in which a
probe can be quietly wrong.

**Two smaller alternatives, if the variant is unwanted:** a `steering::position_replace(recipient,
donor_row, position)` helper mirroring `position_delta`, which would at least put the operation in
one tested place and give the four existing example helpers something to collapse into; or a doc
line on `Replace` pointing at the pattern, which costs nothing and would have saved us the search.

### Scope of this ask

Everything `G` has registered for Q4 patches `ResidPost(L)` and nothing else, so restricting
`PatchAt` to hook points whose dim 1 is the sequence blocks no measurement we currently hold.

The restriction does need to be an explicit, documented error rather than a quiet no-op. It is
that by construction: `apply_intervention` returns `Result<Tensor>` and all 17 call sites take it
with `?`, so an `Err(MIError::Intervention(..))` propagates out of `forward` and cannot degrade
into a silent skip. What still has to be written by hand is the list of hook points `PatchAt`
accepts, since nothing in the type system says it.

## Not asked for

A `Patch` that takes a *range* of positions. Our probe wants exactly one, and every extra degree
of freedom in an intervention API is a degree of freedom in which a probe can be silently wrong.

Key/value patching, which an earlier draft of `G` wanted: inject a circuit's output into one
position's K or V. That is a genuinely separate ask rather than a wider version of this one,
because the `AttnK` / `AttnV` tensors are pre-broadcast. A write there lands on a KV head and fans
out to `n_heads / n_kv_heads` query heads downstream, so even a head-indexed variant at those
points could not express "patch the key seen by query head 17 only". If `PatchAt` is restricted as
above, please leave room for that later variant rather than foreclosing it.
