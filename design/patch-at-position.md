# Design: `Intervention::PatchAt`

**Status:** Implemented **|** **Date:** September 2, 2026 **|** **Relates to:** [`add-at-positions.md`](add-at-positions.md) (the `Add`-side sibling, still Proposed), which parked `ReplaceAtPositions` under "What this does NOT include" **|** **Origin:** askesis `canvas` dogfooding report [`positional-replace-has-no-intervention.md`](../docs/dogfooding-feedbacks/positional-replace-has-no-intervention.md)

## Question

`Intervention::Replace` is whole-tensor. Should overwriting a single sequence
position be a first-class intervention, and if so, at which hook points is it
even meaningful?

## Context

Activation patching is the standard causal instrument in this literature: run a
recipient forward pass, but at one hook point and one position substitute a row
taken from a donor pass, then read whether the prediction moves. Before this
change the crate had no way to say that. A caller had to capture the
recipient's own activation, slice it, splice a donor row in and hand the
reassembled tensor back through `Replace`.

Four examples in this repository had each written that by hand, under two names
and with a third technique nobody had written down:

| Example | Helper | Technique |
|---|---|---|
| `activation_patching.rs` | `patch_position` | host-side mask blend |
| `contrastive_patch.rs` | `patch_position` | host-side mask blend |
| `counterfact_patching.rs` | `replace_position` | host-side mask blend |
| `factual_routing.rs` | `replace_position` | host-side mask blend |

The asymmetry was sharper still because the `Add` side already had a positional
story: `steering::position_delta` builds a `[1, seq_len, hidden]` payload that
is a direction at one position and zero elsewhere, and `Intervention::Add`'s own
doc comment pointed at it. `Replace` had nothing.

## Recommendation

### The variant

```rust
/// Replace a single sequence position, leaving every other position untouched.
PatchAt {
    /// Sequence position to overwrite. Must be in `0..seq_len`.
    position: usize,
    /// Replacement row: `[hidden]`, `[1, 1, hidden]` or `[batch, 1, hidden]`.
    value: Tensor,
},
```

`Intervention` is `#[non_exhaustive]`, so adding a variant is not breaking.

**Exactly one position, not a range.** The reporting probe wanted one, and every
extra degree of freedom in an intervention API is a degree of freedom in which a
probe can be silently wrong. This is also why `PatchAt` is singular where
`add-at-positions.md` proposed a plural `AddAtPositions`: the plural form was
motivated by heterogeneous multi-site injection, which has no counterpart here.

**Three accepted value shapes.** `[hidden]` and `[1, 1, hidden]` broadcast
across the batch, mirroring how `Add` treats a bare direction; `[batch, 1,
hidden]` gives each batch row its own replacement. `[1, 1, hidden]` is accepted
because a donor row taken with `donor.narrow(1, position, 1)` already has that
shape, and rejecting it would put a squeeze at the most common call site.

### The hook-point policy, and why it needed a signature change

`apply_intervention` took `(tensor, intervention)` and never saw the
`HookPoint`. That is fine for every other variant, and fatal for this one,
because **dim 1 does not mean the same thing at every hook point**:

| Hook points | Shape | Dim 1 |
|---|---|---|
| `Embed`, `ResidPre`, `AttnOut`, `ResidMid`, `MlpPre`, `MlpPost`, `MlpOut`, `ResidPost`, `FinalNorm` | `[batch, seq_len, hidden]` | the sequence |
| `AttnQ` | `[batch, n_heads, seq_len, head_dim]` | a query head |
| `AttnK`, `AttnV` | `[batch, n_kv_heads, seq_len, head_dim]` | a KV head |
| `AttnScores`, `AttnPattern` | `[batch, n_heads, seq_len, seq_len]` | a head, and there are two sequence axes |

A `PatchAt` that wrote at dim 1 regardless would overwrite a **head** at the
five attention hook points, and would do so silently whenever the position
happened to be below the head count. That is the failure mode this API exists to
avoid: a plausible figure rather than a crash. The grouped-query split makes it
worse rather than better, since the silent-success window is `p < n_heads` at
`AttnQ` but the tighter `p < n_kv_heads` at `AttnK` and `AttnV`. On
Llama-3.2-1B that is 32 against 8, so one bad position can fail loudly at one
attention hook and pass quietly at another in the same layer.

So `apply_intervention` now takes the hook point:

```rust
pub(crate) fn apply_intervention(
    tensor: &Tensor,
    point: &HookPoint,
    intervention: &Intervention,
) -> Result<Tensor>
```

It is `pub(crate)` with all 17 call sites inside this crate, so no consumer sees
the change. The policy itself is public, as `HookPoint::accepts_positional_patch`,
so a caller can ask before building a spec rather than finding out from a failed
forward.

**The alternative was to have the variant carry its own axis.** Rejected: it
pushes the choice onto the caller, which is one more degree of freedom in which
a probe can be quietly wrong, and it would let `PatchAt` be aimed at
`AttnScores`, where there is no single right answer to aim at.

### Implementation: a masked select, and why not the obvious two

Not the host-side `build_sparse_delta` that `add-at-positions.md` proposes for
the `Add` side: it routes through `to_vec1()`, which forces a device
synchronisation and detaches from the autograd graph.

Not `Tensor::slice_scatter` either, **because it is silently wrong on CUDA when
the donor row is a view**. It routes through `copy_strided_src`, and the two
backends do not agree on what that helper means:

```rust
// cpu_backend/mod.rs:902 -- copies exactly the view's length
StridedBlocks::SingleBlock { start_offset, len } =>
    dst[dst_offset..dst_offset + len]
        .copy_from_slice(&src[start_offset..start_offset + len])

// cuda_backend/mod.rs:1217 -- derives the length from whole-storage sizes
let to_copy = dst.len().saturating_sub(dst_offset)
                 .min(src.len().saturating_sub(src_offset));
```

A donor row is normally a view into a captured activation, which is exactly what
`FullActivationCache::get_position` returns (`narrow(0, p, 1)?.squeeze(0)?`), so
its storage holds the whole donor and `src.len() - src_offset` runs to the end
of it. The copy then overruns into the positions *after* the patch site. This is
a candle bug rather than a misuse: the call is an ordinary narrow view passed to
a public API, and CPU and CUDA return different answers for it.

The three-way `Tensor::cat` the originating report sketched was measured against
the same case and is **correct on both backends**, so it was a usable option;
`where_cond` is preferred over it for the reasons below, not because it is
broken.

It is also the exact failure mode this crate exists to prevent. Caught by
running `examples/activation_patching` on Llama-3.2-1B and comparing against the
pre-`PatchAt` implementation: every position before the last reported 100%
recovery at every layer, including layer 15, where patching a non-final position
cannot affect the logits at all. The unit tests did not catch it because they
built `value` with `Tensor::new`, which owns its storage exactly; only a donor
row taken as a view reproduces it.

So the write is a masked select instead:

```rust
selector.where_cond(&replacement, tensor)
```

with all three operands materialised to the same contiguous
`[batch, seq_len, hidden]` shape. It touches each element once through its own
layout, so it has no dependence on storage provenance; `Op::WhereCond` has a
backward implementation, so a patch inside a tracked forward pass still carries
gradients; and it cannot produce the `0.0 * inf = NaN` that an arithmetic blend
would. The cost is two full-size temporaries, the same order as the mask blend
the four examples were already paying, and less than the whole-tensor `Replace`
it replaces.

`patches_from_a_donor_row_that_is_an_offset_view` (CPU) and
`cuda_patches_from_a_donor_row_that_is_an_offset_view` (GPU, `#[ignore]`d) are
the regression guards. The GPU one fails against a `slice_scatter`
implementation and passes against this one.

Validation order in the private `patch_at` helper: hook-point policy, then an
explicit rank-3 check (defence in depth, so a backend storing an unexpected rank
at an accepting point errors rather than writing at the wrong axis), then
position bounds, then value shape, then dtype coercion. Every rejection is
`MIError::Intervention`, which `src/backend.rs` already documents as what a
backend returns for an invalid intervention.

## As implemented

All four in-repo helpers are gone, and with them the second forward pass each
example ran purely to store the recipient's own activations:

- `activation_patching.rs` runs its corrupted pass with `HookSpec::new()`.
- `contrastive_patch.rs` does the same for the corrupt pass.
- `counterfact_patching.rs` captures only the counterfactual (donor) pass.
- `factual_routing.rs` keeps its `AttnPattern` captures, which the routing
  analysis genuinely needs, and drops the `ResidPost` ones.

## What this does NOT include

**Key/value patching.** Injecting a circuit's output into one position's K or V
is a coherent thing to want, and it is a separate ask rather than a wider
version of this one: `AttnK` and `AttnV` are captured pre-broadcast, so a write
there lands on a KV head and fans out to `n_heads / n_kv_heads` query heads
downstream. Even a head-indexed variant at those points could not express
"patch the key seen by query head 17 only". If that is built later, it needs its
own design note; the restriction here is documented and explicit so that it does
not have to be rediscovered.

**A positional range.** See "Exactly one position" above.

## Tests

In `src/hooks.rs`, gated behind the same backend predicate as
`apply_intervention`:

1. Patches only the target row; the others stay bit-identical.
2. A bare `[hidden]` and a `[1, 1, hidden]` value agree.
3. A bare value broadcasts across `batch = 2`.
4. A `[batch, 1, hidden]` value gives each batch row its own replacement.
5. The result is contiguous, pinning the `.contiguous()` decision.
6. Dtype coercion: an F64 value into an F32 activation.
7. A position past the end is rejected, and the message names `seq_len`.
8. **All five attention hook points are rejected**, and the tensor passed in is
   rank 3, so the rejection is by hook point and not by luck of shape. This is
   the regression the policy exists for.
9. `Custom` is rejected.
10. A rank-4 activation at an accepting hook point is rejected.
11. A value of the wrong shape is rejected.
12. **The donor row is an offset view**, as `FullActivationCache::get_position`
    returns, on CPU and (`#[ignore]`d) on CUDA. This is the pair that catches the
    `copy_strided_src` trap described above.
13. `accepts_positional_patch` is asserted over a table that must equal
    `one_of_each_variant()`, which `declaration_rank` already keeps complete, so
    a new `HookPoint` variant has to be given an explicit answer rather than
    inheriting one.
