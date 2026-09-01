# The interp API forces stringly-typed hook handling downstream

> **Status: IMPLEMENTED in v0.1.24** (2026-09-01). All six items shipped.
> **Item 1:** `HookPoint` derives `PartialOrd` + `Ord`, which closes **item 6**
> by itself. **Item 2:** `HookCache::captures()` / `into_captures()` and
> `HookSpec::captures()`; order is arbitrary and documented as such, with `Ord`
> making `cache.captures().collect::<BTreeMap<_, _>>()` the one-expression
> deterministic walk. **Item 3:** `HookSpec::capture_all` plus
> `FromIterator<HookPoint>`, and `HookSpec: Clone` is now documented as a
> guarantee. **Item 4:** widened to rank-preserving (4a), *not* asserted at rank
> 2 (4b): the two `stoicheia` backends turned out to be the ones not honouring
> the wider contract and now use `broadcast_matmul`, with the rank-2 path
> bit-identical and regression-tested. **Item 5:** the doc line (5a), *not* a
> `HookPoint::Logits` variant (5b), which would have changed `FromStr` and made
> every backend duplicate the largest tensor in the pass.
>
> **Deliberately not done:** no `Extend<HookPoint>` impl, which would collide
> with the inherent `HookSpec::extend`; and the backing stores stay `HashMap` /
> `HashSet`, so the hook fast path is untouched and `RELEASE_TIMINGS.md` needed
> no refresh.
>
> **Item 6 awaits downstream confirmation, which is the acceptance test:**
> `canvas` `capture.rs` drops `CaptureKey`'s cached `to_string()` and keys on
> `HookPoint` directly, and `CaptureSet` drops its "this model's forward emits
> no `{hook}`" error path in favour of `cache.captures()`. Two deletions, both
> compile-checked.

**Date:** September 1, 2026
**Source:** askesis `canvas` leg — building the Measurement `G` capture harness
(`reference/canvas/src/capture.rs`) against `candle-mi`'s `HookSpec` / `HookCache`
**Affected area:** `src/hooks.rs` (`HookPoint`, `HookSpec`, `HookCache`), `src/backend.rs`
(`project_to_vocab`)
**Severity:** Ergonomics — **no defect, nothing blocked.** The harness was built, taps the
decoder's own forward, and its `D11` control passes at `7.5e-8`. Every item below is friction
that produced glue code, not a wrong answer.

---

## Context, so the asks are judged against a real use

`canvas` decodes a masked-diffusion plan grid over 88–136 confidence-ordered rounds. The probe
needs activations **at the round a given grid position was committed**, so the harness decodes
twice: once to learn each position's commit round, then again with a populated `HookSpec` on
exactly those rounds. The library side of that was pleasant: `decode.rs` already called
`MIBackend::forward(model, &ids, &HookSpec::new())` with an empty spec, so tapping was a matter
of handing it a populated one. The six points below are where downstream code had to do work the
API could have done.

## 1. `HookPoint` has no `Ord` — and a determinism contract needs one

`src/hooks.rs:45` derives `Debug, Clone, PartialEq, Eq, Hash`. `canvas`'s determinism contract
(shared with `pweak`) forbids `HashMap` iteration in any result-affecting path, so a capture set
keyed by hook point must be a `BTreeMap`. With no `Ord`, the harness had to invent a key type
carrying a cached `hook.to_string()` purely to obtain a total order.

Every payload in the enum is already `Ord` (`usize`, `String`), so `#[derive(PartialOrd, Ord)]`
is free. **Ask:** derive it.

## 2. `HookCache` cannot be enumerated

`src/hooks.rs:419-483` exposes `new`, `into_output`, `get`, `require`, `store`, `set_output`,
`num_captures` — and no iterator. `num_captures()` reports a count of a collection the caller
cannot walk. A harness that wants *everything that was captured* must keep its own copy of the
request and re-derive the keys, which is also why our code carries a "this model's forward emits
no `{hook}`" error path that would otherwise be unnecessary: absence can only be discovered per
key. `HookSpec` has the same gap — it cannot list its own captures.

**Ask:** `captures(&self) -> impl Iterator<Item = (&HookPoint, &Tensor)>` and
`into_captures(self)` on `HookCache`; a `captures()` view on `HookSpec`.

## 3. `HookSpec::capture` has no bulk form

`src/hooks.rs:293` takes one hook at a time, so building a spec from a `&[HookPoint]` is a `for`
loop with a `.clone()` per element, repeated for every tapped forward.

**Ask:** `capture_all<I: IntoIterator<Item = HookPoint>>`, or `FromIterator<HookPoint>`.

**Related, and worth a line in the docs:** `HookSpec: Clone` (`src/hooks.rs:273`) is what makes a
per-round spec table cheap, but no doc advertises it as a guarantee. We depend on it.

## 4. `project_to_vocab`'s documented shape contradicts a working implementation

`src/backend.rs:74-75` documents `hidden: [batch, hidden_size] -> [batch, vocab_size]`. But
`OthelloGpt`'s implementation (`src/diffusion/othello.rs:674-676`) is `layer_norm` + `Linear`,
both rank-agnostic, so `[batch, seq, hidden]` also works. We obeyed the doc —
`narrow(1, p, 1).squeeze(1).contiguous()` — without being able to tell whether the rank-3 path
is supported or accidental.

**Ask:** either widen the doc to `[.., hidden_size]` and promise rank-preservation, or assert
rank 2 in the implementations. Silent tolerance is the one option that cannot be relied on.

## 5. There is no hook point for the logits, and that lands on the verification control

`HookCache::output()` *is* the logits, so "the model's own logits at position `p`" is reached by
a different route (`output()` + manual `narrow`) from every other activation (`cache.get(&hook)`).
That asymmetry falls exactly on the `D11` capture-verification invariant — recompute the logits
from the captured residual stream and compare — which is the one place a probe most wants the two
quantities to be the same kind of thing.

**Ask:** a `HookPoint::Logits` alias, or an explicit doc line beside `FinalNorm` stating that
`output()` is the logit tap.

## 6. The three together push downstream code into stringly-typed handling

`HookPoint` is `#[non_exhaustive]` (correct — backends extend it), has no `Ord`, and offers
`Display` as its only total operation. Under a lint floor that denies `wildcard_enum_match_arm`,
a downstream crate can never match it exhaustively, so **`to_string()` becomes the only total
operation available on it** — and string keys are precisely what the Grit conventions forbid
elsewhere. Deriving `Ord` (item 1) removes most of this by itself: it gives a total order that is
not a string.

---

## What worked, and is worth keeping

- **Tapping cost one argument.** Because `MIBackend::forward` already takes a `&HookSpec`, the
  untapped path stayed byte-identical: `spec_at(round) -> Option<&HookSpec>` with
  `unwrap_or_default()`. Non-invasive capture (askesis `D10`) was structurally free.
- **The hook-point coverage is right for this work.** `AttnPattern`, `ResidPre`/`ResidPost`,
  `AttnK`/`AttnV`, `FinalNorm` cover Q1–Q4 of the measurement without a single custom tap.
- **`HookSpec` carrying interventions as well as captures** means the patching substep needs no
  second mechanism.
