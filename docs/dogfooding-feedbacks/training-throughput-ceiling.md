# Training is 5.1× slower than `PyTorch`, and the fix candle-mi owns is one `DType` parameter

**Date:** July 29, 2026
**Source:** askesis `canvas` leg — a 10,712,244-parameter MDLM (6L/384d/6H, block 128, vocab 18)
trained on blocksworld plans; RTX 5060 Ti 16 GB, fp32, candle 0.11, candle-mi 0.1.20
**Platform:** **Windows 11**, driver in WDDM mode. This matters for exactly one measurement (J2)
and is called out where it does.
**Affected area:** `OthelloGpt::init` (`src/diffusion/othello.rs:533`), `src/nn_ops.rs`, and the
optimizer story
**Severity:** Throughput ceiling — **not a defect**. The arithmetic is right (candle and `PyTorch`
agree to 9.5e-8 worst-case over 20 steps); it is 5.1× slower to produce. One item here *blocks* an
estimated 1.5–2.5× — estimated, not measured, and §"Item 1" names the ten-minute measurement that
would settle it.

---

## Read this part first: four beliefs about this exact workload, all wrong

Before the numbers, the reason to trust them. Answering "is a faster card worth renting?" produced
four confident hypotheses. All four were measured within the hour. All four were false.

| # | belief | measurement |
|---|---|---|
| J1 | "we're compute-bound, so a faster GPU scales" | **~10% of fp32 peak** (2.3 of ~24 TFLOPS). A 4× card buys far less than 4× |
| J2 | "raise the batch size for utilisation" | **VRAM-bound, hard.** 128 → 31.3k tok/s; 192 → **13.9k** (collapses); 256 OOMs. *Windows-specific mechanism — see below* |
| J3 | "candle-mi's backward-safe composed ops are the cost" | **Composed forward is 0.70× the fused one** — composed is *faster*. `nn_ops` is exonerated |
| J4 | "our `AdamW`/`EMA` are hundreds of tiny kernel launches" | **2.3% and 0.4%.** Fusing them buys nothing |

J3 and J4 were mine, argued from reading candle-mi's source, and I was ready to spend days on
both. A 141-line timing harness refuted them in an afternoon. **Everything below is labelled
MEASURED or HYPOTHESIS, and the hypotheses name the measurement that would settle them.** Please
do not act on a HYPOTHESIS line without taking its measurement first.

**On J2, before anyone re-runs it on Linux.** The 192 → 13.9k *collapse* is WDDM paging device
memory out to system RAM over PCIe, which is a Windows driver behaviour. On Linux the same
over-subscription simply OOMs. **The conclusion is platform-independent — we are VRAM-bound at
128 — but the symptom is not**, so a Linux reader who sees a hard OOM at 192 rather than a soft
2.25× collapse is seeing the same fact through a different driver, not contradicting this table.

## The decomposition — MEASURED

`examples/profile_step.rs` in the canvas crate (askesis, `reference/canvas`), reproducible to
within 2 ms across runs:

| phase | share of step |
|---|---|
| **`backward()`** | **66%** |
| forward | 20% |
| CPU batch preparation (host-side) | 11% |
| `AdamW` | 2.3% |
| `EMA` | 0.4% |

Whole step: **505 ms**, against **99 ms** for the same step in `PyTorch` — **5.1×**. The canvas V9
parity oracle shows the two computing the same function: over 20 recorded steps, 16 of 20 losses
were **bit-identical** and the worst disagreement was **9.5e-8**. So this is a speed gap and not a
correctness one.

For scale, that 9.5e-8 is *smaller than candle's disagreement with itself across devices* — the
same candle code on CPU vs GPU differs by **2.0e-7**, the measured null band. (The house pass bar
is 5e-3 on GPU; all three numbers are distinct and are kept distinct below.)

The backward runs at **3.3× the forward**. The expectation is ~2×, and by arithmetic rather than
folklore: a backward pass computes gradients with respect to both the inputs and the weights, so
it does roughly twice the matmul work of the forward. The excess over 2× is not doing arithmetic.
That ratio, not the absolute number, is the signal: it points at the autograd tape rather than at
the kernels.

**That much is core candle, not candle-mi.** There is no routing around another crate's autograd
from downstream. What follows is the part candle-mi *can* move.

---

## Item 1 — `OthelloGpt::init` hardcodes `DType::F32` (blocking, small, highest leverage)

**MEASURED (the constraint).** `src/diffusion/othello.rs`:

```
550:  Tensor::zeros(dims, DType::F32, device)?
552:  Tensor::ones(dims, DType::F32, device)?
562:  Self::load(config, VarBuilder::from_varmap(varmap, DType::F32, device))
```

`init` takes `(config, varmap, device, seed)` and no dtype. A caller who wants bf16 cannot get it —
`load` accepts any `VarBuilder`, so the capability exists one layer down and `init` is the only
thing withholding it. Note that `attention` already goes out of its way to be dtype-aware
(`othello.rs:326–335` upcasts scores to F32 and casts back), so the model body was written with
mixed precision in mind; only the constructor was not.

**HYPOTHESIS — why this is the highest-leverage item.** Not for the reason one would guess. At 10%
of fp32 peak we are *not* FLOP-bound, so bf16 tensor-core throughput is not the prize and I would
not claim it. The prize is **J2**: we are VRAM-bound at batch 128, and 192 already fails. Halving
activation bytes should move that ceiling, and batch size is the one knob measured to matter.
Estimated 1.5–2.5× compounded — *estimated*; **the measurement that settles it** is
`sweep_batch.sh` re-run under bf16, ten minutes once `init` accepts a dtype.

Note this argument survives the platform caveat: on Windows the 192 failure is a soft WDDM
collapse and on Linux it is a hard OOM, but "we cannot afford the activations" is the same
constraint either way, and halving them relieves it either way.

**Effort:** add `dtype: DType` to `init` (or an `init_with_dtype`, keeping `init` as an F32
shim — candle-mi is published, so the non-breaking form is probably right).

**Consequence worth planning for, not a blocker.** bf16 carries ~3 decimal digits. Our fp32
readings are a 9.5e-8 framework agreement inside a 2.0e-7 cross-device null band; under bf16 both
will grow by orders of magnitude, and a harness carrying fp32-derived expectations will fail
*correctly*. Any parity harness therefore needs a **per-dtype null band, measured rather than
guessed** — the house bar is 5e-3 and even that should be re-measured, not assumed to carry over.
Better to say so in the CHANGELOG than to let a downstream user read a real precision change as a
regression.

## Item 2 — fused kernels *with* a hand-written `bwd()`

**MEASURED (the situation).** `nn_ops.rs` exists because `candle_nn`'s fused `softmax_last_dim`,
`layer_norm` and `rms_norm` are built with `apply_op*_no_bwd`: they record no backprop op, so a
gradient silently stops there. That is canvas defect **C1** — `backward()` returned `Ok` with
gradients for 1 of 29 parameters, and the loss still went down, plateauing at the marginal entropy.
`nn_ops` routes tracked inputs to a composed form that carries a backward. The fix is right and
the crate should keep it.

**HYPOTHESIS.** The composed form trades one tape node for N. J3 established that this costs
nothing in the *forward* (composed is 0.70× fused — faster). It says nothing about the backward,
which is 66% of the step and proportional to tape length. `CustomOp` with an implemented `bwd()`
would give the fused kernel *and* a gradient *and* a single node — strictly better than either
existing path, if tape length is really what dominates.

**I have not measured tape length, and J3 is exactly the shape of error I would be repeating.**
*The measurement that settles it:* an `nsys` profile of the backward, or simply counting nodes and
timing `backward()` alone with composed vs. fused-plus-`no_bwd` forwards. Until that number exists
this item should not be scheduled. **Effort if confirmed:** 2–4 days for the three ops.

## Item 3 — checkpointable `AdamW` (queued for v0.1.21; an *ergonomics* item, not a speed one)

**MEASURED (verified against the crate source, both versions).** Stock `candle_nn::AdamW` keeps
its moments in `struct VarAdamW` — declared without `pub` at `candle-nn/src/optim.rs:104` — and
`step_t` is a private field of `AdamW` itself. `AdamW`'s entire public surface is `new_lr`,
`params` and `set_params`: no moment accessor, no step accessor. Meanwhile `SGD`, **in the same
file**, exposes `pub fn into_inner` at line 73. And `src/optim.rs` is **byte-identical between
0.9.2 and 0.11.0** (`diff` returns nothing), so this is settled API rather than a moving target.

So a training run cannot be resumed with its optimizer state intact, which is what a staged
multi-day run needs. canvas has a transcribed implementation (`canvas/src/optim.rs`, update rule
taken verbatim from candle-nn 0.11.0) held to candle's own trajectory by
`canvas/tests/optim_parity.rs`. It has driven **three resumes** across the four stages of the
40-epoch run that produced the leg's result, plus earlier ones in the v1 run.

**State plainly in the CHANGELOG that this buys no throughput.** `AdamW` is 2.3% of the step (J4).
It belongs in this report only so nobody bundles it into a performance story it cannot support.
Its value is that a staged run resumes exactly; that is worth having on its own terms.

Related and still open upstream: a candle-nn PR adding `AdamW` state accessors
(`step_t`/`set_step_t`, a `moments()` iterator), following `SGD::into_inner`'s precedent. If it
lands, candle-mi deletes its copy.

## Item 4 — the 11% that no GPU will fix

**MEASURED.** Host-side batch preparation is 11% of the step and does **not** shrink with a faster
card. It is already **~4×** `AdamW`+`EMA` combined (11% against 2.7%) — i.e. the largest of the
three things people usually think of optimising, and the only one that a new GPU cannot touch. If
the GPU work alone got 3–4× faster, this 11% would become **~30%** of the new step and the leading
non-backward cost. Prefetching it onto another thread is ordinary work and probably
belongs in the downstream training loop rather than in candle-mi — recorded here so that whoever
reads a post-upgrade profile is not surprised by it.

---

## What NOT to do

Two code changes the measurements have already priced at approximately zero, and one purchase:

- **Do not fuse `AdamW` or `EMA`.** 2.3% and 0.4% — MEASURED. Amdahl caps the whole idea at 2.7%.
- **Do not replace `nn_ops`' composed path with the fused one for speed.** It is 0.70× — MEASURED.
  The composed path is *already* the faster forward, and the fused one silently breaks gradients.
- **Do not expect a datacenter card to scale linearly.** This one is INFERRED, not measured: J1
  puts us at ~10% of fp32 peak, and A100/H100 sell their advantage in tensor-core and fp64 terms
  that an fp32, launch-bound workload cannot spend. Consumer cards looked better per dollar on
  that reasoning — but we have not yet obtained a profile on one, so treat it as an argument.

## Recommendation, in order

1. **Item 1 now.** Small, non-breaking if added as `init_with_dtype`, and it unblocks the one knob
   measured to matter. Ship with a CHANGELOG note about per-dtype parity bands.
2. **Item 3 with it** (v0.1.21, as already planned), labelled as resumability, not speed.
3. **Then measure** the backward with `nsys` before deciding Item 2. If the tape hypothesis holds,
   `CustomOp::bwd()` is candle-mi's largest available win and is worth the 2–4 days. If it does
   not, the remaining gap is core candle's and the honest move is an upstream issue with
   `profile_step.rs` attached — a 5.1× reproducer against `PyTorch` on identical arithmetic is a
   good bug report and costs a day.

## The method note

The instrument that produced every number here is `canvas/examples/profile_step.rs`: **141 lines**,
written in an afternoon, and it overturned four confident readings of candle-mi's own source — two
of them written by the person who had just read that source. Reasoning about performance from code
is not evidence. It is worth keeping such a profiler in candle-mi's `examples/` so the next
throughput question starts with a measurement instead of a hypothesis.

*(An earlier draft of this report said "about fifty lines" in two places. It is 141. The number
was written from memory and never checked — in a document whose entire argument is that unchecked
recollection about code is worthless. Corrected, and left visible.)*
