# Training is 5.1× slower than `PyTorch`, and the fix candle-mi owns is one `DType` parameter

**Date:** July 29, 2026
**Source:** askesis `canvas` leg — a 10.71 M-parameter MDLM (6L/384d/6H, block 128, vocab 18)
trained on blocksworld plans; RTX 5060 Ti 16 GB, fp32, candle 0.11, candle-mi 0.1.20
**Affected area:** `OthelloGpt::init` (`src/diffusion/othello.rs:533`), `src/nn_ops.rs`, and the
optimizer story
**Severity:** Throughput ceiling — **not a defect**. The arithmetic is right (the canvas V9 oracle
agrees with `PyTorch` to 1e-7); it is 5.1× slower to produce. One item here is *blocking* a
straightforward 1.5–2.5× and costs about twenty lines.

---

## Read this part first: four beliefs about this exact workload, all wrong

Before the numbers, the reason to trust them. Answering "is a faster card worth renting?" produced
four confident hypotheses. All four were measured within the hour. All four were false.

| # | belief | measurement |
|---|---|---|
| J1 | "we're compute-bound, so a faster GPU scales" | **~10% of fp32 peak** (2.3 of ~24 TFLOPS). A 4× card buys far less than 4× |
| J2 | "raise the batch size for utilisation" | **VRAM-bound, hard.** 128 → 31.3k tok/s; 192 → **13.9k** (collapses, WDDM spilling over PCIe); 256 OOMs |
| J3 | "candle-mi's backward-safe composed ops are the cost" | **Composed forward is 0.70× the fused one** — composed is *faster*. `nn_ops` is exonerated |
| J4 | "our `AdamW`/`EMA` are hundreds of tiny kernel launches" | **2.3% and 0.4%.** Fusing them buys nothing |

J3 and J4 were mine, argued from reading candle-mi's source, and I was ready to spend days on
both. Fifty lines of timing code refuted them in ten minutes. **Everything below is labelled
MEASURED or HYPOTHESIS, and the hypotheses name the measurement that would settle them.** Please
do not act on a HYPOTHESIS line without taking its measurement first.

## The decomposition — MEASURED

`canvas/examples/profile_step.rs`, reproducible to within 2 ms:

| phase | share of step |
|---|---|
| **`backward()`** | **66%** |
| forward | 20% |
| CPU batch preparation (host-side) | 11% |
| `AdamW` | 2.3% |
| `EMA` | 0.4% |

Whole step: **505 ms**, against **99 ms** for the same step in `PyTorch` — **5.1×**. The canvas V9
parity oracle proves both compute the same function to 1e-7, so this is a speed gap and not a
correctness one.

The backward runs at **3.3× the forward**, where ~2× is the usual expectation. That ratio, not the
absolute number, is the signal: it points at the autograd tape rather than at the kernels.

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
not claim it. The prize is **J2**: we are VRAM-bound at batch 128, and 192 already collapses.
Halving activation bytes should move that ceiling, and batch size is the one knob measured to
matter. Estimated 1.5–2.5× compounded; *the measurement that settles it* is `sweep_batch.sh` run
under bf16, which is ten minutes once `init` accepts a dtype.

**Effort:** add `dtype: DType` to `init` (or an `init_with_dtype`, keeping `init` as an F32
shim — candle-mi is published, so the non-breaking form is probably right).

**Consequence worth planning for, not a blocker.** bf16 carries ~3 decimal digits, so a
cross-backend parity protocol pinned at 2e-7 will fail under it *correctly*. Any parity harness
needs a **per-dtype null band**, measured rather than guessed. Better to say so in the CHANGELOG
than to let a downstream user read a real precision change as a regression.

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

Stock `candle_nn::AdamW` keeps its moments in a private `VarAdamW` with no accessor — unchanged
0.9.2 → 0.11.0, while `SGD` in the same file exposes `into_inner`. So a training run cannot be
resumed with its optimizer state intact, which is what a staged multi-day run needs. canvas has a
transcribed implementation with a parity test (`canvas/src/optim.rs`) that has now driven four real
resume cycles.

**State plainly in the CHANGELOG that this buys no throughput.** `AdamW` is 2.3% of the step (J4).
It belongs in this report only so nobody bundles it into a performance story it cannot support.
Its value is that a staged run resumes exactly; that is worth having on its own terms.

Related and still open upstream: a candle-nn PR adding `AdamW` state accessors
(`step_t`/`set_step_t`, a `moments()` iterator), following `SGD::into_inner`'s precedent. If it
lands, candle-mi deletes its copy.

## Item 4 — the 11% that no GPU will fix

**MEASURED.** Host-side batch preparation is 11% of the step and does **not** shrink with a faster
card. It is currently ~2× `AdamW`+`EMA` combined. On a 3–4× faster GPU it becomes ~30% and the
leading non-backward cost. Prefetching it onto another thread is ordinary work and probably
belongs in the downstream training loop rather than in candle-mi — recorded here so that whoever
reads a post-upgrade profile is not surprised by it.

---

## What NOT to do

Three tempting changes that the measurements have already priced at approximately zero:

- **Do not fuse `AdamW` or `EMA`.** 2.3% and 0.4%.
- **Do not replace `nn_ops`' composed path with the fused one for speed.** It is 0.70× — the
  composed path is already the faster forward, and the fused one silently breaks gradients.
- **Do not rent a bigger datacenter card expecting linear scaling.** At 10% of peak, and fp32,
  consumer cards beat A100/H100 for this workload per dollar.

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

The instrument that produced every number here is `canvas/examples/profile_step.rs`: about fifty
lines, ten minutes to write, and it overturned four confident readings of candle-mi's own source —
two of them written by the person who had just read that source. Reasoning about performance from
code is not evidence. It is worth keeping such a profiler in candle-mi's `examples/` so the next
throughput question starts with a measurement instead of a hypothesis.
