# Fused ops sever autograd in candle-nn: the existing cluster, and what we can add

**Date:** July 30, 2026
**Supersedes:** [v1](candle-fused-ops-drop-gradients-v1-standalone-report.md), kept for provenance
**Status:** not a report to file. A map of what is already upstream, plus the
four things we can contribute to it.

## Why this document was rewritten

V1 was drafted as a standalone bug report, before anyone searched candle's issue
tracker. It would have been the eleventh independent report of the same bug.

The searching also showed that two things V1 treated as findings were already
upstream, and one thing V1 asserted was simply wrong. Those are recorded below
rather than quietly fixed, because the pattern of error is the same one the
throughput report warns about: reasoning about a codebase, or in this case about
a community, without checking first.

## The cluster

Every entry below is open. Comment counts are maintainer plus community
combined.

| # | kind | author | created | comments | subject |
|---|---|---|---|---|---|
| 2168 | issue | agerasev | 2024-05-07 | 4 | `RmsNorm`, no backward when contiguous |
| 2977 | issue | toolness | 2025-06-01 | 0 | `LayerNorm`, when contiguous and `remove_mean` |
| 3011 | issue | tymat | 2025-06-28 | 0 | LayerNorm gradient flow |
| 3526 | PR | Gravirus | 2026-05-08 | 1 | fix `RmsNorm` backward |
| 3568 | issue | nxrobins | 2026-05-28 | 0 | `rope`, `rope_i`, `rope_thd` |
| 3569 | issue | nxrobins | 2026-05-28 | 0 | `softmax_last_dim` |
| 3612 | PR | NahButch | 2026-06-12 | 0 | add backward to fused rope |
| 3613 | PR | NahButch | 2026-06-12 | 0 | add backward to fused layer_norm |
| 3724 | PR | teddytennant | 2026-07-07 | 0 | add backward for `softmax_last_dim` |
| 3752 | issue | ynishi | 2026-07-19 | 1 | LayerNorm; self-closed as duplicate of 3011 |

Two more people reported hitting it in comments on 2168 rather than filing:
`computer-whisperer` (2025-01-08, "had to make my own copy of LayerNorm as a
workaround") and `getupforone` (2025-06-13, "I spent almost three days to figure
out the problem"). With Éric that is eleven people.

The oldest entry is two years and three months old. Four fix PRs are open. Total
maintainer engagement across the whole cluster is a handful of comments.

`ynishi`'s 3752 is worth understanding correctly: it was not rejected. The author
closed it themselves the same day as a duplicate of 3011, with "apologies for the
earlier triage noise". Nobody disputed the substance, and their evidence was
unusually strong (GPT-2 medium, 192 LoRA tensors bit-identical after 199 steps).

## Why it keeps being rediscovered

Four causes, and the last one is the operative one.

**Almost nobody trains in candle.** Seven of candle's 111 example directories
mention `backward`. The fused ops are correct for the great majority of users,
because that majority never calls it. Everyone who trains from scratch is
effectively the first to do so.

**Nothing cheap detects it.** Forward output is numerically correct. `backward()`
returns `Ok`. `step()` returns `Ok`. Checkpoints save. The loss decreases,
because whichever parameters remain connected keep learning. Only a
before-and-after weight diff or a gradient count catches it, and neither is a
common test.

**The contiguity gate makes it look intermittent.** Two codebases doing morally
the same thing take different paths, so the reasonable assumption that somebody
would already have noticed does not hold.

**The issues are titled by symptom, per op, not by cause.** 2168 is "No backward
pass for `RmsNorm` if tensor is contiguous". Someone hitting this through
LayerNorm searches for "candle layer_norm gradient" and does not find it, so they
file their own. That is the duplication mechanism, and it is visible in the title
list above. It also explains why the fixes are per-op: four PRs needing four
separate reviews, against a backlog of roughly 850 open issues.

`nxrobins` is the one exception. Both their titles name the cause and
cross-reference the root, and they maintain standalone reproducer repositories
(`candle-bug-2-rope-no-bwd`, `candle-bug-3-rmsnorm-no-bwd`). They still received
zero replies.

## What v1 got wrong

**`rope_thd` is not unreported.** V1 claimed `rope_thd` and `sdpa` were the two
sites nobody had covered. 3568 explicitly lists `rotary_emb.rs:808` (`rope_thd`)
alongside `rope` and `rope_i`. Only `sdpa` survives that claim.

**V1 missed the module-level path entirely, and 2977 already names it.** V1
framed the problem as seven free functions a user opts into. In fact
`candle_nn::LayerNorm` and `candle_nn::RmsNorm`, which is what people actually
build models from, route into the fused ops themselves:

```rust
// candle-nn 0.11.0, src/layer_norm.rs:118
if x.is_contiguous() && self.remove_mean {
    if let Some(bias) = self.bias.as_ref() {
        return crate::ops::layer_norm(x, &self.weight, bias, self.eps as f32);
    }
}

// src/layer_norm.rs:204
if xs.is_contiguous() {
    crate::ops::rms_norm(xs, &self.0.weight, self.0.eps as f32)
}
```

So a user who never writes `ops::` is affected, and V1's workaround table was
misleading: substituting `rms_norm_slow` does not help someone using the
`RmsNorm` module. Worth being precise about the gates, since they are what makes
the bug intermittent: `RmsNorm` takes the fused path when the input is
contiguous, and `LayerNorm` when the input is contiguous, `remove_mean` is set,
and a bias is present.

This is not our discovery either. `toolness` put it in 2977's title on
2025-06-01, and noted in the same breath that fixing `RmsNorm` would not fix
`LayerNorm` because the gate differs.

**A standalone reproducer is not a contribution here.** V1 treated verifying one
as a selling point. `nxrobins` already publishes two dedicated reproducer
repositories, which is better than a code block in an issue body.

## What is actually ours to add

Four things, none of which appears anywhere in the cluster.

**1. `ops::sdpa` (`ops.rs:1317`).** No issue in this cluster names it. The five
issues that mention `apply_op3_no_bwd` are 3613, 3612, 3568, 2977 and 3752, and
none of them covers scaled dot product attention. Unlike the norms and the two
differentiable rope variants, `sdpa` has no `_slow` twin, so there is no
workaround to point at either.

**2. A systemic fix instead of four per-op ones.** Every open PR implements
`bwd()` for one op. Nobody has proposed dispatching on `Tensor::track_op()`,
which is a single change that would close the whole cluster:

```rust
// candle-core 0.11.0, src/tensor.rs:592
pub fn track_op(&self) -> bool {
    self.is_variable || self.op.is_some()
}
```

That predicate is exactly the condition under which a gradient will be needed, so
the fused kernel can be taken when it is false and the composed form when it is
true, with no API change and no cost to inference. 3568 assumed the opposite,
that "the existence of two variants suggests the no-bwd shape is intentional API
design". It reads more like an omission than a design.

**3. The only throughput number in the cluster.** Four fix PRs sitting unreviewed
for two months suggests an unstated worry that composed paths cost speed. On our
workload the composed forward ran at **0.70x the time of the fused one**, meaning
composed was faster. One workload on one GPU, not a general claim, but it is the
only measurement anyone has offered and it addresses the likely objection head
on.

**4. A regression test that fails on the class, not the instance.** Every report
in the cluster describes the bug. None proposes a test that would have caught it.
Ours is a count: build the model, run the real forward, backward, then assert that
every one of the N parameters received a gradient. Because it is a count it fails
loudly on any future fused-op barrier, including ops that do not exist yet. That
is the shape of test whose absence allowed 1-of-29 training to run and look fine.

## Where to post

**Do not open a general issue.** It would be the eleventh, and the bottleneck
upstream is review capacity, not knowledge of the problem.

**Comment on 2168.** It is the root, the oldest, and the only thread with real
discussion, so it is where the existing audience is. One comment carrying items
2, 3 and 4, cross-referencing 2977, 3011, 3568 and 3569, and noting that the
four open PRs could be collapsed into one.

**Optionally, one narrow issue for `sdpa`.** Scoped to that single function, in
the style of 3568 and 3569, cross-referencing 2168. That is the one genuinely new
site, and a narrow issue is honest where a general one would be a duplicate.

**Keep the AdamW PR separate.** See
[candle-adamw-state-accessors.md](candle-adamw-state-accessors.md). Searches for
AdamW with state, optimizer with resume, and moments with optimizer all return
zero, so that one is genuinely novel and should be filed as a PR.

## For candle-mi's own reference

Independently of anything upstream, two facts from this pass are worth keeping.

The module-level gates above are the reason `nn_ops` has to dispatch rather than
simply choosing composed forms once: any `candle_nn` norm module can silently
switch paths on a contiguity change in the caller.

And nothing here is scheduled against an upstream fix. The `nn_ops` `track_op`
dispatch already ships and makes candle-mi correct today. If the cluster ever
lands, we delete code, which is the good outcome and not a dependency.
