# Upstream contributions

Drafts of issues and pull requests aimed at candle-mi's dependencies, plus a
record of what has already been filed, kept here so the reasoning survives the
wait.

Upstream `huggingface/candle` reacts in weeks rather than days, and outside
contributions can sit for months. Two consequences shape everything in this
directory. Each document is written to stand alone without a maintainer reply,
with the workaround stated up front. And candle-mi never schedules its own work
behind one of these landing: the downstream workaround ships first, and an
upstream fix is a bonus that later lets us delete code.

## Drafts held here

| document | target | kind | status |
|---|---|---|---|
| [candle-cuda-copy-strided-src-overruns.md](candle-cuda-copy-strided-src-overruns.md) | `candle-core` | bug report | **FILED 2026-09-02** as [candle#3940](https://github.com/huggingface/candle/issues/3940), with the fix in [candle#3944](https://github.com/huggingface/candle/pull/3944). CUDA `copy_strided_src` sizes its copy from the storage rather than the source view, so `slice_scatter` with an offset source overruns into the following elements. Silent, GPU-only, and CPU disagrees. Found by `Intervention::PatchAt` returning a plausible-but-wrong causal trace on Llama-3.2-1B; workaround (a masked `where_cond`) already shipped |
| [candle-fused-ops-drop-gradients-v2-cluster-map.md](candle-fused-ops-drop-gradients-v2-cluster-map.md) | `candle-nn` | map plus the posted comment | **comment posted 2026-08-01** as [candle#2168 comment-5150289607](https://github.com/huggingface/candle/issues/2168#issuecomment-5150289607) |
| [candle-fused-ops-drop-gradients-v1-standalone-report.md](candle-fused-ops-drop-gradients-v1-standalone-report.md) | `candle-nn` | bug report | **superseded, do not file** |
| [candle-adamw-state-accessors.md](candle-adamw-state-accessors.md) | `candle-nn` | PR (additive, +80/-0) | **FILED 2026-08-01** as [candle#3819](https://github.com/huggingface/candle/pull/3819). First CI runs expired unapproved after thirty days and were stamped `failure` with zero jobs run, so **rebased onto `638a819a` and re-pushed 2026-08-31** (`67ee82d4`, patch-id unchanged) with [a comment](https://github.com/huggingface/candle/pull/3819#issuecomment-5479119668) explaining the red X. Awaiting workflow approval |
| candle-nn fused ops: `CustomOp::bwd` for `softmax_last_dim` + `layer_norm` | `candle-nn` | PR (the fix for the [gradient-drop cluster](candle-fused-ops-drop-gradients-v2-cluster-map.md): #3011, #3752…) | **FILED 2026-08-01** as [candle#3823](https://github.com/huggingface/candle/pull/3823) — branch `fused-ops-bwd`, independent of #3822; credits and offers deference to the open #3613/#3724. **Rebased onto `638a819a` and re-pushed 2026-08-31** (`dabee405`, patch-id unchanged) hours before its unapproved runs would have expired, with [a comment](https://github.com/huggingface/candle/pull/3823#issuecomment-5479358775). Awaiting workflow approval |
| [candle-gradstore-accumulate.md](candle-gradstore-accumulate.md) | `candle-core` | PR (behaviour-preserving, +115/-133) | **FILED 2026-08-01** as [candle#3822](https://github.com/huggingface/candle/pull/3822). **Rebased onto `638a819a` and re-pushed 2026-08-31** (`e7ea559b`, patch-id unchanged) hours before its unapproved runs would have expired, with [a comment](https://github.com/huggingface/candle/pull/3822#issuecomment-5479358458). Awaiting workflow approval |

On the two versions of the fused-ops document, following the `docs/conventions/`
pattern where the version lives in the filename: v1 was drafted as a standalone
bug report before anyone searched candle's tracker. It turned out the bug has been
reported by ten other people since 2024, with four fix PRs already open, so v1
would have been the eleventh duplicate. V1 also contains two factual errors that
the searching exposed. It is kept unedited for provenance, and v2 records both the
cluster and the errors. Read v2; file nothing from v1.

## Already filed upstream

| # | kind | opened | status |
|---|---|---|---|
| [3368](https://github.com/huggingface/candle/discussions/3368) | discussion, "Interest in a `candle-mi` crate?" | 2026-02-13 | name endorsed by `EricLBuehler` 2026-02-15, who invited the README PR; no candle-side reply since 2026-02-19 |
| [3406](https://github.com/huggingface/candle/pull/3406) | PR, add candle-mi to Useful External Resources | 2026-03-16 | open, solicited in 3368 |
| [3617](https://github.com/huggingface/candle/issues/3617) | issue, unbounded pickle-VM working set (DoS via crafted `.pth`) | 2026-06-13 | open |
| [3940](https://github.com/huggingface/candle/issues/3940) | issue, CUDA `copy_strided_src` over-copies from an offset source view | 2026-09-02 | open; fix proposed the same day as 3944 |
| [3944](https://github.com/huggingface/candle/pull/3944) | PR, size the CUDA copy from the layout rather than the storage (fixes 3940) | 2026-09-02 | open, `MERGEABLE`, +36/-3. Verified on an RTX 5060 Ti: the added `slice_scatter` assertions fail on CUDA without the patch and pass with it; full `candle-core` suite green on CPU and CUDA. Carries a comment flagging the four PRs stuck behind workflow approval |
| [3628](https://github.com/huggingface/candle/pull/3628) | PR, bound the pickle VM's working set and nesting depth | 2026-06-18 | open upstream, but cherry-picked into [`astorise/candle`](https://github.com/astorise/candle) as commit `b196387` with authorship preserved, in that fork's "Pull 9 high-value upstream candle PRs into the fork (Tier 1)" batch (their #37, 2026-07-29). `Sébastien ASTORI` then fixed a clippy lint on top (`2840a5b`). He also filed candle #3688 on pickle DoS, so he is working the same problem and is worth engaging directly |

## The workflow-approval gate

Every one of these is a fork pull request, so its CI run needs a maintainer to
click approve, and a fork cannot approve its own. If nobody clicks within thirty
days, GitHub expires the run and stamps it `conclusion: failure` with zero jobs
executed. The PR then advertises a broken patch to precisely the reviewers whose
attention it is competing for, and nothing notifies the author that the cause was
an unclicked button rather than a real defect. This happened to 3819.

Because upstream reacts in weeks rather than days, this is the default outcome
here, not an edge case. Check the open PRs against it periodically: the deadline
is the run's `created_at` plus thirty days, readable with

```
gh api "repos/huggingface/candle/actions/runs?head_sha=<sha>" --jq '.workflow_runs[] | "\(.name) \(.created_at) \(.conclusion)"'
```

where `action_required` means still pending and `failure` with zero jobs means
expired. A push resets the clock, since it queues a fresh run.

The first sweep, on 2026-08-31, found 3819 already expired and two more within
hours of it: 3822 was due at 15:17 UTC and 3823 at 19:32 UTC the same day, both
still `action_required`. All three were rebased, re-verified and re-pushed. 3406
was due 2026-09-02. The lesson is that the deadlines cluster, because the PRs
were filed in one sitting, so the sweep is worth running as one pass rather than
per PR.

### Correction (2026-09-02): a push does not rescue the pending run

That sweep recorded that 3822 and 3823 were re-pushed before their deadlines "so
neither ever showed a red X". **That was wrong**, and the receipts are the
GitHub notifications that arrived on 2026-08-31:

| run | branch | created | stamped `failure` | jobs |
|---|---|---|---|---|
| `Continuous integration` #8763, `CI / cuda` #5656 | gradstore-accumulate | 2026-08-01 15:17 | 2026-08-31 15:21 | 0 |
| `Continuous integration` #8765, `CI / cuda` #5658 | fused-ops-bwd | 2026-08-01 19:32 | 2026-08-31 19:36 | 0 |

Thirty days plus four minutes, zero jobs, and the re-pushes at 13:43 and 13:44
that same day did **not** stop it. A push does not approve, cancel or renew the
run already queued on the old head; it queues an *additional* run on the new
head. The old one keeps its own clock, expires on schedule, stamps `failure` and
notifies.

What the push does buy is that the expired run is attached to a SHA that is no
longer the PR head, so the red X is not on the current head. Confirmed the same
day: `gh pr view <n> --json statusCheckRollup` is **empty** for 3406, 3819, 3822
and 3823, because an `action_required` run publishes no status check until a
maintainer approves it. A reviewer therefore sees *no* checks rather than failing
ones. That is the whole benefit, and it is worth having, but it is not the same
as "never showed a red X".

The corollary matters for a PR whose head has *not* moved: there, the expiring
run **is** the head's run, so the red X lands on what the reviewer sees.

### State at 2026-09-02

| PR | head last pushed | runs | next expiry |
|---|---|---|---|
| [3406](https://github.com/huggingface/candle/pull/3406) | 2026-08-03 | `action_required` since 2026-08-03 14:15 | **2026-09-02 14:15 UTC**, and the head has not moved since, so this one lands a red X on the current head |
| [3628](https://github.com/huggingface/candle/pull/3628) | 2026-06-18 | none listed; `mergeStateStatus` is `CLEAN` | n/a |
| [3819](https://github.com/huggingface/candle/pull/3819) | 2026-08-31 | `action_required` since 13:32 | 2026-09-30 13:32 UTC |
| [3822](https://github.com/huggingface/candle/pull/3822) | 2026-08-31 | `action_required` since 13:43 | 2026-09-30 13:43 UTC |
| [3823](https://github.com/huggingface/candle/pull/3823) | 2026-08-31 | `action_required` since 13:44 | 2026-09-30 13:44 UTC |

| [3944](https://github.com/huggingface/candle/pull/3944) | 2026-09-02 | `action_required` from opening | 2026-10-02 |

**3406 was deliberately left to expire** at 14:15 UTC on 2026-09-02. Moving its
head would have bought thirty clean days on a one-line README addition with no
reviews in six months, at the price of a standing thirty-day chore on the least
valuable PR of the set. The effort went into 3944 instead.

All five are `MERGEABLE` against current `main`, so none has rotted into a
conflict. None has a review; each of 3406, 3819, 3822 and 3823 carries one
comment, the explanation of the red X. The next sweep is due **2026-09-30**, when
three deadlines fall within twelve minutes of each other.

## When something lands

Record the link and the version here, then delete the corresponding downstream
workaround: the `nn_ops` `track_op` dispatch for the fused-ops cluster, the
vendored checkpointable `AdamW` for the optimizer accessors.
