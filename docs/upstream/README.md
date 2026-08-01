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
| [candle-fused-ops-drop-gradients-v2-cluster-map.md](candle-fused-ops-drop-gradients-v2-cluster-map.md) | `candle-nn` | map plus the posted comment | **comment posted 2026-08-01** as [candle#2168 comment-5150289607](https://github.com/huggingface/candle/issues/2168#issuecomment-5150289607) |
| [candle-fused-ops-drop-gradients-v1-standalone-report.md](candle-fused-ops-drop-gradients-v1-standalone-report.md) | `candle-nn` | bug report | **superseded, do not file** |
| [candle-adamw-state-accessors.md](candle-adamw-state-accessors.md) | `candle-nn` | PR (additive, ~15 lines) | drafted 2026-07-30, ready to file |

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
| [3628](https://github.com/huggingface/candle/pull/3628) | PR, bound the pickle VM's working set and nesting depth | 2026-06-18 | open upstream, but cherry-picked into [`astorise/candle`](https://github.com/astorise/candle) as commit `b196387` with authorship preserved, in that fork's "Pull 9 high-value upstream candle PRs into the fork (Tier 1)" batch (their #37, 2026-07-29). `Sébastien ASTORI` then fixed a clippy lint on top (`2840a5b`). He also filed candle #3688 on pickle DoS, so he is working the same problem and is worth engaging directly |

## When something lands

Record the link and the version here, then delete the corresponding downstream
workaround: the `nn_ops` `track_op` dispatch for the fused-ops cluster, the
vendored checkpointable `AdamW` for the optimizer accessors.
