# Upstream contributions

Drafts of issues and pull requests aimed at candle-mi's dependencies, kept here
so the reasoning survives the wait.

Upstream `huggingface/candle` reacts in weeks rather than days, and a PR of ours
once sat for six weeks before being cherry-picked into a fork rather than merged.
Two consequences shape everything in this directory. Each document is written to
stand alone for months without a maintainer reply, with the workaround stated up
front and any reproducer given as a copy-pasteable standalone crate. And
candle-mi never schedules its own work behind one of these landing: the
downstream workaround ships first, and an upstream fix is a bonus that later lets
us delete code.

| document | target | kind | status |
|---|---|---|---|
| [candle-fused-ops-drop-gradients.md](candle-fused-ops-drop-gradients.md) | `candle-nn` | bug report | drafted 2026-07-30, not filed |
| [candle-adamw-state-accessors.md](candle-adamw-state-accessors.md) | `candle-nn` | PR (additive, ~15 lines) | drafted 2026-07-30, not filed |

Each document names the downstream workaround it corresponds to. When an upstream
fix lands, record the link and the version here, then delete the workaround: the
`nn_ops` `track_op` dispatch for the first, the vendored checkpointable `AdamW`
for the second.
