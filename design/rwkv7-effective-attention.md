# Design: RWKV-7 Effective Attention Formula

**Status:** Implemented — approach 1 (numerical) shipped as
`compute_effective_attention_v7` (`src/rwkv/mod.rs`), exposed via
`HookPoint::RwkvEffectiveAttn`. See "Resolution" below; the "Possible approaches"
section records the design-time options, not open questions.
**Relates to:** Roadmap §8 item 7, Phase 2

## Question

How to derive effective attention for RWKV-7's diag+rank-1 state transition?

## Context

plip-rs Phase 5 derived effective attention for RWKV-6 where the state transition is diagonal: `S_t = diag(w_t) * S_{t-1} + k^T @ v`. The cumulative decay is a simple product of diagonal matrices, computable via log-space prefix sums.

RWKV-7 uses `S_t = (diag(w_t) + a^T @ b) * S_{t-1} + v^T @ k`. The `a^T @ b` rank-1 term makes the transition matrix **non-diagonal**, so the cumulative product is no longer element-wise.

## Challenge

The cumulative transition from step `i` to step `t` is:

```
T(i→t) = Π_{j=i+1}^{t} (diag(w_j) + a_j^T @ b_j)
```

Each factor is diag + rank-1, but their product is **not** diag + rank-1 in general (rank grows with each multiplication). This means the RWKV-6 log-space prefix sum trick doesn't apply.

## Possible approaches

1. **Numerical computation**: Materialise the full `[head_dim, head_dim]` transition matrices and multiply. Cost: O(T^2 * D^2) — feasible for short sequences but expensive.
2. **Low-rank approximation**: Truncate the cumulative product to diag + low-rank after each step. Accuracy depends on spectral properties.
3. **Defer**: Ship RWKV-7 without effective attention initially; add it when the math is worked out.

## Resolution (as implemented)

**Approach 1 (numerical) was chosen and shipped** as
`compute_effective_attention_v7` in `src/rwkv/mod.rs`. It builds the effective
attention matrix row by row from the diag+rank-1 recurrence inputs
(`r, k, w, kk, a`), materialising the cumulative contribution of each source
position to each query position rather than seeking a closed form. The result is
normalised (`ReLU` + L1) and surfaced through `HookPoint::RwkvEffectiveAttn(i)`
with shape `[batch, heads, seq_query, seq_source]`.

This is exact (no low-rank truncation) and validated to 6 decimal places against
the reference — acceptable because MI analyses run on short sequences where the
`O(T² · D²)` cost is not a bottleneck. Approaches 2 (low-rank approximation) and 3
(defer) were not needed.

### Design-time open questions, retrospectively

- *Closed-form for the product of (diag + rank-1) matrices?* — Not required; the
  numerical row-by-row build is exact and fast enough for interpretability
  workloads.
- *Right abstraction for RWKV-7?* — Yes; effective attention transfers cleanly and
  is the primitive exposed to users, consistent with the RWKV-6 path.
