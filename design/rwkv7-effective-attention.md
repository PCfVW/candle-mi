# Design: RWKV-7 Effective Attention Formula

**Status:** Proposed
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

## Open questions

- Is there a closed-form for the product of (diag + rank-1) matrices?
- Does the RWKV-7 paper or fla codebase provide any equivalent computation?
- Is effective attention even the right abstraction for RWKV-7, or should we use a different interpretability primitive?
