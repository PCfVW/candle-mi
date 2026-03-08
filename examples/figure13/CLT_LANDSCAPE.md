# Cross-Layer Transcoder Landscape

Available CLTs for Figure 13 replication and future directions.

## Currently supported (candle-mi presets)

| Preset | Base model | CLT repo | Features | Disk |
|--------|-----------|----------|----------|------|
| `llama3.2-1b-524k` | Llama 3.2 1B | `mntss/clt-llama-3.2-1b-524k` | 524K (16K/layer) | ~8 GB |
| `gemma2-2b-426k` | Gemma 2 2B | `mntss/clt-gemma-2-2b-426k` | 426K (16K/layer) | ~33 GB |
| `gemma2-2b-2.5m` | Gemma 2 2B | `mntss/clt-gemma-2-2b-2.5M` | 2.5M (98K/layer) | ~171 GB |

## Not compatible

| CLT repo | Base model | Why |
|----------|-----------|-----|
| `mntss/clt-131k` | `openai/gpt-oss-20b` | No candle-mi backend for GPT-OSS |

## Requires new backend

| CLT repo | Base model | Features | Blocker |
|----------|-----------|----------|---------|
| `bluelightai/clt-qwen3-0.6b-base-20k` | Qwen3 0.6B | ~573K (20K/layer) | Qwen3 backend (QK-Norm) |
| `bluelightai/clt-qwen3-1.7b-base-20k` | Qwen3 1.7B | ~573K (20K/layer) | Qwen3 backend (QK-Norm) |
| `google/gemma-scope-2-270m-pt` | Gemma 3 270M | TBD | Gemma3 backend |
| `google/gemma-scope-2-270m-it` | Gemma 3 270M | TBD | Gemma3 backend |
| `google/gemma-scope-2-1b-pt` | Gemma 3 1B | TBD | Gemma3 backend |
| `google/gemma-scope-2-1b-it` | Gemma 3 1B | TBD | Gemma3 backend |

## Backend effort estimates

### Qwen3 dense (moderate)

Qwen3 dense is architecturally close to Qwen2. Two changes needed:

1. **Remove attention bias** -- Qwen3 sets `attention_bias: false` (Qwen2 has bias
   on Q, K, V projections). Already config-driven in candle-mi, may work via
   auto-config once the second change is in place.
2. **QK-Norm** -- RMSNorm applied to Q and K heads before RoPE. New component,
   not currently in `GenericTransformer`.

MoE variants (Qwen3 128E/8A) are out of scope.

### Gemma 3 (harder)

Gemma 3 diverges more significantly from Gemma 2:

1. **5:1 local/global attention** -- Gemma 2 alternates 1:1; Gemma 3 uses 5 local
   layers per 1 global layer. Requires new interleaving logic.
2. **Per-layer RoPE theta** -- local layers use 10K, global layers use 1M.
3. **QK-Norm** -- same as Qwen3 (replaces Gemma 2's softcapping).
4. **Remove logit softcapping** -- simplification (less code).
5. **Vision encoder** -- out of scope for text-only MI.

### Shared prerequisite: QK-Norm

Both Qwen3 and Gemma 3 require QK-Norm. Adding this single component to
`GenericTransformer` would unblock Qwen3 dense (and partially unblock Gemma 3).

---

*Last updated: 2026-03-08*
