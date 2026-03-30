# JSON output files should be self-contained

**Date:** March 27, 2026
**Source:** COLM 2026 paper preparation (prolepsis paper)
**Affected example:** `attention_routing`
**Severity:** Design improvement (not a bug)

---

## The problem

When writing Appendix E (full attention routing head tables) of the
COLM 2026 paper, we needed to document the exact experimental
parameters for each Llama routing run: which features were suppressed,
which word was being suppressed, which word was being injected, and
which prompt was used.

The **inject feature** and **prompt** were stored in the JSON output.
The **suppress features** and **suppress/inject words** were not.

This meant that to reconstruct the full experiment specification, we
had to:

1. Cross-reference the JSON output with the CLI command that produced it
2. Look up the correction report (`llama-routing-correction-N4.md`)
3. Look up `rhyme_pairs_llama.json` for the feature-to-word mapping

None of this information was in the output file itself.

## Why this matters

A JSON output file should answer the question: **"What experiment
produced this data?"** without requiring any external context. This is
important for three reasons:

### 1. Reproducibility

A researcher reading the paper's appendix should be able to trace
every number back to a specific output file, and that output file
should contain everything needed to re-run the experiment. If the
suppress features are missing, the experiment cannot be reproduced
from the output file alone.

### 2. Provenance tracking

When you run many experiments over weeks (as we did for the N=4
Llama routing expansion), the CLI commands scroll off the terminal.
The only durable record is the JSON output. If it omits parameters,
you lose provenance.

### 3. Avoiding hallucination-like errors

The original Llama routing result used a **hallucinated feature**
(L5:19894) that was fabricated during a context continuation. One
reason it wasn't caught earlier is that the JSON output didn't
record the suppress features, so there was no easy way to audit
"which features were actually used?" after the fact. If the suppress
features had been in the JSON, the discrepancy with
`rhyme_pairs_llama.json` would have been visible immediately.

## The fix

### What `attention_routing.rs` currently outputs

```json
{
  "model_id": "meta-llama/Llama-3.2-1B",
  "clt_repo": "mntss/clt-llama-3.2-1b-524k",
  "prompt": "The birds were singing in the tree...",
  "feature": "L14:13043",
  "strength": 10.0,
  "planning_site": 14,
  "output_position": 31,
  "n_layers": 16,
  "n_heads": 32,
  "head_deltas": [ ... ]
}
```

### What it should output

```json
{
  "model_id": "meta-llama/Llama-3.2-1B",
  "clt_repo": "mntss/clt-llama-3.2-1b-524k",
  "prompt": "The birds were singing in the tree...",
  "inject_feature": "L14:13043",
  "inject_word": "that",
  "suppress_features": ["L13:30985", "L9:5488", "L14:27874", "L13:32049"],
  "suppress_word": "free",
  "strength": 10.0,
  "planning_site": 14,
  "output_position": 31,
  "n_layers": 16,
  "n_heads": 32,
  "head_deltas": [ ... ]
}
```

The additions are:

| Field | Why |
|-------|-----|
| `inject_word` | Maps the feature ID to a human-readable word |
| `suppress_features` | Records ALL features that were suppressed |
| `suppress_word` | Maps the suppress features to a human-readable word |

### Comparison with `figure13_planning_poems.rs`

The `figure13_planning_poems` example already does this correctly:

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "clt_repo": "mntss/clt-llama-3.2-1b-524k",
  "prompt": "...",
  "suppress_word": "cat",
  "inject_word": "that",
  "suppress_features": [{"layer": 5, "index": 19894}],
  "inject_feature": {"layer": 14, "index": 13043},
  "strength": 15.0,
  "baseline_prob": 0.000017,
  "sweep": [ ... ]
}
```

Every parameter is recorded. The `attention_routing` example should
follow the same pattern.

## General rule for all candle-mi examples

**Every JSON output file should contain all the information needed to
reproduce the experiment that generated it.** Specifically:

1. **Model and CLT identifiers** (already done everywhere)
2. **All intervention parameters:** inject features, suppress features,
   strength, planning site position
3. **Human-readable labels:** the word(s) corresponding to each feature
   ID, so a reader doesn't need to cross-reference `rhyme_pairs_*.json`
4. **The prompt** (already done in most examples)
5. **Metadata:** framework version, timestamp, hardware (optional but
   helpful)

If a parameter is passed on the command line, it should appear in the
output. If someone can't reconstruct the full `cargo run` command from
the JSON file alone, the output is incomplete.
