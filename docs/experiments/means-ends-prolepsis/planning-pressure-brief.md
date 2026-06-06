# Planning pressure — why (and when) a model commits early

*A short pedagogical brief comparing the poetry domain (Anthropic, established) with
an upgraded means-ends prompt (proposed), through one shared lens: the three
ingredients that create **planning pressure**.*

> **Provenance note.** The "planning pressure / three ingredients" framing below is
> **our** hypothesis for *why* early commitment arises — it is **not** stated as
> such in Anthropic's *Planning in Poems* (they describe forward/backward planning,
> the newline planning site, and writing-toward, but do not abstract these
> necessary conditions). The **poetry** cell is Anthropic's established result; the
> **means-ends** couplet is a proposed, **not-yet-tested** upgrade.

## 0. The planning-pressure problem

A language model is trained to predict one token at a time, so its default is to
**improvise**: write each token from the local context as it goes. But sometimes the
token it will eventually have to produce is **pinned by two constraints at once** —
one of them fixed *earlier in the text* — that **cannot be reconciled at the last
moment**. If the model improvised the whole run-up first and only *then* reached for
the final word, it could **paint itself into a corner**: no available word satisfies
both constraints *and* the span it has just committed to. The way out of that bind is
to **decide the target word early**, at a "planning site" before the span, and then
**write the span toward it**. That early, span-shaping decision *is* planning (early
commitment). So the diagnostic question for any domain is: **does the task put two
constraints on a target word across a span, such that improvising at the last token
fails?** If yes, we should expect to see early commitment; if no, we should not.

The same structure can be described as **three ingredients**:

- **① Constrained target** — a single token forced by *two* constraints, at least one set earlier; not locally improvisable.
- **② Intervening span** — tokens between the *planning site* and the target.
- **③ Span shaped by the target** — the run-up is written *toward* the chosen target (edit the target ⇒ the span restructures).

---

## 1. POETRY prompt (Anthropic — established)

```
A rhyming couplet:
He saw a carrot and had to grab it,
His hunger was like a starving ⟨?⟩          →  rabbit
```

The model commits to **rabbit** at the **newline after `grab it,`** and writes the
second line toward it.

- **① Constrained target** — `rabbit` must satisfy **both** *sound* (rhyme with `grab it`, the `-abit` sound, fixed by **line 1**) **and** *sense* (`a starving ⟨?⟩` should be a hungry animal). A purely local guess at the last token risks a word that rhymes but is nonsense, or makes sense but does not rhyme.
- **② Intervening span** — the run-up `His hunger was like a starving …` sits between the planning site (the newline after line 1) and the target word ⟨?⟩.
- **③ Span shaped by the target** — to land coherently on `rabbit`, line 2 is built as a simile (`like a starving …`) that naturally ends on an animal; **editing the planned word** (e.g. to `habit`) **restructures the line** (`His hunger was a powerful habit`) — Anthropic Fig 15 (§4.3) and Fig 16 (§4.4).

> Second constraint here = **sense** (semantic coherence).

---

## 2. MEANS-ENDS prompt (proposed upgrade — versified)

```
A couplet about a lamp:
The room is dim, the mood is low,
We want it bright, so make it ⟨?⟩            →  glow   (brighten ≈ turn it ON)
```
*(The **flip** needs its own line-1 rhyme anchor, since `low` then becomes the
target: "The lamp is harsh, its bulbs **aglow**, / We want it dark, so turn it ⟨?⟩"
→ `low` (dim ≈ turn it OFF). `glow` / `low` / `aglow` all rhyme `-ow`.)*

The plain means-ends prompt (`…We want the room dark. Turn the lamp → off`) has **no**
planning pressure — the action is a deterministic, locally-derivable function of the
goal, and it is the *immediate* next token. Versifying it adds the missing second
constraint, so the action word becomes dual-constrained across a span:

- **① Constrained target** — `glow` must satisfy **both** *sound* (rhyme with `low`, the `-ow` sound, fixed by **line 1**) **and** *goal* (the action that achieves the desired state: want-bright → `glow` ≈ on). A purely local guess at the last token risks a word that rhymes but is the *wrong action*, or the right action but does not rhyme.
- **② Intervening span** — the run-up `We want it bright, so make it …` sits between the planning site (the newline after `low,`) and the target word ⟨?⟩.
- **③ Span shaped by the target** — to land coherently on `glow` vs `low`, line 2 is written toward it; **editing the planned word** (flip the goal) **restructures the line**.

> Second constraint here = **goal** (the means-ends action selection) — the *only*
> change from poetry. Everything else is identical.

> **Craft caveat.** Rhyme forces the target away from the cell's literal `on`/`off`
> to near-synonyms (`glow`/`low` ≈ brighten/dim). Any concrete instance must still be
> validated as we validated the rhyme cells: each target a **single token**, with a
> **clean CLT feature**, and the couplet **competence-gated** (does the base model
> complete it goal-correctly?). `glow`/`low` here is illustrative, not yet verified.

---

## The point of the comparison

| | Constraint A (set early) | Constraint B | Target (rhyme-valid set; B selects) | Planning site |
|---|---|---|---|---|
| **Poetry** | rhyme (`-abit`) | **sense** | `rabbit` / `habit` | newline after line 1 |
| **Means-ends** | rhyme (`-ow`) | **goal** | `glow` / `low` | newline after line 1 |

Poetry and the upgraded means-ends prompt share the **same three ingredients**; they
differ in *one slot* — poetry's second constraint is **sense**, means-ends' is the
**goal**. So the same instruments that found planning in poetry apply unchanged: the
**Figure 13 (§4.2)** position sweep for an early, planning-site spike; the **§4.3**
"intermediate words are written toward the target" test (Fig 14); and the
**§4.3–4.4** edit-the-target-restructures-the-line test (Figs 15–16). If the
means-ends couplet shows that early, planning-site spike, it is **planning in the
same sense Anthropic demonstrated** — obtained by giving the means-ends action a
rhyme to plan around.
