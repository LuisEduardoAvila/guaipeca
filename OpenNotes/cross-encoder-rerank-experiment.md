# Cross-Encoder Re-ranking — Experiment Result (Negative)

**Date:** 2026-10-15
**Status:** Tested, measured, **rejected**. Not implemented. Do not ship without new evidence.
**Context:** Follow-up to the hybrid-default change (`0db7db9`). The eval harness
(`tools/eval_retrieval.py`) flagged three queries that no retrieval mode could fix,
and a cross-encoder re-ranker was the obvious candidate fix.

---

## Summary

A cross-encoder re-ranking stage (`Xenova/ms-marco-MiniLM-L-6-v2`, top-50 → top-k)
was prototyped, wired into `Searcher.search()` and measured against the existing
hybrid baseline on the 25-query golden set. **It degraded every metric.** The
prototype was reverted.

## Measured results

Same index, same 25 scored queries, `top-k=5`:

| mode | recall@5 | hit@1 | MRR |
|---|---|---|---|
| dense | 0.880 | 0.680 | 0.768 |
| **hybrid (shipped)** | **0.920** | **0.800** | **0.848** |
| hybrid + rerank (top-50) | 0.800 | 0.720 | 0.738 |

Full run at `top-k=10`: hybrid 0.960/0.800/0.854 vs hybrid+rerank 0.920/0.720/0.754.

## What it broke

Eight queries that hybrid already ranked correctly were pushed down or lost outright:

| query | hybrid | hybrid+rerank |
|---|---|---|
| "how do I set up single sign-on…" | 1 | **miss** |
| "scheduling a nightly process…" | 1 | 4 |
| "giving a colleague permission to only look at one department's figures" | 1 | 5 |
| "why does my spreadsheet pull the wrong period's data" | 1 | **miss** |
| "designing data entry screens for users" | 1 | **miss** |
| "locking a period so nobody can post adjustments" | 2 | **miss** |

## What it fixed (and why that wasn't enough)

It did rescue two of the three known-hard queries — exactly as predicted:

- "grouping several reports into a collection" (Books): miss → rank 1
- "I need to undo a change someone made last week" (audit trail): rank 5 → rank 1

But it traded two wins for eight losses. Net negative.

## Why it failed (analysis, not yet proven)

The cross-encoder was scoring against the `summary` field — heading + leading
text — to stay consistent with what the embedder sees. Two likely causes:

1. **Out-of-distribution input.** MS MARCO models are trained on short web
   passages. "Section heading + document fragment" is not that shape, and the
   model appears to reward lexical surface overlap with the heading while
   penalising chunks whose *body* answers the query but whose summary does not.
2. **Redundant with existing structure.** Guaipeca's structure-aware chunking
   already encodes the heading/title signal into every chunk. The re-ranker was
   re-deriving a signal the chunking already provides — and doing it worse.

The second point is the more important one: the architecture may make a generic
re-ranker largely redundant, not merely mistuned.

## Latency (measured on Pi 5, ARM64)

- Model size: ~80 MB (same class as the embedder; no new dependency —
  `fastembed`'s `TextCrossEncoder` was already available)
- Warm: **~15 ms/pair**; top-50 ≈ 0.8 s per query
- Cold load: ~2.6 s (one-time, cached)

Latency was *not* the blocker — accuracy was. Worth remembering before anyone
proposes re-ranking again on speed grounds.

## Untried variants (if this is ever revisited)

Neither was run, so both remain open — but note that (2) directly tests the
redundancy theory, and if that theory is right it will also fail:

1. Narrow the candidate pool (top-10 instead of top-50) to limit reordering damage.
2. Score full chunk `text` instead of `summary`.
3. Try a stronger/domain-adjacent model (`BAAI/bge-reranker-base`, ~1 GB — heavy
   for a Pi).

## Recommendation

**Do not implement cross-encoder re-ranking.** Hybrid search covers the ranking
problem adequately for this corpus and deployment. The three hard queries (IPM
insights, Books, audit trail) look like a *corpus/chunking* problem, not a
ranking problem — cheaper avenues to test first:

- Swap the embedding model (`bge-small-en-v1.5`, same 384 dims, no full rebuild).
- Inspect how those specific sections are chunked.

Re-open this note only with a concrete new hypothesis, not a re-run of the same
configuration.
