# Model Alignment Note: bge-base vs bge-small

**Status: DIAGNOSTIC ONLY — no changes made.**
**Date: 2026-09-22**

## Purpose

Compare BAAI/bge-base-en-v1.5 (768-dim, 210MB) vs BAAI/bge-small-en-v1.5
(384-dim, 67MB) as embedding models for Guaipeca. The current default is
all-MiniLM-L6-v2 (384-dim, 90MB).

## Method (for the human to run later)

1. **Create a test config** with bge-base-en-v1.5:
   ```yaml
   embedding:
     model: BAAI/bge-base-en-v1.5
     dimensions: 768
     cache_dir: ~/.guaipeca/models
   ```

2. **Reindex a test corpus** (use a small subset for speed):
   ```bash
   guaipeca index --force --config test-config.yaml
   ```

3. **Run a set of benchmark queries** covering:
   - Exact-token queries (e.g. "EnablePelimNewLogic")
   - Semantic/concept queries (e.g. "how to configure consolidation rules")
   - Mixed queries (e.g. "FCCS currency translation setup")

4. **Compare with bge-small-en-v1.5** (same 384-dim, drop-in replacement):
   ```yaml
   embedding:
     model: BAAI/bge-small-en-v1.5
     dimensions: 384
   ```

5. **Evaluate:**
   - Recall@5: does the correct chunk appear in top-5?
   - MRR (Mean Reciprocal Rank): how high does the correct chunk rank?
   - Index size: FAISS index memory/disk footprint
   - Indexing time: seconds per document
   - Query latency: milliseconds per query

## Expected Trade-offs

| Factor | bge-small (384-dim) | bge-base (768-dim) |
|--------|-------------------|-------------------|
| Model size | 67MB | 210MB |
| Index size | ~same as MiniLM | ~2x (768-dim vectors) |
| Indexing time | ~same | ~1.5-2x slower |
| Query latency | ~same | ~1.5-2x slower |
| Quality | Better than MiniLM | Best (per benchmarks) |

## Important Notes

- **Do NOT change the default embedding model.** This is diagnostic only.
- **Do NOT touch the oracle-epm-docs skill's index.** It uses all-MiniLM-L6-v2.
- **Changing dimensions requires a full reindex** (FAISS vectors must match).
- On Pi 5 (ARM64, 8GB RAM), the 210MB model + 2x index size may be
  significant. Consider available memory before switching.
- bge-small is a drop-in replacement (same 384-dim) — no reindex needed
  beyond the model swap. bge-base requires reindex.

## Recommendation

Start with bge-small-en-v1.5 (same dimensions, better quality than MiniLM,
no reindex cost beyond model swap). Only upgrade to bge-base if quality
gains justify the 2x index size and slower latency on Pi 5.