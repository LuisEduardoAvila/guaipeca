#!/usr/bin/env python3
"""Retrieval quality harness for Guaipeca.

Measures recall@k and MRR for a golden query set against a live Guaipeca
deployment, comparing search modes side by side (dense-only vs hybrid vs
hybrid+toc_rerank).

Why this exists
---------------
Guaipeca's `search.hybrid` and `search.toc_rerank` are OFF by default for
backward compatibility. Before flipping those defaults we need evidence, not
priors: does hybrid actually improve retrieval on THIS user's corpora, or does
it trade keyword precision for recall loss?

Design
------
- Imports the real `Searcher` from the installed package, so the harness
  measures actual production retrieval (chunking, corpus weights, RRF fusion,
  toc_rerank) rather than a reimplementation.
- A golden query is scored as a HIT if ANY expected source file appears in the
  top-k results. This avoids requiring exact chunk-ID stability across modes.
- MRR uses the rank of the first correct result (1-based).
- Modes are measured over the SAME loaded index in one process, so the
  comparison is apples-to-apples (same embedding cache, same model).

Golden set format (YAML or JSON)
--------------------------------
    queries:
      - query: "HFM consolidation journal entries"
        corpus: epm-docs          # optional: restrict search to one corpus
        expect:                   # any of these filenames/paths counts as a hit
          - hfm_admin_guide.pdf
          - consolidation_notes.md
        notes: "should hit the data-audit section"

`expect` entries are matched by basename OR by substring of the result's
source_path / location, case-insensitive.

Usage
-----
    python tools/eval_retrieval.py --queries tools/golden_queries.yaml
    python tools/eval_retrieval.py --queries ... --top-k 5,10 --json out.json
    python tools/eval_retrieval.py --queries ... --modes dense,hybrid

Exit code is 0 always (this is a measurement tool, not a gate); use --fail-on
later if you want CI gating.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Golden set loading
# ---------------------------------------------------------------------------

SUPPORTED_SET_SUFFIXES = (".yaml", ".yml", ".json")


def load_golden_set(path: str) -> list[dict]:
    """Load a golden query set from YAML or JSON.

    Returns a list of normalised query dicts:
        {"query": str, "corpus": str|None, "expect": list[str], "notes": str}
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"error: golden set not found: {path}")

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise SystemExit(
                "error: PyYAML not installed; use a .json golden set or "
                "`pip install pyyaml`"
            )
        data = yaml.safe_load(text)
    else:
        raise SystemExit(
            f"error: unsupported golden set format {suffix!r}; "
            f"use one of {SUPPORTED_SET_SUFFIXES}"
        )

    if isinstance(data, dict):
        queries = data.get("queries", [])
    elif isinstance(data, list):
        queries = data
    else:
        raise SystemExit("error: golden set must be a list or {'queries': [...]}")

    normalised = []
    for i, raw in enumerate(queries):
        if isinstance(raw, str):
            raw = {"query": raw, "expect": []}
        if not isinstance(raw, dict) or not raw.get("query"):
            raise SystemExit(f"error: query #{i} missing 'query' field")

        expect = raw.get("expect") or raw.get("expected") or []
        if isinstance(expect, str):
            expect = [expect]

        normalised.append(
            {
                "query": str(raw["query"]).strip(),
                "corpus": raw.get("corpus") or raw.get("corpora"),
                "expect": [str(e) for e in expect],
                "notes": str(raw.get("notes", "")),
            }
        )
    return normalised


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _result_keys(result: dict) -> list[str]:
    """All lowercased strings a golden 'expect' entry can match against."""
    keys = []
    for field in ("source_path", "location", "filename"):
        val = result.get(field)
        if val:
            keys.append(str(val).lower())
            keys.append(os.path.basename(str(val)).lower())
    return keys


def _matches(result: dict, expected: str) -> bool:
    needle = expected.lower().strip()
    if not needle:
        return False
    return any(needle in key for key in _result_keys(result))


def first_hit_rank(results: list[dict], expect: list[str]) -> int | None:
    """1-based rank of the first result matching any expected source, else None."""
    if not expect:
        return None
    for rank, r in enumerate(results, start=1):
        if any(_matches(r, e) for e in expect):
            return rank
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_mode(
    searcher,
    queries: list[dict],
    hybrid: bool,
    toc_rerank: bool,
    top_k: int,
) -> dict:
    """Run all queries in one mode; return aggregate metrics + per-query detail.

    `toc_rerank` is toggled by mutating the config object the Searcher already
    holds (the Searcher reads `config.search.toc_rerank` at query time), so no
    re-instantiation is needed.
    """
    # Toggle toc_rerank on the live config (read at query time by Searcher).
    searcher.config.search.toc_rerank = toc_rerank

    ranks: list[int | None] = []
    per_query = []

    for q in queries:
        corpora = None
        if q["corpus"]:
            corpora = (
                [c.strip() for c in q["corpus"].split(",")]
                if isinstance(q["corpus"], str)
                else list(q["corpus"])
            )

        res = searcher.search(q["query"], top_k=top_k, corpora=corpora, hybrid=hybrid)
        results = res.get("results", [])
        rank = first_hit_rank(results, q["expect"])

        ranks.append(rank)
        per_query.append(
            {
                "query": q["query"],
                "expect": q["expect"],
                "notes": q["notes"],
                "rank": rank,
                "top_files": [
                    os.path.basename(str(r.get("source_path", ""))) for r in results[:5]
                ],
                "error": res.get("error"),
            }
        )

    scored = [q for q in queries if q["expect"]]
    n = len(scored)
    hits_1 = sum(1 for r in ranks if r == 1)
    hits = sum(1 for r in ranks if r is not None)
    mrr = (sum(1.0 / r for r in ranks if r is not None) / n) if n else 0.0

    return {
        "mode": f"hybrid={hybrid},toc_rerank={toc_rerank}",
        "hybrid": hybrid,
        "toc_rerank": toc_rerank,
        "top_k": top_k,
        "n_scored": n,
        "hits": hits,
        f"recall@{top_k}": round(hits / n, 4) if n else 0.0,
        "hit@1": round(hits_1 / n, 4) if n else 0.0,
        "mrr": round(mrr, 4),
        "per_query": per_query,
    }


def print_report(reports: list[dict], scored_total: int) -> None:
    print()
    print("=" * 64)
    print(f"Guaipeca retrieval eval  (scored queries: {scored_total})")
    print("=" * 64)
    header = f"{'mode':<34} {'recall@k':>9} {'hit@1':>7} {'MRR':>7}"
    print(header)
    print("-" * len(header))
    for rep in reports:
        k = rep["top_k"]
        print(
            f"{rep['mode']:<34} "
            f"{rep[f'recall@{k}']:>9.3f} "
            f"{rep['hit@1']:>7.3f} "
            f"{rep['mrr']:>7.3f}"
        )
    print("-" * len(header))


def print_per_query(reports: list[dict]) -> None:
    print()
    print("Per-query first-hit rank  ('.' = miss)")
    print("-" * 64)
    queries = reports[0]["per_query"] if reports else []
    for i, q in enumerate(queries):
        rank_cells = []
        for rep in reports:
            r = rep["per_query"][i]["rank"]
            rank_cells.append(f"{r}" if r is not None else ".")
        print(f"  [{' '.join(f'{c:>3}' for c in rank_cells)}]  {q['query'][:52]}")
    print()
    print("  column order: " + " | ".join(rep["mode"] for rep in reports))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_top_k(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("error: --top-k must contain at least one integer")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure Guaipeca retrieval quality (recall@k / MRR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="~/.guaipeca/guaipeca.yaml",
                    help="Guaipeca config (default: ~/.guaipeca/guaipeca.yaml)")
    ap.add_argument("--queries", required=True,
                    help="Golden query set (.yaml or .json)")
    ap.add_argument("--top-k", default="5", type=parse_top_k,
                    help="Comma-separated k values to report, e.g. 5,10 (default: 5)")
    ap.add_argument("--modes", default="dense,hybrid",
                    help="Comma-separated modes: dense,hybrid (default: dense,hybrid)")
    ap.add_argument("--toc-rerank", action="store_true",
                    help="Also evaluate with toc_rerank enabled")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="Write full results as JSON to this path")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    # Import here so --help works without the package installed.
    from guaipeca.config import GuaipecaConfig
    from guaipeca.converter import Converter
    from guaipeca.embedding import EmbeddingService
    from guaipeca.search import Searcher

    config_path = os.path.expanduser(args.config)
    config = GuaipecaConfig.from_yaml(config_path)

    queries = load_golden_set(args.queries)
    scored = [q for q in queries if q["expect"]]
    if not queries:
        raise SystemExit("error: golden set is empty")
    if not scored:
        print(
            "warning: no query has an 'expect' list — nothing can be scored. "
            "Add expected source filenames to score recall.",
            file=sys.stderr,
        )

    print(f"Loaded {len(queries)} queries ({len(scored)} scored) from {args.queries}")
    print(f"Config: {config_path}")

    converter = Converter(cache_dir=str(Path(config.indexing.data_dir) / "converted"))
    embedder = EmbeddingService(
        model_name=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
        dimensions=config.embedding.dimensions,
        preprocess=config.embedding.preprocess,
        cache_ttl_seconds=config.embedding.cache_ttl_seconds,
        cache_max_entries=config.embedding.cache_max_entries,
    )
    searcher = Searcher(config=config, embedding_service=embedder, converter=converter)

    mode_specs = []
    for m in args.modes.split(","):
        m = m.strip().lower()
        if m == "dense":
            mode_specs.append((False, False))
        elif m == "hybrid":
            mode_specs.append((True, False))
        else:
            raise SystemExit(f"error: unknown mode {m!r} (use dense,hybrid)")

    if args.toc_rerank:
        mode_specs += [(False, True), (True, True)]
        # dedupe while preserving order
        seen = set()
        mode_specs = [m for m in mode_specs if not (m in seen or seen.add(m))]

    reports = []
    for k in args.top_k:
        for hybrid, toc in mode_specs:
            rep = evaluate_mode(searcher, queries, hybrid=hybrid, toc_rerank=toc, top_k=k)
            reports.append(rep)

    # Print one combined report per k (reports are ordered k-major).
    per_k = len(mode_specs)
    for i, k in enumerate(args.top_k):
        group = reports[i * per_k:(i + 1) * per_k]
        print_report(group, len(scored))

    if args.verbose:
        print_per_query(reports)

    if args.json_out:
        out = {
            "golden_set": args.queries,
            "config": config_path,
            "scored_queries": len(scored),
            "reports": reports,
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
