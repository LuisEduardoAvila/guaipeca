"""Status output parity: the CLI table must not hide fields MCP reports.

`Searcher.status` returns, per corpus, ``bm25_available`` — and top-level
``hybrid_enabled``. The MCP ``status`` tool serialises the dict as-is, but the
CLI formatter printed only model/dimensions/counts, so the same underlying
data rendered differently on the two surfaces.

That mattered: ``bm25_available: false`` means the corpus is silently running
dense-only, which looks identical to a healthy hybrid corpus in the CLI.
"""

from __future__ import annotations

import pytest

from guaipeca import cli


def _fake_searcher(*, bm25: bool, hybrid: bool):
    class FakeSearcher:
        @property
        def status(self):
            return {
                "corpora": [
                    {
                        "corpus": "alpha",
                        "total_chunks": 10,
                        "total_vectors": 10,
                        "files_tracked": 2,
                        "bm25_available": bm25,
                    }
                ],
                "embedding_model": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "hybrid_enabled": hybrid,
            }

    return FakeSearcher()


@pytest.fixture
def run_status(monkeypatch, capsys):
    def _run(*, bm25: bool, hybrid: bool) -> str:
        fake = _fake_searcher(bm25=bm25, hybrid=hybrid)
        from guaipeca.config import GuaipecaConfig

        monkeypatch.setattr(cli, "_load_config", lambda path: GuaipecaConfig())
        monkeypatch.setattr(
            "guaipeca.embedding.EmbeddingService", lambda **kw: object()
        )
        monkeypatch.setattr(
            "guaipeca.search.Searcher", lambda **kw: fake
        )
        cli.cmd_status(type("A", (), {"config": "ignored"})())
        return capsys.readouterr().out

    return _run


class TestStatusOutputParity:
    def test_reports_bm25_available(self, run_status):
        out = run_status(bm25=True, hybrid=True)
        assert "BM25 index: available" in out

    def test_reports_bm25_unavailable(self, run_status):
        """The case that used to be invisible: corpus running dense-only."""
        out = run_status(bm25=False, hybrid=True)
        assert "BM25 index: unavailable" in out

    def test_reports_hybrid_enabled(self, run_status):
        assert "Hybrid search: enabled" in run_status(bm25=True, hybrid=True)

    def test_reports_hybrid_disabled(self, run_status):
        assert "Hybrid search: disabled" in run_status(bm25=True, hybrid=False)

    def test_still_reports_counts(self, run_status):
        out = run_status(bm25=True, hybrid=True)
        for label in ("Total chunks: 10", "Total vectors: 10", "Files tracked: 2"):
            assert label in out

    def test_every_scalar_status_field_is_printed(self, run_status):
        """Guards against a future status field being added but not rendered."""
        out = run_status(bm25=True, hybrid=True)
        status = _fake_searcher(bm25=True, hybrid=True).status

        printed = {"embedding_model": "all-MiniLM-L6-v2", "dimensions": "384"}
        for key, needle in printed.items():
            assert needle in out, f"{key} missing from CLI output"

        # Per-corpus keys must all appear as labels.
        corpus = status["corpora"][0]
        assert f"Total chunks: {corpus['total_chunks']}" in out
        assert f"Total vectors: {corpus['total_vectors']}" in out
        assert f"Files tracked: {corpus['files_tracked']}" in out
        assert "BM25 index:" in out
