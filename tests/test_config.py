"""GuaipecaConfig.from_yaml: single source of truth for defaults (issue #67).

from_yaml used to restate every dataclass default as a literal ``.get()``
fallback, and the copies had drifted: ``search.hybrid`` was ``True`` in the
dataclass but ``False`` in the loader. ``_build()`` now filters YAML keys to
dataclass fields and lets the dataclass defaults apply, so the field
declaration is the only place a default is written.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from guaipeca.config import (
    ChunkingConfig,
    CorpusConfig,
    EmbeddingConfig,
    GuaipecaConfig,
    IndexingConfig,
    SearchConfig,
    ServerConfig,
    UploadConfig,
)


def _full_config_dict(tmp_path) -> dict:
    """A YAML mapping with an explicit value for every field of every section."""
    return {
        "corpora": [
            {
                "name": "docs",
                "path": str(tmp_path / "corpus"),
                "topic": "t",
                "weight": 1.5,
                "extensions": [".md"],
                "max_distance": 0.5,
            }
        ],
        "embedding": {
            "model": "m",
            "dimensions": 384,
            "cache_dir": str(tmp_path / "models"),
            "preprocess": False,
            "cache_ttl_seconds": 60,
            "cache_max_entries": 10,
        },
        "chunking": {
            "max_size": 1000,
            "overlap": 50,
            "table_aware": False,
            "split_on_headings": False,
        },
        "indexing": {
            "data_dir": str(tmp_path / "data"),
            "incremental": False,
            "max_file_size": 1024,
            "exclude_dirs": ["node_modules"],
        },
        "server": {
            "transport": "http",
            "port": 9999,
            "host": "0.0.0.0",
            "auth_token": "tok",
            "max_body_size": 12345,
            "allowed_origins": ["http://localhost:9999"],
        },
        "upload": {
            "allow": ["docs"],
            "max_file_size": 2048,
            "allowed_extensions": [".md"],
            "auto_index": False,
            "replace": {"docs": True},
        },
        "search": {
            "hybrid": False,
            "bm25_weight": 0.5,
            "dense_weight": 1.5,
            "rrf_k": 30,
            "max_chars": 100,
            "toc_rerank": True,
            "toc_rerank_weight": 0.7,
            "fusion_alpha": 0.9,
            "max_distance": 0.25,
        },
    }


def _write(tmp_path, data: dict, name: str = "guaipeca.yaml") -> str:
    path = tmp_path / name
    path.write_text(yaml.dump(data), encoding="utf-8")
    return str(path)


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "guaipeca.config"
    ]


# ---------------------------------------------------------------------------
# Round trip: explicit values for every field load byte-identical
# ---------------------------------------------------------------------------


def test_round_trip_explicit_values(tmp_path):
    """A config that sets every field explicitly loads identically to direct
    construction of the same dataclasses — the loader neither rewrites nor
    re-defaults anything."""
    data = _full_config_dict(tmp_path)
    cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))

    expected = GuaipecaConfig(
        corpora=[CorpusConfig(**data["corpora"][0])],
        embedding=EmbeddingConfig(**data["embedding"]),
        chunking=ChunkingConfig(**data["chunking"]),
        indexing=IndexingConfig(**data["indexing"]),
        server=ServerConfig(**data["server"]),
        upload=UploadConfig(**data["upload"]),
        search=SearchConfig(**data["search"]),
    )
    # Every parsed section must equal direct construction of the same values;
    # config_path is the path from_yaml was given, so it is compared separately.
    assert (cfg.corpora, cfg.embedding, cfg.chunking, cfg.indexing,
            cfg.server, cfg.upload, cfg.search) == (
        expected.corpora, expected.embedding, expected.chunking,
        expected.indexing, expected.server, expected.upload, expected.search,
    )
    assert cfg.config_path == str(tmp_path / "guaipeca.yaml")


def test_round_trip_preserves_explicit_hybrid_false(tmp_path):
    """Explicit search.hybrid: false must stay false (no default bleed-through)."""
    data = _full_config_dict(tmp_path)
    cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
    assert cfg.search.hybrid is False


# ---------------------------------------------------------------------------
# Defaults come from the dataclass, not from loader copies
# ---------------------------------------------------------------------------


def test_omitted_sections_take_dataclass_defaults(tmp_path):
    """Sections the YAML omits must equal the bare dataclass instances."""
    data = {"corpora": [{"name": "t", "path": str(tmp_path)}]}
    cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
    assert cfg.embedding == EmbeddingConfig()
    assert cfg.chunking == ChunkingConfig()
    assert cfg.indexing == IndexingConfig()
    assert cfg.server == ServerConfig()
    assert cfg.upload == UploadConfig()
    assert cfg.search == SearchConfig()


def test_hybrid_default_is_the_dataclass_default(tmp_path):
    """Without a search section, hybrid takes the dataclass default (True),
    not a loader-side copy. The #67 drift (dataclass True, loader False) is
    structurally impossible now."""
    data = {"corpora": [{"name": "t", "path": str(tmp_path)}]}
    cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
    assert cfg.search.hybrid is True
    assert cfg.search.hybrid is SearchConfig().hybrid


def test_partial_section_merges_with_defaults(tmp_path):
    """Keys the YAML sets win; keys it omits fall back to dataclass defaults."""
    data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
            "search": {"rrf_k": 30}}
    cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
    assert cfg.search.rrf_k == 30
    assert cfg.search.hybrid is True
    assert cfg.search.bm25_weight == 1.0


# ---------------------------------------------------------------------------
# Unknown keys warn (naming key and section) instead of being dropped silently
# ---------------------------------------------------------------------------


def test_unknown_section_key_warns(tmp_path, caplog):
    data = {
        "corpora": [{"name": "t", "path": str(tmp_path)}],
        "embedding": {"model": "m", "bogus_key": 1},
        "search": {"rrf_k_typo": 5},
    }
    with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
        cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))

    messages = _warnings(caplog)
    assert any("bogus_key" in m and "embedding" in m for m in messages)
    assert any("rrf_k_typo" in m and "search" in m for m in messages)
    # Unknown keys are dropped, not fatal; known keys still load.
    assert cfg.embedding.model == "m"
    assert cfg.search.rrf_k == 60


def test_unknown_top_level_key_warns(tmp_path, caplog):
    data = {
        "corpora": [{"name": "t", "path": str(tmp_path)}],
        "searching": {"hybrid": True},
    }
    with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
        cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))

    messages = _warnings(caplog)
    assert any("searching" in m for m in messages)
    assert cfg.search.hybrid is True  # dataclass default, not the typo'd section


def test_unknown_corpus_key_warns_with_corpus_name(tmp_path, caplog):
    data = {"corpora": [
        {"name": "docs", "path": str(tmp_path), "extentions": [".md"]},
    ]}
    with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
        cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))

    messages = _warnings(caplog)
    assert any("extentions" in m and "docs" in m for m in messages)
    # Corpus still loads with its default extensions.
    assert cfg.corpora[0].extensions == [".md", ".txt"]


def test_known_keys_do_not_warn(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
        GuaipecaConfig.from_yaml(_write(tmp_path, _full_config_dict(tmp_path)))
    assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# Issue #66+99: config validation rejects wrong types and out-of-range values
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Validation for every field the issue calls out."""

    # fusion_alpha (issue #99)

    def test_fusion_alpha_out_of_range_rejected(self, tmp_path):
        for val in [5.0, -2.0]:
            data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                    "search": {"fusion_alpha": val}}
            with pytest.raises(ValueError, match=r"fusion_alpha.*\[0, 1\]"):
                GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_fusion_alpha_boundaries_accepted(self, tmp_path):
        for val in [0.0, 0.3, 1.0]:
            data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                    "search": {"fusion_alpha": val}}
            cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
            assert cfg.search.fusion_alpha == val

    def test_fusion_alpha_wrong_type_rejected(self, tmp_path):
        for val in ["0.3", True]:
            data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                    "search": {"fusion_alpha": val}}
            with pytest.raises(ValueError, match=r"fusion_alpha"):
                GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # toc_rerank_weight

    def test_toc_rerank_weight_negative_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"toc_rerank_weight": -1.0}}
        with pytest.raises(ValueError, match="toc_rerank_weight"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_toc_rerank_weight_wrong_type_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"toc_rerank_weight": "x"}}
        with pytest.raises(ValueError, match="toc_rerank_weight"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # toc_rerank

    def test_toc_rerank_string_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"toc_rerank": "yes"}}
        with pytest.raises(ValueError, match="toc_rerank must be a boolean"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # corpus.weight

    def test_corpus_weight_negative_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path), "weight": -1}]}
        with pytest.raises(ValueError, match="weight.*>= 0"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_corpus_weight_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path), "weight": True}]}
        with pytest.raises(ValueError, match="weight.*must be a number"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_corpus_weight_zero_accepted(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path), "weight": 0}]}
        cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
        assert cfg.corpora[0].weight == 0

    # rrf_k bool-as-int

    def test_rrf_k_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"rrf_k": True}}
        with pytest.raises(ValueError, match="rrf_k"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # max_body_size upper bound

    def test_max_body_size_exceeds_upper_bound_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "server": {"max_body_size": 2_000_000_000}}
        with pytest.raises(ValueError, match="max_body_size.*1 GB"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # upload.allow type check

    def test_upload_allow_string_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "upload": {"allow": "mydocs"}}
        with pytest.raises(ValueError, match="upload.allow"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # upload.max_file_size

    def test_upload_max_file_size_negative_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "upload": {"max_file_size": -100}}
        with pytest.raises(ValueError, match="upload.max_file_size"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # CorpusConfig.extensions=None coercion

    def test_corpus_extensions_none_coerced(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path), "extensions": None}]}
        cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
        assert cfg.corpora[0].extensions == [".md", ".txt"]

    # Bool-as-int for other integer fields

    def test_chunking_max_size_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "chunking": {"max_size": True}}
        with pytest.raises(ValueError, match="chunking.max_size"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_chunking_overlap_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "chunking": {"overlap": True}}
        with pytest.raises(ValueError, match="chunking.overlap"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_indexing_max_file_size_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "indexing": {"max_file_size": True}}
        with pytest.raises(ValueError, match="indexing.max_file_size"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_server_port_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "server": {"port": True}}
        with pytest.raises(ValueError, match="server.port"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_search_max_chars_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"max_chars": True}}
        with pytest.raises(ValueError, match="max_chars"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # Float fields bool-as-int

    def test_bm25_weight_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"bm25_weight": True}}
        with pytest.raises(ValueError, match="bm25_weight"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    def test_dense_weight_bool_rejected(self, tmp_path):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "search": {"dense_weight": True}}
        with pytest.raises(ValueError, match="dense_weight"):
            GuaipecaConfig.from_yaml(_write(tmp_path, data))

    # max_body_size warning for > 100 MB

    def test_max_body_size_warning_above_100mb(self, tmp_path, caplog):
        data = {"corpora": [{"name": "t", "path": str(tmp_path)}],
                "server": {"max_body_size": 200 * 1024 * 1024}}
        with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
            cfg = GuaipecaConfig.from_yaml(_write(tmp_path, data))
        messages = _warnings(caplog)
        assert any("max_body_size" in m and "100 MB" in m for m in messages)
