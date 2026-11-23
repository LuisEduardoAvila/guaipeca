"""GuaipecaConfig.from_yaml: single source of truth for defaults (issue #67).

from_yaml used to restate every dataclass default as a literal ``.get()``
fallback, and the copies had drifted: ``search.hybrid`` was ``True`` in the
dataclass but ``False`` in the loader. ``_build()`` now filters YAML keys to
dataclass fields and lets the dataclass defaults apply, so the field
declaration is the only place a default is written.
"""

from __future__ import annotations

import logging

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
