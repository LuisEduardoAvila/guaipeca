"""Config file encoding handling.

A guaipeca.yaml written by a different tool — Notepad, a Windows editor, or an
agent whose default encoding is cp1252 — must either load with a warning or
fail with an actionable message. It must never fail with a bare
UnicodeDecodeError raised from inside PyYAML, which names neither the file nor
the encoding.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from guaipeca.config import GuaipecaConfig

# A topic containing an em dash, the character most likely to differ between
# UTF-8 and cp1252 when a config is authored on Windows.
TOPIC = "EPM & Oracle HFM/ARCS — consolidacao"


def _config_dict(tmp_path) -> dict:
    return {
        "corpora": [
            {"name": "test", "path": str(tmp_path), "topic": TOPIC, "extensions": [".md"]}
        ],
    }


def _write(tmp_path, encoding: str, *, bom: bool = False):
    text = yaml.dump(_config_dict(tmp_path), allow_unicode=True)
    path = tmp_path / f"guaipeca-{encoding}.yaml"
    payload = text.encode(encoding)
    if bom and encoding.startswith("utf-8"):
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)
    return path


class TestConfigEncoding:
    def test_utf8_loads(self, tmp_path):
        path = _write(tmp_path, "utf-8")
        cfg = GuaipecaConfig.from_yaml(str(path))
        assert cfg.corpora[0].topic == TOPIC

    def test_utf8_with_bom_loads(self, tmp_path):
        """Notepad writes a UTF-8 BOM; it must not break the loader."""
        path = _write(tmp_path, "utf-8", bom=True)
        cfg = GuaipecaConfig.from_yaml(str(path))
        assert cfg.corpora[0].topic == TOPIC

    def test_cp1252_loads_with_warning(self, tmp_path, caplog):
        """A cp1252 config loads, but is reported so it gets fixed."""
        path = _write(tmp_path, "cp1252")
        with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
            cfg = GuaipecaConfig.from_yaml(str(path))

        assert cfg.corpora[0].topic == TOPIC
        assert any("not UTF-8" in r.getMessage() for r in caplog.records)

    def test_utf8_emits_no_warning(self, tmp_path, caplog):
        """The normal case must stay silent."""
        path = _write(tmp_path, "utf-8")
        with caplog.at_level(logging.WARNING, logger="guaipeca.config"):
            GuaipecaConfig.from_yaml(str(path))

        assert not [r for r in caplog.records if "UTF-8" in r.getMessage()]

    def test_undecodable_reports_file_offset_and_hint(self, tmp_path):
        """Bytes undecodable in every known encoding get an actionable error."""
        path = tmp_path / "broken.yaml"
        # 0x81 is undefined in cp1252 and invalid UTF-8, so both attempts fail.
        path.write_bytes(b"corpora:\n  - name: a\n    path: /tmp\n    topic: \x81end\n")

        with pytest.raises(ValueError) as excinfo:
            GuaipecaConfig.from_yaml(str(path))

        message = str(excinfo.value)
        assert str(path) in message
        assert "UTF-8" in message
        assert "offset" in message
        assert "line" in message

    def test_missing_file_reports_path(self, tmp_path):
        """A missing config names the file rather than raising FileNotFoundError."""
        missing = tmp_path / "nope.yaml"
        with pytest.raises(ValueError, match="Cannot read config file"):
            GuaipecaConfig.from_yaml(str(missing))
