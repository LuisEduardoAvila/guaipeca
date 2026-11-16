"""Server transport config validation.

``server.transport`` accepts exactly ``stdio``, ``http`` and ``both``. ``both``
means "stdio when launched as a subprocess, HTTP when launched as a daemon", not
two simultaneous listeners. The legacy ``sse`` value was removed in the MCP SDK
migration, so it must be rejected with a message that says where to go instead
(``http`` serves ``POST /mcp`` for both protocol eras).
"""

from __future__ import annotations

import pytest
import yaml

from guaipeca.config import GuaipecaConfig


def _write_config(tmp_path, transport: str):
    data = {
        "corpora": [
            {"name": "test", "path": str(tmp_path), "extensions": [".md"]}
        ],
        "server": {"transport": transport},
    }
    path = tmp_path / f"guaipeca-{transport}.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


class TestTransportValidation:
    @pytest.mark.parametrize("transport", ["stdio", "http", "both"])
    def test_transport_accepts_stdio_http_both(self, tmp_path, transport):
        cfg = GuaipecaConfig.from_yaml(str(_write_config(tmp_path, transport)))
        assert cfg.server.transport == transport

    def test_transport_rejects_unknown_value(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            GuaipecaConfig.from_yaml(str(_write_config(tmp_path, "websocket")))

        message = str(excinfo.value)
        assert "websocket" in message
        assert "stdio" in message
        assert "http" in message
        assert "both" in message

    def test_transport_sse_error_explains_removal(self, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            GuaipecaConfig.from_yaml(str(_write_config(tmp_path, "sse")))

        message = str(excinfo.value)
        assert "removed" in message.lower()
        assert "/mcp" in message
