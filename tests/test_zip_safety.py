"""Tests for zip decompression safety (issue #60)."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from guaipeca.converter import Converter


class TestZipSafety:
    """Decompression-bomb protection for .zip files."""

    def _make_zip(self, tmp_path: Path, member_count: int, member_size: int) -> str:
        """Create a zip file with the given number of members, each of member_size bytes."""
        zip_path = str(tmp_path / "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(member_count):
                zf.writestr(f"file_{i}.txt", "x" * member_size)
        return zip_path

    def test_small_zip_converts_successfully(self, tmp_path):
        """A small, legitimate zip file should convert without error."""
        zip_path = self._make_zip(tmp_path, member_count=3, member_size=100)
        converter = Converter()
        # Should not raise
        result = converter._convert_zip_safe(zip_path)
        assert "x" in result

    def test_too_many_members_rejected(self, tmp_path):
        """A zip with >1000 members should be rejected."""
        zip_path = self._make_zip(tmp_path, member_count=1001, member_size=10)
        converter = Converter()
        with pytest.raises(ValueError, match="members"):
            converter._convert_zip_safe(zip_path)

    def test_excessive_uncompressed_size_rejected(self, tmp_path):
        """A zip whose uncompressed size exceeds the cap should be rejected."""
        # Create a zip with 1 member of 300 MB (well above 200 MB cap)
        zip_path = self._make_zip(tmp_path, member_count=1, member_size=300 * 1024 * 1024)
        converter = Converter()
        with pytest.raises(ValueError, match="uncompressed size"):
            converter._convert_zip_safe(zip_path)

    def test_zip_within_cap_converts(self, tmp_path):
        """A zip within both caps should convert."""
        # 10 members of 1 KB each = 10 KB total, well within caps
        zip_path = self._make_zip(tmp_path, member_count=10, member_size=1024)
        converter = Converter()
        result = converter._convert_zip_safe(zip_path)
        assert len(result) > 0
