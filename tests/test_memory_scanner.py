"""Tests for skill.memory_scanner module."""

import tempfile
from pathlib import Path

import pytest

from skill.memory_scanner import (
    MIND_FILES,
    SOUL_FILES,
    MemoryFile,
    MemoryScanner,
    extract_headers,
    extract_key_terms,
)


# ---------------------------------------------------------------------------
# extract_headers
# ---------------------------------------------------------------------------

class TestExtractHeaders:
    def test_basic_headers(self):
        content = "# Title\n## Subtitle\n### Deep\nsome text\n"
        assert extract_headers(content) == ["Title", "Subtitle", "Deep"]

    def test_max_headers_limit(self):
        content = "\n".join(f"# Header {i}" for i in range(30))
        assert len(extract_headers(content, max_headers=5)) == 5

    def test_empty_content(self):
        assert extract_headers("") == []

    def test_skips_empty_header_text(self):
        content = "#\n## \n### Real Header\n"
        assert extract_headers(content) == ["Real Header"]

    def test_skips_single_char_header(self):
        content = "# A\n## BB\n"
        assert extract_headers(content) == ["BB"]


# ---------------------------------------------------------------------------
# extract_key_terms
# ---------------------------------------------------------------------------

class TestExtractKeyTerms:
    def test_bold_terms(self):
        content = "We discussed **budget** and **timeline** today."
        assert extract_key_terms(content) == ["budget", "timeline"]

    def test_max_terms_limit(self):
        content = " ".join(f"**term{i}**" for i in range(20))
        assert len(extract_key_terms(content, max_terms=5)) == 5

    def test_filters_short_terms(self):
        content = "**ab** and **valid term** here."
        assert extract_key_terms(content) == ["valid term"]

    def test_filters_long_terms(self):
        long_term = "x" * 61
        content = f"**{long_term}** and **ok term** here."
        assert extract_key_terms(content) == ["ok term"]

    def test_empty_content(self):
        assert extract_key_terms("") == []


# ---------------------------------------------------------------------------
# MemoryFile
# ---------------------------------------------------------------------------

class TestMemoryFile:
    def test_soul_category(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            soul = ws_path / "SOUL.md"
            soul.write_text("# Soul\nI am an agent.")
            mf = MemoryFile(soul, ws_path)
            assert mf.category == "soul"
            assert mf.rel_path == "SOUL.md"
            assert mf.size > 0

    def test_identity_category(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            f = ws_path / "IDENTITY.md"
            f.write_text("# Identity")
            mf = MemoryFile(f, ws_path)
            assert mf.category == "soul"

    def test_mind_category(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            f = ws_path / "MEMORY.md"
            f.write_text("# Memory index")
            mf = MemoryFile(f, ws_path)
            assert mf.category == "mind"

    def test_long_term_category(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            mem_dir = ws_path / "memory"
            mem_dir.mkdir()
            lt = mem_dir / "LONG_TERM.md"
            lt.write_text("# Long term memories")
            mf = MemoryFile(lt, ws_path)
            assert mf.category == "long-term"

    def test_daily_log_category(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            mem_dir = ws_path / "memory"
            mem_dir.mkdir()
            daily = mem_dir / "2026-03-01.md"
            daily.write_text("# March 1\nMet with **Alice**.")
            mf = MemoryFile(daily, ws_path)
            assert mf.category == "daily-log"

    def test_workspace_category(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            f = ws_path / "notes.md"
            f.write_text("# Random notes")
            mf = MemoryFile(f, ws_path)
            assert mf.category == "workspace"

    def test_to_dict(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            f = ws_path / "MEMORY.md"
            f.write_text("# Mem\n**topic one**\n")
            mf = MemoryFile(f, ws_path)
            d = mf.to_dict()
            assert d["path"] == "MEMORY.md"
            assert d["category"] == "mind"
            assert isinstance(d["chars"], int)
            assert isinstance(d["headers"], list)
            assert isinstance(d["key_terms"], list)

    def test_to_context_block(self):
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            f = ws_path / "SOUL.md"
            f.write_text("Hello world")
            mf = MemoryFile(f, ws_path)
            block = mf.to_context_block()
            assert "=== FILE: SOUL.md" in block
            assert "Hello world" in block


# ---------------------------------------------------------------------------
# MemoryScanner
# ---------------------------------------------------------------------------

class TestMemoryScanner:
    def _make_workspace(self, tmp: str) -> Path:
        """Create a realistic mock workspace."""
        ws = Path(tmp)
        (ws / "SOUL.md").write_text("# Soul\nI am Crick.")
        (ws / "IDENTITY.md").write_text("# Identity")
        (ws / "MEMORY.md").write_text("# Memory index\n**budget**")
        (ws / "USER.md").write_text("# User prefs")
        mem = ws / "memory"
        mem.mkdir()
        (mem / "LONG_TERM.md").write_text("# Long term\n**project alpha**")
        (mem / "2026-03-01.md").write_text("# March 1 log")
        (mem / "2026-03-02.md").write_text("# March 2 log")
        return ws

    def test_scan_memory_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            cats = {f.category for f in scanner.files}
            assert "soul" in cats
            assert "mind" in cats
            assert "long-term" in cats
            assert "daily-log" in cats

    def test_scan_identity_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="identity")
            cats = {f.category for f in scanner.files}
            assert "soul" in cats
            assert "mind" in cats
            assert "daily-log" not in cats
            assert "long-term" not in cats

    def test_scan_project_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "README.md").write_text("# Project readme")
            (ws / "notes.txt").write_text("some notes")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            cats = {f.category for f in scanner.files}
            assert "workspace" in cats
            # project scope alone doesn't run the soul/mind discovery
            assert "soul" not in cats

    def test_scan_all_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            (ws / "extra.txt").write_text("extra file")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="all")
            cats = {f.category for f in scanner.files}
            assert "soul" in cats
            assert "mind" in cats
            assert "long-term" in cats
            assert "workspace" in cats

    def test_empty_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            scanner = MemoryScanner(workspace=Path(tmp))
            scanner.scan(scope="memory")
            assert scanner.files == []

    def test_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            nested = ws / "subdir" / "deeper"
            nested.mkdir(parents=True)
            (nested / "notes.md").write_text("# Nested notes")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert any("subdir" in p for p in paths)

    def test_skips_binary_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "photo.jpg").write_bytes(b"\xff\xd8\xff")
            (ws / "notes.md").write_text("# Notes")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert "photo.jpg" not in paths
            assert "notes.md" in paths

    def test_skips_large_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "huge.md").write_text("x" * 200_000)
            (ws / "small.md").write_text("# Small")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert "huge.md" not in paths
            assert "small.md" in paths

    def test_skips_git_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            git_dir = ws / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("gitconfig")
            (ws / "readme.md").write_text("# Readme")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert not any(".git" in p for p in paths)

    def test_get_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            manifest = scanner.get_manifest()
            assert "MEMORY MANIFEST" in manifest
            assert "Total:" in manifest

    def test_get_manifest_triggers_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            manifest = scanner.get_manifest()
            assert "MEMORY MANIFEST" in manifest

    def test_get_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            ctx = scanner.get_context()
            assert "=== FILE:" in ctx

    def test_get_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            idx = scanner.get_index()
            assert idx["total_files"] > 0
            assert idx["total_chars"] > 0
            assert isinstance(idx["files"], list)
