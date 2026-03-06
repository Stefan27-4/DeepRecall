"""Tests for skill.memory_scanner module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_get_context_auto_scans_if_not_scanned(self):
        """get_context() should call scan() automatically."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            # Do NOT call scan() manually
            ctx = scanner.get_context()
            assert "=== FILE:" in ctx
            assert scanner._scanned

    def test_get_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            idx = scanner.get_index()
            assert idx["total_files"] > 0
            assert idx["total_chars"] > 0
            assert isinstance(idx["files"], list)

    def test_get_index_auto_scans_if_not_scanned(self):
        """get_index() should call scan() automatically."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            # Do NOT call scan() manually
            idx = scanner.get_index()
            assert idx["total_files"] > 0
            assert scanner._scanned

    def test_skips_unreadable_files_gracefully(self):
        """Files that raise an exception during MemoryFile construction are skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "good.md").write_text("# Good file")
            (ws / "bad.md").write_text("# Would fail")

            original_init = __import__("skill.memory_scanner", fromlist=["MemoryFile"]).MemoryFile.__init__

            call_count = [0]

            def patched_init(self, path, workspace):
                call_count[0] += 1
                if path.name == "bad.md":
                    raise PermissionError("cannot read")
                return original_init(self, path, workspace)

            from skill import memory_scanner
            with patch.object(memory_scanner.MemoryFile, "__init__", patched_init):
                scanner = MemoryScanner(workspace=ws)
                scanner.scan(scope="project")

            paths = [f.rel_path for f in scanner.files]
            assert "good.md" in paths
            assert "bad.md" not in paths

    def test_manifest_shows_no_headers_fallback(self):
        """Files with no headers show '(no headers)' in manifest."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "MEMORY.md").write_text("no headers here, just text")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            manifest = scanner.get_manifest()
            assert "(no headers)" in manifest

    def test_scan_is_idempotent(self):
        """Calling scan() twice resets and rebuilds the file list."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            count_first = len(scanner.files)
            scanner.scan(scope="memory")
            count_second = len(scanner.files)
            assert count_first == count_second

    def test_lock_file_skipped(self):
        """Files with .lock extension should be skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "package.lock").write_text("lock content")
            (ws / "notes.md").write_text("# Notes")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert "package.lock" not in paths


# ---------------------------------------------------------------------------
# Additional edge-case and regression tests
# ---------------------------------------------------------------------------

class TestMemoryFileCategoryEdgeCases:
    def test_soul_name_in_subdirectory_is_daily_log(self):
        """memory/SOUL.md must NOT be classified as soul — it's a daily-log."""
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            mem = ws_path / "memory"
            mem.mkdir()
            f = mem / "SOUL.md"
            f.write_text("# Not really a soul file")
            mf = MemoryFile(f, ws_path)
            assert mf.category == "daily-log"

    def test_mind_name_in_subdirectory_is_daily_log(self):
        """memory/MEMORY.md must NOT be classified as mind — it's a daily-log."""
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            mem = ws_path / "memory"
            mem.mkdir()
            f = mem / "MEMORY.md"
            f.write_text("# Not a mind file")
            mf = MemoryFile(f, ws_path)
            assert mf.category == "daily-log"

    def test_rel_path_uses_path_object_not_recomputed(self):
        """rel_path string and MemoryFile.path must be consistent."""
        with tempfile.TemporaryDirectory() as ws:
            ws_path = Path(ws)
            f = ws_path / "SOUL.md"
            f.write_text("# Soul")
            mf = MemoryFile(f, ws_path)
            assert mf.rel_path == str(mf.path.relative_to(ws_path))


class TestMemoryScannerEdgeCases:
    def _make_workspace(self, tmp: str) -> Path:
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

    def test_scan_returns_self_for_chaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            result = scanner.scan(scope="memory")
            assert result is scanner

    def test_chained_scan_get_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            manifest = MemoryScanner(workspace=ws).scan(scope="memory").get_manifest()
            assert "MEMORY MANIFEST" in manifest

    def test_rescan_resets_file_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="all")
            count_all = len(scanner.files)
            scanner.scan(scope="identity")
            count_identity = len(scanner.files)
            assert count_identity < count_all
            assert count_identity > 0

    def test_all_scope_no_duplicate_paths(self):
        """scope='all' must never add the same file twice."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            (ws / "extra.txt").write_text("extra file")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="all")
            paths = [f.path for f in scanner.files]
            assert len(paths) == len(set(paths)), "Duplicate files found in scope='all'"

    def test_skips_node_modules_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            nm = ws / "node_modules" / "some-pkg"
            nm.mkdir(parents=True)
            (nm / "index.js").write_text("module.exports = {}")
            (ws / "app.js").write_text("console.log('hi')")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert not any("node_modules" in p for p in paths)
            assert "app.js" in paths

    def test_skips_pycache_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            pc = ws / "__pycache__"
            pc.mkdir()
            (pc / "module.cpython-312.pyc").write_bytes(b"\x00" * 10)
            (ws / "module.py").write_text("x = 1")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="project")
            paths = [f.rel_path for f in scanner.files]
            assert not any("__pycache__" in p for p in paths)
            assert "module.py" in paths

    def test_daily_log_soul_named_file_not_misclassified(self):
        """A file at memory/SOUL.md must appear as daily-log, not soul."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            mem = ws / "memory"
            mem.mkdir()
            (mem / "SOUL.md").write_text("# Log entry about soul topics")
            scanner = MemoryScanner(workspace=ws)
            scanner.scan(scope="memory")
            categories = {f.category for f in scanner.files}
            assert "daily-log" in categories
            assert "soul" not in categories

    def test_to_dict_key_terms_capped_at_five(self):
        """to_dict() exports at most 5 key_terms for compact index output."""
        with tempfile.TemporaryDirectory() as tmp:
            ws_path = Path(tmp)
            terms = " ".join(f"**term{i:02d}**" for i in range(10))
            f = ws_path / "MEMORY.md"
            f.write_text(terms)
            mf = MemoryFile(f, ws_path)
            assert len(mf.key_terms) == 10
            assert len(mf.to_dict()["key_terms"]) == 5
