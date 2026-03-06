"""Tests for skill.memory_indexer module."""

import tempfile
from pathlib import Path

import pytest

from skill.memory_indexer import (
    build_memory_index,
    extract_topics,
    update_memory_index,
)


# ---------------------------------------------------------------------------
# extract_topics
# ---------------------------------------------------------------------------

class TestExtractTopics:
    def test_extracts_headers(self):
        content = "# Title\n## Section A\nsome text\n### Detail\n"
        info = extract_topics(content, "test.md")
        assert "Title" in info["headers"]
        assert "Section A" in info["headers"]

    def test_extracts_people(self):
        content = "Met with **Alice** and **Bob Smith** today."
        info = extract_topics(content, "test.md")
        assert "Alice" in info["people"]
        assert "Bob Smith" in info["people"]

    def test_filters_non_people_names(self):
        content = "**Done** with **Overview** and **Alice** today."
        info = extract_topics(content, "test.md")
        assert "Alice" in info["people"]
        assert "Done" not in info["people"]
        assert "Overview" not in info["people"]

    def test_extracts_projects(self):
        content = "Working on deeprecall and the IDE integration."
        info = extract_topics(content, "test.md")
        assert "deeprecall" in info["projects"] or "deep recall" in info["projects"]
        assert "ide" in info["projects"]

    def test_extracts_keywords(self):
        content = "The **budget** was discussed. Also **timeline**."
        info = extract_topics(content, "test.md")
        assert "budget" in info["keywords"]
        assert "timeline" in info["keywords"]

    def test_extracts_summary_lines(self):
        content = "✅ Completed the migration to new server\n- [x] Also done\n"
        info = extract_topics(content, "test.md")
        assert len(info["summary_lines"]) >= 1

    def test_empty_content(self):
        info = extract_topics("", "empty.md")
        assert info["headers"] == []
        assert info["people"] == set()
        assert info["projects"] == set()
        assert info["keywords"] == set()
        assert info["summary_lines"] == []

    def test_various_file_formats(self):
        # Plain text without markdown
        content = "Just plain text with **Alice** and deeprecall mentioned."
        info = extract_topics(content, "notes.txt")
        assert "Alice" in info["people"]


# ---------------------------------------------------------------------------
# build_memory_index
# ---------------------------------------------------------------------------

class TestBuildMemoryIndex:
    def _make_workspace(self, tmp: str) -> Path:
        ws = Path(tmp)
        (ws / "MEMORY.md").write_text(
            "# Memory Index\n"
            "## Current Projects\n"
            "**budget** discussion with **Alice**\n"
            "Working on deeprecall\n"
        )
        mem = ws / "memory"
        mem.mkdir()
        (mem / "2026-03-01.md").write_text(
            "# March 1\n"
            "## Morning\n"
            "Met with **Bob** about **timeline**.\n"
            "✅ Finished phase 1\n"
        )
        (mem / "2026-03-02.md").write_text(
            "# March 2\n"
            "## Afternoon\n"
            "Reviewed deeprecall code with **Alice**.\n"
        )
        return ws

    def test_basic_index_building(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "# Memory Index" in index
            assert "## Timeline" in index

    def test_timeline_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "2026-03-01" in index
            assert "2026-03-02" in index

    def test_people_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "## People" in index
            assert "Alice" in index

    def test_projects_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "## Projects" in index
            assert "deeprecall" in index

    def test_topics_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "## Topics" in index

    def test_key_events_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "## Key Events" in index

    def test_empty_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = build_memory_index(workspace=Path(tmp))
            assert "# Memory Index" in index
            assert "## Timeline" in index

    def test_empty_memory_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            mem = ws / "memory"
            mem.mkdir()
            (mem / "2026-01-01.md").write_text("")
            index = build_memory_index(workspace=ws)
            assert "2026-01-01" in index

    def test_memory_md_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = self._make_workspace(tmp)
            index = build_memory_index(workspace=ws)
            assert "## MEMORY.md Sections" in index

    def test_non_date_files_in_long_term_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            mem = ws / "memory"
            mem.mkdir()
            (mem / "LONG_TERM.md").write_text("# Long term\n## Chess Game\nPlayed on Feb 27")
            (mem / "random_notes.md").write_text("# Random")
            index = build_memory_index(workspace=ws)
            # Non-date files should appear in Long-Term section, not Timeline
            assert "LONG_TERM.md" in index
            assert "random_notes.md" in index
            assert "Long-Term Memory Files" in index


# ---------------------------------------------------------------------------
# update_memory_index
# ---------------------------------------------------------------------------

class TestUpdateMemoryIndex:
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "MEMORY.md").write_text("# Mem")
            path = update_memory_index(workspace=ws)
            assert path.exists()
            assert path.name == "MEMORY_INDEX.md"
            content = path.read_text()
            assert "# Memory Index" in content

    def test_max_size_reasonable(self):
        """Index should not be excessively large even with many files."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            mem = ws / "memory"
            mem.mkdir()
            for i in range(50):
                (mem / f"2026-01-{i+1:02d}.md").write_text(
                    f"# Day {i+1}\n**topic{i}** discussion\n✅ Done item {i}\n"
                )
            index = build_memory_index(workspace=ws)
            # Topics limited to top 30
            topic_count = index.count("→ `memory/")
            assert topic_count <= 200  # reasonable bound
