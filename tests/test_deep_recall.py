"""Tests for skill.deep_recall module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

import pytest

from skill.deep_recall import (
    _find_workspace,
    _manager_call,
    _read_file,
    _synthesis_call,
    _worker_call,
    recall,
    recall_deep,
    recall_quick,
)
from skill.provider_bridge import ProviderConfig


def _mock_provider() -> ProviderConfig:
    return ProviderConfig(
        provider="openai",
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        primary_model="openai/gpt-4o",
    )


def _make_workspace(tmp: str) -> Path:
    ws = Path(tmp)
    (ws / "MEMORY.md").write_text("# Memory\n**budget** topic\n## Projects\n")
    mem = ws / "memory"
    mem.mkdir()
    (mem / "2026-03-01.md").write_text(
        "# March 1\nWe decided the budget is $50k.\n**Alice** approved."
    )
    (mem / "2026-03-02.md").write_text(
        "# March 2\nTimeline extended to June.\n"
    )
    (mem / "LONG_TERM.md").write_text("# Long term\nOld decisions here.\n")
    return ws


# ---------------------------------------------------------------------------
# _find_workspace
# ---------------------------------------------------------------------------

class TestFindWorkspace:
    def test_from_env(self):
        with patch.dict("os.environ", {"OPENCLAW_WORKSPACE": "/tmp/test_ws"}):
            assert _find_workspace() == Path("/tmp/test_ws")

    def test_default_fallback(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("skill.deep_recall.Path.exists", return_value=False):
            ws = _find_workspace()
            assert "openclaw" in str(ws).lower()


# ---------------------------------------------------------------------------
# _read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_reads_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "test.md").write_text("hello")
            assert _read_file("test.md", ws) == "hello"

    def test_returns_none_for_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert _read_file("nonexistent.md", Path(tmp)) is None

    def test_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            result = _read_file("../../etc/passwd", ws)
            assert result is None


# ---------------------------------------------------------------------------
# _manager_call
# ---------------------------------------------------------------------------

class TestManagerCall:
    def test_returns_file_list(self):
        provider = _mock_provider()
        mock_response = json.dumps({"files": ["memory/2026-03-01.md"]})

        with patch("skill.deep_recall._chat", return_value=mock_response):
            files = _manager_call("budget?", "index text", 3, provider)

        assert files == ["memory/2026-03-01.md"]

    def test_respects_max_files(self):
        provider = _mock_provider()
        mock_response = json.dumps({
            "files": ["f1.md", "f2.md", "f3.md", "f4.md", "f5.md"]
        })

        with patch("skill.deep_recall._chat", return_value=mock_response):
            files = _manager_call("query", "index", 2, provider)

        assert len(files) <= 2

    def test_handles_invalid_json(self):
        provider = _mock_provider()
        with patch("skill.deep_recall._chat", return_value="not json"):
            files = _manager_call("query", "index", 3, provider)
        assert files == []


# ---------------------------------------------------------------------------
# _worker_call
# ---------------------------------------------------------------------------

class TestWorkerCall:
    def test_extracts_quotes(self):
        provider = _mock_provider()
        mock_response = json.dumps({
            "quotes": [{"text": "Budget is $50k", "line": 2}]
        })

        with patch("skill.deep_recall._chat", return_value=mock_response):
            result = _worker_call("budget?", "memory/2026-03-01.md",
                                  "Budget is $50k\n", provider)

        assert result["file"] == "memory/2026-03-01.md"
        assert len(result["quotes"]) == 1
        assert result["quotes"][0]["text"] == "Budget is $50k"

    def test_handles_api_error(self):
        provider = _mock_provider()
        with patch("skill.deep_recall._chat", side_effect=Exception("API error")):
            result = _worker_call("query", "file.md", "content", provider)
        assert result["quotes"] == []

    def test_anti_hallucination_prompt(self):
        """Verify worker prompt contains exact-quote instructions."""
        provider = _mock_provider()
        mock_response = json.dumps({"quotes": []})

        with patch("skill.deep_recall._chat", return_value=mock_response) as mock_chat:
            _worker_call("query", "file.md", "content", provider)

        call_args = mock_chat.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]
        system_msg = messages[0]["content"]
        assert "exact" in system_msg.lower() or "quote" in system_msg.lower()
        assert "paraphrase" in system_msg.lower() or "EXACTLY" in system_msg


# ---------------------------------------------------------------------------
# _synthesis_call
# ---------------------------------------------------------------------------

class TestSynthesisCall:
    def test_no_quotes_returns_default(self):
        provider = _mock_provider()
        result = _synthesis_call("query", [], provider)
        assert "don't have memories" in result.lower()

    def test_no_quotes_in_results(self):
        provider = _mock_provider()
        worker_results = [{"file": "f.md", "quotes": []}]
        result = _synthesis_call("query", worker_results, provider)
        assert "don't have memories" in result.lower()

    def test_synthesizes_quotes(self):
        provider = _mock_provider()
        worker_results = [{
            "file": "memory/2026-03-01.md",
            "quotes": [{"text": "Budget is $50k", "line": 2}],
        }]
        mock_answer = "The budget was set to $50k (memory/2026-03-01.md:2)."

        with patch("skill.deep_recall._chat", return_value=mock_answer):
            result = _synthesis_call("budget?", worker_results, provider)

        assert "$50k" in result

    def test_handles_synthesis_error(self):
        provider = _mock_provider()
        worker_results = [{
            "file": "f.md",
            "quotes": [{"text": "data", "line": 1}],
        }]
        with patch("skill.deep_recall._chat", side_effect=Exception("timeout")):
            result = _synthesis_call("query", worker_results, provider)
        assert "Synthesis failed" in result


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

class TestRecall:
    def _patch_recall(self, manager_files, worker_quotes, synthesis_answer):
        """Helper to patch all LLM calls for recall()."""
        provider = _mock_provider()

        def fake_chat(messages, prov, json_mode=False):
            system = messages[0]["content"]
            if "file selector" in system.lower() or "memory-file selector" in system.lower():
                return json.dumps({"files": manager_files})
            elif "quote extractor" in system.lower():
                return json.dumps({"quotes": worker_quotes})
            else:
                return synthesis_answer

        return patch("skill.deep_recall.resolve_provider", return_value=provider), \
               patch("skill.deep_recall._chat", side_effect=fake_chat)

    def test_full_recall_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            p1, p2 = self._patch_recall(
                manager_files=["memory/2026-03-01.md"],
                worker_quotes=[{"text": "Budget is $50k", "line": 2}],
                synthesis_answer="Budget was $50k per March 1 notes.",
            )
            with p1, p2:
                result = recall("What is the budget?", workspace=ws)
            assert "50k" in result

    def test_no_memory_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)  # empty workspace
            provider = _mock_provider()
            with patch("skill.deep_recall.resolve_provider", return_value=provider):
                result = recall("query", workspace=ws)
            assert "No memory files" in result

    def test_manager_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()
            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=Exception("API down")):
                result = recall("query", workspace=ws)
            assert "Manager call failed" in result

    def test_no_files_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()
            mock_resp = json.dumps({"files": []})
            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", return_value=mock_resp):
                result = recall("query", workspace=ws)
            assert "No relevant memory files" in result

    def test_provider_resolution_failure(self):
        with patch("skill.deep_recall.resolve_provider",
                   side_effect=RuntimeError("No provider")):
            with pytest.raises(RuntimeError, match="cannot resolve"):
                recall("query")


# ---------------------------------------------------------------------------
# recall_quick
# ---------------------------------------------------------------------------

class TestRecallQuick:
    def test_uses_max_2_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            calls = []

            def fake_chat(messages, prov, json_mode=False):
                system = messages[0]["content"]
                calls.append(system)
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": ["memory/2026-03-01.md"]})
                elif "quote extractor" in system.lower():
                    return json.dumps({"quotes": []})
                return "No memories."

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                result = recall_quick("Who am I?", verbose=False)

            # Check that the manager prompt limited to 2 files
            manager_prompts = [c for c in calls if "file selector" in c.lower()]
            for p in manager_prompts:
                assert "2" in p  # max_files=2 appears in the prompt

    def test_uses_identity_scope(self):
        """recall_quick uses scope='identity'."""
        with patch("skill.deep_recall.recall") as mock_recall:
            mock_recall.return_value = "result"
            recall_quick("test query")
            mock_recall.assert_called_once_with(
                "test query",
                scope="identity",
                verbose=False,
                config_overrides={"max_files": 2},
            )


# ---------------------------------------------------------------------------
# recall_deep
# ---------------------------------------------------------------------------

class TestRecallDeep:
    def test_uses_max_5_files(self):
        """recall_deep should set max_files=5."""
        with patch("skill.deep_recall.recall") as mock_recall:
            mock_recall.return_value = "result"
            recall_deep("summarize everything")
            mock_recall.assert_called_once_with(
                "summarize everything",
                scope="all",
                verbose=False,
                config_overrides={"max_files": 5},
            )

    def test_uses_all_scope(self):
        """recall_deep uses scope='all'."""
        with patch("skill.deep_recall.recall") as mock_recall:
            mock_recall.return_value = "result"
            recall_deep("query")
            call_kwargs = mock_recall.call_args
            assert call_kwargs[1]["scope"] == "all" or call_kwargs[0][1] == "all"


# ---------------------------------------------------------------------------
# Anti-hallucination checks
# ---------------------------------------------------------------------------

class TestAntiHallucination:
    def test_worker_prompt_exact_quote(self):
        """Worker system prompt must instruct LLM to quote exactly."""
        provider = _mock_provider()
        mock_response = json.dumps({"quotes": []})

        with patch("skill.deep_recall._chat", return_value=mock_response) as mock_chat:
            _worker_call("query", "f.md", "content line", provider)

        messages = mock_chat.call_args[0][0]
        system = messages[0]["content"]
        # Must contain anti-hallucination instructions
        assert "exact" in system.lower() or "EXACTLY" in system
        assert "not paraphrase" in system.lower() or "Do not paraphrase" in system

    def test_synthesis_prompt_cite(self):
        """Synthesis prompt must require citations."""
        provider = _mock_provider()
        worker_results = [{
            "file": "f.md",
            "quotes": [{"text": "data", "line": 1}],
        }]

        with patch("skill.deep_recall._chat", return_value="answer") as mock_chat:
            _synthesis_call("query", worker_results, provider)

        messages = mock_chat.call_args[0][0]
        system = messages[0]["content"]
        assert "cite" in system.lower() or "Cite" in system


# ---------------------------------------------------------------------------
# Timeout / error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_worker_timeout_graceful(self):
        """Workers that time out should not crash recall."""
        provider = _mock_provider()
        with patch("skill.deep_recall._chat", side_effect=TimeoutError("timed out")):
            result = _worker_call("q", "f.md", "content", provider)
        assert result["quotes"] == []

    def test_missing_file_in_worker_phase(self):
        """If manager selects a file that doesn't exist, recall should not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            def fake_chat(messages, prov, json_mode=False):
                system = messages[0]["content"]
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": ["nonexistent/ghost.md"]})
                return "No memories."

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                result = recall("query", workspace=ws)
            # Should not crash; returns a coherent message
            assert isinstance(result, str)
