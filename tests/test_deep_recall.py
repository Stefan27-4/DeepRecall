"""Tests for skill.deep_recall module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch, ANY

import pytest

import skill.deep_recall as _dr_module
from skill.deep_recall import (
    _MAX_TOOL_ROUNDS,
    _execute_tool_code,
    _find_workspace,
    _chat,
    _manager_call,
    _read_file,
    _safe_path,
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

    def test_returns_none_on_read_exception(self):
        """If read_text() raises an exception, _read_file returns None."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "test.md").write_text("data")
            with patch("skill.deep_recall.Path.read_text", side_effect=PermissionError("no read")):
                result = _read_file("test.md", ws)
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

    def test_handles_json_in_markdown_fence(self):
        provider = _mock_provider()
        wrapped = '```json\n{"files": ["memory/log.md"]}\n```'
        with patch("skill.deep_recall._chat", return_value=wrapped):
            files = _manager_call("query", "index", 3, provider)
        assert files == ["memory/log.md"]

    def test_handles_json_with_leading_text(self):
        provider = _mock_provider()
        raw = 'Here is the JSON:\n{"files": ["a.md"]}'
        with patch("skill.deep_recall._chat", return_value=raw):
            files = _manager_call("query", "index", 3, provider)
        assert files == ["a.md"]


# ---------------------------------------------------------------------------
# _sanitize_json
# ---------------------------------------------------------------------------

class TestSanitizeJson:
    def test_clean_json_passes_through(self):
        from skill.deep_recall import _sanitize_json
        assert _sanitize_json('{"a": 1}') == '{"a": 1}'

    def test_strips_markdown_json_fence(self):
        from skill.deep_recall import _sanitize_json
        raw = '```json\n{"key": "value"}\n```'
        assert json.loads(_sanitize_json(raw)) == {"key": "value"}

    def test_strips_plain_markdown_fence(self):
        from skill.deep_recall import _sanitize_json
        raw = '```\n[1, 2, 3]\n```'
        assert json.loads(_sanitize_json(raw)) == [1, 2, 3]

    def test_finds_json_after_preamble(self):
        from skill.deep_recall import _sanitize_json
        raw = 'Sure, here is the result:\n{"files": []}'
        assert json.loads(_sanitize_json(raw)) == {"files": []}

    def test_array_json(self):
        from skill.deep_recall import _sanitize_json
        assert json.loads(_sanitize_json('[1, 2]')) == [1, 2]

    def test_empty_string(self):
        from skill.deep_recall import _sanitize_json
        assert _sanitize_json("") == ""

    def test_whitespace_around_json(self):
        from skill.deep_recall import _sanitize_json
        raw = '  \n {"a": 1} \n '
        assert json.loads(_sanitize_json(raw)) == {"a": 1}

    def test_no_json_returns_original(self):
        from skill.deep_recall import _sanitize_json
        raw = "no json here at all"
        assert _sanitize_json(raw) == raw


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

    def test_handles_json_in_markdown_fence(self):
        provider = _mock_provider()
        wrapped = '```json\n{"quotes": [{"text": "hello", "line": 1}]}\n```'
        with patch("skill.deep_recall._chat", return_value=wrapped):
            result = _worker_call("query", "file.md", "content", provider)
        assert len(result["quotes"]) == 1
        assert result["quotes"][0]["text"] == "hello"

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

    def test_worker_model_no_prefix_doubling(self):
        """Worker provider model should never have a doubled prefix."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = ProviderConfig(
                provider="anthropic",
                api_key="sk-test",
                base_url="https://api.anthropic.com/v1",
                primary_model="anthropic/claude-sonnet-4",
            )

            captured_providers = []

            def fake_chat(messages, prov, json_mode=False):
                captured_providers.append(prov)
                system = messages[0]["content"]
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": ["memory/2026-03-01.md"]})
                elif "quote extractor" in system.lower():
                    return json.dumps({"quotes": []})
                return "No memories."

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                recall("query", workspace=ws)

            worker_provs = [p for p in captured_providers
                            if "haiku" in p.primary_model]
            for wp in worker_provs:
                assert not wp.primary_model.startswith("anthropic/anthropic/"), \
                    f"Doubled prefix in worker model: {wp.primary_model}"


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


# ---------------------------------------------------------------------------
# _chat
# ---------------------------------------------------------------------------

class TestChat:
    def test_routes_through_call_llm(self):
        """_chat delegates to provider_bridge.call_llm."""
        provider = _mock_provider()

        with patch("skill.deep_recall.call_llm", return_value="result") as mock_llm:
            result = _chat([{"role": "user", "content": "hi"}], provider)

        mock_llm.assert_called_once()
        assert result == "result"

    def test_passes_json_mode(self):
        provider = _mock_provider()

        with patch("skill.deep_recall.call_llm", return_value="{}") as mock_llm:
            _chat([{"role": "user", "content": "q"}], provider, json_mode=True)

        _, kwargs = mock_llm.call_args
        assert kwargs.get("json_mode") is True

    def test_passes_timeout(self):
        provider = _mock_provider()

        with patch("skill.deep_recall.call_llm", return_value="ok") as mock_llm:
            _chat([{"role": "user", "content": "q"}], provider)

        _, kwargs = mock_llm.call_args
        assert kwargs.get("timeout") == 120


# ---------------------------------------------------------------------------
# _manager_call edge cases
# ---------------------------------------------------------------------------

class TestManagerCallEdgeCases:
    def test_non_list_files_returns_empty(self):
        """If LLM returns files as a non-list, return []."""
        provider = _mock_provider()
        mock_response = json.dumps({"files": "not-a-list"})

        with patch("skill.deep_recall._chat", return_value=mock_response):
            files = _manager_call("query", "index", 3, provider)

        assert files == []

    def test_files_cast_to_str(self):
        """File entries should be converted to strings."""
        provider = _mock_provider()
        mock_response = json.dumps({"files": [123, "memory/log.md"]})

        with patch("skill.deep_recall._chat", return_value=mock_response):
            files = _manager_call("query", "index", 5, provider)

        assert files == ["123", "memory/log.md"]


# ---------------------------------------------------------------------------
# _worker_call edge cases
# ---------------------------------------------------------------------------

class TestWorkerCallEdgeCases:
    def test_non_list_quotes_normalized_to_empty(self):
        """If LLM returns quotes as a non-list, normalize to []."""
        provider = _mock_provider()
        mock_response = json.dumps({"quotes": "invalid"})

        with patch("skill.deep_recall._chat", return_value=mock_response):
            result = _worker_call("q", "f.md", "content", provider)

        assert result["quotes"] == []

    def test_valid_quotes_list_preserved(self):
        provider = _mock_provider()
        quotes = [{"text": "exact text", "line": 5}, {"text": "more text", "line": 10}]
        mock_response = json.dumps({"quotes": quotes})

        with patch("skill.deep_recall._chat", return_value=mock_response):
            result = _worker_call("q", "f.md", "content", provider)

        assert len(result["quotes"]) == 2


# ---------------------------------------------------------------------------
# _find_workspace edge cases
# ---------------------------------------------------------------------------

class TestFindWorkspaceEdgeCases:
    def test_from_config_file_with_workspace_key(self):
        """Reads workspace path from openclaw.json when OPENCLAW_WORKSPACE not set."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_file = base / "openclaw.json"
            config_file.write_text(json.dumps(
                {"agents": {"defaults": {"workspace": str(base / "workspace")}}}
            ))

            real_expanduser = os.path.expanduser

            def fake_expanduser(p: str) -> str:
                if ".openclaw/openclaw.json" in p:
                    return str(config_file)
                return real_expanduser(p)

            with patch.dict("os.environ", {}, clear=True), \
                 patch("os.path.expanduser", side_effect=fake_expanduser):
                result = _find_workspace()

            assert str(base / "workspace") == str(result)

    def test_config_file_with_invalid_json_falls_through(self):
        """If config file has invalid JSON, falls back to default workspace."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bad_config = base / "openclaw.json"
            bad_config.write_text("{bad json")

            real_expanduser = os.path.expanduser

            def fake_expanduser(p: str) -> str:
                if ".openclaw/openclaw.json" in p:
                    return str(bad_config)
                return real_expanduser(p)

            with patch.dict("os.environ", {}, clear=True), \
                 patch("os.path.expanduser", side_effect=fake_expanduser):
                result = _find_workspace()
            # Falls back to default
            assert "openclaw" in str(result).lower()


# ---------------------------------------------------------------------------
# recall() verbose flag
# ---------------------------------------------------------------------------

class TestRecallVerbose:
    def test_verbose_prints_info(self, capsys):
        """recall(verbose=True) should print progress info."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            def fake_chat(messages, prov, json_mode=False):
                system = messages[0]["content"]
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": ["memory/2026-03-01.md"]})
                elif "quote extractor" in system.lower():
                    return json.dumps({"quotes": [{"text": "Budget is $50k", "line": 2}]})
                return "Budget was $50k."

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                recall("What is the budget?", workspace=ws, verbose=True)

            captured = capsys.readouterr()
            assert "DeepRecall" in captured.out
            assert "Provider" in captured.out or "Files" in captured.out

    def test_verbose_no_selected_files(self, capsys):
        """verbose=True with no files selected still prints provider info."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            def fake_chat(messages, prov, json_mode=False):
                return json.dumps({"files": []})

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                result = recall("query", workspace=ws, verbose=True)

            assert "No relevant memory files" in result


# ---------------------------------------------------------------------------
# recall() worker future exception
# ---------------------------------------------------------------------------

class TestWorkerFutureException:
    def test_future_exception_appends_empty_quotes(self):
        """If a worker future raises, recall should still return a string."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            call_count = [0]

            def fake_chat(messages, prov, json_mode=False):
                system = messages[0]["content"]
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": ["memory/2026-03-01.md"]})
                # Synthesis gets no quotes, returns default
                return "I don't have memories that answer this query."

            def fake_worker_call(query, filepath, content, prov):
                raise RuntimeError("worker exploded")

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat), \
                 patch("skill.deep_recall._worker_call", side_effect=fake_worker_call):
                result = recall("query", workspace=ws)

            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _MAX_TOOL_ROUNDS
# ---------------------------------------------------------------------------

class TestMaxToolRounds:
    def test_constant_is_one(self):
        """_MAX_TOOL_ROUNDS must be exactly 1 — the safe default."""
        assert _MAX_TOOL_ROUNDS == 1

    def test_constant_is_integer(self):
        assert isinstance(_MAX_TOOL_ROUNDS, int)


# ---------------------------------------------------------------------------
# _safe_path
# ---------------------------------------------------------------------------

class TestSafePath:
    def test_allows_valid_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "notes.md").write_text("hello")
            result = _safe_path("notes.md", ws)
            assert result is not None
            assert result == (ws / "notes.md").resolve()

    def test_allows_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / "memory").mkdir()
            (ws / "memory" / "log.md").write_text("data")
            result = _safe_path("memory/log.md", ws)
            assert result is not None

    def test_blocks_directory_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _safe_path("../../etc/passwd", Path(tmp))
            assert result is None

    def test_blocks_absolute_path_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _safe_path("/etc/passwd", Path(tmp))
            assert result is None

    def test_returns_none_for_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Null byte — should not crash, just return None
            result = _safe_path("file\x00evil", Path(tmp))
            assert result is None


# ---------------------------------------------------------------------------
# _execute_tool_code (sandbox guard)
# ---------------------------------------------------------------------------

class TestExecuteToolCode:
    def test_refuses_execution(self):
        result = _execute_tool_code("print('hello')", {})
        assert "not permitted" in result.lower() or "sandbox" in result.lower()

    def test_returns_string(self):
        result = _execute_tool_code("1 + 1", {"x": 1})
        assert isinstance(result, str)

    def test_does_not_execute_code(self):
        """The sandbox must never mutate state via exec/eval."""
        sentinel = {"modified": False}
        _execute_tool_code("sentinel['modified'] = True", {"sentinel": sentinel})
        assert sentinel["modified"] is False


# ---------------------------------------------------------------------------
# max_depth parameter
# ---------------------------------------------------------------------------

class TestMaxDepth:
    def test_default_is_one(self):
        import inspect
        sig = inspect.signature(recall)
        assert sig.parameters["max_depth"].default == 1

    def test_max_depth_clamped_to_max_tool_rounds(self):
        """Passing max_depth > _MAX_TOOL_ROUNDS must not raise; behaviour is
        identical to max_depth=1 because the value is clamped internally."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            def fake_chat(messages, prov, json_mode=False):
                system = messages[0]["content"]
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": ["memory/2026-03-01.md"]})
                elif "quote extractor" in system.lower():
                    return json.dumps({"quotes": []})
                return "No memories."

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                # Should not raise, result should be a string
                result = recall("query", workspace=ws, max_depth=99)
            assert isinstance(result, str)

    def test_max_depth_zero_treated_as_one(self):
        """max_depth < 1 is silently raised to 1 (no crash)."""
        with tempfile.TemporaryDirectory() as tmp:
            ws = _make_workspace(tmp)
            provider = _mock_provider()

            def fake_chat(messages, prov, json_mode=False):
                system = messages[0]["content"]
                if "memory-file selector" in system.lower():
                    return json.dumps({"files": []})
                return "No memories."

            with patch("skill.deep_recall.resolve_provider", return_value=provider), \
                 patch("skill.deep_recall._chat", side_effect=fake_chat):
                result = recall("query", workspace=ws, max_depth=0)
            assert isinstance(result, str)
