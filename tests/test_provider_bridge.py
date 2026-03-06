"""Tests for skill.provider_bridge module."""

import json
import tempfile
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skill.provider_bridge import (
    PROVIDER_BASE_URLS,
    PROVIDER_ENV_KEYS,
    ProviderConfig,
    _call_anthropic,
    _call_openai_compatible,
    _get_api_key_from_config,
    _get_api_key_from_env,
    _get_base_url,
    _get_copilot_token,
    _get_primary_model,
    _get_provider_from_model,
    _load_json,
    call_llm,
    make_gemini_native_request,
    make_request,
    resolve_provider,
)


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig:
    def test_basic_creation(self):
        cfg = ProviderConfig(
            provider="openai",
            api_key="sk-test1234abcd",
            base_url="https://api.openai.com/v1",
            primary_model="openai/gpt-4o",
        )
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test1234abcd"

    def test_repr_masks_api_key(self):
        cfg = ProviderConfig(
            provider="openai",
            api_key="sk-test1234abcd",
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        assert "sk-test1234abcd" not in r
        assert "...abcd" in r

    def test_repr_prefix_limited_to_four_chars(self):
        """Repr must not expose more than 4 leading characters of the key."""
        cfg = ProviderConfig(
            provider="openai",
            api_key="sk-test1234abcd",
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        # The 5th character of "sk-test1234abcd" is 'e'; it must not appear
        # adjacent to the known prefix in the output.
        assert "sk-te" not in r

    def test_repr_short_key_fully_masked(self):
        """Keys of 8 chars or fewer must be replaced with '***'."""
        cfg = ProviderConfig(
            provider="openai",
            api_key="sk-short",       # exactly 8 chars
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        assert "sk-short" not in r
        assert "***" in r

    def test_repr_medium_key_fully_masked(self):
        """Keys of 4-8 chars must not reveal partial content."""
        cfg = ProviderConfig(
            provider="openai",
            api_key="tiny",
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        assert "tiny" not in r
        assert "***" in r

    def test_repr_no_key(self):
        cfg = ProviderConfig(
            provider="openai",
            api_key="",
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        assert "[NOT SET]" in r

    def test_default_headers_empty(self):
        cfg = ProviderConfig(
            provider="openai",
            api_key="key",
            base_url="url",
            primary_model="model",
        )
        assert cfg.default_headers == {}


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            result = _load_json(Path(f.name))
        assert result == {"key": "value"}

    def test_missing_file(self):
        assert _load_json(Path("/nonexistent/path.json")) == {}

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            result = _load_json(Path(f.name))
        assert result == {}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_primary_model(self):
        config = {
            "agents": {"defaults": {"model": {"primary": "openai/gpt-4o"}}}
        }
        assert _get_primary_model(config) == "openai/gpt-4o"

    def test_get_primary_model_missing(self):
        assert _get_primary_model({}) is None
        assert _get_primary_model({"agents": {}}) is None

    def test_get_provider_from_model(self):
        assert _get_provider_from_model("anthropic/claude-opus-4") == "anthropic"
        assert _get_provider_from_model("openai/gpt-4o") == "openai"

    def test_get_provider_no_slash(self):
        assert _get_provider_from_model("gpt-4o") is None

    def test_get_api_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fromenv"}):
            assert _get_api_key_from_env("openai") == "sk-fromenv"

    def test_get_api_key_from_env_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _get_api_key_from_env("openai") is None

    def test_get_api_key_from_env_ollama(self):
        # Ollama has no env key
        assert _get_api_key_from_env("ollama") is None

    def test_get_api_key_from_config_env_section(self):
        config = {"env": {"OPENAI_API_KEY": "sk-fromconfig"}}
        assert _get_api_key_from_config(config, "openai") == "sk-fromconfig"

    def test_get_api_key_from_config_provider_section(self):
        config = {
            "models": {"providers": {"anthropic": {"apiKey": "sk-anthro"}}}
        }
        assert _get_api_key_from_config(config, "anthropic") == "sk-anthro"

    def test_get_api_key_from_config_missing(self):
        assert _get_api_key_from_config({}, "openai") is None


# ---------------------------------------------------------------------------
# _get_base_url
# ---------------------------------------------------------------------------

class TestGetBaseUrl:
    def test_known_provider(self):
        url = _get_base_url("openai", {})
        assert url == "https://api.openai.com/v1"

    def test_custom_base_url_from_models_config(self):
        models_config = {
            "providers": {"openai": {"baseUrl": "https://custom.api/v1"}}
        }
        url = _get_base_url("openai", models_config)
        assert url == "https://custom.api/v1"

    def test_custom_base_url_capital(self):
        models_config = {
            "providers": {"openai": {"baseURL": "https://custom2.api/v1"}}
        }
        url = _get_base_url("openai", models_config)
        assert url == "https://custom2.api/v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _get_base_url("totally-unknown-provider", {})


# ---------------------------------------------------------------------------
# _get_copilot_token
# ---------------------------------------------------------------------------

class TestGetCopilotToken:
    def test_valid_token_seconds(self):
        """Token with expiry in seconds format (Unix timestamp)."""
        future = int(time.time()) + 3600
        token_data = {"token": "ghu_valid123", "expiresAt": future}

        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps(token_data))

            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result == "ghu_valid123"

    def test_valid_token_milliseconds(self):
        """Token with expiry in milliseconds format."""
        future_ms = int(time.time() * 1000) + 3_600_000
        token_data = {"token": "ghu_mstoken", "expiresAt": future_ms}

        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps(token_data))

            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result == "ghu_mstoken"

    def test_expired_token_seconds(self):
        past = int(time.time()) - 3600
        token_data = {"token": "ghu_expired", "expiresAt": past}

        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps(token_data))

            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result is None

    def test_expired_token_milliseconds(self):
        past_ms = int(time.time() * 1000) - 3_600_000
        token_data = {"token": "ghu_expired_ms", "expiresAt": past_ms}

        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps(token_data))

            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result is None

    def test_missing_token_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result is None

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            token_file = creds / "github-copilot.token.json"
            token_file.write_text("{bad json")
            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result is None

    def test_config_absolute_credentialsdir_override(self):
        """credentialsDir in config (absolute path) is honoured."""
        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "custom_creds"
            creds.mkdir()
            future = int(time.time()) + 3600
            (creds / "github-copilot.token.json").write_text(
                json.dumps({"token": "ghu_absolute", "expiresAt": future})
            )
            config = {"credentialsDir": str(creds)}
            result = _get_copilot_token(openclaw_config=config)
            assert result == "ghu_absolute"

    def test_config_relative_credentialsdir_resolved_under_openclaw_dir(self):
        """credentialsDir as a relative path is resolved relative to OPENCLAW_DIR."""
        with tempfile.TemporaryDirectory() as tmp:
            openclaw_dir = Path(tmp)
            rel_creds = openclaw_dir / "rel_creds"
            rel_creds.mkdir()
            future = int(time.time()) + 3600
            (rel_creds / "github-copilot.token.json").write_text(
                json.dumps({"token": "ghu_relative", "expiresAt": future})
            )
            config = {"credentialsDir": "rel_creds"}
            with patch("skill.provider_bridge.OPENCLAW_DIR", openclaw_dir):
                result = _get_copilot_token(openclaw_config=config)
            assert result == "ghu_relative"


# ---------------------------------------------------------------------------
# resolve_provider (integration-level, mocked filesystem)
# ---------------------------------------------------------------------------

class TestResolveProvider:
    def _setup_mock_config(self, tmp: str, config: dict, models: dict = None):
        """Set up mock OpenClaw config files."""
        base = Path(tmp)
        config_file = base / "openclaw.json"
        config_file.write_text(json.dumps(config))

        agents_dir = base / "agents" / "main" / "agent"
        agents_dir.mkdir(parents=True)
        models_file = agents_dir / "models.json"
        models_file.write_text(json.dumps(models or {}))

        creds = base / "credentials"
        creds.mkdir()

        return base, config_file, models_file, creds

    def test_resolve_openai_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "openai/gpt-4o"}}},
                "env": {"OPENAI_API_KEY": "sk-test123"},
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = resolve_provider()

            assert result.provider == "openai"
            assert result.api_key == "sk-test123"
            assert "openai" in result.base_url

    def test_resolve_no_primary_model_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, {})

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                with pytest.raises(RuntimeError, match="No primary model"):
                    resolve_provider()

    def test_resolve_no_provider_prefix_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "gpt-4o"}}}
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                with pytest.raises(RuntimeError, match="Cannot determine provider"):
                    resolve_provider()

    def test_resolve_no_api_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "openai/gpt-4o"}}}
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds), \
                 patch.dict("os.environ", {}, clear=True):
                with pytest.raises(RuntimeError, match="No API key"):
                    resolve_provider()

    def test_resolve_ollama_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "ollama/llama3"}}}
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = resolve_provider()

            assert result.provider == "ollama"
            assert result.api_key == "ollama-local"
            assert "localhost" in result.base_url

    def test_resolve_copilot_with_valid_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "github-copilot/gpt-4o"}}}
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            future = int(time.time()) + 3600
            token_data = {"token": "ghu_copilot_tok", "expiresAt": future}
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps(token_data))

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = resolve_provider()

            assert result.provider == "github-copilot"
            assert result.api_key == "ghu_copilot_tok"
            assert result.default_headers  # Copilot sets special headers

    def test_resolve_copilot_expired_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "github-copilot/gpt-4o"}}}
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            past = int(time.time()) - 3600
            token_data = {"token": "ghu_expired", "expiresAt": past}
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps(token_data))

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                with pytest.raises(RuntimeError, match="Copilot token expired"):
                    resolve_provider()

    def test_resolve_api_key_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "agents": {"defaults": {"model": {"primary": "anthropic/claude-3"}}}
            }
            base, cfg_file, models_file, creds = self._setup_mock_config(tmp, config)

            with patch("skill.provider_bridge.CONFIG_FILE", cfg_file), \
                 patch("skill.provider_bridge.MODELS_FILE", models_file), \
                 patch("skill.provider_bridge.CREDENTIALS_DIR", creds), \
                 patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-envkey"}):
                result = resolve_provider()

            assert result.api_key == "sk-envkey"


# ---------------------------------------------------------------------------
# make_gemini_native_request — key in header, not URL
# ---------------------------------------------------------------------------

class TestMakeGeminiNativeRequest:
    def _google_config(self):
        return ProviderConfig(
            provider="google",
            api_key="AIza_test_key_1234",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            primary_model="google/gemini-pro",
        )

    def test_api_key_sent_as_header_not_url_param(self):
        """The API key must appear in x-goog-api-key header, not the request URL."""
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["url"] = url
            captured["headers"] = headers
            return {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}]
            }

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            make_gemini_native_request(self._google_config(), [{"role": "user", "content": "hello"}])

        assert "AIza_test_key_1234" not in captured["url"], \
            "API key must NOT appear in the URL query string"
        assert captured["headers"].get("x-goog-api-key") == "AIza_test_key_1234"

    def test_timeout_forwarded_to_make_request(self):
        """The timeout kwarg must be forwarded to make_request."""
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["timeout"] = timeout
            return {"candidates": []}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            make_gemini_native_request(
                self._google_config(),
                [{"role": "user", "content": "hi"}],
                timeout=120,
            )

        assert captured["timeout"] == 120

    def test_system_message_mapped_to_system_instruction(self):
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["body"] = body
            return {"candidates": []}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            make_gemini_native_request(
                self._google_config(),
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hello"},
                ],
            )

        assert "systemInstruction" in captured["body"]
        assert captured["body"]["systemInstruction"]["parts"][0]["text"] == "You are helpful."

    def test_assistant_role_mapped_to_model(self):
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["body"] = body
            return {"candidates": []}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            make_gemini_native_request(
                self._google_config(),
                [
                    {"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "pong"},
                ],
            )

        roles = [c["role"] for c in captured["body"]["contents"]]
        assert "model" in roles


# ---------------------------------------------------------------------------
# call_llm — timeout propagation and provider routing
# ---------------------------------------------------------------------------

class TestCallLlm:
    def _cfg(self, provider, model):
        return ProviderConfig(
            provider=provider,
            api_key="test-key-abcdefghij",
            base_url=f"https://example.com/v1",
            primary_model=f"{provider}/{model}",
        )

    def test_openai_compatible_timeout_forwarded(self):
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["timeout"] = timeout
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            call_llm([{"role": "user", "content": "hi"}],
                     config=self._cfg("openai", "gpt-4o"),
                     timeout=30)

        assert captured["timeout"] == 30

    def test_anthropic_timeout_forwarded(self):
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["timeout"] = timeout
            return {"content": [{"type": "text", "text": "hi"}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            call_llm([{"role": "user", "content": "hi"}],
                     config=self._cfg("anthropic", "claude-3"),
                     timeout=45)

        assert captured["timeout"] == 45

    def test_gemini_native_timeout_forwarded(self):
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["timeout"] = timeout
            return {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            call_llm([{"role": "user", "content": "hi"}],
                     config=self._cfg("google", "gemini-pro"),
                     native_gemini=True,
                     timeout=90)

        assert captured["timeout"] == 90

    def test_google_openai_compat_path_without_native_flag(self):
        """google provider without native_gemini uses OpenAI-compatible path."""
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["url"] = url
            return {"choices": [{"message": {"content": "ok"}}]}

        cfg = ProviderConfig(
            provider="google",
            api_key="AIza_key_abcdef1234",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            primary_model="google/gemini-pro",
        )
        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            result = call_llm([{"role": "user", "content": "hi"}], config=cfg)

        assert "chat/completions" in captured["url"]
        assert result == "ok"

    def test_ollama_uses_openai_compatible_path(self):
        captured = {}

        def fake_make_request(url, headers, body, timeout=60):
            captured["url"] = url
            return {"choices": [{"message": {"content": "pong"}}]}

        cfg = ProviderConfig(
            provider="ollama",
            api_key="ollama-local",
            base_url="http://localhost:11434/v1",
            primary_model="ollama/llama3",
        )
        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            result = call_llm([{"role": "user", "content": "ping"}], config=cfg)

        assert "chat/completions" in captured["url"]
        assert result == "pong"

    def test_gemini_native_empty_candidates_returns_empty_string(self):
        def fake_make_request(url, headers, body, timeout=60):
            return {"candidates": []}

        with patch("skill.provider_bridge.make_request", side_effect=fake_make_request):
            result = call_llm([{"role": "user", "content": "hi"}],
                              config=self._cfg("google", "gemini-pro"),
                              native_gemini=True)

        assert result == ""


# ---------------------------------------------------------------------------
# make_request — HTTP error surfaces body, not credentials
# ---------------------------------------------------------------------------

class TestMakeRequest:
    def test_http_error_raises_runtime_error(self):
        err = urllib.error.HTTPError(
            url="https://example.com",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        err.read = lambda: b'{"error": "invalid_api_key"}'

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="401"):
                make_request("https://example.com", {}, {})

    def test_url_error_raises_runtime_error(self):
        err = urllib.error.URLError(reason="Name or service not known")

        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match="connection error"):
                make_request("https://example.com", {}, {})

    def test_timeout_default_is_60(self):
        captured = {}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'

        def fake_urlopen(req, timeout):
            captured["timeout"] = timeout
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            make_request("https://example.com", {}, {})

        assert captured["timeout"] == 60

    def test_custom_timeout_passed_through(self):
        captured = {}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'

        def fake_urlopen(req, timeout):
            captured["timeout"] = timeout
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            make_request("https://example.com", {}, {}, timeout=15)

        assert captured["timeout"] == 15


# ---------------------------------------------------------------------------
# ProviderConfig repr edge cases
# ---------------------------------------------------------------------------

class TestProviderConfigReprEdgeCases:
    def test_repr_short_key_shows_stars(self):
        """API key with length <= 8 is replaced with *** (too short to partially show)."""
        cfg = ProviderConfig(
            provider="openai",
            api_key="tiny",
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        assert "tiny" not in r
        assert "***" in r

    def test_repr_long_key_shows_first4_and_last4(self):
        """API key longer than 8 chars shows first4...last4."""
        cfg = ProviderConfig(
            provider="openai",
            api_key="sk-longkey1234567",
            base_url="https://api.openai.com/v1",
            primary_model="gpt-4o",
        )
        r = repr(cfg)
        assert "sk-l" in r    # first 4 chars
        assert "4567" in r    # last 4 chars
        assert "..." in r
        assert "sk-longkey1234567" not in r  # raw key not present


# ---------------------------------------------------------------------------
# _call_anthropic
# ---------------------------------------------------------------------------

class TestCallAnthropic:
    def _anthropic_config(self):
        return ProviderConfig(
            provider="anthropic",
            api_key="sk-ant-key",
            base_url="https://api.anthropic.com/v1",
            primary_model="anthropic/claude-3",
        )

    def test_extracts_text_from_response(self):
        response = {"content": [{"type": "text", "text": "Hello from Claude"}]}

        with patch("skill.provider_bridge.make_request", return_value=response):
            result = _call_anthropic(
                self._anthropic_config(),
                [{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=1024,
            )

        assert result == "Hello from Claude"

    def test_system_message_extracted(self):
        captured_body = {}

        def fake_request(url, headers, body, **kwargs):
            captured_body.update(body)
            return {"content": [{"type": "text", "text": "ok"}]}

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        with patch("skill.provider_bridge.make_request", side_effect=fake_request):
            _call_anthropic(self._anthropic_config(), messages, 0.7, 1024)

        assert "system" in captured_body
        assert captured_body["system"] == "Be concise."
        # System message should NOT be in messages list
        assert not any(m.get("role") == "system" for m in captured_body["messages"])

    def test_uses_anthropic_auth_header(self):
        captured_headers = {}

        def fake_request(url, headers, body, **kwargs):
            captured_headers.update(headers)
            return {"content": [{"type": "text", "text": "ok"}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_request):
            _call_anthropic(
                self._anthropic_config(),
                [{"role": "user", "content": "hi"}],
                0.7,
                1024,
            )

        assert "x-api-key" in captured_headers
        assert captured_headers["x-api-key"] == "sk-ant-key"
        assert "anthropic-version" in captured_headers

    def test_empty_content_returns_empty_string(self):
        with patch("skill.provider_bridge.make_request", return_value={"content": []}):
            result = _call_anthropic(
                self._anthropic_config(),
                [{"role": "user", "content": "hi"}],
                0.7,
                512,
            )
        assert result == ""

    def test_strips_provider_prefix_from_model(self):
        captured_body = {}

        def fake_request(url, headers, body, **kwargs):
            captured_body.update(body)
            return {"content": [{"type": "text", "text": "ok"}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_request):
            _call_anthropic(
                self._anthropic_config(),
                [{"role": "user", "content": "hi"}],
                0.7,
                1024,
            )

        assert captured_body["model"] == "claude-3"


# ---------------------------------------------------------------------------
# _call_openai_compatible
# ---------------------------------------------------------------------------

class TestCallOpenAICompatible:
    def _openai_config(self):
        return ProviderConfig(
            provider="openai",
            api_key="sk-openai",
            base_url="https://api.openai.com/v1",
            primary_model="openai/gpt-4o",
        )

    def test_extracts_message_content(self):
        response = {
            "choices": [{"message": {"content": "Hello from GPT"}}]
        }
        with patch("skill.provider_bridge.make_request", return_value=response):
            result = _call_openai_compatible(
                self._openai_config(),
                [{"role": "user", "content": "Hi"}],
                temperature=0.7,
                max_tokens=1024,
            )
        assert result == "Hello from GPT"

    def test_bearer_auth_header(self):
        captured_headers = {}

        def fake_request(url, headers, body, **kwargs):
            captured_headers.update(headers)
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_request):
            _call_openai_compatible(
                self._openai_config(),
                [{"role": "user", "content": "hi"}],
                0.7,
                1024,
            )

        assert captured_headers["Authorization"] == "Bearer sk-openai"

    def test_empty_choices_returns_empty_string(self):
        with patch("skill.provider_bridge.make_request", return_value={"choices": []}):
            result = _call_openai_compatible(
                self._openai_config(),
                [{"role": "user", "content": "hi"}],
                0.7,
                512,
            )
        assert result == ""

    def test_strips_provider_prefix_from_model(self):
        captured_body = {}

        def fake_request(url, headers, body, **kwargs):
            captured_body.update(body)
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_request):
            _call_openai_compatible(
                self._openai_config(),
                [{"role": "user", "content": "hi"}],
                0.7,
                1024,
            )

        assert captured_body["model"] == "gpt-4o"

    def test_passes_temperature_and_max_tokens(self):
        captured_body = {}

        def fake_request(url, headers, body, **kwargs):
            captured_body.update(body)
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch("skill.provider_bridge.make_request", side_effect=fake_request):
            _call_openai_compatible(
                self._openai_config(),
                [{"role": "user", "content": "hi"}],
                temperature=0.3,
                max_tokens=256,
            )

        assert captured_body["temperature"] == 0.3
        assert captured_body["max_tokens"] == 256


# ---------------------------------------------------------------------------
# _get_copilot_token with credentialsDir override
# ---------------------------------------------------------------------------

class TestGetCopilotTokenCredsDirOverride:
    def test_custom_credentials_dir_from_config(self):
        """credentialsDir key in config overrides CREDENTIALS_DIR."""
        with tempfile.TemporaryDirectory() as tmp:
            custom_creds = Path(tmp) / "custom_creds"
            custom_creds.mkdir()
            token_file = custom_creds / "github-copilot.token.json"
            future = int(time.time()) + 3600
            token_file.write_text(json.dumps({"token": "ghu_custom", "expiresAt": future}))

            config = {"credentialsDir": str(custom_creds)}
            result = _get_copilot_token(openclaw_config=config)
            assert result == "ghu_custom"

    def test_no_expiry_field_returns_token(self):
        """Token without expiresAt field should be returned as valid."""
        with tempfile.TemporaryDirectory() as tmp:
            creds = Path(tmp) / "credentials"
            creds.mkdir()
            token_file = creds / "github-copilot.token.json"
            token_file.write_text(json.dumps({"token": "ghu_noexpiry"}))

            with patch("skill.provider_bridge.CREDENTIALS_DIR", creds):
                result = _get_copilot_token()
            assert result == "ghu_noexpiry"
