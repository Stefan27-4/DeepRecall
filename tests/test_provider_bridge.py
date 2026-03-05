"""Tests for skill.provider_bridge module."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from skill.provider_bridge import (
    PROVIDER_BASE_URLS,
    PROVIDER_ENV_KEYS,
    ProviderConfig,
    _get_api_key_from_config,
    _get_api_key_from_env,
    _get_base_url,
    _get_copilot_token,
    _get_primary_model,
    _get_provider_from_model,
    _load_json,
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
