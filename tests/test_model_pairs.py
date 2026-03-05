"""Tests for skill.model_pairs module."""

import pytest

from skill.model_pairs import MODEL_PAIRS, get_model_pair, get_sub_agent_model


class TestGetSubAgentModel:
    """Test get_sub_agent_model for every catalog entry and edge cases."""

    @pytest.mark.parametrize("primary,expected", list(MODEL_PAIRS.items()))
    def test_every_model_in_catalog(self, primary, expected):
        assert get_sub_agent_model(primary) == expected

    # --- Prefix matching ---

    def test_prefix_match_with_version_suffix(self):
        result = get_sub_agent_model("claude-opus-4-20250514")
        assert result == "claude-sonnet-4"

    def test_prefix_match_with_provider_prefix(self):
        result = get_sub_agent_model("anthropic/claude-opus-4.6")
        assert result == "claude-sonnet-4"

    def test_prefix_not_substring(self):
        """Ensure 'gpt-4o' does NOT match 'gpt-4' — prefix must match from start."""
        result = get_sub_agent_model("gpt-4o-2024-08-06")
        assert result == "gpt-4o-mini"

    def test_longer_prefix_wins(self):
        """'claude-opus-4.6' should match before 'claude-opus-4'."""
        result = get_sub_agent_model("claude-opus-4.6-latest")
        assert result == "claude-sonnet-4"
        result2 = get_sub_agent_model("claude-opus-4-latest")
        assert result2 == "claude-sonnet-4"

    def test_gpt5_prefix(self):
        """'gpt-5' should match models starting with gpt-5."""
        result = get_sub_agent_model("gpt-5-turbo")
        assert result == "gpt-5-mini"

    def test_case_insensitive(self):
        result = get_sub_agent_model("Claude-Opus-4.6")
        assert result == "claude-sonnet-4"

    # --- Unknown model fallback ---

    def test_unknown_model_falls_back_to_same(self):
        result = get_sub_agent_model("some-unknown-model-v2")
        assert result == "some-unknown-model-v2"

    def test_unknown_with_provider_prefix(self):
        result = get_sub_agent_model("custom/my-local-model")
        assert result == "my-local-model"

    # --- Edge cases ---

    def test_empty_string(self):
        result = get_sub_agent_model("")
        assert result == ""

    def test_very_long_model_name(self):
        long_name = "a" * 500
        result = get_sub_agent_model(long_name)
        assert result == long_name

    def test_model_with_slashes(self):
        result = get_sub_agent_model("org/sub/claude-opus-4.6-snapshot")
        # split("/")[-1] strips everything before last slash
        assert result == "claude-sonnet-4"


class TestGetModelPair:
    def test_returns_dict_with_keys(self):
        pair = get_model_pair("claude-opus-4.6")
        assert "primary" in pair
        assert "sub_agent" in pair

    def test_strips_provider_prefix(self):
        pair = get_model_pair("anthropic/claude-opus-4")
        assert pair["primary"] == "claude-opus-4"
        assert pair["sub_agent"] == "claude-sonnet-4"

    def test_no_provider_prefix(self):
        pair = get_model_pair("gpt-4o")
        assert pair["primary"] == "gpt-4o"
        assert pair["sub_agent"] == "gpt-4o-mini"

    def test_unknown_model(self):
        pair = get_model_pair("mystery-model")
        assert pair["primary"] == "mystery-model"
        assert pair["sub_agent"] == "mystery-model"

    @pytest.mark.parametrize("primary", list(MODEL_PAIRS.keys()))
    def test_all_pairs_return_valid_dict(self, primary):
        pair = get_model_pair(primary)
        assert isinstance(pair, dict)
        assert pair["primary"] == primary
        assert pair["sub_agent"] == MODEL_PAIRS[primary]
