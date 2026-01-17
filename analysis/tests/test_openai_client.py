"""
Tests for OpenAI API client for promise classification.

Uses mocks to avoid actual API calls during testing.

Author: Claude Code
Date: 2026-01-17
"""

import os
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from llm_clients.openai_client import (
    classify_promise_openai,
    get_openai_cost_estimate,
)


# =====
# Test classify_promise_openai
# =====
class TestClassifyPromiseOpenAI:
    """Tests for the classify_promise_openai function."""

    def test_raises_error_when_api_key_missing(self):
        """Should raise ValueError when OPENAI_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                classify_promise_openai("test message", [])

    def test_returns_dict_with_required_keys(self):
        """Result should contain 'classification' and 'raw_response' keys."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                # Mock the API response
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "1"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                result = classify_promise_openai("I'll contribute 25", [])

                assert "classification" in result
                assert "raw_response" in result

    def test_classifies_promise_as_1(self):
        """Response '1' should be classified as 1."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "1"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                result = classify_promise_openai("I'll do it", [])

                assert result["classification"] == 1
                assert result["raw_response"] == "1"

    def test_classifies_non_promise_as_0(self):
        """Response '0' should be classified as 0."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "0"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                result = classify_promise_openai("just chatting", [])

                assert result["classification"] == 0
                assert result["raw_response"] == "0"

    def test_parsing_exact_match_only(self):
        """Parsing should only match exact '1', not substring."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]

                # Test that "1" with extra text is NOT classified as 1
                mock_response.choices[0].message.content = "1. This is not a promise"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                result = classify_promise_openai("test", [])

                assert result["classification"] == 0  # Should be 0, not 1

    def test_parsing_handles_whitespace(self):
        """Parsing should strip whitespace before comparison."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "  1  "
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                result = classify_promise_openai("test", [])

                assert result["classification"] == 1

    def test_passes_context_to_prompt(self):
        """Context should be passed to the prompt builder."""
        context = [
            {"sender": "Player A", "body": "Let's all put 25"},
            {"sender": "Player B", "body": "sounds good"},
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                with patch("llm_clients.openai_client.build_classification_prompt") as mock_prompt:
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = "1"
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_openai.return_value = mock_client
                    mock_prompt.return_value = "test prompt"

                    classify_promise_openai("I'm in", context)

                    # Verify context was passed to prompt builder
                    mock_prompt.assert_called_once_with("I'm in", context)

    def test_retries_on_rate_limit_error(self):
        """Should retry when RateLimitError occurs."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                with patch("llm_clients.openai_client.time.sleep"):  # Don't actually sleep
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client

                    # First call raises RateLimitError, second succeeds
                    from openai import RateLimitError
                    mock_response = MagicMock()
                    mock_response.choices = [MagicMock()]
                    mock_response.choices[0].message.content = "1"

                    mock_client.chat.completions.create.side_effect = [
                        RateLimitError("Rate limit", response=MagicMock(), body=None),
                        mock_response
                    ]

                    result = classify_promise_openai("test", [])

                    assert result["classification"] == 1
                    assert mock_client.chat.completions.create.call_count == 2

    def test_returns_error_after_max_retries(self):
        """Should return error response after exhausting retries."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.openai_client.OpenAI") as mock_openai:
                with patch("llm_clients.openai_client.time.sleep"):
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client

                    # Use RateLimitError which we know how to construct
                    from openai import RateLimitError
                    mock_client.chat.completions.create.side_effect = RateLimitError(
                        "Rate limit", response=MagicMock(), body=None
                    )

                    result = classify_promise_openai("test", [])

                    assert result["classification"] == 0
                    assert "Error:" in result["raw_response"]
                    assert mock_client.chat.completions.create.call_count == 3  # MAX_RETRIES


# =====
# Test get_openai_cost_estimate
# =====
class TestGetOpenAICostEstimate:
    """Tests for the get_openai_cost_estimate function."""

    def test_returns_dict_with_required_keys(self):
        """Result should contain token counts and cost estimate."""
        result = get_openai_cost_estimate(100, 5)

        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "estimated_cost_usd" in result

    def test_returns_positive_values(self):
        """All values should be positive."""
        result = get_openai_cost_estimate(100, 5)

        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0
        assert result["estimated_cost_usd"] > 0

    def test_more_messages_increases_cost(self):
        """More messages should increase cost estimate."""
        small = get_openai_cost_estimate(10, 5)
        large = get_openai_cost_estimate(1000, 5)

        assert large["estimated_cost_usd"] > small["estimated_cost_usd"]
        assert large["input_tokens"] > small["input_tokens"]

    def test_more_context_increases_cost(self):
        """More context messages should increase cost."""
        small_context = get_openai_cost_estimate(100, 1)
        large_context = get_openai_cost_estimate(100, 20)

        assert large_context["estimated_cost_usd"] > small_context["estimated_cost_usd"]
        assert large_context["input_tokens"] > small_context["input_tokens"]

    def test_zero_messages_returns_zero_cost(self):
        """Zero messages should result in zero cost."""
        result = get_openai_cost_estimate(0, 5)

        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["estimated_cost_usd"] == 0

    def test_reasonable_cost_for_typical_workload(self):
        """Cost estimate should be reasonable for typical usage."""
        # 5000 messages with average 5 context messages each
        result = get_openai_cost_estimate(5000, 5)

        # Should be less than $1 for this workload with gpt-5-mini
        assert result["estimated_cost_usd"] < 1.0
        assert result["estimated_cost_usd"] > 0
