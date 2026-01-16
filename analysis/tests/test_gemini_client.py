"""
Tests for Gemini API client module.

Uses mocks to avoid actual API calls during testing.

Author: Claude Code
Date: 2026-01-16
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis" / "llm_clients"))

from gemini_client import (
    classify_promise_gemini,
    get_gemini_cost_estimate,
    _build_prompt,
    _format_context,
    _parse_classification,
    _validate_api_key,
)


# =====
# API Key Validation Tests
# =====
class TestApiKeyValidation:
    """Tests for API key validation."""

    def test_raises_when_api_key_missing(self):
        """Should raise ValueError when GEMINI_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the key is not set
            os.environ.pop("GEMINI_API_KEY", None)

            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                _validate_api_key()

    def test_passes_when_api_key_set(self):
        """Should not raise when GEMINI_API_KEY is set."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            # Should not raise
            _validate_api_key()


# =====
# Classification Tests
# =====
class TestClassification:
    """Tests for promise classification function."""

    @patch("gemini_client.genai")
    def test_returns_dict_with_classification(self, mock_genai):
        """Should return dict with 'classification' and 'raw_response' keys."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "1"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = classify_promise_gemini("let's all contribute 25")

        assert "classification" in result
        assert "raw_response" in result

    @patch("gemini_client.genai")
    def test_classification_is_zero_or_one(self, mock_genai):
        """Classification value should be 0 or 1."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "1"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = classify_promise_gemini("I promise to contribute")

        assert result["classification"] in [0, 1]

    @patch("gemini_client.genai")
    def test_returns_zero_on_api_error(self, mock_genai):
        """Should return classification 0 on API error after retries."""
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("gemini_client.time.sleep"):  # Skip actual delay
                result = classify_promise_gemini("test message")

        assert result["classification"] == 0
        assert "Error" in result["raw_response"]


# =====
# Context Formatting Tests
# =====
class TestContextFormatting:
    """Tests for context message formatting."""

    def test_empty_context_returns_empty_string(self):
        """Empty or None context should return empty string."""
        assert _format_context(None) == ""
        assert _format_context([]) == ""

    def test_formats_context_with_sender_and_body(self):
        """Should format context messages with sender and body."""
        context = [
            {"sender": "Alice", "body": "Let's cooperate"},
            {"sender": "Bob", "body": "I agree"},
        ]

        result = _format_context(context)

        assert "Alice: Let's cooperate" in result
        assert "Bob: I agree" in result
        assert "Prior messages" in result

    def test_handles_missing_keys_gracefully(self):
        """Should handle context dicts with missing keys."""
        context = [{"sender": "Alice"}, {"body": "message only"}]

        result = _format_context(context)

        assert "Alice" in result
        assert "Unknown" in result  # Default for missing sender


# =====
# Prompt Building Tests
# =====
class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_prompt_includes_message(self):
        """Prompt should include the message to classify."""
        prompt = _build_prompt("let's do 25")

        assert "let's do 25" in prompt

    def test_prompt_includes_context_when_provided(self):
        """Prompt should include context when provided."""
        context = [{"sender": "Player A", "body": "Should we cooperate?"}]

        prompt = _build_prompt("yes, let's do it", context)

        assert "Player A" in prompt
        assert "Should we cooperate?" in prompt

    def test_prompt_requests_binary_output(self):
        """Prompt should request 0 or 1 classification."""
        prompt = _build_prompt("test message")

        assert "1" in prompt and "0" in prompt
        assert "ONLY" in prompt.upper() or "only" in prompt.lower()


# =====
# Response Parsing Tests
# =====
class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parses_one_as_promise(self):
        """Response containing '1' should parse as 1."""
        assert _parse_classification("1") == 1
        assert _parse_classification("The answer is 1") == 1

    def test_parses_zero_as_not_promise(self):
        """Response containing only '0' or no '1' should parse as 0."""
        assert _parse_classification("0") == 0
        assert _parse_classification("No promise") == 0

    def test_prefers_one_over_zero(self):
        """If both present, should return 1."""
        # Edge case: if response somehow contains both
        assert _parse_classification("0 or 1") == 1


# =====
# Cost Estimation Tests
# =====
class TestCostEstimation:
    """Tests for cost estimation function."""

    def test_returns_expected_keys(self):
        """Should return dict with expected keys."""
        result = get_gemini_cost_estimate(100)

        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "estimated_cost_usd" in result

    def test_scales_with_message_count(self):
        """Cost should increase with more messages."""
        cost_100 = get_gemini_cost_estimate(100)
        cost_1000 = get_gemini_cost_estimate(1000)

        assert cost_1000["estimated_cost_usd"] > cost_100["estimated_cost_usd"]
        assert cost_1000["input_tokens"] > cost_100["input_tokens"]

    def test_context_length_affects_cost(self):
        """Longer context should increase cost."""
        cost_short = get_gemini_cost_estimate(100, avg_context_length=2)
        cost_long = get_gemini_cost_estimate(100, avg_context_length=10)

        assert cost_long["estimated_cost_usd"] > cost_short["estimated_cost_usd"]

    def test_cost_is_positive(self):
        """Estimated cost should be positive for non-zero messages."""
        # Use 1000 messages to ensure cost rounds to non-zero
        result = get_gemini_cost_estimate(1000)

        assert result["estimated_cost_usd"] > 0


# =====
# Integration-style Tests (mocked)
# =====
class TestIntegrationMocked:
    """Integration tests with mocked API calls."""

    @patch("gemini_client.genai")
    def test_full_classification_flow(self, mock_genai):
        """Test complete classification with context."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "1"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        context = [
            {"sender": "A", "body": "Should we all put in 25?"},
            {"sender": "B", "body": "I'm in"},
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = classify_promise_gemini(
                message="Yes, let's do it!",
                context=context,
            )

        assert result["classification"] == 1
        assert result["raw_response"] == "1"

        # Verify the model was called with a prompt
        call_args = mock_model.generate_content.call_args[0][0]
        assert "Yes, let's do it!" in call_args
        assert "A:" in call_args
