"""
Tests for OpenAI API client for promise classification.

Tests use mocks to avoid actual API calls.

Author: Claude Code
Date: 2026-01-16
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

from llm_clients.openai_client import (
    classify_promise_openai,
    get_openai_cost_estimate,
    _build_prompt,
    _format_context,
    _parse_response,
)


# =====
# API Key Validation Tests
# =====
def test_missing_api_key_raises_error():
    """Function raises ValueError when OPENAI_API_KEY not set."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            classify_promise_openai("test message", [])


def test_api_key_from_environment():
    """Function uses API key from environment variable."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "1"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('llm_clients.openai_client.OpenAI', return_value=mock_client):
            result = classify_promise_openai("let's all do 25", [])

            assert result['classification'] in [0, 1]


# =====
# Classification Output Tests
# =====
def test_classification_returns_one():
    """Classification returns 1 when API responds with 1."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "1"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('llm_clients.openai_client.OpenAI', return_value=mock_client):
            result = classify_promise_openai("I promise to contribute", [])

            assert result['classification'] == 1
            assert result['raw_response'] == "1"


def test_classification_returns_zero():
    """Classification returns 0 when API responds with 0."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "0"
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('llm_clients.openai_client.OpenAI', return_value=mock_client):
            result = classify_promise_openai("hello everyone", [])

            assert result['classification'] == 0
            assert result['raw_response'] == "0"


def test_ambiguous_response_defaults_to_zero():
    """Ambiguous responses default to classification 0."""
    assert _parse_response("maybe") == 0
    assert _parse_response("unsure") == 0
    assert _parse_response("") == 0


def test_response_with_one_anywhere():
    """Response containing 1 anywhere is classified as 1."""
    assert _parse_response("The answer is 1") == 1
    assert _parse_response("1 - this is a promise") == 1


# =====
# Context Formatting Tests
# =====
def test_empty_context_formatting():
    """Empty context returns empty string."""
    result = _format_context([])
    assert result == ""


def test_context_with_messages():
    """Context messages are properly formatted."""
    context = [
        {'sender': 'A', 'body': 'hello'},
        {'sender': 'B', 'body': 'lets do 25'},
    ]
    result = _format_context(context)

    assert 'A: "hello"' in result
    assert 'B: "lets do 25"' in result
    assert 'Prior messages' in result


def test_context_with_missing_keys():
    """Context handles missing sender/body keys gracefully."""
    context = [{'sender': 'A'}, {'body': 'test'}]
    result = _format_context(context)

    assert 'A: ""' in result
    assert 'Unknown: "test"' in result


def test_prompt_includes_message():
    """Built prompt includes the message to classify."""
    prompt = _build_prompt("lets all contribute", [])
    assert 'lets all contribute' in prompt


def test_prompt_includes_context():
    """Built prompt includes context when provided."""
    context = [{'sender': 'A', 'body': 'previous message'}]
    prompt = _build_prompt("new message", context)

    assert 'previous message' in prompt
    assert 'new message' in prompt


# =====
# Cost Estimation Tests
# =====
def test_cost_estimate_returns_dict():
    """Cost estimate returns dict with expected keys."""
    result = get_openai_cost_estimate(100, 5)

    assert 'input_tokens' in result
    assert 'output_tokens' in result
    assert 'estimated_cost_usd' in result


def test_cost_estimate_scales_with_messages():
    """More messages means higher cost."""
    cost_100 = get_openai_cost_estimate(100, 5)
    cost_200 = get_openai_cost_estimate(200, 5)

    assert cost_200['estimated_cost_usd'] > cost_100['estimated_cost_usd']
    assert cost_200['input_tokens'] > cost_100['input_tokens']


def test_cost_estimate_scales_with_context():
    """More context means higher cost."""
    cost_short = get_openai_cost_estimate(100, 2)
    cost_long = get_openai_cost_estimate(100, 10)

    assert cost_long['estimated_cost_usd'] > cost_short['estimated_cost_usd']


def test_cost_estimate_is_reasonable():
    """Cost estimate is within reasonable bounds for typical usage."""
    # 1000 messages with 5 context messages each
    result = get_openai_cost_estimate(1000, 5)

    # Should be under $1 for this usage pattern with gpt-5-mini
    assert result['estimated_cost_usd'] < 1.0
    assert result['estimated_cost_usd'] > 0


# =====
# Retry Logic Tests
# =====
def test_retry_on_rate_limit():
    """Function retries on rate limit error."""
    from openai import RateLimitError

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "1"

    # Fail twice, succeed on third try
    mock_client.chat.completions.create.side_effect = [
        RateLimitError("rate limited", response=MagicMock(), body={}),
        RateLimitError("rate limited", response=MagicMock(), body={}),
        mock_response
    ]

    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('llm_clients.openai_client.OpenAI', return_value=mock_client):
            with patch('llm_clients.openai_client.time.sleep'):  # Skip delays
                result = classify_promise_openai("test", [])

                assert result['classification'] == 1
                assert mock_client.chat.completions.create.call_count == 3


def test_returns_zero_after_max_retries():
    """Function returns 0 after exhausting all retries."""
    from openai import RateLimitError

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RateLimitError(
        "rate limited", response=MagicMock(), body={}
    )

    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('llm_clients.openai_client.OpenAI', return_value=mock_client):
            with patch('llm_clients.openai_client.time.sleep'):
                result = classify_promise_openai("test", [])

                assert result['classification'] == 0
                assert 'Error' in result['raw_response']
