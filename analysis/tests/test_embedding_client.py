"""
Tests for OpenAI embeddings client.

Uses mocks to avoid actual API calls during testing.

Author: Claude Code
Date: 2026-03-15
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add derived directory to path (where llm_clients package lives)
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from llm_clients.embedding_client import (
    embed_texts,
    embed_single,
    get_embedding_cost_estimate,
    MAX_BATCH_SIZE,
    MODEL_SMALL,
    MODEL_LARGE,
)


# =====
# Helpers
# =====
def _make_mock_embedding(dim: int = 1536) -> list[float]:
    """Create a fake embedding vector."""
    return [0.1] * dim


def _build_mock_response(n_texts: int, dim: int = 1536) -> MagicMock:
    """Build a mock embeddings API response."""
    mock_response = MagicMock()
    mock_response.data = []
    for _ in range(n_texts):
        item = MagicMock()
        item.embedding = _make_mock_embedding(dim)
        mock_response.data.append(item)
    return mock_response


def _make_rate_limit_error() -> "RateLimitError":
    """Create a RateLimitError for testing."""
    from openai import RateLimitError
    return RateLimitError("Rate limit", response=MagicMock(), body=None)


# =====
# Test _get_client
# =====
class TestGetClient:
    """Tests for client initialization."""

    def test_raises_error_when_api_key_missing(self):
        """Should raise ValueError when OPENAI_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                embed_texts(["test"])

    def test_creates_client_with_api_key(self):
        """Should create client when API key is present."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _build_mock_response(1)
                )
                mock_openai.return_value = mock_client

                embed_texts(["test"])

                mock_openai.assert_called_once_with(api_key="test-key")


# =====
# Test embed_texts
# =====
class TestEmbedTexts:
    """Tests for the embed_texts function."""

    def test_returns_correct_shape(self):
        """Should return one embedding per input text."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _build_mock_response(3)
                )
                mock_openai.return_value = mock_client

                result = embed_texts(["a", "b", "c"])

                assert len(result) == 3
                assert len(result[0]) == 1536

    def test_empty_list_returns_empty(self):
        """Should return empty list for empty input."""
        result = embed_texts([])
        assert result == []

    def test_passes_model_to_api(self):
        """Should pass the specified model to the API call."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _build_mock_response(1)
                )
                mock_openai.return_value = mock_client

                embed_texts(["test"], model=MODEL_LARGE)

                call_kwargs = (
                    mock_client.embeddings.create.call_args.kwargs
                )
                assert call_kwargs["model"] == MODEL_LARGE


# =====
# Test batching
# =====
class TestBatching:
    """Tests for batch splitting when texts exceed MAX_BATCH_SIZE."""

    def test_single_batch_when_under_limit(self):
        """Should make one API call when texts fit in one batch."""
        texts = ["text"] * 10
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _build_mock_response(10)
                )
                mock_openai.return_value = mock_client

                result = embed_texts(texts)

                assert mock_client.embeddings.create.call_count == 1
                assert len(result) == 10

    def test_splits_into_multiple_batches(self):
        """Should split into multiple API calls for large inputs."""
        n_texts = MAX_BATCH_SIZE + 5
        texts = ["text"] * n_texts

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                # First call returns MAX_BATCH_SIZE, second returns 5
                mock_client.embeddings.create.side_effect = [
                    _build_mock_response(MAX_BATCH_SIZE),
                    _build_mock_response(5),
                ]
                mock_openai.return_value = mock_client

                result = embed_texts(texts)

                assert mock_client.embeddings.create.call_count == 2
                assert len(result) == n_texts

    def test_exact_batch_boundary(self):
        """Should handle exact multiples of MAX_BATCH_SIZE."""
        n_texts = MAX_BATCH_SIZE * 2
        texts = ["text"] * n_texts

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.side_effect = [
                    _build_mock_response(MAX_BATCH_SIZE),
                    _build_mock_response(MAX_BATCH_SIZE),
                ]
                mock_openai.return_value = mock_client

                result = embed_texts(texts)

                assert mock_client.embeddings.create.call_count == 2
                assert len(result) == n_texts


# =====
# Test retry logic
# =====
class TestRetryLogic:
    """Tests for retry with exponential backoff."""

    def test_retries_on_rate_limit_error(self):
        """Should retry when RateLimitError occurs."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mk:
                with patch("llm_clients.embedding_client.time.sleep"):
                    client = MagicMock()
                    mk.return_value = client
                    client.embeddings.create.side_effect = [
                        _make_rate_limit_error(),
                        _build_mock_response(1),
                    ]
                    result = embed_texts(["test"])
                    assert len(result) == 1
                    assert client.embeddings.create.call_count == 2

    def test_raises_after_max_retries(self):
        """Should raise RuntimeError after exhausting retries."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mk:
                with patch("llm_clients.embedding_client.time.sleep"):
                    client = MagicMock()
                    mk.return_value = client
                    client.embeddings.create.side_effect = (
                        _make_rate_limit_error()
                    )
                    with pytest.raises(RuntimeError, match="retries"):
                        embed_texts(["test"])
                    assert client.embeddings.create.call_count == 3

    def test_backoff_increases_delay(self):
        """Should use exponential backoff between retries."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mk:
                with patch("llm_clients.embedding_client.time.sleep") as sl:
                    client = MagicMock()
                    mk.return_value = client
                    client.embeddings.create.side_effect = (
                        _make_rate_limit_error()
                    )
                    with pytest.raises(RuntimeError):
                        embed_texts(["test"])
                    delays = [c.args[0] for c in sl.call_args_list]
                    assert delays == [1.0, 2.0]


# =====
# Test embed_single
# =====
class TestEmbedSingle:
    """Tests for the embed_single convenience function."""

    def test_returns_single_vector(self):
        """Should return a single embedding vector, not a list of lists."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _build_mock_response(1)
                )
                mock_openai.return_value = mock_client

                result = embed_single("hello")

                assert isinstance(result, list)
                assert isinstance(result[0], float)
                assert len(result) == 1536


# =====
# Test get_embedding_cost_estimate
# =====
class TestGetEmbeddingCostEstimate:
    """Tests for the cost estimation function."""

    def test_returns_required_keys(self):
        """Result should contain token count, cost, and model."""
        result = get_embedding_cost_estimate(100, 50)

        assert "total_tokens" in result
        assert "estimated_cost_usd" in result
        assert "model" in result

    def test_returns_positive_values(self):
        """All values should be positive for non-zero input."""
        result = get_embedding_cost_estimate(100, 50)

        assert result["total_tokens"] > 0
        assert result["estimated_cost_usd"] > 0

    def test_zero_texts_returns_zero(self):
        """Zero texts should result in zero cost."""
        result = get_embedding_cost_estimate(0, 50)

        assert result["total_tokens"] == 0
        assert result["estimated_cost_usd"] == 0

    def test_more_texts_increases_cost(self):
        """More texts should increase the cost estimate."""
        small = get_embedding_cost_estimate(10, 50)
        large = get_embedding_cost_estimate(1000, 50)

        assert large["estimated_cost_usd"] > small["estimated_cost_usd"]

    def test_large_model_costs_more(self):
        """text-embedding-3-large should cost more than small."""
        small = get_embedding_cost_estimate(100, 50, model=MODEL_SMALL)
        large = get_embedding_cost_estimate(100, 50, model=MODEL_LARGE)

        assert large["estimated_cost_usd"] > small["estimated_cost_usd"]

    def test_default_model_is_small(self):
        """Default model should be text-embedding-3-small."""
        result = get_embedding_cost_estimate(100, 50)

        assert result["model"] == MODEL_SMALL
