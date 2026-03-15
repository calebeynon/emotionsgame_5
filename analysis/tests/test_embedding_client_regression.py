"""
Regression and integration tests for embedding_client.py.

Additional tests beyond the unit tests written by the implementation agent.
Focuses on data integrity, edge cases, and integration between functions.

Author: Claude Code (test-writer)
Date: 2026-03-15
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from llm_clients.embedding_client import (
    COST_PER_1K,
    DIMENSIONS,
    MAX_BATCH_SIZE,
    MAX_RETRIES,
    MODEL_LARGE,
    MODEL_SMALL,
    _split_into_batches,
    embed_single,
    embed_texts,
    get_embedding_cost_estimate,
)


# =====
# Helpers
# =====
def _make_mock_response(n_texts, dim=1536):
    """Build a mock embeddings API response with distinct vectors."""
    mock_response = MagicMock()
    mock_response.data = []
    for i in range(n_texts):
        item = MagicMock()
        item.embedding = [float(i)] * dim
        mock_response.data.append(item)
    return mock_response


# =====
# Regression: constant values match expected configuration
# =====
class TestConstantsRegression:
    """Verify module-level constants match expected values."""

    def test_model_small_name(self):
        """Small model should be text-embedding-3-small."""
        assert MODEL_SMALL == "text-embedding-3-small"

    def test_model_large_name(self):
        """Large model should be text-embedding-3-large."""
        assert MODEL_LARGE == "text-embedding-3-large"

    def test_max_batch_size(self):
        """Max batch size should be 2048 per OpenAI limits."""
        assert MAX_BATCH_SIZE == 2048

    def test_max_retries(self):
        """Max retries should be 3."""
        assert MAX_RETRIES == 3

    def test_small_model_dimensions(self):
        """Small model should produce 1536-dim embeddings."""
        assert DIMENSIONS[MODEL_SMALL] == 1536

    def test_large_model_dimensions(self):
        """Large model should produce 3072-dim embeddings."""
        assert DIMENSIONS[MODEL_LARGE] == 3072

    def test_small_model_cheaper_than_large(self):
        """Small model cost should be less than large model cost."""
        assert COST_PER_1K[MODEL_SMALL] < COST_PER_1K[MODEL_LARGE]


# =====
# Regression: cost estimate arithmetic
# =====
class TestCostEstimateRegression:
    """Verify cost estimate calculations produce known-good values."""

    def test_known_cost_small_model(self):
        """100 texts * 50 tokens = 5000 tokens at $0.02/1M."""
        result = get_embedding_cost_estimate(100, 50, model=MODEL_SMALL)
        expected_tokens = 5000
        expected_cost = (expected_tokens / 1000) * COST_PER_1K[MODEL_SMALL]

        assert result['total_tokens'] == expected_tokens
        assert result['estimated_cost_usd'] == round(expected_cost, 6)

    def test_known_cost_large_model(self):
        """100 texts * 50 tokens = 5000 tokens at $0.13/1M."""
        result = get_embedding_cost_estimate(100, 50, model=MODEL_LARGE)
        expected_tokens = 5000
        expected_cost = (expected_tokens / 1000) * COST_PER_1K[MODEL_LARGE]

        assert result['total_tokens'] == expected_tokens
        assert result['estimated_cost_usd'] == round(expected_cost, 6)

    def test_realistic_workload_cost(self):
        """4700 msgs at avg 20 tokens should cost well under $1."""
        result = get_embedding_cost_estimate(4700, 20, model=MODEL_SMALL)

        assert result['total_tokens'] == 94000
        assert result['estimated_cost_usd'] < 1.0

    def test_unknown_model_uses_small_rate(self):
        """Unknown model should fall back to small model cost rate."""
        result = get_embedding_cost_estimate(100, 50, model="unknown-model")
        expected = get_embedding_cost_estimate(100, 50, model=MODEL_SMALL)

        assert result['estimated_cost_usd'] == expected['estimated_cost_usd']
        assert result['model'] == "unknown-model"


# =====
# Edge cases: _split_into_batches
# =====
class TestSplitIntoBatchesEdgeCases:
    """Edge case tests for batch splitting logic."""

    def test_single_item(self):
        """One text should produce one batch of size 1."""
        batches = _split_into_batches(["hello"])
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_exactly_max_batch_size(self):
        """Exactly MAX_BATCH_SIZE texts should produce one batch."""
        texts = ["t"] * MAX_BATCH_SIZE
        batches = _split_into_batches(texts)
        assert len(batches) == 1
        assert len(batches[0]) == MAX_BATCH_SIZE

    def test_one_over_max_produces_two_batches(self):
        """MAX_BATCH_SIZE + 1 texts should produce two batches."""
        texts = ["t"] * (MAX_BATCH_SIZE + 1)
        batches = _split_into_batches(texts)
        assert len(batches) == 2
        assert len(batches[0]) == MAX_BATCH_SIZE
        assert len(batches[1]) == 1

    def test_empty_list(self):
        """Empty list should produce no batches."""
        batches = _split_into_batches([])
        assert len(batches) == 0

    def test_three_full_batches(self):
        """3 * MAX_BATCH_SIZE texts should produce exactly 3 batches."""
        texts = ["t"] * (MAX_BATCH_SIZE * 3)
        batches = _split_into_batches(texts)
        assert len(batches) == 3
        assert all(len(b) == MAX_BATCH_SIZE for b in batches)

    def test_preserves_text_order(self):
        """Batch splitting should preserve the order of texts."""
        texts = [f"text_{i}" for i in range(10)]
        batches = _split_into_batches(texts)
        flattened = [t for batch in batches for t in batch]
        assert flattened == texts


# =====
# Integration: embed_texts preserves order across batches
# =====
class TestEmbedTextsIntegration:
    """Integration tests for embed_texts with multi-batch scenarios."""

    def test_embeddings_order_matches_input_order(self):
        """Embeddings should correspond to input texts in order."""
        n = 5
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _make_mock_response(n, dim=4)
                )
                mock_openai.return_value = mock_client

                result = embed_texts([f"text_{i}" for i in range(n)])

                # Each vector was created with [float(i)] * dim
                for i in range(n):
                    assert result[i][0] == float(i)

    def test_embed_single_delegates_to_embed_texts(self):
        """embed_single should call embed_texts with a single-element list."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = (
                    _make_mock_response(1, dim=4)
                )
                mock_openai.return_value = mock_client

                result = embed_single("hello")

                call_args = mock_client.embeddings.create.call_args
                assert call_args.kwargs["input"] == ["hello"]
                assert len(result) == 4


# =====
# Edge cases: error handling
# =====
class TestErrorHandlingEdgeCases:
    """Tests for error handling in edge cases."""

    def test_api_error_retries_then_raises(self):
        """APIError should retry and raise RuntimeError on exhaustion."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                with patch("llm_clients.embedding_client.time.sleep"):
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client

                    from openai import APIError
                    mock_client.embeddings.create.side_effect = APIError(
                        "Server error",
                        request=MagicMock(),
                        body=None,
                    )

                    with pytest.raises(RuntimeError, match="retries"):
                        embed_texts(["test"])

    def test_rate_limit_then_success_returns_result(self):
        """One RateLimitError then success should return valid embeddings."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("llm_clients.embedding_client.OpenAI") as mock_openai:
                with patch("llm_clients.embedding_client.time.sleep"):
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client

                    from openai import RateLimitError
                    mock_client.embeddings.create.side_effect = [
                        RateLimitError(
                            "limit", response=MagicMock(), body=None
                        ),
                        _make_mock_response(2, dim=3),
                    ]

                    result = embed_texts(["a", "b"])

                    assert len(result) == 2
                    assert len(result[0]) == 3
