"""
Tests for Task 3: compute_hanaki_embeddings.py output parquet integrity.

Validates schema, embedding dimensions, normalization, metadata alignment,
and helper function correctness against real verified output.

Author: pytest-test-writer
Date: 2026-03-26
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
EMBEDDINGS_FILE = DERIVED_DIR / "hanaki_ozkes_embeddings.parquet"
INPUT_FILE = DERIVED_DIR / "hanaki_ozkes_chat_decisions.parquet"

# Known-good values from verified output
EXPECTED_ROW_COUNT = 8210
EXPECTED_SESSION_COUNT = 23
EXPECTED_EMBEDDING_DIM = 1536
EXPECTED_META_COLS = [
    "session_file", "period", "player_id", "group",
    "Inv", "OtherInv", "Profit", "Chat", "Comp", "chat_text",
]
EXPECTED_TOTAL_COLS = len(EXPECTED_META_COLS) + EXPECTED_EMBEDDING_DIM


# =====
# Fixtures
# =====
@pytest.fixture
def embeddings_df():
    """Load the embeddings parquet as a DataFrame."""
    if not EMBEDDINGS_FILE.exists():
        pytest.skip(f"Not found: {EMBEDDINGS_FILE}")
    return pd.read_parquet(EMBEDDINGS_FILE)


@pytest.fixture
def emb_array(embeddings_df):
    """Extract embedding columns as numpy array."""
    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
    return embeddings_df[emb_cols].values


# =====
# Schema and structure
# =====
class TestEmbeddingsSchema:
    """Verify embeddings parquet has correct schema."""

    def test_file_exists(self):
        """Output parquet must exist."""
        assert EMBEDDINGS_FILE.exists()

    def test_row_count_matches_verified(self, embeddings_df):
        """Regression: 8210 rows (non-empty chat only)."""
        assert len(embeddings_df) == EXPECTED_ROW_COUNT

    def test_session_count(self, embeddings_df):
        """23 sessions have non-empty chat text."""
        assert embeddings_df["session_file"].nunique() == EXPECTED_SESSION_COUNT

    def test_total_column_count(self, embeddings_df):
        """10 metadata columns + 1536 embedding columns."""
        assert len(embeddings_df.columns) == EXPECTED_TOTAL_COLS

    def test_metadata_columns_present(self, embeddings_df):
        """All expected metadata columns present in correct order."""
        actual_meta = list(embeddings_df.columns[:len(EXPECTED_META_COLS)])
        assert actual_meta == EXPECTED_META_COLS

    def test_embedding_columns_sequential(self, embeddings_df):
        """Embedding columns are emb_0 through emb_1535."""
        emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]
        expected = [f"emb_{i}" for i in range(EXPECTED_EMBEDDING_DIM)]
        assert emb_cols == expected

    def test_metadata_dtypes(self, embeddings_df):
        """Metadata columns have correct dtypes."""
        assert embeddings_df["session_file"].dtype == object
        assert embeddings_df["period"].dtype == np.int64
        assert embeddings_df["Inv"].dtype == np.float64
        assert embeddings_df["chat_text"].dtype == object


# =====
# Embedding quality
# =====
class TestEmbeddingQuality:
    """Verify embedding values are well-formed."""

    def test_no_nan_in_embeddings(self, emb_array):
        """No NaN values in any embedding."""
        assert not np.isnan(emb_array).any()

    def test_no_inf_in_embeddings(self, emb_array):
        """No infinite values in any embedding."""
        assert not np.isinf(emb_array).any()

    def test_embeddings_are_approximately_unit_normalized(self, emb_array):
        """OpenAI embeddings should be near unit norm."""
        norms = np.linalg.norm(emb_array, axis=1)
        assert norms.min() > 0.99
        assert norms.max() < 1.01

    def test_embedding_mean_norm_is_one(self, emb_array):
        """Average norm should be very close to 1.0."""
        mean_norm = np.linalg.norm(emb_array, axis=1).mean()
        assert mean_norm == pytest.approx(1.0, abs=0.001)

    def test_embeddings_are_not_all_identical(self, emb_array):
        """Embeddings should not all be the same vector."""
        assert not np.allclose(emb_array[0], emb_array[-1])

    def test_embedding_dimension_matches(self, emb_array):
        """Each embedding should have 1536 dimensions."""
        assert emb_array.shape[1] == EXPECTED_EMBEDDING_DIM


# =====
# Data filtering
# =====
class TestEmbeddingsDataFiltering:
    """Verify only non-empty chat texts are embedded."""

    def test_no_empty_chat_text(self, embeddings_df):
        """All rows should have non-empty chat text."""
        empty = (embeddings_df["chat_text"].str.strip() == "").sum()
        assert empty == 0

    def test_all_chat_1(self, embeddings_df):
        """All rows should have Chat=1."""
        assert (embeddings_df["Chat"] == 1).all()

    def test_row_count_matches_nonempty_chat_from_input(self):
        """Rows should equal non-empty chat count from input parquet."""
        if not INPUT_FILE.exists():
            pytest.skip(f"Input file not found: {INPUT_FILE}")
        input_df = pd.read_parquet(INPUT_FILE)
        nonempty = (input_df["chat_text"].str.strip() != "").sum()
        emb_df = pd.read_parquet(EMBEDDINGS_FILE)
        assert len(emb_df) == nonempty

    def test_investment_range_preserved(self, embeddings_df):
        """Inv values should be in [0, 28] like the input."""
        assert embeddings_df["Inv"].min() >= 0.0
        assert embeddings_df["Inv"].max() <= 28.0


# =====
# Regression values
# =====
class TestEmbeddingsRegression:
    """Regression tests against known-good values."""

    def test_first_row_embedding_values(self, emb_array):
        """First row first 3 embedding values from verified run."""
        assert emb_array[0, 0] == pytest.approx(0.02175903, abs=1e-4)
        assert emb_array[0, 1] == pytest.approx(0.02355957, abs=1e-4)
        assert emb_array[0, 2] == pytest.approx(-0.01276398, abs=1e-4)

    def test_session_211028_1225_not_present(self, embeddings_df):
        """Session 211028_1225 had no non-empty chat, should be absent."""
        sessions = set(embeddings_df["session_file"].unique())
        assert "211028_1225" not in sessions


# =====
# Script helper function tests
# =====
class TestEmbeddingsHelpers:
    """Unit tests for compute_hanaki_embeddings helper functions."""

    def test_build_output_shape(self):
        """_build_output combines metadata with embeddings correctly."""
        from compute_hanaki_embeddings import _build_output

        df = pd.DataFrame({
            "session_file": ["s1"], "period": [1],
            "player_id": [1], "group": [1],
            "Inv": [5.0], "OtherInv": [3.0],
            "Profit": [10.0], "Chat": [1], "Comp": [0],
            "chat_text": ["hello"],
        })
        emb = np.array([[0.1] * 1536])
        result = _build_output(df, emb)
        assert result.shape == (1, 10 + 1536)
        assert result["emb_0"].iloc[0] == pytest.approx(0.1)

    def test_build_output_preserves_metadata(self):
        """_build_output preserves metadata values."""
        from compute_hanaki_embeddings import _build_output

        df = pd.DataFrame({
            "session_file": ["test_session"], "period": [5],
            "player_id": [7], "group": [2],
            "Inv": [14.0], "OtherInv": [12.0],
            "Profit": [20.0], "Chat": [1], "Comp": [1],
            "chat_text": ["bonjour monde"],
        })
        emb = np.zeros((1, 1536))
        result = _build_output(df, emb)
        assert result["session_file"].iloc[0] == "test_session"
        assert result["Inv"].iloc[0] == 14.0
        assert result["chat_text"].iloc[0] == "bonjour monde"
