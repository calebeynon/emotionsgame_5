"""
Tests for CCR chat embedding output (Task 4).

Validates the OpenAI text-embedding-3-small embeddings computed for
CCR group-level chat transcripts. Checks dimensions, normalization,
merge keys, and regression values.

Known data facts (verified from output inspection):
- 115 rows (116 groups minus 1 empty-chat group at session 55, Green)
- 10 metadata columns + 1536 embedding dimensions (emb_0..emb_1535)
- Embedding norms: 0.9995 - 1.0006 (approximately unit-normalized)
- group_key format: "{session}_{red}" for merging with effort data

Author: pytest-test-writer
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# FILE PATHS
EXTERNAL_DIR = Path(__file__).parent.parent / "datastore" / "derived" / "external"
EMBEDDINGS_FILE = EXTERNAL_DIR / "ccr_embeddings_small.parquet"
CHAT_FILE = EXTERNAL_DIR / "ccr_chat_clean.parquet"
EFFORT_GROUP_FILE = EXTERNAL_DIR / "ccr_effort_group.parquet"

# Known-good values from verified output
EXPECTED_ROWS = 115
EXPECTED_EMBEDDING_DIM = 1536
EXPECTED_TOTAL_COLS = 1546  # 10 meta + 1536 embedding
EXPECTED_META_COLS = [
    "session", "red", "run", "ingroup", "commonknow",
    "group_chat_text", "n_messages", "n_words", "n_characters", "group_key",
]
EXPECTED_MISSING_GROUP = "55_0"  # session 55, Green — empty chat


# =====
# Fixtures
# =====
@pytest.fixture
def emb_df():
    """Load the CCR embeddings parquet."""
    if not EMBEDDINGS_FILE.exists():
        pytest.skip(f"Embeddings not found: {EMBEDDINGS_FILE}")
    return pd.read_parquet(EMBEDDINGS_FILE)


@pytest.fixture
def emb_array(emb_df):
    """Extract embedding columns as numpy array."""
    cols = [c for c in emb_df.columns if c.startswith("emb_")]
    return emb_df[cols].values


# =====
# Schema and dimensions
# =====
class TestEmbeddingsSchema:
    """Verify embeddings parquet has correct schema and dimensions."""

    def test_file_exists(self):
        """Embeddings parquet must exist."""
        assert EMBEDDINGS_FILE.exists(), f"Missing: {EMBEDDINGS_FILE}"

    def test_row_count(self, emb_df):
        """Should have 115 rows (116 groups minus 1 empty)."""
        assert len(emb_df) == EXPECTED_ROWS

    def test_total_column_count(self, emb_df):
        """Should have 1546 columns (10 meta + 1536 embeddings)."""
        assert len(emb_df.columns) == EXPECTED_TOTAL_COLS

    def test_embedding_dim_is_1536(self, emb_df):
        """Should have exactly 1536 embedding columns."""
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        assert len(emb_cols) == EXPECTED_EMBEDDING_DIM

    def test_has_all_meta_columns(self, emb_df):
        """Must have all expected metadata columns."""
        for col in EXPECTED_META_COLS:
            assert col in emb_df.columns, f"Missing: {col}"

    def test_embedding_columns_sequential(self, emb_df):
        """Embedding columns must be emb_0 through emb_1535 in order."""
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        expected = [f"emb_{i}" for i in range(EXPECTED_EMBEDDING_DIM)]
        assert emb_cols == expected

    def test_embedding_dtype_float64(self, emb_df):
        """Embedding columns should be float64."""
        assert emb_df["emb_0"].dtype == np.float64


# =====
# Normalization and validity
# =====
class TestEmbeddingsNormalization:
    """Verify embeddings are approximately unit-normalized and valid."""

    def test_norms_near_unity(self, emb_array):
        """All embedding norms should be approximately 1.0."""
        norms = np.linalg.norm(emb_array, axis=1)
        assert norms.min() > 0.999, f"Min norm too low: {norms.min():.6f}"
        assert norms.max() < 1.001, f"Max norm too high: {norms.max():.6f}"

    def test_no_nan_in_embeddings(self, emb_array):
        """No NaN values in embedding columns."""
        assert not np.isnan(emb_array).any(), "Found NaN in embeddings"

    def test_no_inf_in_embeddings(self, emb_array):
        """No infinite values in embedding columns."""
        assert not np.isinf(emb_array).any(), "Found Inf in embeddings"

    def test_no_nan_in_metadata(self, emb_df):
        """No NaN values in metadata columns."""
        meta = emb_df[EXPECTED_META_COLS]
        assert meta.isna().sum().sum() == 0, "Found NaN in metadata"

    def test_no_zero_rows(self, emb_array):
        """No embedding should be the zero vector."""
        norms = np.linalg.norm(emb_array, axis=1)
        assert (norms > 0).all(), "Found zero-vector embedding"


# =====
# Group key and merge integrity
# =====
class TestEmbeddingsMergeKeys:
    """Verify group_key column for downstream merging."""

    def test_group_key_unique(self, emb_df):
        """Each group_key must be unique."""
        assert emb_df["group_key"].nunique() == len(emb_df)

    def test_group_key_format(self, emb_df):
        """group_key should be '{session}_{red}' format."""
        for _, row in emb_df.head(5).iterrows():
            expected = f"{int(row['session'])}_{int(row['red'])}"
            assert row["group_key"] == expected

    def test_missing_group_is_empty_chat(self, emb_df):
        """The missing group (55_0) should be the empty-chat group."""
        assert EXPECTED_MISSING_GROUP not in emb_df["group_key"].values

    def test_covers_58_sessions(self, emb_df):
        """All 58 sessions should be represented."""
        assert emb_df["session"].nunique() == 58

    def test_mergeable_with_effort_group(self, emb_df):
        """group_key should match effort group cross-section."""
        if not EFFORT_GROUP_FILE.exists():
            pytest.skip("Effort group parquet not found")
        effort = pd.read_parquet(EFFORT_GROUP_FILE)
        effort_keys = set(
            effort["session"].astype(int).astype(str)
            + "_"
            + effort["red"].astype(int).astype(str)
        )
        emb_keys = set(emb_df["group_key"])
        # All embedding keys should exist in effort data
        missing = emb_keys - effort_keys
        assert not missing, f"Embedding keys not in effort: {missing}"


# =====
# Regression on known values
# =====
class TestEmbeddingsRegression:
    """Regression tests against verified embedding values."""

    def test_first_row_emb_0(self, emb_df):
        """Regression: first row emb_0 from verified output."""
        assert emb_df["emb_0"].iloc[0] == pytest.approx(0.03509521, abs=1e-4)

    def test_first_row_emb_1(self, emb_df):
        """Regression: first row emb_1 from verified output."""
        assert emb_df["emb_1"].iloc[0] == pytest.approx(0.00184917, abs=1e-4)

    def test_first_row_emb_1535(self, emb_df):
        """Regression: first row last embedding element."""
        assert emb_df["emb_1535"].iloc[0] == pytest.approx(-0.01272583, abs=1e-4)

    def test_first_group_key(self, emb_df):
        """First row should be session 1, Green (red=0)."""
        assert emb_df["group_key"].iloc[0] == "1_0"


# =====
# Consistency with chat parquet
# =====
class TestEmbeddingsConsistency:
    """Cross-validate embeddings against source chat data."""

    def test_row_count_matches_nonempty_chat(self, emb_df):
        """Rows should equal non-empty chat groups in source."""
        if not CHAT_FILE.exists():
            pytest.skip("Chat parquet not found")
        chat = pd.read_parquet(CHAT_FILE)
        text_col = "group_chat_text" if "group_chat_text" in chat.columns else "chat_text"
        nonempty = chat[chat[text_col].str.strip().astype(bool)]
        assert len(emb_df) == len(nonempty)

    def test_metadata_matches_chat_source(self, emb_df):
        """n_words in embeddings should match chat parquet exactly."""
        if not CHAT_FILE.exists():
            pytest.skip("Chat parquet not found")
        chat = pd.read_parquet(CHAT_FILE)
        chat["group_key"] = (
            chat["session"].astype(int).astype(str)
            + "_"
            + chat["red"].astype(int).astype(str)
        )
        merged = emb_df[["group_key", "n_words"]].merge(
            chat[["group_key", "n_words"]],
            on="group_key", suffixes=("_emb", "_chat"),
        )
        assert (merged["n_words_emb"] == merged["n_words_chat"]).all()
