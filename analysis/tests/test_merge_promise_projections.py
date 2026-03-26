"""
Regression and integration tests for promise projection merge in merge_panel_data.py.

Tests the new load_promise_projection_data() and merge_promise_projections()
functions added in Task #4, plus validation that round 1 has NaN promise
projections and that the column order is correct.

Author: Claude Code (test-writer)
Date: 2026-03-15
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from merge_panel_data import (
    EMBEDDING_COLS,
    HOMOGENEITY_EMBEDDING_COLS,
    PROMISE_EMBEDDING_COLS,
    ROUND_LIAR_EMBEDDING_COLS,
    CUMULATIVE_LIAR_EMBEDDING_COLS,
    STATE_MERGE_KEYS,
    _load_projection_data,
    _merge_projection_data,
    _validate_no_suffix_columns,
    _validate_round_1_embeddings,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
PROMISE_PROJ_FILE = DERIVED_DIR / "promise_embedding_projections.csv"
MERGED_FILE = DERIVED_DIR / "merged_panel.csv"


# =====
# Helpers
# =====
def _make_promise_projections_csv(tmp_path, n_players=4, n_rounds=2):
    """Write synthetic promise projections CSV for testing."""
    rows = []
    labels = [chr(65 + i) for i in range(n_players)]
    for rnd in range(1, n_rounds + 1):
        for lbl in labels:
            rows.append({
                'session_code': 's1',
                'segment': 'supergame1',
                'round': rnd,
                'label': lbl,
                'proj_promise_msg_dir_small': 0.5 * rnd,
                'proj_promise_pr_dir_small': 0.4 * rnd,
                'proj_promise_msg_dir_large': 0.6 * rnd,
                'proj_promise_pr_dir_large': 0.3 * rnd,
            })
    path = tmp_path / 'promise_embedding_projections.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_base_panel(n_players=4, rounds=(1, 2)):
    """Create a minimal panel for merge testing."""
    labels = [chr(65 + i) for i in range(n_players)]
    rows = []
    for rnd in rounds:
        for lbl in labels:
            rows.append({
                'session_code': 's1',
                'segment': 'supergame1',
                'round': rnd,
                'label': lbl,
                'page_type': 'Contribute',
            })
    return pd.DataFrame(rows)


# =====
# Regression: PROMISE_EMBEDDING_COLS constant
# =====
class TestPromiseEmbeddingColsConstant:
    """Verify the new PROMISE_EMBEDDING_COLS constant."""

    def test_has_four_columns(self):
        """Should define exactly 4 promise embedding columns."""
        assert len(PROMISE_EMBEDDING_COLS) == 4

    def test_column_names(self):
        """Column names should match expected promise projection names."""
        expected = [
            'proj_promise_msg_dir_small', 'proj_promise_pr_dir_small',
            'proj_promise_msg_dir_large', 'proj_promise_pr_dir_large',
        ]
        assert PROMISE_EMBEDDING_COLS == expected

    def test_does_not_overlap_embedding_cols(self):
        """Promise columns should not overlap with cooperative embedding columns."""
        assert not set(PROMISE_EMBEDDING_COLS) & set(EMBEDDING_COLS)


# =====
# Regression: _load_projection_data
# =====
class TestLoadProjectionData:
    """Tests for loading and aggregating projection data."""

    def test_aggregates_to_player_round(self, tmp_path):
        """Should aggregate message-level data to player-round means."""
        rows = []
        for msg_idx in range(2):
            rows.append({
                'session_code': 's1', 'segment': 'supergame1',
                'round': 2, 'label': 'A',
                'proj_promise_msg_dir_small': 1.0 + msg_idx,
                'proj_promise_pr_dir_small': 2.0 + msg_idx,
                'proj_promise_msg_dir_large': 3.0 + msg_idx,
                'proj_promise_pr_dir_large': 4.0 + msg_idx,
            })
        path = tmp_path / 'promise_embedding_projections.csv'
        pd.DataFrame(rows).to_csv(path, index=False)

        result = _load_projection_data(path, PROMISE_EMBEDDING_COLS)
        assert len(result) == 1
        assert result['proj_promise_msg_dir_small'].iloc[0] == pytest.approx(1.5)

    def test_has_merge_keys(self, tmp_path):
        """Output should have STATE_MERGE_KEYS for joining."""
        path = _make_promise_projections_csv(tmp_path, n_players=2, n_rounds=1)
        result = _load_projection_data(path, PROMISE_EMBEDDING_COLS)
        for key in STATE_MERGE_KEYS:
            assert key in result.columns


# =====
# Regression: _merge_projection_data
# =====
class TestMergeProjectionData:
    """Tests for LEFT JOIN of projections onto panel."""

    def test_left_join_preserves_row_count(self):
        """Merge should not change panel row count."""
        panel = _make_base_panel(n_players=4, rounds=(1, 2))
        proj_df = pd.DataFrame({
            'session_code': ['s1'] * 4,
            'segment': ['supergame1'] * 4,
            'round': [2] * 4,
            'label': ['A', 'B', 'C', 'D'],
            'proj_promise_msg_dir_small': [0.5] * 4,
            'proj_promise_pr_dir_small': [0.4] * 4,
            'proj_promise_msg_dir_large': [0.6] * 4,
            'proj_promise_pr_dir_large': [0.3] * 4,
        })
        merged = _merge_projection_data(panel, proj_df, PROMISE_EMBEDDING_COLS, 'promise')
        assert len(merged) == len(panel)

    def test_round_1_has_nan(self):
        """Round 1 rows should have NaN projections (no prior chat)."""
        panel = _make_base_panel(n_players=2, rounds=(1, 2))
        proj_df = pd.DataFrame({
            'session_code': ['s1'] * 2,
            'segment': ['supergame1'] * 2,
            'round': [2] * 2,
            'label': ['A', 'B'],
            'proj_promise_msg_dir_small': [0.5] * 2,
            'proj_promise_pr_dir_small': [0.4] * 2,
            'proj_promise_msg_dir_large': [0.6] * 2,
            'proj_promise_pr_dir_large': [0.3] * 2,
        })
        merged = _merge_projection_data(panel, proj_df, PROMISE_EMBEDDING_COLS, 'promise')
        r1 = merged[merged['round'] == 1]
        for col in PROMISE_EMBEDDING_COLS:
            assert r1[col].isna().all(), f"Round 1 should have NaN {col}"


# =====
# Regression: _validate_round_1_embeddings includes all projection types
# =====
class TestValidateRound1Embeddings:
    """Test that validation now checks promise projection columns too."""

    def test_passes_when_round_1_all_nan(self):
        """Validation should pass when round 1 has NaN for all embedding cols."""
        all_cols = (EMBEDDING_COLS + PROMISE_EMBEDDING_COLS
                    + HOMOGENEITY_EMBEDDING_COLS + ROUND_LIAR_EMBEDDING_COLS
                    + CUMULATIVE_LIAR_EMBEDDING_COLS)
        panel = pd.DataFrame({
            'round': [1, 2],
            **{col: [np.nan, 0.5] for col in all_cols},
        })
        _validate_round_1_embeddings(panel)

    def test_fails_when_round_1_has_promise_projection(self):
        """Validation should fail if round 1 has non-NaN promise projections."""
        all_cols = (EMBEDDING_COLS + PROMISE_EMBEDDING_COLS
                    + HOMOGENEITY_EMBEDDING_COLS + ROUND_LIAR_EMBEDDING_COLS
                    + CUMULATIVE_LIAR_EMBEDDING_COLS)
        panel = pd.DataFrame({
            'round': [1],
            **{col: [np.nan] for col in all_cols},
        })
        panel.loc[0, 'proj_promise_msg_dir_small'] = 0.5
        with pytest.raises(ValueError, match="Round 1 has non-NaN"):
            _validate_round_1_embeddings(panel)


# =====
# Integration: real data tests (skipped if data not available)
# =====
@pytest.mark.integration
class TestPromiseProjectionIntegration:
    """Integration tests using real merged_panel.csv output."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        """Skip if merged panel has not been generated."""
        if not MERGED_FILE.exists():
            pytest.skip("merged_panel.csv not generated yet")

    @pytest.fixture
    def merged_df(self):
        """Load the merged panel CSV."""
        return pd.read_csv(MERGED_FILE)

    def _require_promise_cols(self, merged_df):
        """Skip test if promise columns are not yet in the CSV."""
        if PROMISE_EMBEDDING_COLS[0] not in merged_df.columns:
            pytest.skip("Promise columns not yet in merged_panel.csv (stale output)")

    def test_promise_columns_present(self, merged_df):
        """All 4 promise embedding columns should be in merged panel."""
        self._require_promise_cols(merged_df)
        for col in PROMISE_EMBEDDING_COLS:
            assert col in merged_df.columns, f"Missing column: {col}"

    def test_round_1_promise_projections_nan(self, merged_df):
        """Round 1 should have NaN promise projections."""
        self._require_promise_cols(merged_df)
        r1 = merged_df[merged_df['round'] == 1]
        if len(r1) == 0:
            pytest.skip("No round 1 rows found")
        for col in PROMISE_EMBEDDING_COLS:
            assert r1[col].isna().all(), (
                f"Round 1 should have NaN {col}, "
                f"found {r1[col].notna().sum()} non-NaN"
            )

    def test_non_round_1_has_some_values(self, merged_df):
        """Non-round-1 game rows should have some non-NaN promise projections."""
        self._require_promise_cols(merged_df)
        game = merged_df[
            (merged_df['page_type'] != 'all_instructions')
            & (merged_df['round'] > 1)
        ]
        if len(game) == 0:
            pytest.skip("No non-round-1 game rows")
        col = PROMISE_EMBEDDING_COLS[0]
        assert game[col].notna().any(), f"Expected some non-NaN values in {col}"

    def test_no_merge_artifact_columns(self, merged_df):
        """No _x or _y suffixed columns should exist."""
        bad = [c for c in merged_df.columns if c.endswith('_x') or c.endswith('_y')]
        assert len(bad) == 0, f"Found merge artifact columns: {bad}"

    def test_promise_cols_come_after_embedding_cols(self, merged_df):
        """Promise columns should appear after cooperative embedding columns."""
        self._require_promise_cols(merged_df)
        cols = list(merged_df.columns)
        if EMBEDDING_COLS[0] not in cols:
            pytest.skip("Embedding columns not found")
        emb_idx = cols.index(EMBEDDING_COLS[-1])
        promise_idx = cols.index(PROMISE_EMBEDDING_COLS[0])
        assert promise_idx > emb_idx, "Promise cols should follow embedding cols"
