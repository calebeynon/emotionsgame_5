"""
Regression and integration tests for compute_embeddings.py.

Additional tests grounded in real data from promise_classifications.csv
and player_state_classification.csv. Tests validate data loading,
merging, and output construction against known-good real results.

Author: Claude Code (test-writer)
Date: 2026-03-15
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from compute_embeddings import (
    ID_COLS,
    build_output_df,
    load_messages,
    load_state_data,
    merge_state_labels,
)

# FILE PATHS
PROMISE_CSV = (
    Path(__file__).parent.parent / 'datastore' / 'derived'
    / 'promise_classifications.csv'
)
STATE_CSV = (
    Path(__file__).parent.parent / 'datastore' / 'derived'
    / 'player_state_classification.csv'
)


def _skip_if_no_data():
    """Skip test if real data files are not available."""
    if not PROMISE_CSV.exists():
        pytest.skip(f"Promise data not found: {PROMISE_CSV}")
    if not STATE_CSV.exists():
        pytest.skip(f"State data not found: {STATE_CSV}")


# =====
# Regression: load_messages against real data
# =====
class TestLoadMessagesRealData:
    """Regression tests for load_messages using real promise_classifications.csv."""

    def test_total_exploded_message_count(self):
        """Real data should produce exactly 4700 individual messages."""
        _skip_if_no_data()
        result = load_messages()
        assert len(result) == 4700

    def test_columns_match_expected(self):
        """Exploded DataFrame should have exactly the ID_COLS columns."""
        _skip_if_no_data()
        result = load_messages()
        expected_cols = {
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'message_index', 'message_text',
        }
        assert set(result.columns) == expected_cols

    def test_sessions_match_expected(self):
        """Should contain all 10 known sessions."""
        _skip_if_no_data()
        result = load_messages()
        expected_sessions = {
            '6sdkxl2q', '6ucza025', '6uv359rf', 'iiu3xixz',
            'irrzlgk2', 'j3ki5tli', 'r5dj4yfl', 'sa7mprty',
            'sylq2syi', 'umbzdj98',
        }
        assert set(result['session_code'].unique()) == expected_sessions

    def test_treatments_are_1_or_2(self):
        """Treatments should only be 1 or 2."""
        _skip_if_no_data()
        result = load_messages()
        assert set(result['treatment'].unique()) == {1, 2}

    def test_segments_are_supergames(self):
        """Segments should be supergame1 through supergame5."""
        _skip_if_no_data()
        result = load_messages()
        expected = {f'supergame{i}' for i in range(1, 6)}
        assert set(result['segment'].unique()) == expected

    def test_known_message_text(self):
        """First message from sa7mprty/supergame1/r2/A should match."""
        _skip_if_no_data()
        result = load_messages()
        row = result[
            (result['session_code'] == 'sa7mprty')
            & (result['segment'] == 'supergame1')
            & (result['round'] == 2)
            & (result['label'] == 'A')
            & (result['message_index'] == 0)
        ]
        assert len(row) == 1
        expected_text = (
            "Hey if everyone keeps in 25 in the group, "
            "earning is 40 each"
        )
        assert row.iloc[0]['message_text'] == expected_text

    def test_no_empty_message_texts(self):
        """No message_text should be null or empty string."""
        _skip_if_no_data()
        result = load_messages()
        assert result['message_text'].notna().all()
        assert (result['message_text'].str.len() > 0).all()

    def test_message_index_starts_at_zero(self):
        """All message_index values should start at 0."""
        _skip_if_no_data()
        result = load_messages()
        assert result['message_index'].min() == 0


# =====
# Regression: load_state_data against real data
# =====
class TestLoadStateDataRealData:
    """Regression tests for load_state_data using real CSV."""

    def test_column_renaming(self):
        """round_num and group_id should be renamed."""
        _skip_if_no_data()
        result = load_state_data()
        assert 'round' in result.columns
        assert 'group' in result.columns
        assert 'round_num' not in result.columns
        assert 'group_id' not in result.columns

    def test_has_player_state_column(self):
        """Should include player_state column."""
        _skip_if_no_data()
        result = load_state_data()
        assert 'player_state' in result.columns

    def test_player_states_are_valid(self):
        """player_state should only be cooperative or noncooperative."""
        _skip_if_no_data()
        result = load_state_data()
        valid = {'cooperative', 'noncooperative'}
        assert set(result['player_state'].unique()) == valid

    def test_row_count(self):
        """State classification should have 3520 rows."""
        _skip_if_no_data()
        result = load_state_data()
        assert len(result) == 3520


# =====
# Regression: merge_state_labels with real data
# =====
class TestMergeStateLabelsRealData:
    """Regression tests for merge using real data files."""

    def test_full_merge_preserves_all_messages(self):
        """LEFT JOIN should not drop any message rows."""
        _skip_if_no_data()
        messages = load_messages()
        states = load_state_data()
        merged = merge_state_labels(messages, states)
        assert len(merged) == len(messages)

    def test_full_merge_all_matched(self):
        """All 4700 messages should match a state label."""
        _skip_if_no_data()
        messages = load_messages()
        states = load_state_data()
        merged = merge_state_labels(messages, states)
        assert merged['player_state'].notna().all()

    def test_known_player_state(self):
        """sa7mprty/supergame1/r2/A should be cooperative."""
        _skip_if_no_data()
        messages = load_messages()
        states = load_state_data()
        merged = merge_state_labels(messages, states)

        row = merged[
            (merged['session_code'] == 'sa7mprty')
            & (merged['segment'] == 'supergame1')
            & (merged['round'] == 2)
            & (merged['label'] == 'A')
        ]
        assert all(row['player_state'] == 'cooperative')

    def test_merge_adds_only_player_state(self):
        """Merge should add only the player_state column."""
        _skip_if_no_data()
        messages = load_messages()
        states = load_state_data()
        merged = merge_state_labels(messages, states)

        new_cols = set(merged.columns) - set(messages.columns)
        assert new_cols == {'player_state'}


# =====
# Edge cases: merge with partial state data
# =====
class TestMergeStateLabelsEdgeCases:
    """Edge case tests for merge_state_labels."""

    def test_empty_state_df_produces_all_nan(self):
        """Empty state DataFrame should leave all player_state as NaN."""
        messages_df = pd.DataFrame([{
            'session_code': 'x', 'treatment': 1,
            'segment': 'supergame1', 'round': 2,
            'group': 1, 'label': 'A',
            'message_index': 0, 'message_text': 'hi',
        }])
        state_df = pd.DataFrame(columns=[
            'session_code', 'segment', 'round',
            'group', 'label', 'player_state',
        ])

        result = merge_state_labels(messages_df, state_df)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]['player_state'])

    def test_duplicate_state_rows_cause_expansion(self):
        """Duplicate state rows should cause row expansion (data issue)."""
        messages_df = pd.DataFrame([{
            'session_code': 'x', 'treatment': 1,
            'segment': 'supergame1', 'round': 2,
            'group': 1, 'label': 'A',
            'message_index': 0, 'message_text': 'hi',
        }])
        state_df = pd.DataFrame([
            {
                'session_code': 'x', 'segment': 'supergame1',
                'round': 2, 'group': 1, 'label': 'A',
                'player_state': 'cooperative',
            },
            {
                'session_code': 'x', 'segment': 'supergame1',
                'round': 2, 'group': 1, 'label': 'A',
                'player_state': 'noncooperative',
            },
        ])

        result = merge_state_labels(messages_df, state_df)
        # LEFT JOIN with 2 matching rows -> 2 output rows
        assert len(result) == 2


# =====
# Regression: build_output_df structure
# =====
class TestBuildOutputDfRegression:
    """Regression tests for build_output_df with realistic dimensions."""

    def test_small_model_dimensions(self):
        """Output should have 1536 emb columns for small model."""
        n = 3
        dim = 1536
        msgs = pd.DataFrame({
            'session_code': ['a'] * n,
            'treatment': [1] * n,
            'segment': ['supergame1'] * n,
            'round': [2] * n,
            'group': [1] * n,
            'label': ['A'] * n,
            'message_index': list(range(n)),
            'message_text': ['hi'] * n,
            'player_state': ['cooperative'] * n,
        })
        embeddings = np.random.rand(n, dim)

        result = build_output_df(msgs, embeddings, 'text-embedding-3-small')
        emb_cols = [c for c in result.columns if c.startswith('emb_')]

        assert len(emb_cols) == dim
        assert emb_cols[0] == 'emb_0'
        assert emb_cols[-1] == f'emb_{dim - 1}'

    def test_large_model_dimensions(self):
        """Output should have 3072 emb columns for large model."""
        n = 2
        dim = 3072
        msgs = pd.DataFrame({
            'session_code': ['a'] * n,
            'treatment': [1] * n,
            'segment': ['supergame1'] * n,
            'round': [2] * n,
            'group': [1] * n,
            'label': ['A'] * n,
            'message_index': list(range(n)),
            'message_text': ['hi'] * n,
            'player_state': ['cooperative'] * n,
        })
        embeddings = np.random.rand(n, dim)

        result = build_output_df(msgs, embeddings, 'text-embedding-3-large')
        emb_cols = [c for c in result.columns if c.startswith('emb_')]

        assert len(emb_cols) == dim

    def test_id_cols_constant_matches_expected(self):
        """ID_COLS constant should match expected column list."""
        expected = [
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'message_index', 'message_text',
        ]
        assert ID_COLS == expected

    def test_output_dtypes_are_numeric_for_embeddings(self):
        """Embedding columns should be float dtype."""
        n = 2
        msgs = pd.DataFrame({
            'session_code': ['a'] * n,
            'treatment': [1] * n,
            'segment': ['supergame1'] * n,
            'round': [2] * n,
            'group': [1] * n,
            'label': ['A'] * n,
            'message_index': list(range(n)),
            'message_text': ['hi'] * n,
            'player_state': ['cooperative'] * n,
        })
        embeddings = np.array([[1.5, 2.5], [3.5, 4.5]])

        result = build_output_df(msgs, embeddings, 'test')

        assert result['emb_0'].dtype == np.float64
        assert result['emb_1'].dtype == np.float64


# =====
# Integration: full pipeline (load -> merge -> build_output)
# =====
class TestFullPipelineIntegration:
    """Integration test covering load -> merge -> build_output."""

    def test_pipeline_produces_correct_output(self):
        """Full pipeline should produce DataFrame with correct structure."""
        _skip_if_no_data()
        messages = load_messages()
        states = load_state_data()
        merged = merge_state_labels(messages, states)

        # Use small fake embeddings to test structure
        n = len(merged)
        dim = 4
        fake_embeddings = np.random.rand(n, dim)

        result = build_output_df(merged, fake_embeddings, 'test-model')

        # Check shape
        assert result.shape[0] == 4700
        # ID_COLS(8) + player_state(1) + model(1) + emb_dims(4) = 14
        assert result.shape[1] == 14

        # Check no NaN in required columns
        assert result['session_code'].notna().all()
        assert result['message_text'].notna().all()
        assert result['model'].notna().all()
        assert result['player_state'].notna().all()

        # Check embedding columns exist
        for i in range(dim):
            assert f'emb_{i}' in result.columns

    def test_pipeline_message_count_per_session(self):
        """Verify message counts per session are reasonable."""
        _skip_if_no_data()
        messages = load_messages()
        session_counts = messages.groupby('session_code').size()

        # Each session has 16 players, so should have some messages
        assert (session_counts > 0).all()
        # No single session should have more than half the messages
        assert (session_counts < 2350).all()
