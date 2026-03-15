"""
Tests for compute_embeddings.py.

Tests data loading, state merging, and output construction
using synthetic data with no API calls.

Author: Claude Code
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
    build_output_df,
    load_messages,
    load_state_data,
    merge_state_labels,
)

# =====
# Fixtures
# =====
SAMPLE_PROMISE_ROWS = [
    {
        'session_code': 'abc',
        'treatment': 1,
        'segment': 'supergame1',
        'round': 2,
        'group': 1,
        'label': 'A',
        'participant_id': 1,
        'contribution': 25,
        'payoff': 40,
        'message_count': 2,
        'promise_count': 1,
        'promise_percentage': 50.0,
        'messages': json.dumps(['hello world', 'lets cooperate']),
        'classifications': json.dumps([0, 1]),
    },
    {
        'session_code': 'abc',
        'treatment': 1,
        'segment': 'supergame1',
        'round': 2,
        'group': 1,
        'label': 'E',
        'participant_id': 5,
        'contribution': 20,
        'payoff': 38,
        'message_count': 1,
        'promise_count': 0,
        'promise_percentage': 0.0,
        'messages': json.dumps(['ok sounds good']),
        'classifications': json.dumps([0]),
    },
]

SAMPLE_STATE_ROWS = [
    {
        'session_code': 'abc',
        'treatment': 1,
        'segment': 'supergame1',
        'round_num': 2,
        'group_id': 1,
        'label': 'A',
        'contribution': 25,
        'others_total_contribution': 70,
        'player_state': 'cooperative',
        'player_behavior': 'cooperative',
        'made_promise': True,
        'others_threshold': 60,
        'player_threshold': 20,
    },
]


@pytest.fixture
def sample_promise_csv(tmp_path):
    """Write sample promise data to a temp CSV and return path."""
    df = pd.DataFrame(SAMPLE_PROMISE_ROWS)
    path = tmp_path / 'promise_classifications.csv'
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_state_csv(tmp_path):
    """Write sample state data to a temp CSV and return path."""
    df = pd.DataFrame(SAMPLE_STATE_ROWS)
    path = tmp_path / 'player_state_classification.csv'
    df.to_csv(path, index=False)
    return path


EXPLODED_MSG_ROWS = [
    {'session_code': 'abc', 'treatment': 1, 'segment': 'supergame1',
     'round': 2, 'group': 1, 'label': 'A',
     'message_index': 0, 'message_text': 'hello world'},
    {'session_code': 'abc', 'treatment': 1, 'segment': 'supergame1',
     'round': 2, 'group': 1, 'label': 'A',
     'message_index': 1, 'message_text': 'lets cooperate'},
    {'session_code': 'abc', 'treatment': 1, 'segment': 'supergame1',
     'round': 2, 'group': 1, 'label': 'E',
     'message_index': 0, 'message_text': 'ok sounds good'},
]


@pytest.fixture
def sample_messages_df():
    """Return an exploded messages DataFrame (3 rows)."""
    return pd.DataFrame(EXPLODED_MSG_ROWS)


# =====
# Tests for load_messages
# =====
class TestLoadMessages:
    """Tests for load_messages() JSON exploding logic."""

    def test_explodes_json_to_individual_rows(self, sample_promise_csv):
        """Two player-rounds with 2+1 messages should produce 3 rows."""
        with patch('compute_embeddings.INPUT_FILE', sample_promise_csv):
            result = load_messages()

        assert len(result) == 3

    def test_preserves_metadata_columns(self, sample_promise_csv):
        """Each exploded row should carry session/segment/round metadata."""
        with patch('compute_embeddings.INPUT_FILE', sample_promise_csv):
            result = load_messages()

        expected_cols = {
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'message_index', 'message_text',
        }
        assert expected_cols == set(result.columns)

    def test_message_index_resets_per_player(self, sample_promise_csv):
        """Message index should start at 0 for each player-round."""
        with patch('compute_embeddings.INPUT_FILE', sample_promise_csv):
            result = load_messages()

        player_a = result[result['label'] == 'A']
        player_e = result[result['label'] == 'E']

        assert list(player_a['message_index']) == [0, 1]
        assert list(player_e['message_index']) == [0]

    def test_message_text_matches_json(self, sample_promise_csv):
        """Exploded text should match original JSON content."""
        with patch('compute_embeddings.INPUT_FILE', sample_promise_csv):
            result = load_messages()

        texts = result['message_text'].tolist()
        assert 'hello world' in texts
        assert 'lets cooperate' in texts
        assert 'ok sounds good' in texts


# =====
# Tests for load_state_data
# =====
class TestLoadStateData:
    """Tests for load_state_data() column renaming."""

    def test_renames_round_num_to_round(self, sample_state_csv):
        """round_num should be renamed to round."""
        with patch('compute_embeddings.STATE_FILE', sample_state_csv):
            result = load_state_data()

        assert 'round' in result.columns
        assert 'round_num' not in result.columns

    def test_renames_group_id_to_group(self, sample_state_csv):
        """group_id should be renamed to group."""
        with patch('compute_embeddings.STATE_FILE', sample_state_csv):
            result = load_state_data()

        assert 'group' in result.columns
        assert 'group_id' not in result.columns


# =====
# Tests for merge_state_labels
# =====
class TestMergeStateLabels:
    """Tests for merge_state_labels() LEFT JOIN logic."""

    def test_adds_player_state_column(self, sample_messages_df):
        """Merged result should include player_state column."""
        state_df = pd.DataFrame([{
            'session_code': 'abc', 'segment': 'supergame1',
            'round': 2, 'group': 1, 'label': 'A',
            'player_state': 'cooperative',
        }])

        result = merge_state_labels(sample_messages_df, state_df)
        assert 'player_state' in result.columns

    def test_matched_rows_get_state(self, sample_messages_df):
        """Player A rows should get 'cooperative' state."""
        state_df = pd.DataFrame([{
            'session_code': 'abc', 'segment': 'supergame1',
            'round': 2, 'group': 1, 'label': 'A',
            'player_state': 'cooperative',
        }])

        result = merge_state_labels(sample_messages_df, state_df)
        player_a = result[result['label'] == 'A']
        assert all(player_a['player_state'] == 'cooperative')

    def test_unmatched_rows_get_nan(self, sample_messages_df):
        """Player E (not in state data) should have NaN state."""
        state_df = pd.DataFrame([{
            'session_code': 'abc', 'segment': 'supergame1',
            'round': 2, 'group': 1, 'label': 'A',
            'player_state': 'cooperative',
        }])

        result = merge_state_labels(sample_messages_df, state_df)
        player_e = result[result['label'] == 'E']
        assert all(player_e['player_state'].isna())

    def test_preserves_all_message_rows(self, sample_messages_df):
        """LEFT JOIN should not drop any message rows."""
        state_df = pd.DataFrame([{
            'session_code': 'abc', 'segment': 'supergame1',
            'round': 2, 'group': 1, 'label': 'A',
            'player_state': 'cooperative',
        }])

        result = merge_state_labels(sample_messages_df, state_df)
        assert len(result) == len(sample_messages_df)


# =====
# Tests for build_output_df
# =====
class TestBuildOutputDf:
    """Tests for build_output_df() combining metadata + embeddings."""

    def test_correct_shape(self, sample_messages_df):
        """Output should have n_messages rows and id+state+model+emb cols."""
        sample_messages_df['player_state'] = 'cooperative'
        embeddings = np.random.rand(3, 5)

        result = build_output_df(sample_messages_df, embeddings, 'test-model')

        assert result.shape[0] == 3
        # ID_COLS(8) + player_state(1) + model(1) + emb_dims(5) = 15
        assert result.shape[1] == 15

    def test_embedding_column_names(self, sample_messages_df):
        """Embedding columns should be named emb_0, emb_1, etc."""
        sample_messages_df['player_state'] = 'cooperative'
        embeddings = np.random.rand(3, 4)

        result = build_output_df(sample_messages_df, embeddings, 'test-model')
        emb_cols = [c for c in result.columns if c.startswith('emb_')]

        assert emb_cols == ['emb_0', 'emb_1', 'emb_2', 'emb_3']

    def test_model_column_value(self, sample_messages_df):
        """Model column should contain the model name."""
        sample_messages_df['player_state'] = 'cooperative'
        embeddings = np.random.rand(3, 2)

        result = build_output_df(sample_messages_df, embeddings, 'my-model')
        assert all(result['model'] == 'my-model')

    def test_metadata_preserved(self, sample_messages_df):
        """Metadata columns should pass through unchanged."""
        sample_messages_df['player_state'] = 'cooperative'
        embeddings = np.random.rand(3, 2)

        result = build_output_df(sample_messages_df, embeddings, 'test')

        assert list(result['label']) == ['A', 'A', 'E']
        assert list(result['message_text']) == [
            'hello world', 'lets cooperate', 'ok sounds good',
        ]

    def test_embedding_values_match_input(self, sample_messages_df):
        """Embedding values should match the numpy array passed in."""
        sample_messages_df['player_state'] = 'cooperative'
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = build_output_df(sample_messages_df, embeddings, 'test')

        assert result['emb_0'].tolist() == [1.0, 3.0, 5.0]
        assert result['emb_1'].tolist() == [2.0, 4.0, 6.0]
