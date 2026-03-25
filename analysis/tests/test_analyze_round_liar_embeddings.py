"""
Tests for analyze_round_liar_embeddings.py.

Tests round-liar label computation from lied_this_round_20 boolean,
label merging, centroid computation, direction normalization, and
projection output using synthetic data with no file I/O or API calls.

Author: Claude Code
Date: 2026-03-21
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from analyze_round_liar_embeddings import (
    compute_round_liar_centroids,
    compute_round_liar_labels,
    merge_round_liar_labels,
    RLIAR_COL,
    STATE_LIAR,
    STATE_NON_LIAR,
    JOIN_KEYS,
    ID_COLS,
    PR_ID_COLS,
    PROBE_PHRASES,
    _build_output,
    _build_cross_level_output,
)
from analyze_embeddings import compute_difference_vector, project_onto_direction


# =====
# Synthetic data builders
# =====
def _make_state_df(rows_spec):
    """Build player-level DataFrame with lied_this_round_20 column.

    Args:
        rows_spec: list of (session, segment, round, group, label, lied_bool)
    """
    rows = []
    for session, segment, rnd, group, label, lied in rows_spec:
        rows.append({
            'session_code': session, 'segment': segment,
            'round': rnd, 'group': group,
            'label': label, 'lied_this_round_20': lied,
        })
    return pd.DataFrame(rows)


def _make_embeddings_and_labels(n_liar=3, n_non_liar=3, dim=4):
    """Create synthetic embeddings with known separation by liar label."""
    rng = np.random.RandomState(42)
    liar = rng.normal(loc=1.0, scale=0.1, size=(n_liar, dim))
    non_liar = rng.normal(loc=-1.0, scale=0.1, size=(n_non_liar, dim))
    embeddings = np.vstack([liar, non_liar])
    labels = np.array(
        [STATE_LIAR] * n_liar + [STATE_NON_LIAR] * n_non_liar
    )
    return embeddings, labels


def _make_metadata(n, with_liar=True):
    """Create minimal metadata DataFrame for n messages."""
    meta = pd.DataFrame({
        'session_code': ['s1'] * n, 'treatment': [1] * n,
        'segment': ['supergame1'] * n, 'round': [2] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
        'message_index': range(n),
        'message_text': [f'msg {i}' for i in range(n)],
        'player_state': ['cooperative'] * (n // 2)
        + ['noncooperative'] * (n - n // 2),
    })
    if with_liar:
        meta[RLIAR_COL] = (
            [STATE_LIAR] * (n // 2)
            + [STATE_NON_LIAR] * (n - n // 2)
        )
    return meta


# =====
# TestComputeLiarLabels
# =====
class TestComputeLiarLabels:
    """Tests for round-liar label computation from boolean column."""

    def test_true_maps_to_liar(self):
        """lied_this_round_20=True -> 'liar' label."""
        df = _make_state_df([('s1', 'sg1', 2, 1, 'A', True)])
        result = compute_round_liar_labels(df)
        assert result[RLIAR_COL].iloc[0] == STATE_LIAR

    def test_false_maps_to_non_liar(self):
        """lied_this_round_20=False -> 'non_liar' label."""
        df = _make_state_df([('s1', 'sg1', 2, 1, 'A', False)])
        result = compute_round_liar_labels(df)
        assert result[RLIAR_COL].iloc[0] == STATE_NON_LIAR

    def test_mixed_labels(self):
        """Mixed True/False should produce corresponding labels."""
        df = _make_state_df([
            ('s1', 'sg1', 2, 1, 'A', True),
            ('s1', 'sg1', 2, 1, 'E', False),
            ('s1', 'sg1', 2, 1, 'J', True),
            ('s1', 'sg1', 2, 1, 'N', False),
        ])
        result = compute_round_liar_labels(df)
        expected = [STATE_LIAR, STATE_NON_LIAR, STATE_LIAR, STATE_NON_LIAR]
        assert result[RLIAR_COL].tolist() == expected

    def test_output_columns(self):
        """Output should contain JOIN_KEYS and RLIAR_COL."""
        df = _make_state_df([('s1', 'sg1', 2, 1, 'A', True)])
        result = compute_round_liar_labels(df)
        for key in JOIN_KEYS:
            assert key in result.columns
        assert RLIAR_COL in result.columns

    def test_all_same_label(self):
        """All True -> all 'liar'; All False -> all 'non_liar'."""
        df_t = _make_state_df([
            ('s1', 'sg1', 2, 1, 'A', True),
            ('s1', 'sg1', 2, 1, 'E', True),
        ])
        assert (compute_round_liar_labels(df_t)[RLIAR_COL] == STATE_LIAR).all()
        df_f = _make_state_df([
            ('s1', 'sg1', 2, 1, 'A', False),
            ('s1', 'sg1', 2, 1, 'E', False),
        ])
        assert (compute_round_liar_labels(df_f)[RLIAR_COL] == STATE_NON_LIAR).all()


# =====
# TestMergeLiarLabels
# =====
class TestMergeLiarLabels:
    """Tests for merge_round_liar_labels."""

    def test_adds_liar_column(self):
        """Merged metadata should have RLIAR_COL."""
        meta = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['sg1'],
            'round': [2], 'group': [1], 'label': ['A'],
        })
        liar_df = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['sg1'],
            'round': [2], 'group': [1], 'label': ['A'],
            RLIAR_COL: [STATE_LIAR],
        })
        result = merge_round_liar_labels(meta, liar_df)
        assert RLIAR_COL in result.columns
        assert result[RLIAR_COL].iloc[0] == STATE_LIAR

    def test_left_join_preserves_unmatched(self):
        """Unmatched rows should have NaN for RLIAR_COL."""
        meta = pd.DataFrame({
            'session_code': ['s1', 's1'], 'segment': ['sg1', 'sg1'],
            'round': [2, 3], 'group': [1, 1], 'label': ['A', 'A'],
        })
        liar_df = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['sg1'],
            'round': [2], 'group': [1], 'label': ['A'],
            RLIAR_COL: [STATE_LIAR],
        })
        result = merge_round_liar_labels(meta, liar_df)
        assert len(result) == 2
        assert pd.isna(result[RLIAR_COL].iloc[1])

    def test_preserves_row_count(self):
        """Merge should not add or lose rows."""
        meta = _make_metadata(6, with_liar=False)
        liar_df = pd.DataFrame({
            'session_code': ['s1'] * 6, 'segment': ['supergame1'] * 6,
            'round': [2] * 6, 'group': [1, 1, 1, 2, 2, 2],
            'label': [chr(65 + i) for i in range(6)],
            RLIAR_COL: [STATE_LIAR] * 3 + [STATE_NON_LIAR] * 3,
        })
        result = merge_round_liar_labels(meta, liar_df)
        assert len(result) == 6


# =====
# TestCentroidComputation
# =====
class TestCentroidComputation:
    """Tests for compute_round_liar_centroids."""

    def test_centroids_are_mean_of_correct_subsets(self):
        """Each centroid should be the mean of its group."""
        emb = np.array([
            [2.0, 0.0], [4.0, 0.0],
            [0.0, 2.0], [0.0, 4.0],
        ])
        labels = np.array([
            STATE_LIAR, STATE_LIAR, STATE_NON_LIAR, STATE_NON_LIAR,
        ])
        l_c, nl_c = compute_round_liar_centroids(emb, labels)
        np.testing.assert_array_equal(l_c, [3.0, 0.0])
        np.testing.assert_array_equal(nl_c, [0.0, 3.0])

    def test_single_point_per_class(self):
        """Single point should be its own centroid."""
        emb = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        labels = np.array([STATE_LIAR, STATE_NON_LIAR])
        l_c, nl_c = compute_round_liar_centroids(emb, labels)
        np.testing.assert_array_equal(l_c, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(nl_c, [-1.0, -2.0, -3.0])

    def test_all_same_label_raises_error(self):
        """All liar labels should raise ValueError (empty non-liar mask)."""
        emb = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        labels = np.array([STATE_LIAR, STATE_LIAR, STATE_LIAR])
        with pytest.raises(ValueError, match="No"):
            compute_round_liar_centroids(emb, labels)


# =====
# TestDirectionVector
# =====
class TestDirectionVector:
    """Tests for direction vector computation using reused functions."""

    def test_direction_is_normalized(self):
        """Direction vector should be unit length."""
        emb, labels = _make_embeddings_and_labels()
        l_c, nl_c = compute_round_liar_centroids(emb, labels)
        direction = compute_difference_vector(l_c, nl_c)
        np.testing.assert_almost_equal(np.linalg.norm(direction), 1.0)

    def test_direction_points_toward_liar(self):
        """Direction should point from non-liar toward liar."""
        emb, labels = _make_embeddings_and_labels()
        l_c, nl_c = compute_round_liar_centroids(emb, labels)
        direction = compute_difference_vector(l_c, nl_c)
        assert direction[0] > 0


# =====
# TestProjectionOutput
# =====
class TestProjectionOutput:
    """Tests for projection column names and values."""

    def test_projection_values_are_scalar_dot_products(self):
        """Projection values should match manual dot product computation."""
        rng = np.random.RandomState(42)
        n_rows, n_dims = 4, 5
        embeddings = rng.randn(n_rows, n_dims)
        direction = rng.randn(n_dims)
        direction = direction / np.linalg.norm(direction)
        projections = project_onto_direction(embeddings, direction)
        for i in range(n_rows):
            expected = np.dot(embeddings[i], direction)
            assert projections[i] == pytest.approx(expected, abs=1e-10)

    def test_liar_projects_higher(self):
        """Liar embeddings should project higher than non-liar."""
        rng = np.random.RandomState(42)
        n, dim = 50, 32
        emb = np.vstack([
            rng.normal(loc=2.0, scale=0.5, size=(n, dim)),
            rng.normal(loc=-2.0, scale=0.5, size=(n, dim)),
        ])
        labels = np.array([STATE_LIAR] * n + [STATE_NON_LIAR] * n)
        l_c, nl_c = compute_round_liar_centroids(emb, labels)
        direction = compute_difference_vector(l_c, nl_c)
        projections = project_onto_direction(emb, direction)
        assert projections[:n].mean() > projections[n:].mean()

    def test_build_output_has_rliar_column(self):
        """Output should include proj_rliar_pr_dir_{suffix} and RLIAR_COL."""
        meta = _make_metadata(4)
        projections = np.array([1.0, 2.0, -1.0, -2.0])
        result = _build_output(meta, projections, 'small', ID_COLS)
        assert 'proj_rliar_pr_dir_small' in result.columns
        assert RLIAR_COL in result.columns
        assert 'player_state' in result.columns

    def test_cross_level_output_columns(self):
        """Cross-level output should have all 4 proj_rliar columns."""
        n = 4
        meta = _make_metadata(n)
        results = {
            'small': (meta, np.ones(n), np.ones(n) * 2),
            'large': (meta, np.ones(n) * 3, np.ones(n) * 4),
        }
        out = _build_cross_level_output(results)
        for col in ['proj_rliar_msg_dir_small', 'proj_rliar_pr_dir_small',
                     'proj_rliar_msg_dir_large', 'proj_rliar_pr_dir_large']:
            assert col in out.columns


# =====
# TestConstants
# =====
class TestConstants:
    """Verify module constants match expected values."""

    def test_state_labels(self):
        """Liar state labels should match expected strings."""
        assert STATE_LIAR == 'liar'
        assert STATE_NON_LIAR == 'non_liar'

    def test_rliar_col(self):
        """Round-liar column name should match expected value."""
        assert RLIAR_COL == 'round_liar_label'

    def test_probe_phrases_are_nonempty(self):
        """Should have at least one probe phrase."""
        assert len(PROBE_PHRASES) > 0

    def test_join_keys(self):
        """Join keys for merging liar labels."""
        expected = ['session_code', 'segment', 'round', 'group', 'label']
        assert JOIN_KEYS == expected
