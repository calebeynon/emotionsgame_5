"""
Tests for analyze_homogeneity_embeddings.py.

Tests homogeneity label computation, label merging, centroid
computation, and projection output using synthetic data with
no file I/O or API calls.

Author: Claude Code
Date: 2026-03-20
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))


from analyze_homogeneity_embeddings import (
    _assign_labels,
    _compute_contribution_range,
    compute_homogeneity_centroids,
    merge_homogeneity_labels,
    GROUP_KEYS,
    HOMOG_COL,
)
from analyze_embeddings import compute_difference_vector, project_onto_direction


# =====
# Synthetic data builders
# =====
def _make_contribution_df(contributions_by_group):
    """Build player-level DataFrame with columns matching GROUP_KEYS.

    Args:
        contributions_by_group: list of (session, segment, round, group,
            [(label, contribution), ...]) tuples.
    """
    rows = []
    for session, segment, rnd, gid, players in contributions_by_group:
        for label, contrib in players:
            rows.append({
                'session_code': session,
                'segment': segment,
                'round': rnd,
                'group': gid,
                'label': label,
                'contribution': contrib,
            })
    return pd.DataFrame(rows)


def _compute_labels_from_df(df):
    """Compose internal functions to compute homogeneity labels."""
    group_range = _compute_contribution_range(df)
    group_range[HOMOG_COL] = _assign_labels(group_range['contribution_range'])
    return group_range[GROUP_KEYS + [HOMOG_COL]]


def _make_embeddings_df(n_rows, n_dims=10, seed=42):
    """Build synthetic embedding DataFrame with metadata columns."""
    rng = np.random.RandomState(seed)
    rows = []
    labels = ['A', 'E', 'J', 'N']
    for i in range(n_rows):
        row = {
            'session_code': 's1',
            'treatment': 1,
            'segment': 'supergame1',
            'round': (i // 4) + 2,
            'group': (i % 4) + 1,
            'label': labels[i % 4],
        }
        for d in range(n_dims):
            row[f'emb_{d}'] = rng.randn()
        rows.append(row)
    return pd.DataFrame(rows)


# =====
# TestComputeHomogeneityLabels
# =====
class TestComputeHomogeneityLabels:
    """Tests for homogeneity label computation from contribution data."""

    def test_all_same_is_homogeneous(self):
        """All players contribute same amount -> homogeneous."""
        df = _make_contribution_df([
            ('s1', 'supergame1', 2, 1,
             [('A', 25), ('E', 25), ('J', 25), ('N', 25)]),
        ])
        result = _compute_labels_from_df(df)
        assert result[HOMOG_COL].iloc[0] == 'homogeneous'

    def test_within_1_ecu_is_homogeneous(self):
        """Contributions within 1 ECU spread -> homogeneous."""
        df = _make_contribution_df([
            ('s1', 'supergame1', 2, 1,
             [('A', 24), ('E', 25), ('J', 25), ('N', 24)]),
        ])
        result = _compute_labels_from_df(df)
        assert result[HOMOG_COL].iloc[0] == 'homogeneous'

    def test_spread_greater_than_1_is_heterogeneous(self):
        """Contributions with spread > 1 ECU -> heterogeneous."""
        df = _make_contribution_df([
            ('s1', 'supergame1', 2, 1,
             [('A', 0), ('E', 25), ('J', 10), ('N', 15)]),
        ])
        result = _compute_labels_from_df(df)
        assert result[HOMOG_COL].iloc[0] == 'heterogeneous'

    def test_edge_case_range_exactly_1(self):
        """Range exactly 1 -> homogeneous (boundary inclusive)."""
        df = _make_contribution_df([
            ('s1', 'supergame1', 2, 1,
             [('A', 24), ('E', 25), ('J', 25), ('N', 25)]),
        ])
        result = _compute_labels_from_df(df)
        assert result[HOMOG_COL].iloc[0] == 'homogeneous'

    def test_edge_case_range_exactly_2(self):
        """Range exactly 2 -> heterogeneous."""
        df = _make_contribution_df([
            ('s1', 'supergame1', 2, 1,
             [('A', 23), ('E', 25), ('J', 24), ('N', 24)]),
        ])
        result = _compute_labels_from_df(df)
        assert result[HOMOG_COL].iloc[0] == 'heterogeneous'

    def test_one_row_per_group_round(self):
        """Output should have one row per group-round combination."""
        df = _make_contribution_df([
            ('s1', 'supergame1', 2, 1,
             [('A', 25), ('E', 25), ('J', 25), ('N', 25)]),
            ('s1', 'supergame1', 2, 2,
             [('A', 0), ('E', 25), ('J', 10), ('N', 5)]),
            ('s1', 'supergame1', 3, 1,
             [('A', 20), ('E', 20), ('J', 20), ('N', 20)]),
        ])
        result = _compute_labels_from_df(df)
        assert len(result) == 3


# =====
# TestMergeHomogeneityLabels
# =====
class TestMergeHomogeneityLabels:
    """Tests for merge_homogeneity_labels."""

    def test_left_join_preserves_all_embedding_rows(self):
        """LEFT JOIN should keep all rows from embeddings DataFrame."""
        emb_df = _make_embeddings_df(n_rows=8)
        homog_df = pd.DataFrame([
            {'session_code': 's1', 'segment': 'supergame1',
             'round': 2, 'group': 1, 'homogeneity_label': 'homogeneous'},
            {'session_code': 's1', 'segment': 'supergame1',
             'round': 2, 'group': 2, 'homogeneity_label': 'heterogeneous'},
        ])
        result = merge_homogeneity_labels(emb_df, homog_df)
        assert len(result) == len(emb_df)

    def test_label_column_populated(self):
        """Homogeneity label column should be present after merge."""
        emb_df = _make_embeddings_df(n_rows=4)
        homog_df = pd.DataFrame([
            {'session_code': 's1', 'segment': 'supergame1',
             'round': 2, 'group': 1, 'homogeneity_label': 'homogeneous'},
            {'session_code': 's1', 'segment': 'supergame1',
             'round': 2, 'group': 2, 'homogeneity_label': 'heterogeneous'},
            {'session_code': 's1', 'segment': 'supergame1',
             'round': 2, 'group': 3, 'homogeneity_label': 'homogeneous'},
            {'session_code': 's1', 'segment': 'supergame1',
             'round': 2, 'group': 4, 'homogeneity_label': 'heterogeneous'},
        ])
        result = merge_homogeneity_labels(emb_df, homog_df)
        assert 'homogeneity_label' in result.columns
        assert result['homogeneity_label'].notna().any()


# =====
# TestCentroidComputation
# =====
class TestCentroidComputation:
    """Tests for compute_homogeneity_centroids."""

    def test_centroids_are_mean_of_correct_subsets(self):
        """Centroids should be the mean of homogeneous/heterogeneous embeddings."""
        rng = np.random.RandomState(42)
        homog_embs = rng.randn(5, 10) + 2.0
        heterog_embs = rng.randn(5, 10) - 2.0
        embeddings = np.vstack([homog_embs, heterog_embs])
        labels = np.array(['homogeneous'] * 5 + ['heterogeneous'] * 5)

        homog_c, heterog_c = compute_homogeneity_centroids(
            embeddings, labels,
        )
        np.testing.assert_array_almost_equal(
            homog_c, homog_embs.mean(axis=0),
        )
        np.testing.assert_array_almost_equal(
            heterog_c, heterog_embs.mean(axis=0),
        )

    def test_direction_vector_is_normalized(self):
        """Direction from centroids should be unit length."""
        rng = np.random.RandomState(42)
        n_dims = 10
        homog_c = rng.randn(n_dims) + 1.0
        heterog_c = rng.randn(n_dims) - 1.0
        direction = compute_difference_vector(homog_c, heterog_c)
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)


# =====
# TestProjectionOutput
# =====
class TestProjectionOutput:
    """Tests for projection column names and values."""

    def test_expected_column_names(self):
        """Projection output should contain expected column names."""
        rng = np.random.RandomState(42)
        n_rows, n_dims = 8, 10
        embeddings = rng.randn(n_rows, n_dims)
        direction = rng.randn(n_dims)
        direction = direction / np.linalg.norm(direction)

        projections = project_onto_direction(embeddings, direction)

        meta = _make_embeddings_df(n_rows=n_rows)
        id_cols = ['session_code', 'segment', 'round', 'group', 'label']
        out = meta[id_cols].copy()
        out['proj_homog_msg_dir_small'] = projections

        assert 'proj_homog_msg_dir_small' in out.columns

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
