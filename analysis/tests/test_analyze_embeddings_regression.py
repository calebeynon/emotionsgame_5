"""
Regression and integration tests for analyze_embeddings.py.

Additional tests beyond implementation agent's unit tests.
Focuses on mathematical correctness, edge cases, integration
between analysis functions, and output structure validation.

Author: Claude Code (test-writer)
Date: 2026-03-15
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from analyze_embeddings import (
    ID_COLS,
    PROBE_PHRASES,
    STATE_COOPERATIVE,
    STATE_NONCOOPERATIVE,
    _cosine_similarities,
    _merge_projections,
    build_projection_csv,
    compute_centroids,
    compute_difference_vector,
    compute_group_round_embeddings,
    project_onto_direction,
    rank_messages,
)


# =====
# Helpers
# =====
def _make_metadata(n, n_groups=1, states=None):
    """Create metadata DataFrame for n messages."""
    if states is None:
        states = [STATE_COOPERATIVE] * (n // 2) + [STATE_NONCOOPERATIVE] * (n - n // 2)
    return pd.DataFrame({
        'session_code': ['s1'] * n,
        'treatment': [1] * n,
        'segment': ['supergame1'] * n,
        'round': [2] * n,
        'group': [(i % n_groups) + 1 for i in range(n)],
        'label': [chr(65 + i) for i in range(n)],
        'message_index': list(range(n)),
        'message_text': [f'msg {i}' for i in range(n)],
        'player_state': states,
    })


# =====
# Regression: compute_centroids mathematical properties
# =====
class TestComputeCentroidsRegression:
    """Mathematical correctness tests for centroid computation."""

    def test_single_cooperative_point(self):
        """Single cooperative point should be its own centroid."""
        emb = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        labels = np.array([STATE_COOPERATIVE, STATE_NONCOOPERATIVE])
        coop_c, noncoop_c = compute_centroids(emb, labels)

        np.testing.assert_array_equal(coop_c, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(noncoop_c, [-1.0, -2.0, -3.0])

    def test_centroid_known_values(self):
        """Verify against hand-computed centroids."""
        emb = np.array([
            [2.0, 0.0],  # cooperative
            [4.0, 0.0],  # cooperative
            [0.0, 2.0],  # noncooperative
            [0.0, 4.0],  # noncooperative
        ])
        labels = np.array([
            STATE_COOPERATIVE, STATE_COOPERATIVE,
            STATE_NONCOOPERATIVE, STATE_NONCOOPERATIVE,
        ])
        coop_c, noncoop_c = compute_centroids(emb, labels)

        np.testing.assert_array_equal(coop_c, [3.0, 0.0])
        np.testing.assert_array_equal(noncoop_c, [0.0, 3.0])

    def test_many_dimensions(self):
        """Should work with high-dimensional embeddings."""
        dim = 1536
        emb = np.vstack([
            np.ones((5, dim)),
            -np.ones((5, dim)),
        ])
        labels = np.array(
            [STATE_COOPERATIVE] * 5 + [STATE_NONCOOPERATIVE] * 5
        )
        coop_c, noncoop_c = compute_centroids(emb, labels)

        assert coop_c.shape == (dim,)
        np.testing.assert_array_almost_equal(coop_c, np.ones(dim))
        np.testing.assert_array_almost_equal(noncoop_c, -np.ones(dim))


# =====
# Regression: compute_difference_vector
# =====
class TestComputeDifferenceVectorRegression:
    """Mathematical correctness tests for direction vector."""

    def test_known_direction_2d(self):
        """2D example: (1,0) - (-1,0) = (2,0), normalized = (1,0)."""
        coop = np.array([1.0, 0.0])
        noncoop = np.array([-1.0, 0.0])
        direction = compute_difference_vector(coop, noncoop)
        np.testing.assert_array_almost_equal(direction, [1.0, 0.0])

    def test_known_direction_diagonal(self):
        """Diagonal: (1,1) - (-1,-1) = (2,2), normalized = (1/sqrt2, 1/sqrt2)."""
        coop = np.array([1.0, 1.0])
        noncoop = np.array([-1.0, -1.0])
        direction = compute_difference_vector(coop, noncoop)
        expected = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(direction, expected)

    def test_norm_is_always_one(self):
        """Output norm should always be 1 (unless zero vector)."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            coop = rng.normal(size=50)
            noncoop = rng.normal(size=50)
            direction = compute_difference_vector(coop, noncoop)
            np.testing.assert_almost_equal(
                np.linalg.norm(direction), 1.0, decimal=10
            )

    def test_asymmetry(self):
        """Swapping arguments should reverse direction."""
        coop = np.array([3.0, 1.0])
        noncoop = np.array([1.0, 3.0])
        d1 = compute_difference_vector(coop, noncoop)
        d2 = compute_difference_vector(noncoop, coop)
        np.testing.assert_array_almost_equal(d1, -d2)


# =====
# Regression: project_onto_direction
# =====
class TestProjectOntoDirectionRegression:
    """Mathematical correctness tests for dot-product projection."""

    def test_orthogonal_projects_to_zero(self):
        """Orthogonal vector should project to zero."""
        emb = np.array([[0.0, 1.0]])
        direction = np.array([1.0, 0.0])
        result = project_onto_direction(emb, direction)
        assert result[0] == pytest.approx(0.0)

    def test_parallel_projects_to_magnitude(self):
        """Parallel vector should project to its magnitude."""
        emb = np.array([[3.0, 0.0]])
        direction = np.array([1.0, 0.0])
        result = project_onto_direction(emb, direction)
        assert result[0] == pytest.approx(3.0)

    def test_antiparallel_projects_negative(self):
        """Anti-parallel vector should project to negative magnitude."""
        emb = np.array([[-5.0, 0.0]])
        direction = np.array([1.0, 0.0])
        result = project_onto_direction(emb, direction)
        assert result[0] == pytest.approx(-5.0)

    def test_full_pipeline_cooperative_higher(self):
        """End-to-end: cooperative messages should project higher."""
        rng = np.random.default_rng(42)
        n_coop, n_noncoop, dim = 50, 50, 100
        coop_emb = rng.normal(loc=1.0, scale=0.3, size=(n_coop, dim))
        noncoop_emb = rng.normal(loc=-1.0, scale=0.3, size=(n_noncoop, dim))
        emb = np.vstack([coop_emb, noncoop_emb])
        labels = np.array(
            [STATE_COOPERATIVE] * n_coop
            + [STATE_NONCOOPERATIVE] * n_noncoop
        )

        coop_c, noncoop_c = compute_centroids(emb, labels)
        direction = compute_difference_vector(coop_c, noncoop_c)
        projections = project_onto_direction(emb, direction)

        coop_mean = projections[:n_coop].mean()
        noncoop_mean = projections[n_coop:].mean()
        assert coop_mean > noncoop_mean
        # With 50 samples each, separation should be clear
        assert coop_mean - noncoop_mean > 1.0


# =====
# Regression: _cosine_similarities
# =====
class TestCosineSimilarities:
    """Tests for the cosine similarity helper."""

    def test_identical_vectors_similarity_one(self):
        """Identical normalized vectors should have similarity 1."""
        direction = np.array([1.0, 0.0, 0.0])
        emb = np.array([[1.0, 0.0, 0.0]])
        result = _cosine_similarities(emb, direction)
        assert result[0] == pytest.approx(1.0)

    def test_opposite_vectors_similarity_negative_one(self):
        """Opposite vectors should have similarity -1."""
        direction = np.array([1.0, 0.0])
        emb = np.array([[-1.0, 0.0]])
        result = _cosine_similarities(emb, direction)
        assert result[0] == pytest.approx(-1.0)

    def test_orthogonal_similarity_zero(self):
        """Orthogonal vectors should have similarity 0."""
        direction = np.array([1.0, 0.0])
        emb = np.array([[0.0, 1.0]])
        result = _cosine_similarities(emb, direction)
        assert result[0] == pytest.approx(0.0)

    def test_handles_zero_embedding(self):
        """Zero-norm embedding should not cause division by zero."""
        direction = np.array([1.0, 0.0])
        emb = np.array([[0.0, 0.0]])
        result = _cosine_similarities(emb, direction)
        # Zero vector has cosine sim = 0 (due to zero-norm guard)
        assert result[0] == pytest.approx(0.0)

    def test_batch_computation(self):
        """Should compute similarities for multiple embeddings at once."""
        direction = np.array([1.0, 0.0])
        emb = np.array([
            [1.0, 0.0],   # sim = 1
            [0.0, 1.0],   # sim = 0
            [-1.0, 0.0],  # sim = -1
        ])
        result = _cosine_similarities(emb, direction)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, -1.0])


# =====
# Edge cases: compute_group_round_embeddings
# =====
class TestGroupRoundEmbeddingsEdgeCases:
    """Edge case tests for group-round aggregation."""

    def test_single_message_per_group(self):
        """Each group with one message should return that embedding."""
        meta = pd.DataFrame({
            'session_code': ['s1', 's1'],
            'segment': ['sg1', 'sg1'],
            'round': [1, 1],
            'group': [1, 2],
            'label': ['A', 'B'],
            'message_index': [0, 0],
            'message_text': ['a', 'b'],
            'player_state': [STATE_COOPERATIVE, STATE_NONCOOPERATIVE],
        })
        emb = np.array([[1.0, 2.0], [3.0, 4.0]])
        group_meta, group_emb = compute_group_round_embeddings(meta, emb)

        assert len(group_meta) == 2
        assert group_emb.shape == (2, 2)

    def test_multiple_rounds_same_group(self):
        """Same group in different rounds should produce separate entries."""
        meta = pd.DataFrame({
            'session_code': ['s1', 's1'],
            'segment': ['sg1', 'sg1'],
            'round': [1, 2],
            'group': [1, 1],
            'label': ['A', 'A'],
            'message_index': [0, 0],
            'message_text': ['a', 'b'],
            'player_state': [STATE_COOPERATIVE, STATE_COOPERATIVE],
        })
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        group_meta, group_emb = compute_group_round_embeddings(meta, emb)

        assert len(group_meta) == 2


# =====
# Edge cases: rank_messages
# =====
class TestRankMessagesEdgeCases:
    """Edge case tests for message ranking."""

    def test_n_larger_than_data(self):
        """Requesting more messages than available should not error."""
        meta = _make_metadata(3)
        proj = np.array([1.0, 0.0, -1.0])
        top_coop, top_noncoop = rank_messages(meta, proj, n=10)
        assert len(top_coop) == 3
        assert len(top_noncoop) == 3

    def test_all_equal_projections(self):
        """All equal projections should still return results."""
        meta = _make_metadata(4)
        proj = np.array([0.0, 0.0, 0.0, 0.0])
        top_coop, top_noncoop = rank_messages(meta, proj, n=2)
        assert len(top_coop) == 2
        assert len(top_noncoop) == 2

    def test_ranking_order_is_descending(self):
        """Top cooperative should be sorted descending."""
        meta = _make_metadata(5)
        proj = np.array([3.0, 1.0, 5.0, -2.0, -4.0])
        top_coop, _ = rank_messages(meta, proj, n=3)
        values = top_coop['projection'].tolist()
        assert values == sorted(values, reverse=True)


# =====
# Regression: build_projection_csv
# =====
class TestBuildProjectionCsvRegression:
    """Regression tests for output DataFrame construction."""

    def test_small_suffix_column_name(self):
        """Small model should produce proj_pr_dir_small column."""
        meta = _make_metadata(2)
        proj = np.array([1.0, -1.0])
        result = build_projection_csv(meta, proj, 'small')
        assert 'proj_pr_dir_small' in result.columns

    def test_large_suffix_column_name(self):
        """Large model should produce proj_pr_dir_large column."""
        meta = _make_metadata(2)
        proj = np.array([1.0, -1.0])
        result = build_projection_csv(meta, proj, 'large')
        assert 'proj_pr_dir_large' in result.columns

    def test_includes_player_state(self):
        """Output should include player_state column."""
        meta = _make_metadata(2)
        proj = np.array([1.0, -1.0])
        result = build_projection_csv(meta, proj, 'small')
        assert 'player_state' in result.columns

    def test_does_not_include_extra_columns(self):
        """Output should have exactly ID_COLS + player_state + projection."""
        meta = _make_metadata(2)
        proj = np.array([1.0, -1.0])
        result = build_projection_csv(meta, proj, 'small')
        expected = set(ID_COLS + ['player_state', 'proj_pr_dir_small'])
        assert set(result.columns) == expected


# =====
# Regression: _merge_projections
# =====
class TestMergeProjections:
    """Tests for merging small and large projection DataFrames."""

    def test_produces_both_score_columns(self):
        """Merged output should have both projection score columns."""
        meta = _make_metadata(3)
        proj_s = np.array([1.0, 0.0, -1.0])
        proj_l = np.array([2.0, 0.5, -2.0])
        df_small = build_projection_csv(meta, proj_s, 'small')
        df_large = build_projection_csv(meta, proj_l, 'large')

        merged = _merge_projections(df_small, df_large)

        assert 'proj_pr_dir_small' in merged.columns
        assert 'proj_pr_dir_large' in merged.columns

    def test_preserves_row_count(self):
        """Merge should not add or remove rows."""
        meta = _make_metadata(5)
        proj = np.zeros(5)
        df_small = build_projection_csv(meta, proj, 'small')
        df_large = build_projection_csv(meta, proj + 1, 'large')

        merged = _merge_projections(df_small, df_large)
        assert len(merged) == 5

    def test_values_match_inputs(self):
        """Merged values should match original projections."""
        meta = _make_metadata(3)
        proj_s = np.array([1.5, 2.5, 3.5])
        proj_l = np.array([-1.5, -2.5, -3.5])
        df_small = build_projection_csv(meta, proj_s, 'small')
        df_large = build_projection_csv(meta, proj_l, 'large')

        merged = _merge_projections(df_small, df_large)

        np.testing.assert_array_almost_equal(
            merged['proj_pr_dir_small'].values, proj_s
        )
        np.testing.assert_array_almost_equal(
            merged['proj_pr_dir_large'].values, proj_l
        )


# =====
# Integration: full analysis pipeline
# =====
class TestFullAnalysisPipeline:
    """Integration tests for the complete centroid->projection pipeline."""

    def test_pipeline_separates_cooperative_from_noncooperative(self):
        """Full pipeline should assign higher scores to cooperative msgs."""
        rng = np.random.default_rng(99)
        n = 20
        dim = 32
        coop_emb = rng.normal(loc=2.0, scale=0.5, size=(n, dim))
        noncoop_emb = rng.normal(loc=-2.0, scale=0.5, size=(n, dim))
        emb = np.vstack([coop_emb, noncoop_emb])

        labels = np.array(
            [STATE_COOPERATIVE] * n + [STATE_NONCOOPERATIVE] * n
        )
        coop_c, noncoop_c = compute_centroids(emb, labels)
        direction = compute_difference_vector(coop_c, noncoop_c)
        projections = project_onto_direction(emb, direction)

        # Every cooperative should score higher than every noncooperative
        coop_min = projections[:n].min()
        noncoop_max = projections[n:].max()
        assert coop_min > noncoop_max

    def test_pipeline_output_structure(self):
        """Pipeline output CSV should have correct column set."""
        n = 10
        meta = _make_metadata(n)
        emb = np.random.randn(n, 8)
        labels = meta['player_state'].values

        coop_c, noncoop_c = compute_centroids(emb, labels)
        direction = compute_difference_vector(coop_c, noncoop_c)
        proj = project_onto_direction(emb, direction)

        result_small = build_projection_csv(meta, proj, 'small')
        result_large = build_projection_csv(meta, proj * 0.5, 'large')
        merged = _merge_projections(result_small, result_large)

        expected_cols = set(
            ID_COLS + ['player_state',
                       'proj_pr_dir_small',
                       'proj_pr_dir_large']
        )
        assert set(merged.columns) == expected_cols
        assert len(merged) == n


# =====
# Constants validation
# =====
class TestConstantsRegression:
    """Verify module constants match expected values."""

    def test_probe_phrases_count(self):
        """Should have 6 probe phrases."""
        assert len(PROBE_PHRASES) == 6

    def test_probe_phrases_content(self):
        """Verify key probe phrases are present."""
        assert "let's all contribute 25" in PROBE_PHRASES
        assert "I'm going to free ride" in PROBE_PHRASES
        assert "we should cooperate" in PROBE_PHRASES

    def test_state_labels(self):
        """State labels should match expected strings."""
        assert STATE_COOPERATIVE == 'cooperative'
        assert STATE_NONCOOPERATIVE == 'noncooperative'

    def test_id_cols_matches_compute_embeddings(self):
        """ID_COLS should match those in compute_embeddings module."""
        expected = [
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'message_index', 'message_text',
        ]
        assert ID_COLS == expected
