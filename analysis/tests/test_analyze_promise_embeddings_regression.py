"""
Regression and integration tests for analyze_promise_embeddings.py.

Covers mathematical correctness of promise centroid computation,
direction vectors, projections, cross-level structure, output columns,
and label merging. Uses synthetic numpy arrays -- no API calls or file I/O.

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

from analyze_promise_embeddings import (
    ID_COLS,
    JOIN_KEYS,
    PR_ID_COLS,
    PROBE_PHRASES,
    PROMISE_COL,
    STATE_NO_PROMISE,
    STATE_PROMISE,
    _build_cross_level_output,
    _build_output,
    _merge_projections,
    compute_promise_centroids,
    load_promise_labels,
    merge_promise_labels,
    probe_phrase_validation,
)
from analyze_embeddings import (
    compute_difference_vector,
    project_onto_direction,
    _cosine_similarities,
)


# =====
# Helpers
# =====
def _make_embeddings_and_labels(n_promise=3, n_no_promise=3, dim=4):
    """Create synthetic embeddings with known separation by promise label."""
    rng = np.random.default_rng(42)
    promise = rng.normal(loc=1.0, scale=0.1, size=(n_promise, dim))
    no_promise = rng.normal(loc=-1.0, scale=0.1, size=(n_no_promise, dim))
    embeddings = np.vstack([promise, no_promise])
    labels = np.array(
        [STATE_PROMISE] * n_promise + [STATE_NO_PROMISE] * n_no_promise
    )
    return embeddings, labels


def _make_metadata(n, with_promise=True):
    """Create minimal metadata DataFrame for n messages."""
    meta = pd.DataFrame({
        'session_code': ['sess1'] * n,
        'treatment': [1] * n,
        'segment': ['supergame1'] * n,
        'round': [2] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
        'message_index': range(n),
        'message_text': [f'message {i}' for i in range(n)],
        'player_state': ['cooperative'] * (n // 2) + ['noncooperative'] * (n - n // 2),
    })
    if with_promise:
        meta[PROMISE_COL] = (
            [STATE_PROMISE] * (n // 2)
            + [STATE_NO_PROMISE] * (n - n // 2)
        )
    return meta


def _make_pr_metadata(n, with_promise=True):
    """Create player-round level metadata DataFrame."""
    meta = pd.DataFrame({
        'session_code': ['sess1'] * n,
        'treatment': [1] * n,
        'segment': ['supergame1'] * n,
        'round': [2] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
        'combined_text': [f'combined text {i}' for i in range(n)],
        'player_state': ['cooperative'] * (n // 2) + ['noncooperative'] * (n - n // 2),
    })
    if with_promise:
        meta[PROMISE_COL] = (
            [STATE_PROMISE] * (n // 2)
            + [STATE_NO_PROMISE] * (n - n // 2)
        )
    return meta


# =====
# Regression: constants
# =====
class TestConstants:
    """Verify module constants match expected values."""

    def test_state_labels(self):
        """Promise state labels should match expected strings."""
        assert STATE_PROMISE == 'promise'
        assert STATE_NO_PROMISE == 'no_promise'

    def test_probe_phrases_count(self):
        """Should have 12 promise probe phrases."""
        assert len(PROBE_PHRASES) == 12

    def test_probe_phrases_content(self):
        """Probe phrases should include key promise/no-promise exemplars."""
        assert "I promise to put in 25" in PROBE_PHRASES
        assert "I'm not making any promises" in PROBE_PHRASES
        assert "no guarantees from me" in PROBE_PHRASES

    def test_id_cols_matches_message_level(self):
        """ID_COLS should match the message-level columns."""
        expected = [
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'message_index', 'message_text',
        ]
        assert ID_COLS == expected

    def test_pr_id_cols_matches_player_round(self):
        """PR_ID_COLS should match the player-round columns."""
        expected = [
            'session_code', 'treatment', 'segment', 'round',
            'group', 'label', 'combined_text',
        ]
        assert PR_ID_COLS == expected

    def test_join_keys(self):
        """Join keys for merging promise labels."""
        expected = ['session_code', 'segment', 'round', 'group', 'label']
        assert JOIN_KEYS == expected


# =====
# Regression: compute_promise_centroids
# =====
class TestComputePromiseCentroids:
    """Mathematical correctness for promise centroid computation."""

    def test_returns_two_vectors(self):
        """Should return promise and no-promise centroids."""
        emb, labels = _make_embeddings_and_labels()
        p_c, np_c = compute_promise_centroids(emb, labels)
        assert p_c.shape == (4,)
        assert np_c.shape == (4,)

    def test_centroids_are_separated(self):
        """Promise centroid should differ from no-promise."""
        emb, labels = _make_embeddings_and_labels()
        p_c, np_c = compute_promise_centroids(emb, labels)
        distance = np.linalg.norm(p_c - np_c)
        assert distance > 1.0

    def test_centroid_is_mean_of_group(self):
        """Each centroid should be the mean of its group."""
        emb = np.array([
            [2.0, 0.0],   # promise
            [4.0, 0.0],   # promise
            [0.0, 2.0],   # no-promise
            [0.0, 4.0],   # no-promise
        ])
        labels = np.array([
            STATE_PROMISE, STATE_PROMISE,
            STATE_NO_PROMISE, STATE_NO_PROMISE,
        ])
        p_c, np_c = compute_promise_centroids(emb, labels)

        np.testing.assert_array_equal(p_c, [3.0, 0.0])
        np.testing.assert_array_equal(np_c, [0.0, 3.0])

    def test_single_point_per_class(self):
        """Single point should be its own centroid."""
        emb = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        labels = np.array([STATE_PROMISE, STATE_NO_PROMISE])
        p_c, np_c = compute_promise_centroids(emb, labels)

        np.testing.assert_array_equal(p_c, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(np_c, [-1.0, -2.0, -3.0])

    def test_high_dimensional(self):
        """Should work with high-dimensional embeddings."""
        dim = 1536
        emb = np.vstack([np.ones((5, dim)), -np.ones((5, dim))])
        labels = np.array(
            [STATE_PROMISE] * 5 + [STATE_NO_PROMISE] * 5
        )
        p_c, np_c = compute_promise_centroids(emb, labels)

        assert p_c.shape == (dim,)
        np.testing.assert_array_almost_equal(p_c, np.ones(dim))
        np.testing.assert_array_almost_equal(np_c, -np.ones(dim))


# =====
# Regression: direction vector with promise labels
# =====
class TestPromiseDirectionVector:
    """Tests for promise direction vector computation using reused functions."""

    def test_direction_points_toward_promise(self):
        """Direction should point from no-promise toward promise."""
        emb, labels = _make_embeddings_and_labels()
        p_c, np_c = compute_promise_centroids(emb, labels)
        direction = compute_difference_vector(p_c, np_c)

        # Promise centroid is near +1, no-promise near -1, so direction[0] > 0
        assert direction[0] > 0

    def test_direction_unit_length(self):
        """Direction vector should have unit norm."""
        emb, labels = _make_embeddings_and_labels()
        p_c, np_c = compute_promise_centroids(emb, labels)
        direction = compute_difference_vector(p_c, np_c)
        np.testing.assert_almost_equal(np.linalg.norm(direction), 1.0)


# =====
# Regression: projection separates promise from no-promise
# =====
class TestPromiseProjection:
    """End-to-end: projections should separate promise/no-promise."""

    def test_promise_projects_higher(self):
        """Promise embeddings should project higher than no-promise."""
        rng = np.random.default_rng(42)
        n = 50
        dim = 32
        promise_emb = rng.normal(loc=2.0, scale=0.5, size=(n, dim))
        no_promise_emb = rng.normal(loc=-2.0, scale=0.5, size=(n, dim))
        emb = np.vstack([promise_emb, no_promise_emb])
        labels = np.array(
            [STATE_PROMISE] * n + [STATE_NO_PROMISE] * n
        )

        p_c, np_c = compute_promise_centroids(emb, labels)
        direction = compute_difference_vector(p_c, np_c)
        projections = project_onto_direction(emb, direction)

        promise_mean = projections[:n].mean()
        no_promise_mean = projections[n:].mean()
        assert promise_mean > no_promise_mean
        assert promise_mean - no_promise_mean > 1.0

    def test_clear_separation_all_promise_above(self):
        """With well-separated data, every promise should score above every no-promise."""
        rng = np.random.default_rng(99)
        n = 20
        dim = 32
        promise_emb = rng.normal(loc=3.0, scale=0.3, size=(n, dim))
        no_promise_emb = rng.normal(loc=-3.0, scale=0.3, size=(n, dim))
        emb = np.vstack([promise_emb, no_promise_emb])
        labels = np.array(
            [STATE_PROMISE] * n + [STATE_NO_PROMISE] * n
        )

        p_c, np_c = compute_promise_centroids(emb, labels)
        direction = compute_difference_vector(p_c, np_c)
        projections = project_onto_direction(emb, direction)

        assert projections[:n].min() > projections[n:].max()


# =====
# Regression: merge_promise_labels
# =====
class TestMergePromiseLabels:
    """Tests for merging promise labels onto metadata."""

    def test_adds_promise_column(self):
        """Merged metadata should have promise_label column."""
        meta = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['sg1'],
            'round': [2], 'group': [1], 'label': ['A'],
        })
        promise_df = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['sg1'],
            'round': [2], 'group': [1], 'label': ['A'],
            PROMISE_COL: [STATE_PROMISE],
        })
        result = merge_promise_labels(meta, promise_df)
        assert PROMISE_COL in result.columns
        assert result[PROMISE_COL].iloc[0] == STATE_PROMISE

    def test_left_join_preserves_unmatched(self):
        """Unmatched rows should have NaN for promise_label."""
        meta = pd.DataFrame({
            'session_code': ['s1', 's1'], 'segment': ['sg1', 'sg1'],
            'round': [2, 3], 'group': [1, 1], 'label': ['A', 'A'],
        })
        promise_df = pd.DataFrame({
            'session_code': ['s1'], 'segment': ['sg1'],
            'round': [2], 'group': [1], 'label': ['A'],
            PROMISE_COL: [STATE_PROMISE],
        })
        result = merge_promise_labels(meta, promise_df)
        assert len(result) == 2
        assert pd.isna(result[PROMISE_COL].iloc[1])

    def test_preserves_row_count(self):
        """Merge should not add or lose rows."""
        meta = _make_metadata(6, with_promise=False)
        promise_df = pd.DataFrame({
            'session_code': ['sess1'] * 6,
            'segment': ['supergame1'] * 6,
            'round': [2] * 6,
            'group': [1, 1, 1, 2, 2, 2],
            'label': [chr(65 + i) for i in range(6)],
            PROMISE_COL: [STATE_PROMISE] * 3 + [STATE_NO_PROMISE] * 3,
        })
        result = merge_promise_labels(meta, promise_df)
        assert len(result) == 6


# =====
# Regression: _build_output
# =====
class TestBuildOutput:
    """Tests for output DataFrame construction."""

    def test_has_promise_projection_column(self):
        """Output should include proj_promise_pr_dir_{suffix} column."""
        meta = _make_metadata(4)
        projections = np.array([1.0, 2.0, -1.0, -2.0])
        result = _build_output(meta, projections, 'small', ID_COLS)
        assert 'proj_promise_pr_dir_small' in result.columns

    def test_includes_promise_label(self):
        """Output should include promise_label column."""
        meta = _make_metadata(4)
        projections = np.zeros(4)
        result = _build_output(meta, projections, 'small', ID_COLS)
        assert PROMISE_COL in result.columns

    def test_includes_player_state(self):
        """Output should include player_state column."""
        meta = _make_metadata(4)
        projections = np.zeros(4)
        result = _build_output(meta, projections, 'small', ID_COLS)
        assert 'player_state' in result.columns

    def test_row_count_matches_input(self):
        """Output should have same number of rows as input."""
        meta = _make_metadata(6)
        projections = np.zeros(6)
        result = _build_output(meta, projections, 'large', ID_COLS)
        assert len(result) == 6

    def test_pr_level_output(self):
        """Player-round level output should use PR_ID_COLS."""
        meta = _make_pr_metadata(4)
        projections = np.zeros(4)
        result = _build_output(meta, projections, 'small', PR_ID_COLS)
        assert 'combined_text' in result.columns
        assert 'message_index' not in result.columns


# =====
# Regression: _merge_projections
# =====
class TestMergeProjections:
    """Tests for merging small and large projections."""

    def test_produces_both_score_columns(self):
        """Merged output should have both small and large projections."""
        meta = _make_pr_metadata(3)
        proj_s = np.array([1.0, 0.0, -1.0])
        proj_l = np.array([2.0, 0.5, -2.0])
        df_small = _build_output(meta, proj_s, 'small', PR_ID_COLS)
        df_large = _build_output(meta, proj_l, 'large', PR_ID_COLS)

        merged = _merge_projections(df_small, df_large, PR_ID_COLS)
        assert 'proj_promise_pr_dir_small' in merged.columns
        assert 'proj_promise_pr_dir_large' in merged.columns

    def test_preserves_row_count(self):
        """Merge should not add or remove rows."""
        meta = _make_pr_metadata(5)
        proj = np.zeros(5)
        df_small = _build_output(meta, proj, 'small', PR_ID_COLS)
        df_large = _build_output(meta, proj + 1, 'large', PR_ID_COLS)

        merged = _merge_projections(df_small, df_large, PR_ID_COLS)
        assert len(merged) == 5

    def test_values_match_inputs(self):
        """Merged values should match original projections."""
        meta = _make_pr_metadata(3)
        proj_s = np.array([1.5, 2.5, 3.5])
        proj_l = np.array([-1.5, -2.5, -3.5])
        df_small = _build_output(meta, proj_s, 'small', PR_ID_COLS)
        df_large = _build_output(meta, proj_l, 'large', PR_ID_COLS)

        merged = _merge_projections(df_small, df_large, PR_ID_COLS)

        np.testing.assert_array_almost_equal(
            merged['proj_promise_pr_dir_small'].values, proj_s
        )
        np.testing.assert_array_almost_equal(
            merged['proj_promise_pr_dir_large'].values, proj_l
        )


# =====
# Regression: _build_cross_level_output
# =====
class TestBuildCrossLevelOutput:
    """Tests for the cross-level output builder."""

    def test_has_four_projection_columns(self):
        """Output should have all 4 cross-level projection columns."""
        n = 4
        meta = _make_metadata(n)
        results = {
            'small': (meta, np.ones(n), np.ones(n) * 2),
            'large': (meta, np.ones(n) * 3, np.ones(n) * 4),
        }
        out = _build_cross_level_output(results)

        expected_cols = [
            'proj_promise_msg_dir_small', 'proj_promise_pr_dir_small',
            'proj_promise_msg_dir_large', 'proj_promise_pr_dir_large',
        ]
        for col in expected_cols:
            assert col in out.columns

    def test_includes_both_state_columns(self):
        """Output should have both player_state and promise_label."""
        n = 4
        meta = _make_metadata(n)
        results = {
            'small': (meta, np.zeros(n), np.zeros(n)),
            'large': (meta, np.zeros(n), np.zeros(n)),
        }
        out = _build_cross_level_output(results)

        assert 'player_state' in out.columns
        assert PROMISE_COL in out.columns

    def test_row_count(self):
        """Output should match the number of messages."""
        n = 6
        meta = _make_metadata(n)
        results = {
            'small': (meta, np.zeros(n), np.zeros(n)),
            'large': (meta, np.zeros(n), np.zeros(n)),
        }
        out = _build_cross_level_output(results)
        assert len(out) == n

    def test_projection_values_are_correct(self):
        """Projection values should match the provided arrays."""
        n = 3
        meta = _make_metadata(n)
        msg_s = np.array([1.0, 2.0, 3.0])
        pr_s = np.array([4.0, 5.0, 6.0])
        msg_l = np.array([7.0, 8.0, 9.0])
        pr_l = np.array([10.0, 11.0, 12.0])
        results = {
            'small': (meta, msg_s, pr_s),
            'large': (meta, msg_l, pr_l),
        }
        out = _build_cross_level_output(results)

        np.testing.assert_array_equal(
            out['proj_promise_msg_dir_small'].values, msg_s
        )
        np.testing.assert_array_equal(
            out['proj_promise_pr_dir_small'].values, pr_s
        )
        np.testing.assert_array_equal(
            out['proj_promise_msg_dir_large'].values, msg_l
        )
        np.testing.assert_array_equal(
            out['proj_promise_pr_dir_large'].values, pr_l
        )


# =====
# Regression: probe_phrase_validation
# =====
class TestProbePhraseValidation:
    """Tests for probe phrase validation."""

    def test_returns_all_probes(self):
        """Should return one row per probe phrase."""
        direction = np.random.randn(1536)
        direction /= np.linalg.norm(direction)

        with patch("analyze_promise_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(12, 1536).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert len(result) == 12

    def test_has_similarity_column(self):
        """Output should have cosine_similarity column."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_promise_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(12, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert 'cosine_similarity' in result.columns

    def test_similarities_in_valid_range(self):
        """Cosine similarities should be in [-1, 1]."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_promise_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(12, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert (result['cosine_similarity'] >= -1.01).all()
            assert (result['cosine_similarity'] <= 1.01).all()

    def test_sorted_by_similarity(self):
        """Output should be sorted by cosine_similarity descending."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_promise_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(12, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            values = result['cosine_similarity'].tolist()
            assert values == sorted(values, reverse=True)


# =====
# Integration: full promise analysis pipeline
# =====
class TestFullPromisePipeline:
    """Integration tests for the complete promise centroid->projection pipeline."""

    def test_pipeline_separates_promise_from_no_promise(self):
        """Full pipeline should assign higher scores to promise messages."""
        rng = np.random.default_rng(42)
        n = 30
        dim = 32
        promise_emb = rng.normal(loc=2.0, scale=0.5, size=(n, dim))
        no_promise_emb = rng.normal(loc=-2.0, scale=0.5, size=(n, dim))
        emb = np.vstack([promise_emb, no_promise_emb])
        labels = np.array(
            [STATE_PROMISE] * n + [STATE_NO_PROMISE] * n
        )

        p_c, np_c = compute_promise_centroids(emb, labels)
        direction = compute_difference_vector(p_c, np_c)
        projections = project_onto_direction(emb, direction)

        promise_min = projections[:n].min()
        no_promise_max = projections[n:].max()
        assert promise_min > no_promise_max

    def test_pipeline_output_structure(self):
        """Pipeline output should have correct column set."""
        n = 10
        meta = _make_metadata(n)
        emb = np.random.randn(n, 8)

        labels = meta[PROMISE_COL].values
        p_c, np_c = compute_promise_centroids(emb, labels)
        direction = compute_difference_vector(p_c, np_c)
        proj = project_onto_direction(emb, direction)

        result_small = _build_output(meta, proj, 'small', ID_COLS)
        result_large = _build_output(meta, proj * 0.5, 'large', ID_COLS)

        assert PROMISE_COL in result_small.columns
        assert 'proj_promise_pr_dir_small' in result_small.columns
        assert 'proj_promise_pr_dir_large' in result_large.columns

    def test_cross_level_output_column_names(self):
        """Cross-level output should match expected promise column names."""
        n = 6
        meta = _make_metadata(n)
        results = {
            'small': (meta, np.zeros(n), np.zeros(n)),
            'large': (meta, np.zeros(n), np.zeros(n)),
        }
        out = _build_cross_level_output(results)

        # Check that all 4 promise projection columns are present
        promise_cols = [c for c in out.columns if c.startswith('proj_promise_')]
        assert len(promise_cols) == 4


# =====
# Edge cases
# =====
class TestEdgeCases:
    """Edge case tests for promise embedding analysis."""

    def test_all_same_label_centroids(self):
        """All promise labels should produce one valid centroid."""
        emb = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        labels = np.array([STATE_PROMISE, STATE_PROMISE, STATE_PROMISE])
        p_c, np_c = compute_promise_centroids(emb, labels)

        np.testing.assert_array_equal(p_c, [2.0, 0.0])
        # np_c will be NaN due to empty slice
        assert np.isnan(np_c).all()

    def test_cosine_with_zero_embedding(self):
        """Zero-norm embedding should have zero cosine similarity."""
        direction = np.array([1.0, 0.0])
        emb = np.array([[0.0, 0.0]])
        result = _cosine_similarities(emb, direction)
        assert result[0] == pytest.approx(0.0)
