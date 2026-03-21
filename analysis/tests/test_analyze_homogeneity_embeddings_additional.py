"""
Additional tests for analyze_homogeneity_embeddings.py.

Covers output construction, cross-level output, projection merging,
probe phrase validation, end-to-end pipeline separation, constants,
and edge cases. Complements test_analyze_homogeneity_embeddings.py.

Author: Claude Code (test-writer)
Date: 2026-03-20
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from analyze_homogeneity_embeddings import (
    _build_cross_level_output,
    _build_output,
    compute_homogeneity_centroids,
    probe_phrase_validation,
    GROUP_KEYS,
    HOMOG_COL,
    ID_COLS,
    PR_ID_COLS,
    PROBE_PHRASES,
    STATE_HETEROGENEOUS,
    STATE_HOMOGENEOUS,
)
from analyze_embeddings import compute_difference_vector, project_onto_direction


# =====
# Helpers
# =====
def _make_msg_metadata(n, with_homog=True):
    """Create message-level metadata DataFrame."""
    meta = pd.DataFrame({
        'session_code': ['s1'] * n,
        'treatment': [1] * n,
        'segment': ['supergame1'] * n,
        'round': [2] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
        'message_index': range(n),
        'message_text': [f'message {i}' for i in range(n)],
        'player_state': ['cooperative'] * n,
    })
    if with_homog:
        meta[HOMOG_COL] = (
            [STATE_HOMOGENEOUS] * (n // 2)
            + [STATE_HETEROGENEOUS] * (n - n // 2)
        )
    return meta


def _make_pr_metadata(n, with_homog=True):
    """Create player-round level metadata DataFrame."""
    meta = pd.DataFrame({
        'session_code': ['s1'] * n,
        'treatment': [1] * n,
        'segment': ['supergame1'] * n,
        'round': [2] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
        'combined_text': [f'combined text {i}' for i in range(n)],
        'player_state': ['cooperative'] * n,
    })
    if with_homog:
        meta[HOMOG_COL] = (
            [STATE_HOMOGENEOUS] * (n // 2)
            + [STATE_HETEROGENEOUS] * (n - n // 2)
        )
    return meta


# =====
# Constants
# =====
class TestConstants:
    """Verify module constants match expected values."""

    def test_state_labels(self):
        """State labels should match expected strings."""
        assert STATE_HOMOGENEOUS == 'homogeneous'
        assert STATE_HETEROGENEOUS == 'heterogeneous'

    def test_homog_col_name(self):
        """Column name constant should be 'homogeneity_label'."""
        assert HOMOG_COL == 'homogeneity_label'

    def test_group_keys(self):
        """GROUP_KEYS should match expected merge columns."""
        expected = ['session_code', 'segment', 'round', 'group']
        assert GROUP_KEYS == expected

    def test_probe_phrases_count(self):
        """Should have 11 homogeneity-related probe phrases."""
        assert len(PROBE_PHRASES) == 11

    def test_probe_phrases_content(self):
        """Probe phrases should include key homogeneity exemplars."""
        assert "let's all put in the same amount" in PROBE_PHRASES
        assert "I'm going my own way" in PROBE_PHRASES
        assert "we should all contribute equally" in PROBE_PHRASES

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


# =====
# _build_output
# =====
class TestBuildOutput:
    """Tests for output DataFrame construction."""

    def test_has_homog_projection_column(self):
        """Output should include proj_homog_pr_dir_{suffix} column."""
        meta = _make_pr_metadata(4)
        projections = np.array([1.0, 2.0, -1.0, -2.0])
        result = _build_output(meta, projections, 'small', PR_ID_COLS)
        assert 'proj_homog_pr_dir_small' in result.columns

    def test_includes_homogeneity_label(self):
        """Output should include homogeneity_label column."""
        meta = _make_pr_metadata(4)
        projections = np.zeros(4)
        result = _build_output(meta, projections, 'small', PR_ID_COLS)
        assert HOMOG_COL in result.columns

    def test_includes_player_state(self):
        """Output should include player_state column."""
        meta = _make_pr_metadata(4)
        projections = np.zeros(4)
        result = _build_output(meta, projections, 'small', PR_ID_COLS)
        assert 'player_state' in result.columns

    def test_row_count_matches_input(self):
        """Output should have same number of rows as input."""
        meta = _make_pr_metadata(6)
        projections = np.zeros(6)
        result = _build_output(meta, projections, 'large', PR_ID_COLS)
        assert len(result) == 6

    def test_pr_level_output(self):
        """Player-round level output should use PR_ID_COLS."""
        meta = _make_pr_metadata(4)
        projections = np.zeros(4)
        result = _build_output(meta, projections, 'small', PR_ID_COLS)
        assert 'combined_text' in result.columns
        assert 'message_index' not in result.columns


# =====
# _build_cross_level_output
# =====
class TestBuildCrossLevelOutput:
    """Tests for the cross-level output builder."""

    def test_has_four_projection_columns(self):
        """Output should have all 4 cross-level projection columns."""
        n = 4
        meta = _make_msg_metadata(n)
        results = {
            'small': (meta, np.ones(n), np.ones(n) * 2),
            'large': (meta, np.ones(n) * 3, np.ones(n) * 4),
        }
        out = _build_cross_level_output(results)

        expected_cols = [
            'proj_homog_msg_dir_small', 'proj_homog_pr_dir_small',
            'proj_homog_msg_dir_large', 'proj_homog_pr_dir_large',
        ]
        for col in expected_cols:
            assert col in out.columns

    def test_includes_both_state_columns(self):
        """Output should have both player_state and homogeneity_label."""
        n = 4
        meta = _make_msg_metadata(n)
        results = {
            'small': (meta, np.zeros(n), np.zeros(n)),
            'large': (meta, np.zeros(n), np.zeros(n)),
        }
        out = _build_cross_level_output(results)

        assert 'player_state' in out.columns
        assert HOMOG_COL in out.columns

    def test_row_count(self):
        """Output should match the number of messages."""
        n = 6
        meta = _make_msg_metadata(n)
        results = {
            'small': (meta, np.zeros(n), np.zeros(n)),
            'large': (meta, np.zeros(n), np.zeros(n)),
        }
        out = _build_cross_level_output(results)
        assert len(out) == n

    def test_projection_values_are_correct(self):
        """Projection values should match the provided arrays."""
        n = 3
        meta = _make_msg_metadata(n)
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
            out['proj_homog_msg_dir_small'].values, msg_s,
        )
        np.testing.assert_array_equal(
            out['proj_homog_pr_dir_small'].values, pr_s,
        )
        np.testing.assert_array_equal(
            out['proj_homog_msg_dir_large'].values, msg_l,
        )
        np.testing.assert_array_equal(
            out['proj_homog_pr_dir_large'].values, pr_l,
        )


# =====
# probe_phrase_validation
# =====
class TestProbePhraseValidation:
    """Tests for probe phrase validation."""

    def test_returns_all_probes(self):
        """Should return one row per probe phrase."""
        direction = np.random.randn(1536)
        direction /= np.linalg.norm(direction)

        with patch("analyze_homogeneity_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(11, 1536).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert len(result) == 11

    def test_has_similarity_column(self):
        """Output should have cosine_similarity column."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_homogeneity_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(11, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert 'cosine_similarity' in result.columns

    def test_similarities_in_valid_range(self):
        """Cosine similarities should be in [-1, 1]."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_homogeneity_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(11, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert (result['cosine_similarity'] >= -1.01).all()
            assert (result['cosine_similarity'] <= 1.01).all()

    def test_sorted_by_similarity(self):
        """Output should be sorted by cosine_similarity descending."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_homogeneity_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(11, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            values = result['cosine_similarity'].tolist()
            assert values == sorted(values, reverse=True)


# =====
# End-to-end pipeline
# =====
class TestFullHomogeneityPipeline:
    """Integration: centroid -> direction -> projection separation."""

    def test_pipeline_separates_homogeneous_from_heterogeneous(self):
        """Homogeneous embeddings should project higher than heterogeneous."""
        rng = np.random.default_rng(42)
        n = 50
        dim = 32
        homog_emb = rng.normal(loc=2.0, scale=0.5, size=(n, dim))
        heterog_emb = rng.normal(loc=-2.0, scale=0.5, size=(n, dim))
        emb = np.vstack([homog_emb, heterog_emb])
        labels = np.array(
            [STATE_HOMOGENEOUS] * n + [STATE_HETEROGENEOUS] * n,
        )

        h_c, het_c = compute_homogeneity_centroids(emb, labels)
        direction = compute_difference_vector(h_c, het_c)
        projections = project_onto_direction(emb, direction)

        homog_mean = projections[:n].mean()
        heterog_mean = projections[n:].mean()
        assert homog_mean > heterog_mean
        assert homog_mean - heterog_mean > 1.0

    def test_clear_separation_all_homog_above(self):
        """With well-separated data, every homog score above every heterog."""
        rng = np.random.default_rng(99)
        n = 20
        dim = 32
        homog_emb = rng.normal(loc=3.0, scale=0.3, size=(n, dim))
        heterog_emb = rng.normal(loc=-3.0, scale=0.3, size=(n, dim))
        emb = np.vstack([homog_emb, heterog_emb])
        labels = np.array(
            [STATE_HOMOGENEOUS] * n + [STATE_HETEROGENEOUS] * n,
        )

        h_c, het_c = compute_homogeneity_centroids(emb, labels)
        direction = compute_difference_vector(h_c, het_c)
        projections = project_onto_direction(emb, direction)

        assert projections[:n].min() > projections[n:].max()

    def test_cross_level_output_column_names(self):
        """Cross-level output should match expected homog column names."""
        n = 6
        meta = _make_msg_metadata(n)
        results = {
            'small': (meta, np.zeros(n), np.zeros(n)),
            'large': (meta, np.zeros(n), np.zeros(n)),
        }
        out = _build_cross_level_output(results)
        homog_cols = [c for c in out.columns if c.startswith('proj_homog_')]
        assert len(homog_cols) == 4


# =====
# Edge cases
# =====
class TestEdgeCases:
    """Edge case tests for homogeneity embedding analysis."""

    def test_all_same_label_centroids(self):
        """All homogeneous labels should produce one valid centroid."""
        emb = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        labels = np.array([
            STATE_HOMOGENEOUS, STATE_HOMOGENEOUS, STATE_HOMOGENEOUS,
        ])
        h_c, het_c = compute_homogeneity_centroids(emb, labels)

        np.testing.assert_array_equal(h_c, [2.0, 0.0])
        assert np.isnan(het_c).all()

    def test_single_point_per_class(self):
        """Single point should be its own centroid."""
        emb = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        labels = np.array([STATE_HOMOGENEOUS, STATE_HETEROGENEOUS])
        h_c, het_c = compute_homogeneity_centroids(emb, labels)

        np.testing.assert_array_equal(h_c, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(het_c, [-1.0, -2.0, -3.0])

    def test_high_dimensional_centroids(self):
        """Should work with high-dimensional embeddings."""
        dim = 1536
        emb = np.vstack([np.ones((5, dim)), -np.ones((5, dim))])
        labels = np.array(
            [STATE_HOMOGENEOUS] * 5 + [STATE_HETEROGENEOUS] * 5,
        )
        h_c, het_c = compute_homogeneity_centroids(emb, labels)

        assert h_c.shape == (dim,)
        np.testing.assert_array_almost_equal(h_c, np.ones(dim))
        np.testing.assert_array_almost_equal(het_c, -np.ones(dim))
