"""
Tests for embedding analysis functions.

Uses synthetic numpy arrays — no API calls or file I/O.

Author: Claude Code
Date: 2026-03-15
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from analyze_embeddings import (
    compute_centroids,
    compute_difference_vector,
    project_onto_direction,
    compute_group_round_embeddings,
    rank_messages,
    probe_phrase_validation,
    build_projection_csv,
    STATE_COOPERATIVE,
    STATE_NONCOOPERATIVE,
    ID_COLS,
)


# =====
# Fixtures
# =====
def _make_embeddings_and_labels(n_coop=3, n_noncoop=3, dim=4):
    """Create synthetic embeddings with known centroid separation."""
    rng = np.random.default_rng(42)
    coop = rng.normal(loc=1.0, scale=0.1, size=(n_coop, dim))
    noncoop = rng.normal(loc=-1.0, scale=0.1, size=(n_noncoop, dim))
    embeddings = np.vstack([coop, noncoop])

    labels = np.array(
        [STATE_COOPERATIVE] * n_coop
        + [STATE_NONCOOPERATIVE] * n_noncoop
    )
    return embeddings, labels


def _make_metadata(n: int) -> pd.DataFrame:
    """Create minimal metadata DataFrame for n messages."""
    return pd.DataFrame({
        'session_code': ['sess1'] * n,
        'treatment': [1] * n,
        'segment': ['supergame1'] * n,
        'round': [1] * n,
        'group': [1] * (n // 2) + [2] * (n - n // 2),
        'label': [chr(65 + i) for i in range(n)],
        'message_index': range(n),
        'message_text': [f'message {i}' for i in range(n)],
        'player_state': (
            [STATE_COOPERATIVE] * (n // 2)
            + [STATE_NONCOOPERATIVE] * (n - n // 2)
        ),
    })


# =====
# Test compute_centroids
# =====
class TestComputeCentroids:
    """Tests for centroid computation."""

    def test_returns_two_vectors(self):
        """Should return cooperative and non-cooperative centroids."""
        emb, labels = _make_embeddings_and_labels()
        coop_c, noncoop_c = compute_centroids(emb, labels)
        assert coop_c.shape == (4,)
        assert noncoop_c.shape == (4,)

    def test_centroids_are_separated(self):
        """Cooperative centroid should differ from non-cooperative."""
        emb, labels = _make_embeddings_and_labels()
        coop_c, noncoop_c = compute_centroids(emb, labels)
        distance = np.linalg.norm(coop_c - noncoop_c)
        assert distance > 1.0

    def test_centroid_is_mean_of_group(self):
        """Each centroid should be the mean of its group."""
        emb, labels = _make_embeddings_and_labels(n_coop=2, n_noncoop=2)
        coop_c, _ = compute_centroids(emb, labels)
        expected = emb[:2].mean(axis=0)
        np.testing.assert_array_almost_equal(coop_c, expected)


# =====
# Test compute_difference_vector
# =====
class TestComputeDifferenceVector:
    """Tests for normalized difference vector."""

    def test_output_is_unit_length(self):
        """Direction vector should have unit norm."""
        coop_c = np.array([1.0, 0.0, 0.0])
        noncoop_c = np.array([-1.0, 0.0, 0.0])
        direction = compute_difference_vector(coop_c, noncoop_c)
        np.testing.assert_almost_equal(np.linalg.norm(direction), 1.0)

    def test_points_toward_cooperative(self):
        """Direction should point from non-cooperative toward cooperative."""
        coop_c = np.array([1.0, 0.0])
        noncoop_c = np.array([-1.0, 0.0])
        direction = compute_difference_vector(coop_c, noncoop_c)
        assert direction[0] > 0

    def test_handles_identical_centroids(self):
        """Should return zero vector if centroids are identical."""
        c = np.array([1.0, 2.0])
        direction = compute_difference_vector(c, c)
        np.testing.assert_array_equal(direction, np.zeros(2))


# =====
# Test project_onto_direction
# =====
class TestProjectOntoDirection:
    """Tests for dot-product projection."""

    def test_cooperative_scores_higher(self):
        """Cooperative embeddings should project higher."""
        emb, labels = _make_embeddings_and_labels()
        coop_c, noncoop_c = compute_centroids(emb, labels)
        direction = compute_difference_vector(coop_c, noncoop_c)
        projections = project_onto_direction(emb, direction)

        coop_mean = projections[:3].mean()
        noncoop_mean = projections[3:].mean()
        assert coop_mean > noncoop_mean

    def test_output_shape(self):
        """Should return one score per embedding."""
        emb = np.random.randn(10, 4)
        direction = np.array([1.0, 0.0, 0.0, 0.0])
        projections = project_onto_direction(emb, direction)
        assert projections.shape == (10,)

    def test_known_values(self):
        """Verify with hand-computed dot products."""
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        direction = np.array([1.0, 0.0])
        result = project_onto_direction(emb, direction)
        np.testing.assert_array_equal(result, [1.0, 0.0])


# =====
# Test compute_group_round_embeddings
# =====
class TestComputeGroupRoundEmbeddings:
    """Tests for group-round aggregation."""

    def test_reduces_to_group_count(self):
        """Should return one row per unique group-round."""
        meta = _make_metadata(6)
        emb = np.random.randn(6, 4)
        group_meta, group_emb = compute_group_round_embeddings(meta, emb)
        assert len(group_meta) == 2  # Two groups
        assert group_emb.shape == (2, 4)

    def test_group_embedding_is_mean(self):
        """Group embedding should be mean of member embeddings."""
        meta = _make_metadata(4)
        emb = np.array([
            [1.0, 0.0], [3.0, 0.0],  # group 1
            [0.0, 1.0], [0.0, 3.0],  # group 2
        ])
        _, group_emb = compute_group_round_embeddings(meta, emb)
        # Sort by first dimension to ensure consistent ordering
        group_emb = group_emb[group_emb[:, 0].argsort()[::-1]]
        np.testing.assert_array_almost_equal(
            group_emb[0], [2.0, 0.0]
        )


# =====
# Test rank_messages
# =====
class TestRankMessages:
    """Tests for message ranking."""

    def test_returns_correct_count(self):
        """Should return n messages in each direction."""
        meta = _make_metadata(6)
        projections = np.array([5, 4, 3, -3, -4, -5], dtype=float)
        top_coop, top_noncoop = rank_messages(meta, projections, n=2)
        assert len(top_coop) == 2
        assert len(top_noncoop) == 2

    def test_top_cooperative_has_highest(self):
        """Top cooperative should contain highest projections."""
        meta = _make_metadata(6)
        projections = np.array([5, 4, 3, -3, -4, -5], dtype=float)
        top_coop, _ = rank_messages(meta, projections, n=2)
        assert top_coop['projection'].iloc[0] == 5.0

    def test_top_noncooperative_has_lowest(self):
        """Top non-cooperative should contain lowest projections."""
        meta = _make_metadata(6)
        projections = np.array([5, 4, 3, -3, -4, -5], dtype=float)
        _, top_noncoop = rank_messages(meta, projections, n=2)
        assert top_noncoop['projection'].iloc[-1] == -5.0


# =====
# Test probe_phrase_validation
# =====
class TestProbePhraseValidation:
    """Tests for probe phrase validation."""

    def test_returns_all_probes(self):
        """Should return one row per probe phrase."""
        direction = np.random.randn(1536)
        direction /= np.linalg.norm(direction)

        with patch("analyze_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(6, 1536).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert len(result) == 6

    def test_has_similarity_column(self):
        """Output should have cosine_similarity column."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(6, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert 'cosine_similarity' in result.columns

    def test_similarities_in_valid_range(self):
        """Cosine similarities should be in [-1, 1]."""
        direction = np.random.randn(4)
        direction /= np.linalg.norm(direction)

        with patch("analyze_embeddings.embed_texts") as mock:
            mock.return_value = np.random.randn(6, 4).tolist()
            result = probe_phrase_validation(direction, "test-model")
            assert (result['cosine_similarity'] >= -1.01).all()
            assert (result['cosine_similarity'] <= 1.01).all()


# =====
# Test build_projection_csv
# =====
class TestBuildProjectionCsv:
    """Tests for output DataFrame construction."""

    def test_has_projection_column(self):
        """Output should include proj_pr_dir_{suffix} column."""
        meta = _make_metadata(4)
        projections = np.array([1.0, 2.0, -1.0, -2.0])
        result = build_projection_csv(meta, projections, 'small')
        assert 'proj_pr_dir_small' in result.columns

    def test_preserves_id_columns(self):
        """All ID columns should be present."""
        meta = _make_metadata(4)
        projections = np.array([1.0, 2.0, -1.0, -2.0])
        result = build_projection_csv(meta, projections, 'small')
        for col in ID_COLS:
            assert col in result.columns

    def test_row_count_matches_input(self):
        """Output should have same number of rows as input."""
        meta = _make_metadata(6)
        projections = np.zeros(6)
        result = build_projection_csv(meta, projections, 'large')
        assert len(result) == 6
