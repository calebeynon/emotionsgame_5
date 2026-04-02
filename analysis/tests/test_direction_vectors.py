"""
Tests for Task 2: cache_direction_vectors.py output .npy file integrity.

Validates file existence, shapes, normalization, distinctness, regression
values, and helper function correctness against real verified output.

Author: pytest-test-writer
Date: 2026-03-26
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
DIRECTION_VECTORS_DIR = DERIVED_DIR / "direction_vectors"

# Known-good values from verified output
EXPECTED_DIRECTION_NAMES = [
    "cooperative", "cumulative_liar", "homogeneity", "promise", "round_liar",
]
EXPECTED_EMBEDDING_DIM = 1536


# =====
# Fixtures
# =====
@pytest.fixture
def direction_vectors():
    """Load all 5 direction vectors as a dict of numpy arrays."""
    if not DIRECTION_VECTORS_DIR.exists():
        pytest.skip(f"Dir not found: {DIRECTION_VECTORS_DIR}")
    vectors = {}
    for name in EXPECTED_DIRECTION_NAMES:
        path = DIRECTION_VECTORS_DIR / f"{name}.npy"
        if not path.exists():
            pytest.skip(f"Missing: {path}")
        vectors[name] = np.load(path)
    return vectors


# =====
# File existence and shapes
# =====
class TestDirectionVectorFiles:
    """Verify all 5 .npy files exist with correct shapes."""

    def test_direction_vectors_dir_exists(self):
        """Output directory must exist."""
        assert DIRECTION_VECTORS_DIR.exists()

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_npy_file_exists(self, name):
        """Each .npy file must exist."""
        assert (DIRECTION_VECTORS_DIR / f"{name}.npy").exists()

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_npy_shape_is_1536(self, name):
        """Each vector must have shape (1536,)."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert v.shape == (EXPECTED_EMBEDDING_DIM,)

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_npy_dtype_is_float64(self, name):
        """Direction vectors should be float64."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert v.dtype == np.float64

    def test_exactly_five_vectors(self):
        """There should be exactly 5 .npy files."""
        npy_files = sorted(DIRECTION_VECTORS_DIR.glob("*.npy"))
        assert sorted(f.stem for f in npy_files) == sorted(EXPECTED_DIRECTION_NAMES)


# =====
# Normalization
# =====
class TestDirectionVectorNormalization:
    """Verify each direction vector is unit-normalized."""

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_unit_norm(self, name):
        """Each direction vector norm must be approximately 1.0."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_no_nan_values(self, name):
        """No NaN values in direction vectors."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert not np.isnan(v).any()

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_no_inf_values(self, name):
        """No infinite values in direction vectors."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert not np.isinf(v).any()


# =====
# Distinctness
# =====
class TestDirectionVectorDistinctness:
    """Verify direction vectors are distinct from each other."""

    def test_all_vectors_are_distinct(self, direction_vectors):
        """No two direction vectors should be identical."""
        names = list(direction_vectors.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                cos_sim = np.dot(direction_vectors[n1], direction_vectors[n2])
                assert abs(cos_sim) < 0.999, (
                    f"{n1} vs {n2}: cosine sim = {cos_sim:.4f}"
                )

    def test_cooperative_vs_homogeneity_correlated(self, direction_vectors):
        """Cooperative and homogeneity are correlated but not identical."""
        cos_sim = np.dot(
            direction_vectors["cooperative"],
            direction_vectors["homogeneity"],
        )
        assert cos_sim == pytest.approx(0.9121, abs=0.01)

    def test_cooperative_vs_promise_near_orthogonal(self, direction_vectors):
        """Cooperative and promise should be near-orthogonal."""
        cos_sim = np.dot(
            direction_vectors["cooperative"],
            direction_vectors["promise"],
        )
        assert abs(cos_sim) < 0.1


# =====
# Regression on known values
# =====
class TestDirectionVectorRegression:
    """Regression tests against known-good element values."""

    def test_cooperative_first_elements(self, direction_vectors):
        """Cooperative vector first 3 elements from verified run."""
        v = direction_vectors["cooperative"]
        assert v[0] == pytest.approx(-0.02725520, abs=1e-5)
        assert v[1] == pytest.approx(-0.00766506, abs=1e-5)
        assert v[2] == pytest.approx(-0.04702951, abs=1e-5)

    def test_promise_first_elements(self, direction_vectors):
        """Promise vector first 3 elements from verified run."""
        v = direction_vectors["promise"]
        assert v[0] == pytest.approx(-0.01111069, abs=1e-5)
        assert v[1] == pytest.approx(0.05503330, abs=1e-5)
        assert v[2] == pytest.approx(0.00492019, abs=1e-5)

    def test_round_liar_last_element(self, direction_vectors):
        """Round-liar vector last element from verified run."""
        v = direction_vectors["round_liar"]
        assert v[-1] == pytest.approx(-0.03425808, abs=1e-5)

    def test_cumulative_liar_last_element(self, direction_vectors):
        """Cumulative-liar vector last element from verified run."""
        v = direction_vectors["cumulative_liar"]
        assert v[-1] == pytest.approx(0.00423274, abs=1e-5)


# =====
# Helper function unit tests
# =====
class TestDirectionVectorHelpers:
    """Unit tests for cache_direction_vectors helper functions."""

    def test_compute_centroid_basic(self):
        """compute_centroid returns mean of selected rows."""
        from cache_direction_vectors import compute_centroid

        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = np.array([True, False, True])
        expected = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(
            compute_centroid(embeddings, mask), expected,
        )

    def test_compute_centroid_empty_mask_raises(self):
        """compute_centroid raises ValueError for all-False mask."""
        from cache_direction_vectors import compute_centroid

        with pytest.raises(ValueError, match="No True values"):
            compute_centroid(np.array([[1.0, 2.0]]), np.array([False]))

    def test_normalized_direction_is_unit(self):
        """compute_normalized_direction returns a unit vector."""
        from cache_direction_vectors import compute_normalized_direction

        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = compute_normalized_direction(
            embeddings, np.array([True, False]), np.array([False, True]),
        )
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-10)

    def test_normalized_direction_correct_sign(self):
        """Direction should point from negative toward positive centroid."""
        from cache_direction_vectors import compute_normalized_direction

        embeddings = np.array([
            [10.0, 0.0], [12.0, 0.0],
            [0.0, 0.0], [0.0, 0.0],
        ])
        result = compute_normalized_direction(
            embeddings,
            np.array([True, True, False, False]),
            np.array([False, False, True, True]),
        )
        assert result[0] > 0

    def test_identical_centroids_raises(self):
        """Identical centroids should raise ValueError (zero norm)."""
        from cache_direction_vectors import compute_normalized_direction

        embeddings = np.array([[1.0, 2.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="zero norm"):
            compute_normalized_direction(
                embeddings, np.array([True, False]), np.array([False, True]),
            )

    def test_load_embeddings_separates_meta_and_vectors(self):
        """load_embeddings returns (meta_df, embeddings_array)."""
        from cache_direction_vectors import load_embeddings

        emb_path = DERIVED_DIR / "embeddings_player_round_small.parquet"
        if not emb_path.exists():
            pytest.skip("Embeddings parquet not available")
        meta, emb = load_embeddings(emb_path)
        assert isinstance(meta, pd.DataFrame)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (len(meta), EXPECTED_EMBEDDING_DIM)
        assert not any(c.startswith("emb_") for c in meta.columns)


# =====
# Reproducibility
# =====
def _get_compute_fn(name):
    """Return the direction-computation function for the given name."""
    from cache_direction_vectors import (
        compute_cooperative_direction,
        compute_cumulative_liar_direction,
        compute_homogeneity_direction,
        compute_promise_direction,
        compute_round_liar_direction,
    )

    return {
        "cooperative": compute_cooperative_direction,
        "promise": compute_promise_direction,
        "homogeneity": compute_homogeneity_direction,
        "round_liar": compute_round_liar_direction,
        "cumulative_liar": compute_cumulative_liar_direction,
    }[name]


class TestDirectionVectorReproducibility:
    """Verify re-running the script produces identical output."""

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_recompute_matches_cached(self, name):
        """Re-deriving a direction vector must match the saved .npy file."""
        from cache_direction_vectors import load_embeddings

        emb_path = DERIVED_DIR / "embeddings_player_round_small.parquet"
        if not emb_path.exists():
            pytest.skip("Embeddings parquet not available")

        meta, embeddings = load_embeddings(emb_path)
        recomputed = _get_compute_fn(name)(meta, embeddings)
        cached = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        np.testing.assert_array_almost_equal(
            recomputed, cached, decimal=10,
        )
