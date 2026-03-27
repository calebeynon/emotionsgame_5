"""
Tests for CCR direction vector .npy files (Task 1).

Verifies cached direction vectors used for external validation
projections: file existence, shape (1536), unit normalization,
distinctness, and regression on known element values.

Author: pytest-test-writer
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pytest

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
        pytest.skip(f"Direction vectors dir not found: {DIRECTION_VECTORS_DIR}")
    vectors = {}
    for name in EXPECTED_DIRECTION_NAMES:
        path = DIRECTION_VECTORS_DIR / f"{name}.npy"
        if not path.exists():
            pytest.skip(f"Missing direction vector: {path}")
        vectors[name] = np.load(path)
    return vectors


# =====
# File existence and shapes
# =====
class TestDirectionVectorFiles:
    """Verify all 5 direction vector .npy files exist with correct shapes."""

    def test_direction_vectors_dir_exists(self):
        """Output directory must exist."""
        assert DIRECTION_VECTORS_DIR.exists()

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_npy_file_exists(self, name):
        """Each direction vector .npy file must exist."""
        path = DIRECTION_VECTORS_DIR / f"{name}.npy"
        assert path.exists(), f"Missing: {path}"

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_npy_shape_is_1536(self, name):
        """Each vector must have shape (1536,) for text-embedding-3-small."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert v.shape == (EXPECTED_EMBEDDING_DIM,), f"{name}: {v.shape}"

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_npy_dtype_is_float64(self, name):
        """Direction vectors should be float64."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert v.dtype == np.float64, f"{name}: {v.dtype}"

    def test_exactly_five_vectors(self):
        """There should be exactly 5 .npy files."""
        npy_files = sorted(DIRECTION_VECTORS_DIR.glob("*.npy"))
        names = [f.stem for f in npy_files]
        assert sorted(names) == sorted(EXPECTED_DIRECTION_NAMES)


# =====
# Normalization and validity
# =====
class TestDirectionVectorNormalization:
    """Verify each direction vector is unit-normalized and valid."""

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_unit_norm(self, name):
        """Each direction vector norm must be approximately 1.0."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        norm = np.linalg.norm(v)
        assert norm == pytest.approx(1.0, abs=1e-6), f"{name}: norm={norm}"

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_all_elements_nonzero(self, name):
        """All 1536 elements should be nonzero (dense embedding space)."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert np.count_nonzero(v) == EXPECTED_EMBEDDING_DIM

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_no_nan_values(self, name):
        """No NaN values in direction vectors."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert not np.isnan(v).any(), f"{name} has NaN values"

    @pytest.mark.parametrize("name", EXPECTED_DIRECTION_NAMES)
    def test_no_inf_values(self, name):
        """No infinite values in direction vectors."""
        v = np.load(DIRECTION_VECTORS_DIR / f"{name}.npy")
        assert not np.isinf(v).any(), f"{name} has infinite values"


# =====
# Distinctness
# =====
class TestDirectionVectorDistinctness:
    """Verify direction vectors are distinct from each other."""

    def test_all_vectors_are_distinct(self, direction_vectors):
        """No two direction vectors should be near-identical."""
        names = list(direction_vectors.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                cos_sim = np.dot(direction_vectors[n1], direction_vectors[n2])
                assert abs(cos_sim) < 0.999, (
                    f"{n1} vs {n2}: cosine similarity = {cos_sim:.4f}"
                )


# =====
# Regression on known element values
# =====
class TestDirectionVectorRegression:
    """Regression tests: compare against known-good element values."""

    def test_cooperative_first_elements(self, direction_vectors):
        """Regression: cooperative vector first 3 elements from verified run."""
        v = direction_vectors["cooperative"]
        assert v[0] == pytest.approx(-0.02725520, abs=1e-5)
        assert v[1] == pytest.approx(-0.00766506, abs=1e-5)
        assert v[2] == pytest.approx(-0.04702951, abs=1e-5)

    def test_promise_first_elements(self, direction_vectors):
        """Regression: promise vector first 3 elements from verified run."""
        v = direction_vectors["promise"]
        assert v[0] == pytest.approx(-0.01111069, abs=1e-5)
        assert v[1] == pytest.approx(0.05503330, abs=1e-5)
        assert v[2] == pytest.approx(0.00492019, abs=1e-5)

    def test_round_liar_last_element(self, direction_vectors):
        """Regression: round_liar vector last element from verified run."""
        v = direction_vectors["round_liar"]
        assert v[-1] == pytest.approx(-0.03425808, abs=1e-5)

    def test_cumulative_liar_last_element(self, direction_vectors):
        """Regression: cumulative_liar vector last element."""
        v = direction_vectors["cumulative_liar"]
        assert v[-1] == pytest.approx(0.00423274, abs=1e-5)

    def test_homogeneity_first_element(self, direction_vectors):
        """Regression: homogeneity vector first element from verified run."""
        v = direction_vectors["homogeneity"]
        assert v[0] == pytest.approx(-0.01785697, abs=1e-5)
