"""
Tests for Task 4: project_hanaki_embeddings.py output CSV integrity.

Validates schema, projection values, PairAveCho computation, alignment
with embeddings input, and helper function correctness.

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
PROJECTIONS_FILE = DERIVED_DIR / "hanaki_ozkes_projections.csv"
EMBEDDINGS_FILE = DERIVED_DIR / "hanaki_ozkes_embeddings.parquet"
DIRECTION_DIR = DERIVED_DIR / "direction_vectors"

# Known-good values from verified output
EXPECTED_ROW_COUNT = 8210
EXPECTED_COLUMNS = [
    "session_file", "period", "player_id", "group", "chat_text",
    "Inv", "OtherInv", "PairAveCho", "Profit", "Chat", "Comp",
    "proj_cooperative", "proj_promise", "proj_homogeneity",
    "proj_round_liar", "proj_cumulative_liar",
]
PROJECTION_COLS = [
    "proj_cooperative", "proj_promise", "proj_homogeneity",
    "proj_round_liar", "proj_cumulative_liar",
]


# =====
# Fixtures
# =====
@pytest.fixture
def proj_df():
    """Load the projections CSV as a DataFrame."""
    if not PROJECTIONS_FILE.exists():
        pytest.skip(f"Not found: {PROJECTIONS_FILE}")
    return pd.read_csv(PROJECTIONS_FILE)


# =====
# Schema and structure
# =====
class TestProjectionsSchema:
    """Verify projections CSV has correct schema."""

    def test_file_exists(self):
        """Output CSV must exist."""
        assert PROJECTIONS_FILE.exists()

    def test_row_count_matches_embeddings(self, proj_df):
        """Row count should match embeddings input (8210)."""
        assert len(proj_df) == EXPECTED_ROW_COUNT

    def test_expected_columns(self, proj_df):
        """CSV must contain exactly the expected columns in order."""
        assert list(proj_df.columns) == EXPECTED_COLUMNS

    def test_column_count(self, proj_df):
        """16 columns: 11 metadata + 5 projections."""
        assert len(proj_df.columns) == 16

    def test_no_null_values(self, proj_df):
        """No null values in any column."""
        assert proj_df.isna().sum().sum() == 0


# =====
# Projection values
# =====
class TestProjectionValues:
    """Verify projection values are reasonable."""

    @pytest.mark.parametrize("col", PROJECTION_COLS)
    def test_projection_is_finite(self, proj_df, col):
        """All projection values must be finite."""
        assert np.isfinite(proj_df[col]).all()

    @pytest.mark.parametrize("col", PROJECTION_COLS)
    def test_projection_range(self, proj_df, col):
        """Projections should be in a reasonable range (unit vectors dotted
        with unit embeddings => projections in [-1, 1])."""
        assert proj_df[col].min() >= -1.0
        assert proj_df[col].max() <= 1.0

    def test_projections_have_variance(self, proj_df):
        """Each projection should have non-zero variance."""
        for col in PROJECTION_COLS:
            assert proj_df[col].std() > 0.01, f"{col} has near-zero variance"

    def test_projections_not_all_identical(self, proj_df):
        """Different projection axes should produce different values."""
        corr = proj_df[PROJECTION_COLS].corr()
        for i, c1 in enumerate(PROJECTION_COLS):
            for c2 in PROJECTION_COLS[i + 1:]:
                assert abs(corr.loc[c1, c2]) < 0.999


# =====
# PairAveCho computation
# =====
class TestPairAveCho:
    """Verify PairAveCho = (Inv + OtherInv) / 2."""

    def test_pair_ave_cho_formula(self, proj_df):
        """PairAveCho must equal (Inv + OtherInv) / 2."""
        expected = (proj_df["Inv"] + proj_df["OtherInv"]) / 2
        pd.testing.assert_series_equal(
            proj_df["PairAveCho"], expected, check_names=False,
        )

    def test_pair_ave_cho_range(self, proj_df):
        """PairAveCho should be in [0, 28]."""
        assert proj_df["PairAveCho"].min() >= 0.0
        assert proj_df["PairAveCho"].max() <= 28.0


# =====
# Data alignment with embeddings
# =====
class TestProjectionsAlignment:
    """Verify projections align with embeddings input."""

    def test_session_count_matches_embeddings(self, proj_df):
        """Same number of sessions as embeddings."""
        assert proj_df["session_file"].nunique() == 23

    def test_all_chat_1(self, proj_df):
        """All rows should have Chat=1."""
        assert (proj_df["Chat"] == 1).all()

    def test_investment_range(self, proj_df):
        """Inv should be in [0, 28]."""
        assert proj_df["Inv"].min() >= 0.0
        assert proj_df["Inv"].max() <= 28.0


# =====
# Regression values
# =====
class TestProjectionsRegression:
    """Regression tests against known-good values."""

    def test_first_row_proj_cooperative(self, proj_df):
        """First row proj_cooperative from verified run."""
        assert proj_df["proj_cooperative"].iloc[0] == pytest.approx(
            0.08915487, abs=1e-4,
        )

    def test_first_row_proj_promise(self, proj_df):
        """First row proj_promise from verified run."""
        assert proj_df["proj_promise"].iloc[0] == pytest.approx(
            0.07423218, abs=1e-4,
        )

    def test_first_row_proj_cumulative_liar(self, proj_df):
        """First row proj_cumulative_liar from verified run."""
        assert proj_df["proj_cumulative_liar"].iloc[0] == pytest.approx(
            -0.09206078, abs=1e-4,
        )

    def test_projection_means(self, proj_df):
        """Projection means from verified run."""
        assert proj_df["proj_cooperative"].mean() == pytest.approx(
            0.0002, abs=0.01,
        )
        assert proj_df["proj_promise"].mean() == pytest.approx(
            0.1647, abs=0.01,
        )


# =====
# Helper function unit tests
# =====
class TestProjectionHelpers:
    """Unit tests for project_hanaki_embeddings helper functions."""

    def test_compute_all_projections_shape(self):
        """compute_all_projections returns correct shape."""
        from project_hanaki_embeddings import compute_all_projections

        embeddings = np.random.randn(10, 1536)
        directions = {
            name: np.random.randn(1536)
            for name in [
                "cooperative", "promise", "homogeneity",
                "round_liar", "cumulative_liar",
            ]
        }
        result = compute_all_projections(embeddings, directions)
        assert result.shape == (10, 5)

    def test_compute_all_projections_dot_product(self):
        """Projections should equal dot product of embeddings and directions."""
        from project_hanaki_embeddings import compute_all_projections

        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        directions = {
            "cooperative": np.array([1.0, 0.0, 0.0]),
            "promise": np.array([0.0, 1.0, 0.0]),
            "homogeneity": np.array([0.0, 0.0, 1.0]),
            "round_liar": np.array([1.0, 1.0, 0.0]),
            "cumulative_liar": np.array([0.5, 0.5, 0.5]),
        }
        result = compute_all_projections(embeddings, directions)
        assert result["proj_cooperative"].iloc[0] == pytest.approx(1.0)
        assert result["proj_cooperative"].iloc[1] == pytest.approx(0.0)
        assert result["proj_promise"].iloc[0] == pytest.approx(0.0)
        assert result["proj_promise"].iloc[1] == pytest.approx(1.0)

    def test_build_output_adds_pair_ave_cho(self):
        """build_output must add PairAveCho column."""
        from project_hanaki_embeddings import build_output

        meta = pd.DataFrame({
            "session_file": ["s1"], "period": [1],
            "player_id": [1], "group": [1],
            "chat_text": ["hi"], "Inv": [10.0],
            "OtherInv": [20.0], "Profit": [5.0],
            "Chat": [1], "Comp": [0],
        })
        proj = pd.DataFrame({
            f"proj_{n}": [0.1]
            for n in [
                "cooperative", "promise", "homogeneity",
                "round_liar", "cumulative_liar",
            ]
        })
        result = build_output(meta, proj)
        assert result["PairAveCho"].iloc[0] == pytest.approx(15.0)
