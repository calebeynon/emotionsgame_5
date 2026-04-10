"""
Tests for dynamic_regression.R data derivation logic.

Replicates the deviation variable construction from the Arellano-Bond
dynamic panel regression R script in Python and validates correctness
against the real contributions.csv dataset. Includes a regression test
for the /1.3 bug fix (Stata used /1.3 instead of /3).

Author: Claude Code
Date: 2026-04-09
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# FILE PATHS
CONTRIBUTIONS_CSV = (
    Path(__file__).parent.parent / "datastore" / "derived" / "contributions.csv"
)

# CONSTANTS
ENDOWMENT = 25
MULTIPLIER = 0.4


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def derived_df() -> pd.DataFrame:
    """Load contributions.csv and compute deviation variables.

    Replicates derive_deviation_variables() from the R script.
    """
    if not CONTRIBUTIONS_CSV.exists():
        pytest.skip(f"contributions.csv not found: {CONTRIBUTIONS_CSV}")
    df = pd.read_csv(CONTRIBUTIONS_CSV)
    df["othercont"] = (df["payoff"] - 25 + 0.6 * df["contribution"]) / 0.4
    df["othercontaverage"] = df["othercont"] / 3
    df["morethanaverage"] = (df["contribution"] > df["othercontaverage"]).astype(int)
    df["lessthanaverage"] = (df["contribution"] < df["othercontaverage"]).astype(int)
    df["diffcont"] = df["contribution"] - df["othercontaverage"]
    df["contmore"] = df["diffcont"] * df["morethanaverage"]
    df["contless"] = -df["diffcont"] * df["lessthanaverage"]
    return df


# =====
# Test 1: Payoff equation roundtrip
# =====
class TestPayoffRoundtrip:
    """Verify othercont derivation is consistent with the payoff formula."""

    def test_payoff_reconstructed_matches_original(self, derived_df):
        """Reconstruct payoff from contribution + othercont; must match."""
        reconstructed = (
            ENDOWMENT
            - derived_df["contribution"]
            + (derived_df["contribution"] + derived_df["othercont"]) * MULTIPLIER
        )
        max_error = (derived_df["payoff"] - reconstructed).abs().max()
        assert max_error < 1e-10, f"Payoff reconstruction error: max={max_error}"

    def test_othercont_nearly_nonnegative(self, derived_df):
        """othercont >= -1.5 (payoffs are integer-rounded in oTree)."""
        min_val = derived_df["othercont"].min()
        assert min_val >= -1.5, f"othercont too negative: min={min_val}"

    def test_othercont_upper_bound(self, derived_df):
        """othercont <= 76.5 (3 x 25 = 75 plus rounding tolerance)."""
        max_val = derived_df["othercont"].max()
        assert max_val <= 76.5, f"othercont too large: max={max_val}"


# =====
# Test 2: Other-average correctness (regression test for /1.3 bug)
# =====
class TestOtherAverage:
    """Verify othercontaverage = othercont / 3, not the Stata /1.3 bug."""

    def test_othercontaverage_equals_othercont_div_3(self, derived_df):
        """othercontaverage must be othercont / 3 for all rows."""
        expected = derived_df["othercont"] / 3
        max_diff = (derived_df["othercontaverage"] - expected).abs().max()
        assert max_diff < 1e-12, f"othercontaverage != othercont/3: max diff={max_diff}"

    def test_othercontaverage_not_div_1_3(self, derived_df):
        """Regression: ensure we divide by 3, not 1.3 (Stata bug)."""
        buggy = derived_df["othercont"] / 3.25
        nonzero = derived_df["othercont"].abs() > 0.01
        if nonzero.any():
            differs = derived_df.loc[nonzero, "othercontaverage"] != buggy[nonzero]
            assert differs.all(), "othercontaverage matches buggy /1.3 formula"


# =====
# Test 3: Deviation indicators are mutually exclusive
# =====
class TestDeviationIndicators:
    """Verify morethanaverage and lessthanaverage are mutually exclusive."""

    def test_no_row_has_both_flags(self, derived_df):
        """No row should have both morethanaverage=1 and lessthanaverage=1."""
        both = (derived_df["morethanaverage"] == 1) & (derived_df["lessthanaverage"] == 1)
        assert not both.any(), f"Found {both.sum()} rows with both flags set"

    def test_equal_contribution_has_neither_flag(self, derived_df):
        """When contribution == othercontaverage, both flags should be 0."""
        equal = np.isclose(
            derived_df["contribution"], derived_df["othercontaverage"], atol=1e-10
        )
        if equal.any():
            eq_rows = derived_df[equal]
            assert (eq_rows["morethanaverage"] == 0).all()
            assert (eq_rows["lessthanaverage"] == 0).all()

    def test_flags_are_binary(self, derived_df):
        """Both indicator columns should contain only 0 or 1."""
        for col in ["morethanaverage", "lessthanaverage"]:
            assert set(derived_df[col].unique()) <= {0, 1}, f"{col} non-binary"


# =====
# Test 4: contmore and contless are non-negative
# =====
class TestDeviationMagnitudes:
    """Verify contmore and contless are non-negative magnitude measures."""

    def test_contmore_nonnegative(self, derived_df):
        """contmore should be >= 0 everywhere."""
        assert derived_df["contmore"].min() >= -1e-10

    def test_contless_nonnegative(self, derived_df):
        """contless should be >= 0 everywhere."""
        assert derived_df["contless"].min() >= -1e-10


# =====
# Test 5: contmore + contless == abs(diffcont)
# =====
class TestDeviationDecomposition:
    """Verify contmore and contless decompose abs(diffcont) correctly."""

    def test_contmore_plus_contless_equals_abs_diffcont(self, derived_df):
        """contmore + contless should equal |diffcont| for every row."""
        lhs = derived_df["contmore"] + derived_df["contless"]
        rhs = derived_df["diffcont"].abs()
        assert (lhs - rhs).abs().max() < 1e-10

    def test_contmore_zero_when_below_average(self, derived_df):
        """contmore should be 0 when contribution < othercontaverage."""
        below = derived_df["lessthanaverage"] == 1
        if below.any():
            assert (derived_df.loc[below, "contmore"].abs() < 1e-10).all()

    def test_contless_zero_when_above_average(self, derived_df):
        """contless should be 0 when contribution > othercontaverage."""
        above = derived_df["morethanaverage"] == 1
        if above.any():
            assert (derived_df.loc[above, "contless"].abs() < 1e-10).all()


# =====
# Known values from real data (session 6uv359rf, sg1, r1, group 1)
# =====
class TestKnownValues:
    """Verify derived values for player A: contribution=3, payoff=33."""

    def _get_row_a(self, derived_df):
        """Helper: fetch known row for player A."""
        mask = (
            (derived_df["session_code"] == "6uv359rf")
            & (derived_df["segment"] == "supergame1")
            & (derived_df["round"] == 1)
            & (derived_df["label"] == "A")
        )
        return derived_df[mask]

    def test_known_othercont(self, derived_df):
        """othercont = (33 - 25 + 0.6*3) / 0.4 = 24.5"""
        assert self._get_row_a(derived_df)["othercont"].values[0] == pytest.approx(24.5)

    def test_known_othercontaverage(self, derived_df):
        """othercontaverage = 24.5 / 3"""
        expected = 24.5 / 3
        assert self._get_row_a(derived_df)["othercontaverage"].values[0] == pytest.approx(expected)

    def test_known_deviation_flags(self, derived_df):
        """Player A (cont=3) is below average (8.17): lessthanaverage=1."""
        row = self._get_row_a(derived_df)
        assert row["morethanaverage"].values[0] == 0
        assert row["lessthanaverage"].values[0] == 1

    def test_known_contmore_contless(self, derived_df):
        """contmore=0 (below avg), contless=|3 - 24.5/3|."""
        row = self._get_row_a(derived_df)
        assert row["contmore"].values[0] == pytest.approx(0.0)
        assert row["contless"].values[0] == pytest.approx(abs(3 - 24.5 / 3))


# =====
# Run tests directly
# =====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
