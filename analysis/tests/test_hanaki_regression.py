"""
Tests for Task 5: regress_hanaki_projections.R output LaTeX table integrity.

Validates that output tables exist, have correct structure, contain expected
coefficients, and report correct sample sizes.

Author: pytest-test-writer
Date: 2026-03-26
"""

import re
from pathlib import Path

import pytest

# FILE PATHS
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "tables"
INV_TABLE = OUTPUT_DIR / "hanaki_external_validation_inv.tex"
PAIR_TABLE = OUTPUT_DIR / "hanaki_external_validation_pair.tex"

# Expected sample size from projections CSV
EXPECTED_N = "7,846"

# Expected variable labels in output
EXPECTED_VARS = [
    "Cooperative", "Promise", "Homogeneity", "Round Liar", "Cumulative Liar",
    "Log(Word Count)",
]


# =====
# Fixtures
# =====
@pytest.fixture
def inv_tex():
    """Read individual investment table as text."""
    if not INV_TABLE.exists():
        pytest.skip(f"Not found: {INV_TABLE}")
    return INV_TABLE.read_text()


@pytest.fixture
def pair_tex():
    """Read pair average investment table as text."""
    if not PAIR_TABLE.exists():
        pytest.skip(f"Not found: {PAIR_TABLE}")
    return PAIR_TABLE.read_text()


# =====
# File existence
# =====
class TestRegressionFilesExist:
    """Verify both LaTeX table files exist."""

    def test_inv_table_exists(self):
        """Individual investment table must exist."""
        assert INV_TABLE.exists()

    def test_pair_table_exists(self):
        """Pair average investment table must exist."""
        assert PAIR_TABLE.exists()


# =====
# LaTeX structure
# =====
class TestRegressionTableStructure:
    """Verify LaTeX tables have correct structure."""

    def test_inv_is_valid_latex_tabular(self, inv_tex):
        """Inv table should have begin/end tabular environment."""
        assert r"\begin{tabular}" in inv_tex
        assert r"\end{tabular}" in inv_tex

    def test_pair_is_valid_latex_tabular(self, pair_tex):
        """Pair table should have begin/end tabular environment."""
        assert r"\begin{tabular}" in pair_tex
        assert r"\end{tabular}" in pair_tex

    def test_inv_has_6_model_columns(self, inv_tex):
        """Inv table should have 6 columns (5 univariate + 1 multivariate)."""
        assert r"\begin{tabular}{lcccccc}" in inv_tex

    def test_pair_has_6_model_columns(self, pair_tex):
        """Pair table should have 6 columns."""
        assert r"\begin{tabular}{lcccccc}" in pair_tex

    def test_inv_dependent_variable_label(self, inv_tex):
        """Inv table DV should be 'Inv'."""
        assert r"\multicolumn{6}{c}{Inv}" in inv_tex

    def test_pair_dependent_variable_label(self, pair_tex):
        """Pair table DV should be 'PairAveCho'."""
        assert r"\multicolumn{6}{c}{PairAveCho}" in pair_tex


# =====
# Variable names and fixed effects
# =====
class TestRegressionContent:
    """Verify tables contain expected variables and FEs."""

    @pytest.mark.parametrize("var", EXPECTED_VARS)
    def test_inv_table_has_variable(self, inv_tex, var):
        """Each projection variable should appear in the inv table."""
        assert var in inv_tex

    @pytest.mark.parametrize("var", EXPECTED_VARS)
    def test_pair_table_has_variable(self, pair_tex, var):
        """Each projection variable should appear in the pair table."""
        assert var in pair_tex

    def test_inv_has_session_fe(self, inv_tex):
        """Session fixed effects should be marked 'Yes'."""
        assert r"session\_file" in inv_tex

    def test_inv_has_period_fe(self, inv_tex):
        """Period fixed effects should be marked 'Yes'."""
        assert "period" in inv_tex

    def test_inv_has_clustered_se(self, inv_tex):
        """Clustered standard errors (pair_id) should be noted."""
        assert r"pair\_id" in inv_tex


# =====
# Sample sizes
# =====
class TestRegressionSampleSize:
    """Verify observations count matches expected N."""

    def test_inv_observations_7846(self, inv_tex):
        """All Inv models should report 7,846 observations."""
        obs_line = [l for l in inv_tex.splitlines() if "Observations" in l]
        assert len(obs_line) == 1
        counts = re.findall(r"[\d,]+", obs_line[0])
        for c in counts:
            assert c == EXPECTED_N

    def test_pair_observations_7846(self, pair_tex):
        """All Pair models should report 7,846 observations."""
        obs_line = [l for l in pair_tex.splitlines() if "Observations" in l]
        assert len(obs_line) == 1
        counts = re.findall(r"[\d,]+", obs_line[0])
        for c in counts:
            assert c == EXPECTED_N


# =====
# Regression coefficient validation
# =====
class TestRegressionCoefficients:
    """Validate specific regression coefficients from verified output."""

    def test_inv_promise_significant(self, inv_tex):
        """Promise should be significant in the Inv univariate model."""
        lines = inv_tex.splitlines()
        promise_lines = [l for l in lines if l.strip().startswith("Promise")]
        assert any("***" in l or "**" in l for l in promise_lines)

    def test_inv_round_liar_significant_negative(self, inv_tex):
        """Round Liar should be significant and negative in univariate."""
        lines = inv_tex.splitlines()
        liar_lines = [l for l in lines if "Round Liar" in l]
        assert any("-" in l for l in liar_lines)
        assert any("***" in l or "**" in l for l in liar_lines)

    def test_inv_r2_reasonable(self, inv_tex):
        """R-squared should be between 0.05 and 0.25 for Inv models."""
        r2_line = [l for l in inv_tex.splitlines() if "R$^2$" in l]
        assert len(r2_line) >= 1
        r2_values = re.findall(r"0\.\d{4,5}", r2_line[0])
        for val in r2_values:
            r2 = float(val)
            assert 0.05 < r2 < 0.25, f"Unusual R2: {r2}"

    def test_pair_promise_significant(self, pair_tex):
        """Promise should be significant in Pair univariate model."""
        lines = pair_tex.splitlines()
        promise_lines = [l for l in lines if l.strip().startswith("Promise")]
        assert any("***" in l or "**" in l for l in promise_lines)
