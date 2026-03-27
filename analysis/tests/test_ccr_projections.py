"""
Tests for CCR projection outputs and regression tables (Task 5).

Validates embedding projections onto 5 direction vectors and the
LaTeX regression tables produced by the R script.

Known data facts (verified from output inspection):
- Panel: 34,800 rows (full data.dta), 300 NaN projections (session 55 Green)
- Group: 116 rows, 1 NaN projection row (group_key 55_0, empty chat)
- 5 projection columns: proj_{cooperative,promise,homogeneity,round_liar,cumulative_liar}
- 6 LaTeX tables produced by external_validation_ccr.R

Author: pytest-test-writer
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# FILE PATHS
EXTERNAL_DIR = Path(__file__).parent.parent / "datastore" / "derived" / "external"
PROJ_PANEL_FILE = EXTERNAL_DIR / "ccr_projections_panel.parquet"
PROJ_GROUP_FILE = EXTERNAL_DIR / "ccr_projections_group.parquet"
TABLES_DIR = Path(__file__).parent.parent / "output" / "tables"
R_SCRIPT = Path(__file__).parent.parent / "analysis" / "external_validation_ccr.R"

# Known-good values
EXPECTED_PANEL_ROWS = 34800
EXPECTED_GROUP_ROWS = 116
EXPECTED_NAN_GROUP = "55_0"
EXPECTED_NAN_PANEL_ROWS = 300  # 6 subjects x 50 periods
EXPECTED_PROJ_COLS = [
    "proj_cooperative", "proj_promise", "proj_homogeneity",
    "proj_round_liar", "proj_cumulative_liar",
]
EXPECTED_LATEX_FILES = [
    "external_validation_ccr.tex",
    "external_validation_ccr_by_site.tex",
    "external_validation_ccr_detailed.tex",
    "external_validation_ccr_group.tex",
    "external_validation_ccr_interactions.tex",
    "external_validation_ccr_period1.tex",
]


# =====
# Fixtures
# =====
@pytest.fixture
def panel_df():
    """Load the CCR projections panel parquet."""
    if not PROJ_PANEL_FILE.exists():
        pytest.skip(f"Panel not found: {PROJ_PANEL_FILE}")
    return pd.read_parquet(PROJ_PANEL_FILE)


@pytest.fixture
def group_df():
    """Load the CCR projections group parquet."""
    if not PROJ_GROUP_FILE.exists():
        pytest.skip(f"Group not found: {PROJ_GROUP_FILE}")
    return pd.read_parquet(PROJ_GROUP_FILE)


# =====
# Panel schema and dimensions
# =====
class TestPanelSchema:
    """Verify panel projection parquet has correct structure."""

    def test_file_exists(self):
        """Panel parquet must exist."""
        assert PROJ_PANEL_FILE.exists()

    def test_panel_row_count(self, panel_df):
        """Panel should have 34,800 rows (full data.dta)."""
        assert len(panel_df) == EXPECTED_PANEL_ROWS

    def test_has_all_projection_columns(self, panel_df):
        """Must have all 5 projection columns."""
        for col in EXPECTED_PROJ_COLS:
            assert col in panel_df.columns, f"Missing: {col}"

    def test_has_effort_column(self, panel_df):
        """Must have effort outcome column."""
        assert "effort" in panel_df.columns

    def test_has_merge_keys(self, panel_df):
        """Must have group_key and session columns."""
        for col in ["group_key", "session", "subject", "period"]:
            assert col in panel_df.columns, f"Missing: {col}"

    def test_has_treatment_indicators(self, panel_df):
        """Must have ingroup, commonknow, and run columns."""
        for col in ["ingroup", "commonknow", "run"]:
            assert col in panel_df.columns, f"Missing: {col}"


# =====
# Panel projection values
# =====
class TestPanelProjectionValues:
    """Verify projection values in the panel are valid."""

    def test_nan_count_is_300(self, panel_df):
        """Exactly 300 rows should have NaN projections (session 55 Green)."""
        nan_count = panel_df[EXPECTED_PROJ_COLS[0]].isna().sum()
        assert nan_count == EXPECTED_NAN_PANEL_ROWS

    def test_nan_rows_are_group_55_0(self, panel_df):
        """NaN projections should only be in group 55_0."""
        nan_mask = panel_df[EXPECTED_PROJ_COLS[0]].isna()
        nan_keys = panel_df.loc[nan_mask, "group_key"].unique()
        assert list(nan_keys) == [EXPECTED_NAN_GROUP]

    def test_all_proj_cols_nan_together(self, panel_df):
        """All 5 projection columns should be NaN for the same rows."""
        nan_masks = [panel_df[c].isna() for c in EXPECTED_PROJ_COLS]
        for mask in nan_masks[1:]:
            assert (mask == nan_masks[0]).all()

    def test_no_inf_in_projections(self, panel_df):
        """No infinite values in projection columns."""
        for col in EXPECTED_PROJ_COLS:
            valid = panel_df[col].dropna()
            assert not np.isinf(valid).any(), f"Inf in {col}"

    def test_valid_projections_are_bounded(self, panel_df):
        """Valid projection values should be in a reasonable range."""
        for col in EXPECTED_PROJ_COLS:
            valid = panel_df[col].dropna()
            assert valid.min() > -1.0, f"{col} min too low: {valid.min()}"
            assert valid.max() < 1.0, f"{col} max too high: {valid.max()}"


# =====
# Group schema and dimensions
# =====
class TestGroupSchema:
    """Verify group projection parquet has correct structure."""

    def test_file_exists(self):
        """Group parquet must exist."""
        assert PROJ_GROUP_FILE.exists()

    def test_group_row_count(self, group_df):
        """Group should have 116 rows (58 sessions x 2 colors)."""
        assert len(group_df) == EXPECTED_GROUP_ROWS

    def test_has_all_projection_columns(self, group_df):
        """Must have all 5 projection columns."""
        for col in EXPECTED_PROJ_COLS:
            assert col in group_df.columns, f"Missing: {col}"

    def test_has_effort_aggregates(self, group_df):
        """Must have effort aggregate columns."""
        assert "mean_effort" in group_df.columns

    def test_has_merge_keys(self, group_df):
        """Must have group_key, session, and red columns."""
        for col in ["group_key", "session", "red"]:
            assert col in group_df.columns, f"Missing: {col}"

    def test_group_key_unique(self, group_df):
        """Each group_key must be unique."""
        assert group_df["group_key"].nunique() == len(group_df)


# =====
# Group projection values
# =====
class TestGroupProjectionValues:
    """Verify projection values in the group cross-section."""

    def test_one_nan_group(self, group_df):
        """Exactly 1 group should have NaN projections (55_0)."""
        nan_count = group_df[EXPECTED_PROJ_COLS[0]].isna().sum()
        assert nan_count == 1

    def test_nan_group_is_55_0(self, group_df):
        """NaN projection group should be 55_0."""
        nan_mask = group_df[EXPECTED_PROJ_COLS[0]].isna()
        nan_key = group_df.loc[nan_mask, "group_key"].iloc[0]
        assert nan_key == EXPECTED_NAN_GROUP

    def test_no_inf_in_projections(self, group_df):
        """No infinite values in projection columns."""
        for col in EXPECTED_PROJ_COLS:
            valid = group_df[col].dropna()
            assert not np.isinf(valid).any(), f"Inf in {col}"

    def test_effort_range_valid(self, group_df):
        """Mean effort should be in [110, 170]."""
        assert group_df["mean_effort"].min() >= 110
        assert group_df["mean_effort"].max() <= 170


# =====
# Cross-validation between panel and group
# =====
class TestPanelGroupConsistency:
    """Cross-validate panel and group projections."""

    def test_group_keys_match(self, panel_df, group_df):
        """Panel group_keys should be a subset of group group_keys."""
        panel_keys = set(panel_df["group_key"].unique())
        group_keys = set(group_df["group_key"].unique())
        missing = panel_keys - group_keys
        assert not missing, f"Panel keys not in group: {missing}"

    def test_session_counts_match(self, panel_df, group_df):
        """Both should cover 58 sessions."""
        assert panel_df["session"].nunique() == 58
        assert group_df["session"].nunique() == 58


# =====
# LaTeX regression tables
# =====
class TestLatexTables:
    """Verify regression tables were produced by the R script."""

    @pytest.mark.parametrize("filename", EXPECTED_LATEX_FILES)
    def test_latex_file_exists(self, filename):
        """Each expected LaTeX table file must exist."""
        path = TABLES_DIR / filename
        assert path.exists(), f"Missing: {path}"

    @pytest.mark.parametrize("filename", EXPECTED_LATEX_FILES)
    def test_latex_file_nonempty(self, filename):
        """LaTeX files must have content."""
        path = TABLES_DIR / filename
        if not path.exists():
            pytest.skip(f"Not found: {path}")
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("filename", EXPECTED_LATEX_FILES)
    def test_latex_has_tabular(self, filename):
        """LaTeX files should contain tabular or table environment."""
        path = TABLES_DIR / filename
        if not path.exists():
            pytest.skip(f"Not found: {path}")
        text = path.read_text()
        has_table = "tabular" in text or "\\begin{table}" in text
        assert has_table, f"{filename} missing tabular/table environment"

    def test_r_script_exists(self):
        """R regression script must exist."""
        assert R_SCRIPT.exists()
