"""
Purpose: Tests for Issue #39 emotion-sentiment analysis R scripts.
         Verifies that R scripts run without errors and produce valid outputs.
         Covers issue_39_common.R, issue_39_plot_dotplots.R,
         and issue_39_regression_decomposition.R.
Author: Claude Code
Date: 2026-03-14
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "merged_panel.csv"
BEHAVIOR_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "behavior_classifications.csv"
WORKING_DIR = Path(__file__).resolve().parent.parent

# R SCRIPT PATHS
COMMON_SCRIPT = ANALYSIS_DIR / "issue_39_common.R"
DOTPLOT_SCRIPT = ANALYSIS_DIR / "issue_39_plot_dotplots.R"
DECOMPOSITION_SCRIPT = ANALYSIS_DIR / "issue_39_regression_decomposition.R"

# EXPECTED OUTPUT FILES
DOTPLOT_FILES = [
    "emotion_sentiment_gap_by_cooperative_state.png",
    "emotion_sentiment_gap_by_liar_status.png",
    "emotion_sentiment_gap_by_sucker_status.png",
]

DECOMPOSITION_TABLES = [
    "emotion_sentiment_orthogonal.tex",
    "emotion_sentiment_deception.tex",
    "emotion_sentiment_deception_descriptive.tex",
]

# KNOWN DATA PROPERTIES
EXPECTED_CONTRIBUTE_ROWS = 3520
EXPECTED_SESSIONS = 10
EXPECTED_LABELS = 16
MIN_TEX_SIZE = 200


# =====
# Helpers
# =====
def run_r_script(script_path, timeout=120):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def run_r_code(code, timeout=30):
    """Run inline R code and return the completed process."""
    return subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


# =====
# Precondition checks
# =====
class TestPreconditions:
    """Verify data files and R scripts exist."""

    def test_data_file_exists(self):
        assert DATA_FILE.exists()

    def test_behavior_file_exists(self):
        assert BEHAVIOR_FILE.exists()

    def test_common_script_exists(self):
        assert COMMON_SCRIPT.exists()

    def test_dotplot_script_exists(self):
        assert DOTPLOT_SCRIPT.exists()


# =====
# issue_39_common.R tests
# =====
class TestCommonUtilities:
    """Test shared utilities file."""

    def test_sources_without_error(self):
        result = run_r_code('source("analysis/issue_39_common.R")')
        assert result.returncode == 0, result.stderr

    def test_load_contribute_data(self):
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
cat("ROWS:", nrow(dt), "\\n")
cat("HAS_CLUSTER:", "cluster_id" %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"ROWS: {EXPECTED_CONTRIBUTE_ROWS}" in result.stdout

    def test_merge_behavior_classifications(self):
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- merge_behavior_classifications(dt)
cat("HAS_LIAR:", "is_liar_20" %in% names(dt), "\\n")
cat("HAS_SUCKER:", "is_sucker_20" %in% names(dt), "\\n")
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_LIAR: TRUE" in result.stdout
        assert "HAS_SUCKER: TRUE" in result.stdout
        assert f"ROWS: {EXPECTED_CONTRIBUTE_ROWS}" in result.stdout

    def test_compute_zscores(self):
        result = run_r_code("""
source("analysis/issue_39_common.R")
dt <- load_contribute_data()
dt <- compute_zscores(dt)
cat("HAS_VALZ:", "valence_z" %in% names(dt), "\\n")
cat("HAS_CMPZ:", "compound_z" %in% names(dt), "\\n")
cat("HAS_GAP:", "zscore_gap" %in% names(dt), "\\n")
complete <- dt[!is.na(valence_z) & !is.na(compound_z)]
cat("VALMEAN:", round(mean(complete$valence_z), 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_VALZ: TRUE" in result.stdout
        assert "HAS_GAP: TRUE" in result.stdout
        assert "VALMEAN: 0" in result.stdout


# =====
# issue_39_plot_dotplots.R tests
# =====
class TestDotPlots:
    """Test dot plot script execution and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(DOTPLOT_SCRIPT)
        assert result.returncode == 0, (
            f"Failed (exit {result.returncode}):\n"
            f"stderr: {result.stderr}"
        )

    @pytest.mark.parametrize("filename", DOTPLOT_FILES)
    def test_plot_file_exists(self, filename):
        assert (PLOT_DIR / filename).exists()

    @pytest.mark.parametrize("filename", DOTPLOT_FILES)
    def test_plot_file_nonempty(self, filename):
        path = PLOT_DIR / filename
        assert path.exists()
        assert path.stat().st_size > 10_000


# =====
# Regression tests
# =====
class TestDecompositionRegression:
    """Test decomposition and deception regressions."""

    def test_script_runs(self):
        result = run_r_script(DECOMPOSITION_SCRIPT, timeout=180)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("filename", DECOMPOSITION_TABLES)
    def test_table_exists(self, filename):
        path = TABLE_DIR / filename
        assert path.exists()
        assert path.stat().st_size >= MIN_TEX_SIZE


# =====
# Data integrity tests
# =====
class TestDataIntegrity:
    """Verify input data assumptions."""

    @pytest.fixture()
    def contribute_df(self):
        df = pd.read_csv(DATA_FILE)
        return df[df["page_type"] == "Contribute"]

    def test_contribute_row_count(self, contribute_df):
        assert len(contribute_df) == EXPECTED_CONTRIBUTE_ROWS

    def test_contribution_range(self, contribute_df):
        valid = contribute_df["contribution"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 25

    def test_sentiment_compound_range(self, contribute_df):
        valid = contribute_df["sentiment_compound_mean"].dropna()
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0
