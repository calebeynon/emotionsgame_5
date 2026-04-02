"""
Purpose: Tests for Issue #46 sentiment-contribution regression.
Author: Claude Code
Date: 2026-04-01
"""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
WORKING_DIR = Path(__file__).resolve().parent.parent
SENTIMENT_CSV = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "sentiment_scores.csv"
R_SCRIPT = ANALYSIS_DIR / "sentiment_contribution_regression.R"
OUTPUT_TEX = TABLE_DIR / "sentiment_contribution_regression.tex"

EXPECTED_ROWS = 2298
MIN_TEX_SIZE = 200


def run_r_script(script_path, timeout=120):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


# =====
# Precondition checks
# =====
class TestPreconditions:
    """Verify input data and R script exist."""

    def test_sentiment_csv_exists(self):
        assert SENTIMENT_CSV.exists()

    def test_r_script_exists(self):
        assert R_SCRIPT.exists()


# =====
# Input data validation
# =====
@pytest.fixture()
def sentiment_df():
    """Load sentiment_scores.csv."""
    return pd.read_csv(SENTIMENT_CSV)


class TestSentimentScoresCSV:
    """Validate input data schema and value ranges."""

    def test_row_count(self, sentiment_df):
        assert len(sentiment_df) == EXPECTED_ROWS

    def test_required_columns(self, sentiment_df):
        required = [
            "session_code", "treatment", "segment", "round", "group",
            "label", "contribution", "message_count", "sentiment_compound_mean",
        ]
        missing = set(required) - set(sentiment_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_sentiment_range(self, sentiment_df):
        valid = sentiment_df["sentiment_compound_mean"].dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_contribution_range(self, sentiment_df):
        valid = sentiment_df["contribution"].dropna()
        assert (valid >= 0).all() and (valid <= 25).all()

    def test_treatment_values(self, sentiment_df):
        assert set(sentiment_df["treatment"].unique()) == {1, 2}

    def test_segment_values(self, sentiment_df):
        expected = {f"supergame{i}" for i in range(1, 6)}
        assert set(sentiment_df["segment"].unique()) == expected

    def test_no_nan_contribution(self, sentiment_df):
        assert sentiment_df["contribution"].notna().all()

    def test_cluster_id_construction(self, sentiment_df):
        """Verify cluster_id can be constructed and produces reasonable unique count."""
        cluster_ids = (
            sentiment_df["session_code"].astype(str) + "_" +
            sentiment_df["segment"].astype(str) + "_" +
            sentiment_df["group"].astype(str)
        )
        n_clusters = cluster_ids.nunique()
        # With multiple sessions x 5 supergames x 4 groups, expect dozens of clusters
        assert n_clusters >= 20, f"Only {n_clusters} unique clusters"


# =====
# R script integration tests
# =====
class TestRegressionOutputTable:
    """Test that the R script runs and produces valid output."""

    def test_r_script_runs_successfully(self):
        result = run_r_script(R_SCRIPT)
        assert result.returncode == 0, f"R script failed:\n{result.stderr}"

    def test_tex_file_exists(self):
        assert OUTPUT_TEX.exists(), f"Expected output: {OUTPUT_TEX}"

    def test_tex_file_minimum_size(self):
        assert OUTPUT_TEX.stat().st_size >= MIN_TEX_SIZE

    def test_tex_contains_sentiment(self):
        content = OUTPUT_TEX.read_text()
        assert "Sentiment" in content

    def test_tex_contains_treatment(self):
        content = OUTPUT_TEX.read_text()
        assert "Treatment" in content

    def test_tex_contains_message_count(self):
        content = OUTPUT_TEX.read_text()
        assert "Message Count" in content

    def test_tex_has_two_model_columns(self):
        """Verify the table has both Baseline and With Message Count columns."""
        content = OUTPUT_TEX.read_text()
        assert "Baseline" in content
        assert "With Message Count" in content
