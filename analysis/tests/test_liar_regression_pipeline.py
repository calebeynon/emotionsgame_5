"""
Tests for liar regression input and plot data pipeline (issue #53).

Validates the regression data pipeline (issue_53_liar_regression.R),
the plot/summary pipeline (issue_53_liar_plots.py), and output files.
All expected values manually verified from behavior_classifications.csv
and merged_panel.csv.

Author: Claude Code
Date: 2026-04-09
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
BEHAVIOR_CSV = DERIVED_DIR / "behavior_classifications.csv"
LIAR_BUCKETS_CSV = DERIVED_DIR / "liar_buckets.csv"
MERGED_PANEL = DERIVED_DIR / "merged_panel.csv"
PLOTS_DIR = Path(__file__).parent.parent / "output" / "plots"
TABLES_DIR = Path(__file__).parent.parent / "output" / "summary_statistics"
REGRESSION_TEX = Path(__file__).parent.parent / "output" / "tables" / "liar_conditional_probability.tex"

# REGRESSION CONSTANTS (manually verified)
EXPECTED_REGRESSION_OBS = 2720
EXPECTED_REGRESSION_PARTICIPANTS = 160
EXPECTED_TRANSITIONS = {(0, 0): 2563, (0, 1): 83, (1, 0): 63, (1, 1): 11}

# PLOT CONSTANTS (manually verified)
EXPECTED_PLOT_ROWS = 2720
EXPECTED_PLOT_BUCKETS = {"never": 1785, "one_time": 612, "moderate": 238, "severe": 85}
EXPECTED_SENTIMENT_ROWS = 2298
EXPECTED_EMOTION_ROWS = 2160


# =====
# Fixtures
# =====
@pytest.fixture
def behavior_df():
    if not BEHAVIOR_CSV.exists():
        pytest.skip(f"Not found: {BEHAVIOR_CSV}")
    return pd.read_csv(BEHAVIOR_CSV)


@pytest.fixture
def regression_data(behavior_df):
    """Reproduce the regression input from issue_53_liar_regression.R."""
    df = behavior_df.copy()
    df["lied_this_round_20"] = df["lied_this_round_20"].fillna(False)
    df["lied"] = df["lied_this_round_20"].astype(int)
    df = df.sort_values(["session_code", "label", "segment", "round"])
    df["lied_prev"] = df.groupby(["session_code", "label", "segment"])["lied"].shift(1)
    return df.dropna(subset=["lied_prev"])


@pytest.fixture
def plot_data():
    """Reproduce the plot data pipeline from issue_53_liar_plots.py."""
    if not MERGED_PANEL.exists() or not LIAR_BUCKETS_CSV.exists():
        pytest.skip("Missing input files")
    panel = pd.read_csv(MERGED_PANEL)
    buckets = pd.read_csv(LIAR_BUCKETS_CSV, usecols=["session_code", "label", "liar_bucket"])
    merged = panel.merge(buckets, on=["session_code", "label"], how="left")
    results = merged[merged["page_type"] == "Results"].copy()
    return results[results["round"] > 1]


# =====
# Regression input tests
# =====
class TestRegressionInput:
    def test_observation_count(self, regression_data):
        assert len(regression_data) == EXPECTED_REGRESSION_OBS

    def test_participant_count(self, regression_data):
        n = regression_data.groupby(["session_code", "label"]).ngroups
        assert n == EXPECTED_REGRESSION_PARTICIPANTS

    def test_transition_matrix(self, regression_data):
        ct = pd.crosstab(
            regression_data["lied_prev"].astype(int),
            regression_data["lied"].astype(int),
        )
        for (prev, curr), expected in EXPECTED_TRANSITIONS.items():
            assert ct.loc[prev, curr] == expected

    def test_conditional_probability_ratio(self, regression_data):
        """P(lie|prev_lie) / P(lie|prev_no_lie) ~ 4.74x."""
        ct = pd.crosstab(
            regression_data["lied_prev"].astype(int),
            regression_data["lied"].astype(int),
            margins=True,
        )
        ratio = (ct.loc[1, 1] / ct.loc[1, "All"]) / (ct.loc[0, 1] / ct.loc[0, "All"])
        assert abs(ratio - 4.74) < 0.1

    def test_no_first_rounds(self, regression_data):
        first_rounds = regression_data.groupby(
            ["session_code", "label", "segment"]
        )["round"].min()
        assert (first_rounds > 1).all()

    def test_iiu3xixz_L_transitions(self, regression_data):
        """Most prolific liar: 17 obs, transition counts (0->0):5 (0->1):5 (1->0):3 (1->1):4."""
        sub = regression_data[
            (regression_data["session_code"] == "iiu3xixz")
            & (regression_data["label"] == "L")
        ]
        assert len(sub) == 17
        ct = pd.crosstab(sub["lied_prev"].astype(int), sub["lied"].astype(int))
        assert ct.loc[0, 0] == 5
        assert ct.loc[0, 1] == 5
        assert ct.loc[1, 0] == 3
        assert ct.loc[1, 1] == 4


# =====
# Plot data pipeline tests
# =====
class TestPlotDataPipeline:
    def test_row_count(self, plot_data):
        assert len(plot_data) == EXPECTED_PLOT_ROWS

    def test_bucket_distribution(self, plot_data):
        actual = plot_data["liar_bucket"].value_counts().to_dict()
        for bucket, expected in EXPECTED_PLOT_BUCKETS.items():
            assert actual.get(bucket, 0) == expected

    def test_no_null_buckets(self, plot_data):
        assert plot_data["liar_bucket"].isna().sum() == 0

    def test_sentiment_availability(self, plot_data):
        assert plot_data["sentiment_compound_mean"].notna().sum() == EXPECTED_SENTIMENT_ROWS

    def test_emotion_availability(self, plot_data):
        for col in ["emotion_anger", "emotion_contempt", "emotion_joy",
                     "emotion_sadness", "emotion_surprise"]:
            assert plot_data[col].notna().sum() == EXPECTED_EMOTION_ROWS

    def test_only_results_pages(self, plot_data):
        assert (plot_data["page_type"] == "Results").all()

    def test_no_round_1(self, plot_data):
        assert (plot_data["round"] > 1).all()


# =====
# Summary table spot checks
# =====
class TestSummaryTable:
    def test_sentiment_never_mean(self, plot_data):
        """Sentiment mean for 'never' bucket: 0.125."""
        val = plot_data[plot_data["liar_bucket"] == "never"]["sentiment_compound_mean"].dropna().mean()
        assert abs(val - 0.125) < 0.001

    def test_joy_severe_mean(self, plot_data):
        """Joy mean for 'severe' bucket: 14.562."""
        val = plot_data[plot_data["liar_bucket"] == "severe"]["emotion_joy"].dropna().mean()
        assert abs(val - 14.562) < 0.001


# =====
# Output file tests
# =====
class TestOutputFiles:
    def test_sentiment_plot_exists(self):
        assert (PLOTS_DIR / "sentiment_by_liar_bucket.png").exists()

    def test_emotions_plot_exists(self):
        assert (PLOTS_DIR / "emotions_by_liar_bucket.png").exists()

    def test_summary_tex_valid(self):
        path = TABLES_DIR / "liar_bucket_summary.tex"
        if not path.exists():
            pytest.skip("Not generated")
        content = path.read_text()
        for label in ["Never", "One-Time", "Moderate", "Severe",
                       "Sentiment", "Anger", "Contempt", "Joy", "Sadness", "Surprise"]:
            assert label in content, f"Missing: {label}"

    def test_regression_tex_key_values(self):
        if not REGRESSION_TEX.exists():
            pytest.skip("Not generated")
        content = REGRESSION_TEX.read_text()
        assert "1.448" in content, "Missing coefficient"
        assert "2,720" in content, "Missing obs count"
        assert "Lied Previous Round" in content
        assert "Female" in content
        assert "Treatment" in content
