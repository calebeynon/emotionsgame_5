"""
Tests for liar bucket classification (issue #53).

Validates liar_buckets.py output CSV and issue_53_liar_plots.py outputs.
Grounded in real data from behavior_classifications.csv.

Author: pytest-test-writer
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
PLOTS_DIR = Path(__file__).parent.parent / "output" / "plots"
TABLES_DIR = Path(__file__).parent.parent / "output" / "summary_statistics"

SENTIMENT_PLOT = PLOTS_DIR / "sentiment_by_liar_bucket.png"
EMOTIONS_PLOT = PLOTS_DIR / "emotions_by_liar_bucket.png"
SUMMARY_TEX = TABLES_DIR / "liar_bucket_summary.tex"

# EXPECTED VALUES (derived from real behavior_classifications.csv)
EXPECTED_ROW_COUNT = 160
EXPECTED_SESSION_COUNT = 10
EXPECTED_TOTAL_LIES = 94
VALID_BUCKETS = {"never", "one_time", "moderate", "severe"}
EXPECTED_BUCKET_COUNTS = {
    "never": 105,
    "one_time": 36,
    "moderate": 14,
    "severe": 5,
}
EXPECTED_COLUMNS = [
    "session_code", "treatment", "label", "participant_id",
    "lie_count", "liar_bucket",
]
EXPECTED_SESSIONS = sorted([
    "6sdkxl2q", "6ucza025", "6uv359rf", "iiu3xixz", "irrzlgk2",
    "j3ki5tli", "r5dj4yfl", "sa7mprty", "sylq2syi", "umbzdj98",
])
VALID_LABELS = sorted([
    "A", "B", "C", "D", "E", "F", "G", "H",
    "J", "K", "L", "M", "N", "P", "Q", "R",
])


# =====
# Fixtures
# =====
@pytest.fixture
def behavior_df():
    """Load source behavior classifications CSV."""
    if not BEHAVIOR_CSV.exists():
        pytest.skip(f"Source data not found: {BEHAVIOR_CSV}")
    return pd.read_csv(BEHAVIOR_CSV)


@pytest.fixture
def liar_buckets_df():
    """Load liar buckets output CSV."""
    if not LIAR_BUCKETS_CSV.exists():
        pytest.skip(f"Liar buckets output not found: {LIAR_BUCKETS_CSV}")
    return pd.read_csv(LIAR_BUCKETS_CSV)


# =====
# Output existence tests
# =====
class TestOutputExists:
    """Verify output file exists."""

    def test_liar_buckets_csv_exists(self):
        """liar_buckets.csv must be generated."""
        assert LIAR_BUCKETS_CSV.exists(), (
            f"Output file missing: {LIAR_BUCKETS_CSV}. "
            "Run liar_buckets.py first."
        )


# =====
# Schema tests
# =====
class TestSchema:
    """Verify output CSV schema and structure."""

    def test_required_columns_present(self, liar_buckets_df):
        """All required columns must exist."""
        for col in EXPECTED_COLUMNS:
            assert col in liar_buckets_df.columns, (
                f"Missing column: {col}"
            )

    def test_no_extra_columns(self, liar_buckets_df):
        """No unexpected columns should be present."""
        extra = set(liar_buckets_df.columns) - set(EXPECTED_COLUMNS)
        assert len(extra) == 0, f"Unexpected columns: {extra}"

    def test_exactly_160_rows(self, liar_buckets_df):
        """Must have exactly 160 rows (10 sessions x 16 participants)."""
        assert len(liar_buckets_df) == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows, got {len(liar_buckets_df)}"
        )


# =====
# Data integrity tests
# =====
class TestDataIntegrity:
    """Verify data values and constraints."""

    def test_no_duplicate_participants(self, liar_buckets_df):
        """Each (session_code, label) pair must be unique."""
        dupes = liar_buckets_df.duplicated(
            subset=["session_code", "label"], keep=False
        )
        assert not dupes.any(), (
            f"Found {dupes.sum()} duplicate participant rows"
        )

    def test_all_sessions_represented(self, liar_buckets_df):
        """All 10 sessions must be present."""
        actual = sorted(liar_buckets_df["session_code"].unique())
        assert actual == EXPECTED_SESSIONS, (
            f"Session mismatch.\n"
            f"Missing: {set(EXPECTED_SESSIONS) - set(actual)}\n"
            f"Extra: {set(actual) - set(EXPECTED_SESSIONS)}"
        )

    def test_session_count(self, liar_buckets_df):
        """Exactly 10 unique sessions."""
        assert liar_buckets_df["session_code"].nunique() == EXPECTED_SESSION_COUNT

    def test_16_participants_per_session(self, liar_buckets_df):
        """Each session must have exactly 16 participants."""
        counts = liar_buckets_df.groupby("session_code").size()
        bad = counts[counts != 16]
        assert bad.empty, (
            f"Sessions with wrong participant count: {bad.to_dict()}"
        )

    def test_valid_labels_only(self, liar_buckets_df):
        """Labels must be from A-R (skipping I and O)."""
        actual = sorted(liar_buckets_df["label"].unique())
        assert actual == VALID_LABELS, (
            f"Label mismatch. Got: {actual}, expected: {VALID_LABELS}"
        )

    def test_treatments_are_1_or_2(self, liar_buckets_df):
        """Treatment values must be 1 or 2."""
        valid = {1, 2}
        actual = set(liar_buckets_df["treatment"].unique())
        assert actual.issubset(valid), (
            f"Invalid treatments: {actual - valid}"
        )


# =====
# Lie count tests
# =====
class TestLieCount:
    """Verify lie_count values and consistency."""

    def test_lie_count_non_negative(self, liar_buckets_df):
        """lie_count must be >= 0."""
        assert (liar_buckets_df["lie_count"] >= 0).all(), (
            "Found negative lie_count values"
        )

    def test_lie_count_is_integer(self, liar_buckets_df):
        """lie_count must be integer-valued."""
        values = liar_buckets_df["lie_count"]
        assert (values == values.astype(int)).all(), (
            "lie_count contains non-integer values"
        )

    def test_total_lies_matches_source(self, liar_buckets_df):
        """Sum of lie_count must match total lied_this_round_20 in source."""
        total = int(liar_buckets_df["lie_count"].sum())
        assert total == EXPECTED_TOTAL_LIES, (
            f"Total lies {total} != expected {EXPECTED_TOTAL_LIES} "
            f"from behavior_classifications.csv"
        )

    def test_lie_count_matches_source_per_participant(
        self, liar_buckets_df, behavior_df
    ):
        """Each participant's lie_count must match source data."""
        mismatches = _find_lie_count_mismatches(liar_buckets_df, behavior_df)
        assert mismatches.empty, (
            f"Lie count mismatches for {len(mismatches)} participants:\n"
            f"{mismatches[['session_code', 'label', 'lie_count', 'expected_lie_count']].head(10)}"
        )


def _find_lie_count_mismatches(liar_buckets_df, behavior_df):
    """Compare per-participant lie counts between output and source."""
    source_lies = (
        behavior_df
        .groupby(["session_code", "label"])["lied_this_round_20"]
        .sum()
        .reset_index()
    )
    source_lies.columns = ["session_code", "label", "expected_lie_count"]
    merged = liar_buckets_df.merge(
        source_lies, on=["session_code", "label"], how="left"
    )
    return merged[merged["lie_count"] != merged["expected_lie_count"]]


# =====
# Bucket assignment tests
# =====
class TestBucketAssignment:
    """Verify liar_bucket values and mapping logic."""

    def test_valid_bucket_values_only(self, liar_buckets_df):
        """liar_bucket must be one of the four valid values."""
        actual = set(liar_buckets_df["liar_bucket"].unique())
        assert actual.issubset(VALID_BUCKETS), (
            f"Invalid bucket values: {actual - VALID_BUCKETS}"
        )

    def test_never_bucket_means_zero_lies(self, liar_buckets_df):
        """Participants with liar_bucket='never' must have lie_count=0."""
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "never"]
        assert (subset["lie_count"] == 0).all(), (
            "Found 'never' bucket with non-zero lie_count"
        )

    def test_one_time_bucket_means_one_lie(self, liar_buckets_df):
        """Participants with liar_bucket='one_time' must have lie_count=1."""
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "one_time"]
        assert (subset["lie_count"] == 1).all(), (
            "Found 'one_time' bucket with lie_count != 1"
        )

    def test_moderate_bucket_means_2_or_3_lies(self, liar_buckets_df):
        """Participants with liar_bucket='moderate' must have lie_count in {2, 3}."""
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "moderate"]
        assert subset["lie_count"].isin([2, 3]).all(), (
            "Found 'moderate' bucket with lie_count outside {2, 3}"
        )

    def test_severe_bucket_means_4_plus_lies(self, liar_buckets_df):
        """Participants with liar_bucket='severe' must have lie_count >= 4."""
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "severe"]
        assert (subset["lie_count"] >= 4).all(), (
            "Found 'severe' bucket with lie_count < 4"
        )

    @pytest.mark.parametrize("lie_count,expected_bucket", [
        (0, "never"),
        (1, "one_time"),
        (2, "moderate"),
        (3, "moderate"),
        (4, "severe"),
        (5, "severe"),
        (9, "severe"),
    ])
    def test_bucket_mapping_exhaustive(
        self, liar_buckets_df, lie_count, expected_bucket
    ):
        """Verify bucket assignment for each lie_count value present."""
        subset = liar_buckets_df[liar_buckets_df["lie_count"] == lie_count]
        if subset.empty:
            pytest.skip(f"No participants with lie_count={lie_count}")
        assert (subset["liar_bucket"] == expected_bucket).all(), (
            f"lie_count={lie_count} should map to '{expected_bucket}', "
            f"got: {subset['liar_bucket'].unique()}"
        )


# =====
# Regression tests against real data
# =====
class TestRealDataRegression:
    """Regression tests grounded in verified real data."""

    def test_bucket_distribution_matches_expected(self, liar_buckets_df):
        """Bucket counts must match values derived from source data."""
        actual = liar_buckets_df["liar_bucket"].value_counts().to_dict()
        for bucket, expected_count in EXPECTED_BUCKET_COUNTS.items():
            assert actual.get(bucket, 0) == expected_count, (
                f"Bucket '{bucket}': expected {expected_count}, "
                f"got {actual.get(bucket, 0)}"
            )

    def test_total_bucket_count(self, liar_buckets_df):
        """Sum of all bucket counts must equal 160."""
        total = liar_buckets_df["liar_bucket"].value_counts().sum()
        assert total == EXPECTED_ROW_COUNT

    def test_participant_ids_are_valid(self, liar_buckets_df, behavior_df):
        """All participant_ids must match those in behavior_classifications."""
        source_ids = set(
            behavior_df.groupby(["session_code", "participant_id"])
            .size()
            .reset_index()
            .apply(
                lambda r: (r["session_code"], r["participant_id"]), axis=1
            )
        )
        output_ids = set(
            liar_buckets_df.apply(
                lambda r: (r["session_code"], r["participant_id"]), axis=1
            )
        )
        missing = source_ids - output_ids
        extra = output_ids - source_ids
        assert not missing, f"Missing participants: {missing}"
        assert not extra, f"Extra participants: {extra}"


# =====
# Plot output tests (Task #3)
# =====
class TestPlotOutputs:
    """Verify plot and LaTeX outputs from issue_53_liar_plots.py."""

    def test_sentiment_plot_exists(self):
        """Sentiment box plot PNG must be generated."""
        assert SENTIMENT_PLOT.exists(), (
            f"Missing plot: {SENTIMENT_PLOT}"
        )

    def test_emotions_plot_exists(self):
        """Emotions faceted box plot PNG must be generated."""
        assert EMOTIONS_PLOT.exists(), (
            f"Missing plot: {EMOTIONS_PLOT}"
        )

    def test_summary_tex_exists(self):
        """Summary statistics LaTeX table must be generated."""
        assert SUMMARY_TEX.exists(), (
            f"Missing LaTeX file: {SUMMARY_TEX}"
        )

    def test_sentiment_plot_not_empty(self):
        """Sentiment plot file must have non-zero size."""
        if not SENTIMENT_PLOT.exists():
            pytest.skip("Sentiment plot not yet generated")
        assert SENTIMENT_PLOT.stat().st_size > 0, (
            "Sentiment plot file is empty"
        )

    def test_emotions_plot_not_empty(self):
        """Emotions plot file must have non-zero size."""
        if not EMOTIONS_PLOT.exists():
            pytest.skip("Emotions plot not yet generated")
        assert EMOTIONS_PLOT.stat().st_size > 0, (
            "Emotions plot file is empty"
        )

    def test_summary_tex_valid_latex(self):
        """LaTeX file must contain valid LaTeX table markers."""
        if not SUMMARY_TEX.exists():
            pytest.skip("Summary tex not yet generated")
        content = SUMMARY_TEX.read_text()
        assert "\\begin{tabular" in content or "\\begin{table" in content, (
            "LaTeX file does not contain tabular or table environment"
        )
        assert "\\end{tabular" in content or "\\end{table" in content, (
            "LaTeX file is missing closing tabular/table environment"
        )
