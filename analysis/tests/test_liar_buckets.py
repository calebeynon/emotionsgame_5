"""
Tests for liar bucket classification (issue #53).

Validates liar_buckets.py output CSV: schema, integrity, bucket logic,
and manually traced participant values. All expected values verified by
tracing behavior_classifications.csv round-by-round.

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

# MANUALLY VERIFIED CONSTANTS
EXPECTED_ROW_COUNT = 160
EXPECTED_TOTAL_LIES = 94
VALID_BUCKETS = {"never", "one_time", "moderate", "severe"}
EXPECTED_BUCKET_COUNTS = {"never": 105, "one_time": 36, "moderate": 14, "severe": 5}
EXPECTED_LIE_COUNT_DISTRIBUTION = {
    0: 105, 1: 36, 2: 12, 3: 2, 4: 2, 5: 1, 6: 1, 9: 1,
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

# MANUALLY TRACED PARTICIPANTS
# Each verified by reading every round's lied_this_round_20 in behavior_classifications.csv
TRACED_PARTICIPANTS = [
    # (session_code, label, lie_count, bucket)
    ("iiu3xixz", "L", 9, "severe"),   # most prolific liar
    ("irrzlgk2", "N", 6, "severe"),
    ("umbzdj98", "N", 5, "severe"),
    ("irrzlgk2", "K", 4, "severe"),
    ("j3ki5tli", "P", 4, "severe"),
    ("irrzlgk2", "M", 3, "moderate"),  # lies in sg1r3, sg4r2, sg4r5
    ("irrzlgk2", "P", 3, "moderate"),
    ("6sdkxl2q", "D", 2, "moderate"),
    ("6sdkxl2q", "A", 1, "one_time"),  # single lie in sg4r7
    ("6sdkxl2q", "B", 0, "never"),     # zero lies across 22 rounds
]


# =====
# Fixtures
# =====
@pytest.fixture
def behavior_df():
    if not BEHAVIOR_CSV.exists():
        pytest.skip(f"Source data not found: {BEHAVIOR_CSV}")
    return pd.read_csv(BEHAVIOR_CSV)


@pytest.fixture
def liar_buckets_df():
    if not LIAR_BUCKETS_CSV.exists():
        pytest.skip(f"Output not found: {LIAR_BUCKETS_CSV}")
    return pd.read_csv(LIAR_BUCKETS_CSV)


# =====
# Schema and integrity
# =====
class TestSchema:
    def test_required_columns(self, liar_buckets_df):
        for col in EXPECTED_COLUMNS:
            assert col in liar_buckets_df.columns, f"Missing: {col}"

    def test_no_extra_columns(self, liar_buckets_df):
        extra = set(liar_buckets_df.columns) - set(EXPECTED_COLUMNS)
        assert not extra, f"Unexpected: {extra}"

    def test_exactly_160_rows(self, liar_buckets_df):
        assert len(liar_buckets_df) == EXPECTED_ROW_COUNT

    def test_no_nulls(self, liar_buckets_df):
        nulls = liar_buckets_df.isnull().sum()
        bad = nulls[nulls > 0]
        assert bad.empty, f"Nulls: {bad.to_dict()}"


class TestDataIntegrity:
    def test_no_duplicate_participants(self, liar_buckets_df):
        dupes = liar_buckets_df.duplicated(subset=["session_code", "label"], keep=False)
        assert not dupes.any()

    def test_all_sessions_present(self, liar_buckets_df):
        assert sorted(liar_buckets_df["session_code"].unique()) == EXPECTED_SESSIONS

    def test_16_per_session(self, liar_buckets_df):
        counts = liar_buckets_df.groupby("session_code").size()
        bad = counts[counts != 16]
        assert bad.empty, f"Wrong count: {bad.to_dict()}"

    def test_valid_labels(self, liar_buckets_df):
        assert sorted(liar_buckets_df["label"].unique()) == VALID_LABELS

    def test_treatments(self, liar_buckets_df):
        assert set(liar_buckets_df["treatment"].unique()) == {1, 2}


# =====
# Manually traced participants
# =====
class TestTracedParticipants:
    @pytest.mark.parametrize(
        "session_code,label,expected_lies,expected_bucket",
        TRACED_PARTICIPANTS,
        ids=[f"{s}/{l}" for s, l, _, _ in TRACED_PARTICIPANTS],
    )
    def test_lie_count_and_bucket(
        self, liar_buckets_df, session_code, label, expected_lies, expected_bucket
    ):
        row = liar_buckets_df[
            (liar_buckets_df["session_code"] == session_code)
            & (liar_buckets_df["label"] == label)
        ]
        assert len(row) == 1
        assert int(row["lie_count"].values[0]) == expected_lies
        assert row["liar_bucket"].values[0] == expected_bucket

    def test_most_prolific_liar(self, liar_buckets_df):
        max_row = liar_buckets_df.loc[liar_buckets_df["lie_count"].idxmax()]
        assert max_row["session_code"] == "iiu3xixz"
        assert max_row["label"] == "L"
        assert int(max_row["lie_count"]) == 9

    def test_all_lie_counts_match_source(self, liar_buckets_df, behavior_df):
        """Every participant's lie_count matches sum of lied_this_round_20."""
        behavior_df["lied_this_round_20"] = behavior_df["lied_this_round_20"].fillna(False)
        source = (
            behavior_df.groupby(["session_code", "label"])["lied_this_round_20"]
            .sum().reset_index()
            .rename(columns={"lied_this_round_20": "expected"})
        )
        merged = liar_buckets_df.merge(source, on=["session_code", "label"])
        merged["expected"] = merged["expected"].astype(int)
        mismatches = merged[merged["lie_count"] != merged["expected"]]
        assert mismatches.empty, f"{len(mismatches)} mismatches"


# =====
# Bucket assignment logic
# =====
class TestBucketAssignment:
    def test_never_means_zero(self, liar_buckets_df):
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "never"]
        assert (subset["lie_count"] == 0).all()

    def test_one_time_means_one(self, liar_buckets_df):
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "one_time"]
        assert (subset["lie_count"] == 1).all()

    def test_moderate_means_2_or_3(self, liar_buckets_df):
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "moderate"]
        assert subset["lie_count"].isin([2, 3]).all()

    def test_severe_means_4_plus(self, liar_buckets_df):
        subset = liar_buckets_df[liar_buckets_df["liar_bucket"] == "severe"]
        assert (subset["lie_count"] >= 4).all()

    def test_bucket_distribution(self, liar_buckets_df):
        actual = liar_buckets_df["liar_bucket"].value_counts().to_dict()
        for bucket, expected in EXPECTED_BUCKET_COUNTS.items():
            assert actual.get(bucket, 0) == expected

    def test_lie_count_histogram(self, liar_buckets_df):
        actual = liar_buckets_df["lie_count"].value_counts().to_dict()
        for count, expected_n in EXPECTED_LIE_COUNT_DISTRIBUTION.items():
            assert actual.get(count, 0) == expected_n

    def test_total_lies(self, liar_buckets_df):
        assert int(liar_buckets_df["lie_count"].sum()) == EXPECTED_TOTAL_LIES
