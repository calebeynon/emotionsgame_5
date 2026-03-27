"""
Tests for Task 1: preprocess_hanaki_ozkes.py output parquet integrity.

Validates schema, data ranges, chat text quality, session composition,
and helper function correctness against real verified output.

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
PARQUET_FILE = DERIVED_DIR / "hanaki_ozkes_chat_decisions.parquet"

# Known-good values from verified output
EXPECTED_COLUMNS = [
    "session_file", "period", "player_id", "group",
    "Inv", "OtherInv", "Profit", "Chat", "Comp", "chat_text",
]
EXPECTED_SESSION_COUNT = 22
EXPECTED_ROW_COUNT = 11842


# =====
# Fixtures
# =====
@pytest.fixture
def parquet_df():
    """Load the preprocessed parquet as a DataFrame."""
    if not PARQUET_FILE.exists():
        pytest.skip(f"Parquet not found: {PARQUET_FILE}")
    return pd.read_parquet(PARQUET_FILE)


# =====
# Schema and structure
# =====
class TestPreprocessSchema:
    """Verify parquet output has correct schema."""

    def test_parquet_file_exists(self):
        """Output parquet must exist."""
        assert PARQUET_FILE.exists(), f"Missing: {PARQUET_FILE}"

    def test_expected_columns(self, parquet_df):
        """Parquet must contain exactly the expected columns."""
        assert list(parquet_df.columns) == EXPECTED_COLUMNS

    def test_row_count_matches_verified(self, parquet_df):
        """Regression: verified output has 6382 rows."""
        assert len(parquet_df) == EXPECTED_ROW_COUNT

    def test_session_count_matches_verified(self, parquet_df):
        """Regression: verified output has 22 unique sessions."""
        assert parquet_df["session_file"].nunique() == EXPECTED_SESSION_COUNT

    def test_column_dtypes(self, parquet_df):
        """Verify column data types are correct."""
        assert parquet_df["session_file"].dtype == object
        assert parquet_df["period"].dtype == np.int64
        assert parquet_df["player_id"].dtype == np.int64
        assert parquet_df["Inv"].dtype == np.float64
        assert parquet_df["Chat"].dtype == np.int64
        assert parquet_df["chat_text"].dtype == object


# =====
# Data integrity
# =====
class TestPreprocessDataIntegrity:
    """Verify data values are within expected ranges."""

    def test_investment_range_0_to_28(self, parquet_df):
        """Investment must be in [0, 28] (Hanaki endowment is 28)."""
        assert parquet_df["Inv"].min() >= 0.0
        assert parquet_df["Inv"].max() <= 28.0

    def test_other_investment_range_0_to_28(self, parquet_df):
        """Other investment must be in [0, 28]."""
        assert parquet_df["OtherInv"].dropna().min() >= 0.0
        assert parquet_df["OtherInv"].dropna().max() <= 28.0

    def test_only_chat_1_sessions(self, parquet_df):
        """All rows must have Chat=1 (only chat sessions included)."""
        assert (parquet_df["Chat"] == 1).all()

    def test_comp_is_0_or_1(self, parquet_df):
        """Comp column must be 0 or 1."""
        assert set(parquet_df["Comp"].unique()) == {0, 1}

    def test_no_null_values(self, parquet_df):
        """No columns should have null values."""
        assert parquet_df.isna().sum().sum() == 0

    def test_no_underscore_sessions(self, parquet_df):
        """No session IDs ending with underscore (duplicates)."""
        sessions = parquet_df["session_file"].unique()
        underscore = [s for s in sessions if s.endswith("_")]
        assert underscore == [], f"Duplicate sessions: {underscore}"

    def test_period_range(self, parquet_df):
        """Periods should be in [0, 30]."""
        assert parquet_df["period"].min() >= 0
        assert parquet_df["period"].max() <= 30

    def test_no_aggregate_files(self, parquet_df):
        """Aggregate files must not appear as sessions."""
        sessions = set(parquet_df["session_file"].unique())
        assert "Data_for_Stata" not in sessions
        assert "Top200_Alln" not in sessions


# =====
# Chat text quality
# =====
class TestPreprocessChatText:
    """Verify chat text is correctly parsed and merged."""

    def test_chat_text_is_string(self, parquet_df):
        """All chat_text values must be strings."""
        assert parquet_df["chat_text"].apply(type).eq(str).all()

    def test_nonempty_chat_count_matches_verified(self, parquet_df):
        """Regression: verified output has 7748 non-empty chat rows."""
        nonempty = (parquet_df["chat_text"] != "").sum()
        assert nonempty == 7748

    def test_chat_text_not_all_empty(self, parquet_df):
        """More than half of rows should have non-empty chat text."""
        empty_ratio = (parquet_df["chat_text"] == "").mean()
        assert empty_ratio < 0.5, f"Too many empty: {empty_ratio:.2%}"

    def test_nonempty_chat_has_actual_text(self, parquet_df):
        """Non-empty chat_text must not be whitespace-only."""
        nonempty = parquet_df[parquet_df["chat_text"] != ""]["chat_text"]
        whitespace_only = (nonempty.str.strip() == "").sum()
        assert whitespace_only == 0


# =====
# Session-level validation
# =====
class TestPreprocessSessions:
    """Verify session-level properties from the verified output."""

    def test_known_sessions_present(self, parquet_df):
        """Core Chat=1 sessions from BehaviorCode.R must be present."""
        sessions = set(parquet_df["session_file"].unique())
        core_chat1 = {
            "160503_0820", "160510_0920", "160511_0853",
            "160511_1426", "160512_1420", "160513_0850",
            "160519_0934", "161018_0903", "161019_0824",
            "161019_1411", "170510_0938", "170510_1319",
        }
        missing = core_chat1 - sessions
        assert not missing, f"Missing sessions: {missing}"

    def test_session_160503_0820_row_count(self, parquet_df):
        """Regression: session 160503_0820 has 372 rows."""
        count = len(parquet_df[parquet_df["session_file"] == "160503_0820"])
        assert count == 372

    def test_session_160503_0820_subjects(self, parquet_df):
        """Session 160503_0820 has 12 subjects at period 0."""
        p0 = parquet_df[
            (parquet_df["session_file"] == "160503_0820")
            & (parquet_df["period"] == 0)
        ]
        assert p0["player_id"].nunique() == 12

    def test_all_sessions_have_multiple_periods(self, parquet_df):
        """Every session should span multiple periods (0-30)."""
        periods = parquet_df.groupby("session_file")["period"].nunique()
        too_few = periods[periods < 2]
        assert too_few.empty, f"Sessions with < 2 periods: {too_few.to_dict()}"

    def test_all_sessions_have_31_periods(self, parquet_df):
        """Each session should have periods 0-30 (31 periods total)."""
        periods = parquet_df.groupby("session_file")["period"].nunique()
        not_31 = periods[periods != 31]
        assert not_31.empty, f"Sessions without 31 periods: {not_31.to_dict()}"


# =====
# Helper function unit tests
# =====
class TestPreprocessHelpers:
    """Unit tests for preprocessing helper functions."""

    def test_safe_numeric_valid(self):
        """_safe_numeric converts valid numbers."""
        from preprocess_hanaki_ozkes import _safe_numeric

        assert _safe_numeric("3.14") == pytest.approx(3.14)
        assert _safe_numeric("0") == 0.0
        assert _safe_numeric("-28") == -28.0

    def test_safe_numeric_invalid(self):
        """_safe_numeric returns NaN for non-numeric strings."""
        from preprocess_hanaki_ozkes import _safe_numeric

        assert np.isnan(_safe_numeric("-"))
        assert np.isnan(_safe_numeric(""))
        assert np.isnan(_safe_numeric("abc"))

    def test_collect_session_files_skips_underscore(self):
        """_collect_session_files must skip files ending with _."""
        from preprocess_hanaki_ozkes import _collect_session_files

        stems = [f.stem for f in _collect_session_files()]
        assert not any(s.endswith("_") for s in stems)

    def test_collect_session_files_returns_only_chat1(self):
        """_collect_session_files returns only Chat=1 sessions."""
        from preprocess_hanaki_ozkes import TREATMENT_MAP, _collect_session_files

        for f in _collect_session_files():
            assert f.stem in TREATMENT_MAP

    def test_treatment_map_all_chat1(self):
        """All TREATMENT_MAP entries must have Chat=1."""
        from preprocess_hanaki_ozkes import TREATMENT_MAP

        for sid, (chat, _) in TREATMENT_MAP.items():
            assert chat == 1, f"{sid} has Chat={chat}"

    def test_aggregate_chat_concatenates(self):
        """_aggregate_chat joins multiple messages per player-period."""
        from preprocess_hanaki_ozkes import _aggregate_chat

        chat_df = pd.DataFrame({
            "session_file": ["s1", "s1", "s1"],
            "period": [1, 1, 2],
            "player_id": [1, 1, 1],
            "chat_text": ["hello", "world", "solo"],
        })
        result = _aggregate_chat(chat_df)
        row = result[
            (result["session_file"] == "s1") & (result["period"] == 1)
        ]
        assert row["chat_text"].iloc[0] == "hello world"
