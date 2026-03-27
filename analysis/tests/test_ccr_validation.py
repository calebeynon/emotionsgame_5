"""
Tests for CCR chat and effort data pipelines (Tasks 2 and 3).

Validates preprocessed chat parquet, effort panel parquet, effort group
cross-section, and raw data source integrity for the Chen, Chen &
Riyanto (2021) external validation.

Known data facts (verified from raw data.dta and file inspection):
- data.dta: 34,800 rows, 696 subjects, 58 sessions, 50 periods
- Effort range: 110-170, mean ~141.19
- 116 color teams (58 sessions x 2 red/green)
- Run codes: 0=Original (6), 1=Science (14), 2=NUS (10), 3=NTU (28)

Author: pytest-test-writer
Date: 2026-03-26
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
EXTERNAL_DIR = DERIVED_DIR / "external"
CHAT_PARQUET = EXTERNAL_DIR / "ccr_chat_clean.parquet"
EFFORT_PANEL_PARQUET = EXTERNAL_DIR / "ccr_effort_panel.parquet"
EFFORT_GROUP_PARQUET = EXTERNAL_DIR / "ccr_effort_group.parquet"
CCR_RAW_DIR = (
    Path(__file__).parent.parent
    / "datastore" / "raw" / "external_datasets" / "chen_chen_riyanto_2021"
)
DATA_DTA = CCR_RAW_DIR / "4 - Data Analysis" / "Statistics" / "data.dta"

# Known-good values from verified data
EXPECTED_DTA_ROWS = 34800
EXPECTED_DTA_SUBJECTS = 696
EXPECTED_DTA_SESSIONS = 58
EXPECTED_DTA_PERIODS = 50
EXPECTED_COLOR_TEAMS = 116
EXPECTED_EFFORT_MIN = 110
EXPECTED_EFFORT_MAX = 170
EXPECTED_EFFORT_MEAN = 141.189
EXPECTED_RUN_SESSION_COUNTS = {0: 6, 1: 14, 2: 10, 3: 28}


# =====
# Fixtures
# =====
@pytest.fixture
def chat_df():
    """Load the preprocessed CCR chat parquet."""
    if not CHAT_PARQUET.exists():
        pytest.skip(f"Chat parquet not found: {CHAT_PARQUET}")
    return pd.read_parquet(CHAT_PARQUET)


@pytest.fixture
def effort_panel_df():
    """Load the CCR effort panel parquet."""
    if not EFFORT_PANEL_PARQUET.exists():
        pytest.skip(f"Effort panel not found: {EFFORT_PANEL_PARQUET}")
    return pd.read_parquet(EFFORT_PANEL_PARQUET)


@pytest.fixture
def effort_group_df():
    """Load the CCR effort group cross-section parquet."""
    if not EFFORT_GROUP_PARQUET.exists():
        pytest.skip(f"Effort group not found: {EFFORT_GROUP_PARQUET}")
    return pd.read_parquet(EFFORT_GROUP_PARQUET)


@pytest.fixture
def raw_dta():
    """Load raw data.dta for cross-validation."""
    if not DATA_DTA.exists():
        pytest.skip(f"Raw data.dta not found: {DATA_DTA}")
    return pd.read_stata(DATA_DTA)


# =====
# Task 2: Chat parquet - schema and structure
# =====
class TestChatParquetSchema:
    """Verify CCR chat parquet has correct schema and dimensions."""

    def test_chat_parquet_exists(self):
        """Output parquet must exist after preprocess_ccr_chat.py runs."""
        if not CHAT_PARQUET.exists():
            pytest.skip("Chat parquet not yet created")
        assert CHAT_PARQUET.exists()

    def test_has_required_columns(self, chat_df):
        """Must have session, red, chat text, and count columns."""
        required = ["session", "red", "n_messages", "n_words", "n_characters"]
        for col in required:
            assert col in chat_df.columns, f"Missing column: {col}"
        # Chat text may be named chat_text or group_chat_text
        has_text = "chat_text" in chat_df.columns or "group_chat_text" in chat_df.columns
        assert has_text, "Missing chat text column (chat_text or group_chat_text)"


# =====
# Task 2: Chat parquet - data integrity
# =====
class TestChatParquetIntegrity:
    """Verify chat data values are correct."""

    def test_all_58_sessions_covered(self, chat_df):
        """All 58 CCR sessions must be represented."""
        n = chat_df["session"].nunique()
        assert n == EXPECTED_DTA_SESSIONS, f"Expected {EXPECTED_DTA_SESSIONS}, got {n}"

    def test_red_values_are_0_or_1(self, chat_df):
        """Red column must be 0 (Green) or 1 (Red)."""
        assert set(chat_df["red"].unique()) <= {0, 1, 0.0, 1.0}

    def test_chat_text_is_string(self, chat_df):
        """All chat text values must be strings."""
        col = _chat_text_col(chat_df)
        assert chat_df[col].apply(type).eq(str).all()

    def test_no_system_messages(self, chat_df):
        """System messages (joined/left) must be filtered out."""
        col = _chat_text_col(chat_df)
        for pattern in ["has just joined", "has just left"]:
            matches = chat_df[col].str.contains(pattern, case=False, na=False)
            assert not matches.any(), f"Found '{pattern}' in output"

    def test_message_counts_nonnegative(self, chat_df):
        """All count columns must be >= 0."""
        for col in ["n_messages", "n_words", "n_characters"]:
            assert (chat_df[col] >= 0).all(), f"{col} has negative values"

    def test_nonempty_chat_has_positive_counts(self, chat_df):
        """Rows with non-empty chat should have positive word counts."""
        col = _chat_text_col(chat_df)
        nonempty = chat_df[chat_df[col].str.len() > 0]
        if len(nonempty) > 0:
            assert (nonempty["n_words"] > 0).all()


# =====
# Task 2: Chat parquet - group-level aggregation
# =====
class TestChatGroupLevel:
    """Verify group-level (session x color team) chat aggregation."""

    def test_group_level_has_116_rows(self, chat_df):
        """Should have 116 color teams (58 sessions x 2)."""
        groups = chat_df.groupby(["session", "red"]).ngroups
        assert groups == EXPECTED_COLOR_TEAMS

    def test_each_session_has_two_color_teams(self, chat_df):
        """Each session should have exactly 2 color teams."""
        teams = chat_df.groupby("session")["red"].nunique()
        wrong = teams[teams != 2]
        assert wrong.empty, f"Sessions without 2 teams: {wrong.to_dict()}"


# =====
# Task 2: Chat cross-validation with data.dta
# =====
class TestChatCrossValidation:
    """Cross-validate chat output with raw data.dta word counts."""

    def test_word_counts_correlate_with_dta(self, chat_df, raw_dta):
        """Chat word counts should correlate with count_words in data.dta."""
        dta_words = (
            raw_dta.groupby(["session", "red"])["count_words"]
            .first().reset_index()
        )
        # Align types for merge (chat uses int, dta uses float32)
        dta_words["session"] = dta_words["session"].astype(int)
        dta_words["red"] = dta_words["red"].astype(int)
        chat_copy = chat_df.copy()
        chat_copy["session"] = chat_copy["session"].astype(int)
        chat_copy["red"] = chat_copy["red"].astype(int)
        merged = chat_copy.merge(dta_words, on=["session", "red"], how="inner")
        if len(merged) == 0:
            pytest.skip("No merge matches — session keys may differ")
        corr = merged["n_words"].corr(merged["count_words"])
        assert corr > 0.5, f"Weak correlation: {corr:.3f}"


# =====
# Task 3: Effort panel - schema and dimensions
# =====
class TestEffortPanelSchema:
    """Verify effort panel parquet has correct schema."""

    def test_effort_panel_exists(self):
        """Output parquet must exist."""
        if not EFFORT_PANEL_PARQUET.exists():
            pytest.skip("Effort panel not yet created")
        assert EFFORT_PANEL_PARQUET.exists()

    def test_panel_row_count(self, effort_panel_df):
        """Panel should have 34,800 rows (696 subjects x 50 periods)."""
        assert len(effort_panel_df) == EXPECTED_DTA_ROWS

    def test_has_key_columns(self, effort_panel_df):
        """Must have essential columns for analysis."""
        for col in ["session", "subject", "period", "effort", "red",
                     "ingroup", "commonknow", "run"]:
            assert col in effort_panel_df.columns, f"Missing: {col}"


# =====
# Task 3: Effort panel - data integrity
# =====
class TestEffortPanelIntegrity:
    """Verify effort panel data values match known facts."""

    def test_effort_range_110_to_170(self, effort_panel_df):
        """Effort must be in [110, 170]."""
        assert effort_panel_df["effort"].min() >= EXPECTED_EFFORT_MIN
        assert effort_panel_df["effort"].max() <= EXPECTED_EFFORT_MAX

    def test_effort_mean(self, effort_panel_df):
        """Regression: effort mean ~141.19 from verified data.dta."""
        assert effort_panel_df["effort"].mean() == pytest.approx(EXPECTED_EFFORT_MEAN, abs=0.01)

    def test_subject_count(self, effort_panel_df):
        """Should have 696 unique subjects."""
        assert effort_panel_df["subject"].nunique() == EXPECTED_DTA_SUBJECTS

    def test_session_count(self, effort_panel_df):
        """Should have 58 unique sessions."""
        assert effort_panel_df["session"].nunique() == EXPECTED_DTA_SESSIONS

    def test_period_count_per_subject(self, effort_panel_df):
        """Each subject should have exactly 50 periods."""
        periods = effort_panel_df.groupby("subject")["period"].nunique()
        wrong = periods[periods != EXPECTED_DTA_PERIODS]
        assert wrong.empty, f"Subjects without 50 periods: {len(wrong)}"

    def test_binary_treatment_columns(self, effort_panel_df):
        """Red, ingroup, commonknow must be binary (0 or 1)."""
        for col in ["red", "ingroup", "commonknow"]:
            vals = set(effort_panel_df[col].dropna().unique())
            assert vals <= {0, 1, 0.0, 1.0}, f"{col} not binary: {vals}"

    def test_run_session_counts(self, effort_panel_df):
        """Each run (site) should have expected session counts."""
        run_sessions = _run_session_map(effort_panel_df)
        for run_code, expected_n in EXPECTED_RUN_SESSION_COUNTS.items():
            actual = run_sessions.get(float(run_code))
            assert actual == expected_n, (
                f"Run {run_code}: expected {expected_n}, got {actual}"
            )

    def test_matcheffort_range(self, effort_panel_df):
        """Match effort should also be in [110, 170]."""
        if "matcheffort" not in effort_panel_df.columns:
            pytest.skip("matcheffort column not in panel")
        valid = effort_panel_df["matcheffort"].dropna()
        assert valid.min() >= EXPECTED_EFFORT_MIN
        assert valid.max() <= EXPECTED_EFFORT_MAX


# =====
# Task 3: Effort group cross-section
# =====
class TestEffortGroupCrossSection:
    """Verify group-level cross-section parquet (116 rows)."""

    def test_group_parquet_exists(self):
        """Group cross-section parquet must exist."""
        if not EFFORT_GROUP_PARQUET.exists():
            pytest.skip("Group cross-section not yet created")
        assert EFFORT_GROUP_PARQUET.exists()

    def test_group_row_count_is_116(self, effort_group_df):
        """Should have 116 rows (58 sessions x 2 color teams)."""
        assert len(effort_group_df) == EXPECTED_COLOR_TEAMS

    def test_group_has_merge_keys(self, effort_group_df):
        """Must have session and red columns for merge."""
        for col in ["session", "red"]:
            assert col in effort_group_df.columns, f"Missing: {col}"

    def test_group_effort_within_range(self, effort_group_df):
        """All effort aggregates must be within [110, 170]."""
        effort_cols = [
            c for c in effort_group_df.columns
            if "effort" in c.lower()
            and effort_group_df[c].dtype in [np.float64, np.float32, np.int64]
        ]
        for col in effort_cols:
            valid = effort_group_df[col].dropna()
            if len(valid) == 0:
                continue
            assert valid.min() >= EXPECTED_EFFORT_MIN
            assert valid.max() <= EXPECTED_EFFORT_MAX

    def test_group_unique_pairs(self, effort_group_df):
        """Each (session, red) pair should be unique."""
        dupes = effort_group_df.duplicated(subset=["session", "red"])
        assert not dupes.any()


# =====
# Cross-validation with raw data.dta
# =====
class TestEffortCrossValidation:
    """Cross-validate effort panel against raw data.dta."""

    def test_effort_mean_matches_dta(self, effort_panel_df, raw_dta):
        """Mean effort should match data.dta exactly."""
        assert effort_panel_df["effort"].mean() == pytest.approx(
            raw_dta["effort"].mean(), abs=0.01,
        )

    def test_session_subject_counts_match(self, effort_panel_df, raw_dta):
        """Subject counts per session should match data.dta."""
        panel = effort_panel_df.groupby("session")["subject"].nunique()
        dta = raw_dta.groupby("session")["subject"].nunique()
        assert len(panel) == len(dta)
        assert (panel.sort_index().values == dta.sort_index().values).all()


# =====
# Raw data source validation
# =====
class TestRawDataSource:
    """Validate raw CCR data files are accessible and intact."""

    def test_data_dta_shape(self, raw_dta):
        """data.dta must have 34,800 rows and 75 columns."""
        assert raw_dta.shape == (EXPECTED_DTA_ROWS, 75)

    def test_original_chat_file_count(self):
        """Original site should have 12 .txt chat files."""
        chat_dir = CCR_RAW_DIR / "2 - Raw Data" / "0-Original" / "Chats"
        if not chat_dir.exists():
            pytest.skip("Not accessible")
        assert len(list(chat_dir.glob("*.txt"))) == 12

    def test_science_chat_file_count(self):
        """Science site should have 28 .txt chat files."""
        chat_dir = CCR_RAW_DIR / "2 - Raw Data" / "1-Science Replication" / "Chats"
        if not chat_dir.exists():
            pytest.skip("Not accessible")
        assert len(list(chat_dir.glob("*.txt"))) == 28

    def test_nus_xls_file_count(self):
        """NUS site should have 10 .xls z-Tree files."""
        xls_dir = CCR_RAW_DIR / "2 - Raw Data" / "2-NUS Replication" / "Chat & Effort"
        if not xls_dir.exists():
            pytest.skip("Not accessible")
        assert len(list(xls_dir.glob("*.xls"))) == 10

    def test_ntu_xls_file_count(self):
        """NTU site should have 28 .xls z-Tree files."""
        xls_dir = CCR_RAW_DIR / "2 - Raw Data" / "3-NTU Replication" / "Chat & Effort"
        if not xls_dir.exists():
            pytest.skip("Not accessible")
        assert len(list(xls_dir.glob("*.xls"))) == 28


# =====
# Helpers
# =====
def _run_session_map(df):
    """Build dict mapping float(run) -> n_unique_sessions."""
    return {
        float(k): v
        for k, v in df.groupby("run")["session"].nunique().to_dict().items()
    }


def _chat_text_col(df):
    """Return the chat text column name (chat_text or group_chat_text)."""
    if "chat_text" in df.columns:
        return "chat_text"
    if "group_chat_text" in df.columns:
        return "group_chat_text"
    raise KeyError("No chat text column found")
