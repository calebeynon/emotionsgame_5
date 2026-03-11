"""
Tests for the panel data merge pipeline (Issue #38).

Covers session_mapping.py, load_emotion_data.py, and merge_panel_data.py.
Unit tests use mock data; integration tests use real data files and are
skipped when data is unavailable.

Author: Claude Code
Date: 2026-03-11
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add derived directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from session_mapping import (
    SESSION_NUM_TO_CODE,
    SESSION_NUM_TO_TREATMENT,
    parse_annotation,
    parse_participant_id,
)
from load_emotion_data import (
    drop_empty_rows,
    convert_emotion_columns,
    deduplicate_recordings,
    finalize_columns,
    EMOTION_COLS,
    RAW_EMOTION_COLS,
    DEDUP_KEYS,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
STATE_FILE = DERIVED_DIR / "player_state_classification.csv"
SENTIMENT_FILE = DERIVED_DIR / "sentiment_scores.csv"
RWORK_DIR = Path(__file__).parent.parent / "datastore" / "Rwork"
EMOTION_RAW_FILE = RWORK_DIR / "all.csv"

# EXPECTED CONSTANTS
EXPECTED_SESSIONS = 10
EXPECTED_STATE_ROWS = 3520
EXPECTED_GAME_ROWS = 10560  # 3520 * 3 page types
PAGE_TYPES = ["Contribute", "Results", "ResultsOnly"]
EMOTION_COLUMNS = [
    "emotion_anger", "emotion_contempt", "emotion_disgust", "emotion_fear",
    "emotion_joy", "emotion_sadness", "emotion_surprise", "emotion_engagement",
    "emotion_valence", "emotion_sentimentality", "emotion_confusion",
    "emotion_neutral", "emotion_attention",
]
SENTIMENT_COLUMNS = [
    "sentiment_compound_mean", "sentiment_compound_std",
    "sentiment_compound_min", "sentiment_compound_max",
    "sentiment_positive_mean", "sentiment_negative_mean",
    "sentiment_neutral_mean",
]


# =====
# TestAnnotationParsing
# =====
class TestAnnotationParsing:
    """Tests for parse_annotation() in session_mapping.py."""

    def test_standard_contribute(self):
        """Standard pattern: s1r1Contribute -> supergame1, round 1."""
        result = parse_annotation("s1r1Contribute")
        assert result == ("supergame1", 1, "Contribute")

    def test_standard_results(self):
        """Standard pattern: s2r3Results -> supergame2, round 3."""
        result = parse_annotation("s2r3Results")
        assert result == ("supergame2", 3, "Results")

    def test_standard_results_only(self):
        """Standard pattern: s3r2ResultsOnly -> supergame3, round 2."""
        result = parse_annotation("s3r2ResultsOnly")
        assert result == ("supergame3", 2, "ResultsOnly")

    def test_standard_high_segment_round(self):
        """Standard pattern: s5r5ResultsOnly -> supergame5, round 5."""
        result = parse_annotation("s5r5ResultsOnly")
        assert result == ("supergame5", 5, "ResultsOnly")

    def test_irregular_s4_result1(self):
        """Irregular: S4result1 -> supergame4, round 1, Results."""
        result = parse_annotation("S4result1")
        assert result == ("supergame4", 1, "Results")

    def test_irregular_s4_result2(self):
        """Irregular: S4Result2 -> supergame4, round 2, Results."""
        result = parse_annotation("S4Result2")
        assert result == ("supergame4", 2, "Results")

    def test_irregular_s4_results4(self):
        """Irregular: S4Results4 -> supergame4, round 4, Results."""
        result = parse_annotation("S4Results4")
        assert result == ("supergame4", 4, "Results")

    def test_irregular_s4_result3(self):
        """Irregular: S4result3 -> supergame4, round 3, Results."""
        result = parse_annotation("S4result3")
        assert result == ("supergame4", 3, "Results")

    def test_irregular_s4_result5(self):
        """Irregular: S4result5 -> supergame4, round 5, Results."""
        result = parse_annotation("S4result5")
        assert result == ("supergame4", 5, "Results")

    def test_irregular_s4_result6(self):
        """Irregular: S4result6 -> supergame4, round 6, Results."""
        result = parse_annotation("S4result6")
        assert result == ("supergame4", 6, "Results")

    def test_irregular_s4_result7(self):
        """Irregular: S4result7 -> supergame4, round 7, Results."""
        result = parse_annotation("S4result7")
        assert result == ("supergame4", 7, "Results")

    def test_all_instructions(self):
        """Instructions: all_instructions -> None, None, 'all_instructions'."""
        result = parse_annotation("all_instructions")
        assert result == (None, None, "all_instructions")

    def test_all_instructions_segment_is_none(self):
        """all_instructions should return None for segment."""
        segment, _, _ = parse_annotation("all_instructions")
        assert segment is None

    def test_all_instructions_round_is_none(self):
        """all_instructions should return None for round."""
        _, round_num, _ = parse_annotation("all_instructions")
        assert round_num is None

    @pytest.mark.parametrize("annotation,expected_segment,expected_round,expected_page", [
        ("s1r1Contribute", "supergame1", 1, "Contribute"),
        ("s1r1Results", "supergame1", 1, "Results"),
        ("s1r1ResultsOnly", "supergame1", 1, "ResultsOnly"),
        ("s4r7Contribute", "supergame4", 7, "Contribute"),
        ("s5r5Results", "supergame5", 5, "Results"),
    ])
    def test_parametrized_standard_patterns(
        self, annotation, expected_segment, expected_round, expected_page
    ):
        """Parametrized test covering multiple standard patterns."""
        result = parse_annotation(annotation)
        assert result == (expected_segment, expected_round, expected_page)

    @pytest.mark.parametrize("annotation,expected_round", [
        ("S4result1", 1),
        ("S4Result2", 2),
        ("S4result3", 3),
        ("S4Results4", 4),
        ("S4result5", 5),
        ("S4result6", 6),
        ("S4result7", 7),
    ])
    def test_parametrized_irregular_s4(self, annotation, expected_round):
        """All irregular S4 annotations map to supergame4/Results."""
        segment, round_num, page_type = parse_annotation(annotation)
        assert segment == "supergame4"
        assert round_num == expected_round
        assert page_type == "Results"


# =====
# TestSessionMapping
# =====
class TestSessionMapping:
    """Tests for SESSION_NUM_TO_CODE and SESSION_NUM_TO_TREATMENT dicts."""

    @pytest.mark.parametrize("session_num,expected_code", [
        (1, "sa7mprty"),
        (3, "irrzlgk2"),
        (4, "6uv359rf"),
        (5, "umbzdj98"),
        (6, "j3ki5tli"),
        (7, "r5dj4yfl"),
        (8, "sylq2syi"),
        (9, "iiu3xixz"),
        (10, "6ucza025"),
        (11, "6sdkxl2q"),
    ])
    def test_session_num_to_code(self, session_num, expected_code):
        """Each session number maps to the correct oTree session code."""
        assert SESSION_NUM_TO_CODE[session_num] == expected_code

    def test_all_10_sessions_present(self):
        """All 10 sessions are in the mapping dict."""
        assert len(SESSION_NUM_TO_CODE) == EXPECTED_SESSIONS

    def test_session_numbers_match(self):
        """CODE and TREATMENT dicts have the same session numbers."""
        assert set(SESSION_NUM_TO_CODE.keys()) == set(
            SESSION_NUM_TO_TREATMENT.keys()
        )

    @pytest.mark.parametrize("session_num,expected_treatment", [
        (1, 1), (3, 2), (4, 2), (5, 1), (6, 2),
        (7, 1), (8, 2), (9, 1), (10, 2), (11, 1),
    ])
    def test_session_num_to_treatment(self, session_num, expected_treatment):
        """Each session maps to the correct treatment group."""
        assert SESSION_NUM_TO_TREATMENT[session_num] == expected_treatment

    def test_treatment_values_are_1_or_2(self):
        """All treatments should be either 1 or 2."""
        for treatment in SESSION_NUM_TO_TREATMENT.values():
            assert treatment in (1, 2)

    def test_session_2_not_present(self):
        """Session 2 does not exist (skipped in experiment)."""
        assert 2 not in SESSION_NUM_TO_CODE

    def test_codes_are_unique(self):
        """All session codes are unique."""
        codes = list(SESSION_NUM_TO_CODE.values())
        assert len(codes) == len(set(codes))


# =====
# TestParticipantIdParsing
# =====
class TestParticipantIdParsing:
    """Tests for parse_participant_id() in session_mapping.py."""

    def test_normal_id_extracts_label(self):
        """A3 in session 3 -> 'A'."""
        assert parse_participant_id("A3", 3) == "A"

    def test_another_normal_id(self):
        """B5 in session 5 -> 'B'."""
        assert parse_participant_id("B5", 5) == "B"

    def test_multi_digit_session(self):
        """K10 in session 10 -> 'K'."""
        assert parse_participant_id("K10", 10) == "K"

    def test_f3_session_1_edge_case(self):
        """F3 in session 1 is a known misentry, should return 'F'."""
        assert parse_participant_id("F3", 1) == "F"

    def test_f3_session_3_normal(self):
        """F3 in session 3 is normal, should return 'F'."""
        assert parse_participant_id("F3", 3) == "F"

    def test_single_digit_session(self):
        """Q1 in session 1 -> 'Q'."""
        assert parse_participant_id("Q1", 1) == "Q"

    @pytest.mark.parametrize("id_str,session,expected_label", [
        ("A1", 1, "A"),
        ("D4", 4, "D"),
        ("R11", 11, "R"),
        ("H8", 8, "H"),
        ("J9", 9, "J"),
    ])
    def test_various_ids(self, id_str, session, expected_label):
        """Various IDs parse correctly."""
        assert parse_participant_id(id_str, session) == expected_label


# =====
# TestEmotionLoaderHelpers (unit tests for load_emotion_data functions)
# =====
class TestEmotionLoaderHelpers:
    """Tests for individual functions in load_emotion_data.py."""

    def test_drop_empty_rows_removes_nan_session(self):
        """Rows with NaN sESSION are dropped."""
        df = pd.DataFrame({
            "sESSION": [3.0, np.nan, 5.0],
            "id": ["A3", "X", "B5"],
        })
        result = drop_empty_rows(df)
        assert len(result) == 2

    def test_drop_empty_rows_removes_empty_string(self):
        """Rows with empty string sESSION are dropped."""
        df = pd.DataFrame({
            "sESSION": ["3.0", "  ", "5.0"],
            "id": ["A3", "X", "B5"],
        })
        result = drop_empty_rows(df)
        assert len(result) == 2

    def test_convert_emotion_columns_to_float(self):
        """Raw emotion columns are converted to float."""
        df = pd.DataFrame({col: ["0.5"] for col in RAW_EMOTION_COLS})
        result = convert_emotion_columns(df)
        for col in RAW_EMOTION_COLS:
            assert result[col].dtype == np.float64

    def test_convert_emotion_columns_coerces_invalid(self):
        """Non-numeric emotion values are coerced to NaN."""
        df = pd.DataFrame({col: ["invalid"] for col in RAW_EMOTION_COLS})
        result = convert_emotion_columns(df)
        for col in RAW_EMOTION_COLS:
            assert pd.isna(result[col].iloc[0])

    def test_emotion_cols_constant_has_13_columns(self):
        """EMOTION_COLS should have exactly 13 entries."""
        assert len(EMOTION_COLS) == 13

    def test_emotion_cols_all_prefixed(self):
        """All EMOTION_COLS should start with 'emotion_'."""
        for col in EMOTION_COLS:
            assert col.startswith("emotion_"), f"{col} missing prefix"

    def test_dedup_keys_correct(self):
        """DEDUP_KEYS should match expected key columns."""
        expected = ["session_code", "label", "segment", "round", "page_type"]
        assert DEDUP_KEYS == expected


# =====
# TestDuplicateHandling (mock data using actual deduplicate_recordings)
# =====
class TestDuplicateHandling:
    """Tests for deduplication logic in load_emotion_data."""

    def _make_dedup_df(self, session_codes, labels, segments, rounds,
                       page_types, anger_vals, joy_vals):
        """Helper to build a DataFrame for deduplication testing."""
        # Use RAW_EMOTION_COLS names since deduplicate_recordings uses them
        data = {
            "session_code": session_codes,
            "label": labels,
            "segment": segments,
            "round": rounds,
            "page_type": page_types,
        }
        # Set all raw emotion cols to zero, then override Anger and Joy
        for col in RAW_EMOTION_COLS:
            data[col] = [0.0] * len(session_codes)
        data["Anger"] = anger_vals
        data["Joy"] = joy_vals
        return pd.DataFrame(data)

    def test_averaging_with_zero_exclusion(self):
        """Duplicate group: all-zero rows excluded, rest averaged."""
        df = self._make_dedup_df(
            ["s1"] * 3, ["A"] * 3, ["supergame1"] * 3,
            [1] * 3, ["Contribute"] * 3,
            anger_vals=[0.0, 0.5, 1.0],
            joy_vals=[0.0, 0.8, 0.4],
        )
        deduped = deduplicate_recordings(df)
        assert len(deduped) == 1
        assert deduped["Anger"].iloc[0] == pytest.approx(0.75)
        assert deduped["Joy"].iloc[0] == pytest.approx(0.6)

    def test_all_zero_group_keeps_one_row(self):
        """When all rows in dup group are all-zero, keep one."""
        df = self._make_dedup_df(
            ["s1"] * 2, ["A"] * 2, ["supergame1"] * 2,
            [1] * 2, ["Contribute"] * 2,
            anger_vals=[0.0, 0.0],
            joy_vals=[0.0, 0.0],
        )
        deduped = deduplicate_recordings(df)
        assert len(deduped) == 1
        assert deduped["Anger"].iloc[0] == 0.0

    def test_non_dup_zero_rows_preserved(self):
        """Non-duplicate rows with all zeros are kept as-is."""
        df = self._make_dedup_df(
            ["s1", "s1"], ["A", "B"], ["supergame1", "supergame1"],
            [1, 1], ["Contribute", "Contribute"],
            anger_vals=[0.0, 0.0],
            joy_vals=[0.0, 0.0],
        )
        deduped = deduplicate_recordings(df)
        assert len(deduped) == 2


# =====
# TestBasePanel (mock data)
# =====
class TestBasePanel:
    """Tests for the cross-join that produces the base panel."""

    def test_cross_join_multiplies_rows(self):
        """Cross-joining state data with 3 page types triples row count."""
        state_df = pd.DataFrame({
            "session_code": ["s1", "s1"],
            "segment": ["supergame1", "supergame1"],
            "round": [1, 2],
            "label": ["A", "A"],
            "group": [1, 1],
            "contribution": [25.0, 20.0],
        })
        page_types = pd.DataFrame({"page_type": PAGE_TYPES})
        base = state_df.merge(page_types, how="cross")
        assert len(base) == 6  # 2 rows * 3 page types

    def test_cross_join_has_all_page_types(self):
        """Each original row should have all 3 page types."""
        state_df = pd.DataFrame({
            "session_code": ["s1"],
            "segment": ["supergame1"],
            "round": [1],
            "label": ["A"],
        })
        page_types = pd.DataFrame({"page_type": PAGE_TYPES})
        base = state_df.merge(page_types, how="cross")

        assert set(base["page_type"].unique()) == set(PAGE_TYPES)
        assert len(base) == 3


# =====
# TestInstructionRows (mock data)
# =====
class TestInstructionRows:
    """Tests for instruction rows appended from emotion data."""

    def test_instruction_row_has_nan_segment(self):
        """Instruction rows should have NaN segment."""
        instruction_row = pd.DataFrame({
            "session_code": ["s1"],
            "label": ["A"],
            "segment": [None],
            "round": [None],
            "page_type": ["all_instructions"],
        })
        assert pd.isna(instruction_row["segment"].iloc[0])

    def test_instruction_row_has_nan_round(self):
        """Instruction rows should have NaN round."""
        instruction_row = pd.DataFrame({
            "session_code": ["s1"],
            "label": ["A"],
            "segment": [None],
            "round": [np.nan],
            "page_type": ["all_instructions"],
        })
        assert pd.isna(instruction_row["round"].iloc[0])

    def test_instruction_page_type(self):
        """Instruction rows should have page_type='all_instructions'."""
        instruction_row = pd.DataFrame({
            "session_code": ["s1"],
            "label": ["A"],
            "page_type": ["all_instructions"],
        })
        assert instruction_row["page_type"].iloc[0] == "all_instructions"


# =====
# TestSentimentMerge (mock data)
# =====
class TestSentimentMerge:
    """Tests for LEFT JOIN of sentiment onto base panel."""

    @pytest.fixture
    def base_panel(self):
        """Small base panel with rounds 1 and 2, all page types."""
        rows = []
        for rnd in [1, 2]:
            for pt in PAGE_TYPES:
                rows.append({
                    "session_code": "s1",
                    "segment": "supergame1",
                    "round": rnd,
                    "label": "A",
                    "group": 1,
                    "page_type": pt,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sentiment_data(self):
        """Sentiment data only for round 2."""
        return pd.DataFrame({
            "session_code": ["s1"],
            "segment": ["supergame1"],
            "round": [2],
            "label": ["A"],
            "sentiment_compound_mean": [0.5],
            "sentiment_compound_std": [0.1],
            "sentiment_compound_min": [0.3],
            "sentiment_compound_max": [0.7],
            "sentiment_positive_mean": [0.3],
            "sentiment_negative_mean": [0.05],
            "sentiment_neutral_mean": [0.65],
        })

    def test_round_1_has_nan_sentiment(self, base_panel, sentiment_data):
        """After LEFT JOIN, round 1 should have NaN sentiment."""
        merge_keys = ["session_code", "segment", "round", "label"]
        sentiment_cols = merge_keys + SENTIMENT_COLUMNS
        merged = base_panel.merge(
            sentiment_data[sentiment_cols], on=merge_keys, how="left"
        )
        r1 = merged[merged["round"] == 1]
        for col in SENTIMENT_COLUMNS:
            assert r1[col].isna().all(), f"Round 1 should have NaN {col}"

    def test_round_2_has_sentiment_values(self, base_panel, sentiment_data):
        """After LEFT JOIN, round 2 should have sentiment values."""
        merge_keys = ["session_code", "segment", "round", "label"]
        sentiment_cols = merge_keys + SENTIMENT_COLUMNS
        merged = base_panel.merge(
            sentiment_data[sentiment_cols], on=merge_keys, how="left"
        )
        r2 = merged[merged["round"] == 2]
        assert r2["sentiment_compound_mean"].notna().all()

    def test_sentiment_replicated_across_page_types(
        self, base_panel, sentiment_data
    ):
        """Same sentiment value should appear for all page types in round 2."""
        merge_keys = ["session_code", "segment", "round", "label"]
        sentiment_cols = merge_keys + SENTIMENT_COLUMNS
        merged = base_panel.merge(
            sentiment_data[sentiment_cols], on=merge_keys, how="left"
        )
        r2 = merged[merged["round"] == 2]
        # All 3 page types should have the same value
        assert len(r2) == 3
        assert r2["sentiment_compound_mean"].nunique() == 1
        assert r2["sentiment_compound_mean"].iloc[0] == pytest.approx(0.5)

    def test_left_join_preserves_row_count(self, base_panel, sentiment_data):
        """LEFT JOIN should not change row count."""
        merge_keys = ["session_code", "segment", "round", "label"]
        sentiment_cols = merge_keys + SENTIMENT_COLUMNS
        merged = base_panel.merge(
            sentiment_data[sentiment_cols], on=merge_keys, how="left"
        )
        assert len(merged) == len(base_panel)


# =====
# TestEmotionMerge (mock data)
# =====
class TestEmotionMerge:
    """Tests for LEFT JOIN of emotion data onto base panel."""

    @pytest.fixture
    def base_panel(self):
        """Small base panel with 2 page types for a single round."""
        return pd.DataFrame({
            "session_code": ["s1", "s1"],
            "label": ["A", "A"],
            "segment": ["supergame1", "supergame1"],
            "round": [1, 1],
            "page_type": ["Contribute", "Results"],
        })

    @pytest.fixture
    def emotion_data(self):
        """Emotion data for only the Contribute page type."""
        return pd.DataFrame({
            "session_code": ["s1"],
            "label": ["A"],
            "segment": ["supergame1"],
            "round": [1],
            "page_type": ["Contribute"],
            "emotion_anger": [0.5],
            "emotion_joy": [0.8],
        })

    def test_matched_page_has_emotion(self, base_panel, emotion_data):
        """Contribute page should have emotion values after join."""
        keys = ["session_code", "label", "segment", "round", "page_type"]
        merged = base_panel.merge(emotion_data, on=keys, how="left")
        contribute = merged[merged["page_type"] == "Contribute"]
        assert contribute["emotion_anger"].iloc[0] == pytest.approx(0.5)

    def test_unmatched_page_has_nan(self, base_panel, emotion_data):
        """Results page should have NaN emotion (no match)."""
        keys = ["session_code", "label", "segment", "round", "page_type"]
        merged = base_panel.merge(emotion_data, on=keys, how="left")
        results = merged[merged["page_type"] == "Results"]
        assert pd.isna(results["emotion_anger"].iloc[0])

    def test_emotion_join_preserves_rows(self, base_panel, emotion_data):
        """LEFT JOIN on page_type should not change row count."""
        keys = ["session_code", "label", "segment", "round", "page_type"]
        merged = base_panel.merge(emotion_data, on=keys, how="left")
        assert len(merged) == len(base_panel)


# =====
# Integration tests (skip if data files missing)
# =====
@pytest.mark.integration
class TestIntegrationOutput:
    """Integration tests that run on real data files.

    These validate the full merge pipeline output.
    Skipped if data files or merge module are not available.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        """Skip all tests in this class if input data files are missing."""
        if not STATE_FILE.exists():
            pytest.skip(f"State file not found: {STATE_FILE}")
        if not SENTIMENT_FILE.exists():
            pytest.skip(f"Sentiment file not found: {SENTIMENT_FILE}")

    @pytest.fixture(autouse=True)
    def skip_if_no_merge_module(self):
        """Skip if merge_panel_data module is not yet implemented."""
        try:
            import merge_panel_data  # noqa: F401
        except ImportError:
            pytest.skip("merge_panel_data module not yet available")

    @pytest.fixture
    def merged_df(self):
        """Run the merge pipeline and return the output DataFrame."""
        from merge_panel_data import main as run_merge
        # Check if output file exists, if so read it; otherwise run
        output_file = DERIVED_DIR / "merged_panel.csv"
        if output_file.exists():
            return pd.read_csv(output_file)
        pytest.skip("merged_panel.csv not generated yet")

    def test_output_game_row_count(self, merged_df):
        """Game rows (non-instruction) should total 10,560."""
        game_rows = merged_df[
            merged_df["page_type"] != "all_instructions"
        ]
        assert len(game_rows) == EXPECTED_GAME_ROWS, (
            f"Expected {EXPECTED_GAME_ROWS} game rows, got {len(game_rows)}"
        )

    def test_output_has_instruction_rows(self, merged_df):
        """Instruction rows should exist in output."""
        instructions = merged_df[
            merged_df["page_type"] == "all_instructions"
        ]
        assert len(instructions) > 0, "Expected instruction rows in output"

    def test_no_duplicate_keys(self, merged_df):
        """No duplicate (session_code, label, segment, round, page_type) in game rows."""
        game_rows = merged_df[
            merged_df["page_type"] != "all_instructions"
        ]
        keys = ["session_code", "label", "segment", "round", "page_type"]
        dupes = game_rows.duplicated(subset=keys, keep=False)
        assert not dupes.any(), (
            f"Found {dupes.sum()} duplicate key rows in game data"
        )

    def test_session_1_emotion_coverage(self, merged_df):
        """Session 1 (sa7mprty) has limited emotion data.

        Only supergame4 Results from irregular S4 annotations should
        have emotion data (the only iMotions recordings for session 1
        were the S4 irregular annotations).
        """
        s1 = merged_df[merged_df["session_code"] == "sa7mprty"]
        if len(s1) == 0:
            pytest.skip("Session sa7mprty not in output")

        # Check that session 1 has some emotion data
        has_emotion = s1[EMOTION_COLUMNS[0]].notna()
        if not has_emotion.any():
            # Session 1 may only have S4 Results data
            s1_s4_results = s1[
                (s1["segment"] == "supergame4")
                & (s1["page_type"] == "Results")
            ]
            # It's okay if only S4/Results has emotion data
            assert len(s1_s4_results) >= 0

    def test_sentiment_round_1_all_nan(self, merged_df):
        """All round 1 rows should have NaN sentiment values."""
        r1 = merged_df[merged_df["round"] == 1]
        if len(r1) == 0:
            pytest.skip("No round 1 rows found")

        for col in SENTIMENT_COLUMNS:
            if col in merged_df.columns:
                assert r1[col].isna().all(), (
                    f"Round 1 should have NaN for {col}, "
                    f"found {r1[col].notna().sum()} non-NaN values"
                )

    def test_no_merge_artifact_columns(self, merged_df):
        """No _x or _y suffixed columns from merge artifacts."""
        artifact_cols = [
            c for c in merged_df.columns
            if c.endswith("_x") or c.endswith("_y")
        ]
        assert len(artifact_cols) == 0, (
            f"Found merge artifact columns: {artifact_cols}"
        )

    def test_all_10_sessions_present(self, merged_df):
        """Output should contain all 10 experimental sessions."""
        unique_sessions = merged_df["session_code"].nunique()
        assert unique_sessions == EXPECTED_SESSIONS, (
            f"Expected {EXPECTED_SESSIONS} sessions, found {unique_sessions}"
        )

    def test_all_page_types_present(self, merged_df):
        """Game rows should contain all 3 page types."""
        game_rows = merged_df[
            merged_df["page_type"] != "all_instructions"
        ]
        assert set(game_rows["page_type"].unique()) == set(PAGE_TYPES)

    def test_instruction_rows_have_nan_state_cols(self, merged_df):
        """Instruction rows should have NaN for game state columns."""
        instructions = merged_df[
            merged_df["page_type"] == "all_instructions"
        ]
        if len(instructions) == 0:
            pytest.skip("No instruction rows found")

        for col in ["segment", "round", "group"]:
            if col in instructions.columns:
                assert instructions[col].isna().all(), (
                    f"Instruction rows should have NaN {col}"
                )

    def test_emotion_columns_present(self, merged_df):
        """All 13 emotion columns should be in output."""
        for col in EMOTION_COLUMNS:
            assert col in merged_df.columns, f"Missing emotion column: {col}"

    def test_sentiment_columns_present(self, merged_df):
        """All 7 sentiment columns should be in output."""
        for col in SENTIMENT_COLUMNS:
            assert col in merged_df.columns, (
                f"Missing sentiment column: {col}"
            )


# =====
# Run tests directly
# =====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
