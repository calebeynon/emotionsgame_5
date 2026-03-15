"""
Unit tests for the panel data merge pipeline (Issue #38).

Covers session_mapping.py and load_emotion_data.py using mock data only.
No data files required.

Author: Claude Code
Date: 2026-03-11
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from session_mapping import (
    SESSION_NUM_TO_CODE,
    SESSION_NUM_TO_TREATMENT,
    parse_annotation,
    parse_participant_id,
)
from load_emotion_data import (
    convert_emotion_columns,
    deduplicate_recordings,
    drop_empty_rows,
    DEDUP_KEYS,
    EMOTION_COLS,
    RAW_EMOTION_COLS,
)

# EXPECTED CONSTANTS
EXPECTED_SESSIONS = 10
PAGE_TYPES = ["Contribute", "Results", "ResultsOnly"]
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

    def test_all_instructions(self):
        """Instructions: all_instructions -> None, None, 'all_instructions'."""
        assert parse_annotation("all_instructions") == (None, None, "all_instructions")

    def test_all_instructions_fields_are_none(self):
        """all_instructions should return None for segment and round."""
        segment, round_num, _ = parse_annotation("all_instructions")
        assert segment is None
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
        assert parse_annotation(annotation) == (
            expected_segment, expected_round, expected_page
        )

    @pytest.mark.parametrize("annotation,expected_round", [
        ("S4result1", 1), ("S4Result2", 2), ("S4result3", 3),
        ("S4Results4", 4), ("S4result5", 5), ("S4result6", 6), ("S4result7", 7),
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
        (1, "sa7mprty"), (3, "irrzlgk2"), (4, "6uv359rf"), (5, "umbzdj98"),
        (6, "j3ki5tli"), (7, "r5dj4yfl"), (8, "sylq2syi"), (9, "iiu3xixz"),
        (10, "6ucza025"), (11, "6sdkxl2q"),
    ])
    def test_session_num_to_code(self, session_num, expected_code):
        """Each session number maps to the correct oTree session code."""
        assert SESSION_NUM_TO_CODE[session_num] == expected_code

    def test_all_10_sessions_present(self):
        """All 10 sessions are in the mapping dict."""
        assert len(SESSION_NUM_TO_CODE) == EXPECTED_SESSIONS

    def test_session_numbers_match(self):
        """CODE and TREATMENT dicts have the same session numbers."""
        assert set(SESSION_NUM_TO_CODE.keys()) == set(SESSION_NUM_TO_TREATMENT.keys())

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

    def test_multi_digit_session(self):
        """K10 in session 10 -> 'K'."""
        assert parse_participant_id("K10", 10) == "K"

    def test_f3_session_1_edge_case(self):
        """F3 in session 1 is a known misentry, should return 'F'."""
        assert parse_participant_id("F3", 1) == "F"

    def test_f3_session_3_normal(self):
        """F3 in session 3 is normal, should return 'F'."""
        assert parse_participant_id("F3", 3) == "F"

    @pytest.mark.parametrize("id_str,session,expected_label", [
        ("A1", 1, "A"), ("D4", 4, "D"), ("R11", 11, "R"),
        ("H8", 8, "H"), ("J9", 9, "J"),
    ])
    def test_various_ids(self, id_str, session, expected_label):
        """Various IDs parse correctly."""
        assert parse_participant_id(id_str, session) == expected_label


# =====
# TestEmotionLoaderHelpers
# =====
class TestEmotionLoaderHelpers:
    """Tests for individual functions in load_emotion_data.py."""

    def test_drop_empty_rows_removes_nan_session(self):
        """Rows with NaN sESSION are dropped."""
        df = pd.DataFrame({"sESSION": [3.0, np.nan, 5.0], "id": ["A3", "X", "B5"]})
        assert len(drop_empty_rows(df)) == 2

    def test_drop_empty_rows_removes_empty_string(self):
        """Rows with empty string sESSION are dropped."""
        df = pd.DataFrame({"sESSION": ["3.0", "  ", "5.0"], "id": ["A3", "X", "B5"]})
        assert len(drop_empty_rows(df)) == 2

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
        assert DEDUP_KEYS == ["session_code", "label", "segment", "round", "page_type"]


# =====
# TestDuplicateHandling
# =====
class TestDuplicateHandling:
    """Tests for deduplication logic in load_emotion_data."""

    def _make_dedup_df(self, session_codes, labels, segments, rounds,
                       page_types, anger_vals, joy_vals):
        """Helper to build a DataFrame for deduplication testing."""
        data = {
            "session_code": session_codes, "label": labels,
            "segment": segments, "round": rounds, "page_type": page_types,
        }
        for col in RAW_EMOTION_COLS:
            data[col] = [0.0] * len(session_codes)
        data["Anger"] = anger_vals
        data["Joy"] = joy_vals
        return pd.DataFrame(data)

    def test_averaging_with_zero_exclusion(self):
        """Duplicate group: all-zero rows excluded, rest averaged."""
        df = self._make_dedup_df(
            ["s1"] * 3, ["A"] * 3, ["supergame1"] * 3, [1] * 3, ["Contribute"] * 3,
            anger_vals=[0.0, 0.5, 1.0], joy_vals=[0.0, 0.8, 0.4],
        )
        deduped = deduplicate_recordings(df)
        assert len(deduped) == 1
        assert deduped["Anger"].iloc[0] == pytest.approx(0.75)
        assert deduped["Joy"].iloc[0] == pytest.approx(0.6)

    def test_all_zero_group_keeps_one_row(self):
        """When all rows in dup group are all-zero, keep one."""
        df = self._make_dedup_df(
            ["s1"] * 2, ["A"] * 2, ["supergame1"] * 2, [1] * 2, ["Contribute"] * 2,
            anger_vals=[0.0, 0.0], joy_vals=[0.0, 0.0],
        )
        deduped = deduplicate_recordings(df)
        assert len(deduped) == 1
        assert deduped["Anger"].iloc[0] == 0.0

    def test_non_dup_zero_rows_preserved(self):
        """Non-duplicate rows with all zeros are kept as-is."""
        df = self._make_dedup_df(
            ["s1", "s1"], ["A", "B"], ["supergame1", "supergame1"],
            [1, 1], ["Contribute", "Contribute"],
            anger_vals=[0.0, 0.0], joy_vals=[0.0, 0.0],
        )
        assert len(deduplicate_recordings(df)) == 2


# =====
# TestBasePanel
# =====
class TestBasePanel:
    """Tests for the cross-join that produces the base panel."""

    def test_cross_join_row_count_and_page_types(self):
        """Cross-join with 3 page types: row count triples, all types present."""
        state_df = pd.DataFrame({
            "session_code": ["s1"], "segment": ["supergame1"],
            "round": [1], "label": ["A"],
        })
        base = state_df.merge(pd.DataFrame({"page_type": PAGE_TYPES}), how="cross")
        assert len(base) == 3
        assert set(base["page_type"].unique()) == set(PAGE_TYPES)


# =====
# TestSentimentMerge
# =====
class TestSentimentMerge:
    """Tests for LEFT JOIN of sentiment onto base panel."""

    @pytest.fixture
    def base_panel(self):
        """Small base panel with rounds 1 and 2, all page types."""
        rows = [
            {"session_code": "s1", "segment": "supergame1",
             "round": rnd, "label": "A", "group": 1, "page_type": pt}
            for rnd in [1, 2] for pt in PAGE_TYPES
        ]
        return pd.DataFrame(rows)

    @pytest.fixture
    def sentiment_data(self):
        """Sentiment data only for round 2."""
        return pd.DataFrame({
            "session_code": ["s1"], "segment": ["supergame1"],
            "round": [2], "label": ["A"],
            "sentiment_compound_mean": [0.5], "sentiment_compound_std": [0.1],
            "sentiment_compound_min": [0.3], "sentiment_compound_max": [0.7],
            "sentiment_positive_mean": [0.3], "sentiment_negative_mean": [0.05],
            "sentiment_neutral_mean": [0.65],
        })

    def _merge(self, base_panel, sentiment_data):
        """Helper: LEFT JOIN sentiment onto panel."""
        keys = ["session_code", "segment", "round", "label"]
        cols = keys + SENTIMENT_COLUMNS
        return base_panel.merge(sentiment_data[cols], on=keys, how="left")

    def test_round_1_has_nan_sentiment(self, base_panel, sentiment_data):
        """After LEFT JOIN, round 1 should have NaN sentiment."""
        merged = self._merge(base_panel, sentiment_data)
        r1 = merged[merged["round"] == 1]
        for col in SENTIMENT_COLUMNS:
            assert r1[col].isna().all(), f"Round 1 should have NaN {col}"

    def test_round_2_has_sentiment_values(self, base_panel, sentiment_data):
        """After LEFT JOIN, round 2 should have sentiment values."""
        merged = self._merge(base_panel, sentiment_data)
        assert merged[merged["round"] == 2]["sentiment_compound_mean"].notna().all()

    def test_sentiment_replicated_across_page_types(self, base_panel, sentiment_data):
        """Same sentiment value should appear for all page types in round 2."""
        merged = self._merge(base_panel, sentiment_data)
        r2 = merged[merged["round"] == 2]
        assert len(r2) == 3
        assert r2["sentiment_compound_mean"].nunique() == 1
        assert r2["sentiment_compound_mean"].iloc[0] == pytest.approx(0.5)

    def test_left_join_preserves_row_count(self, base_panel, sentiment_data):
        """LEFT JOIN should not change row count."""
        merged = self._merge(base_panel, sentiment_data)
        assert len(merged) == len(base_panel)


# =====
# TestEmotionMerge
# =====
class TestEmotionMerge:
    """Tests for LEFT JOIN of emotion data onto base panel."""

    @pytest.fixture
    def base_panel(self):
        """Small base panel with 2 page types for a single round."""
        return pd.DataFrame({
            "session_code": ["s1", "s1"], "label": ["A", "A"],
            "segment": ["supergame1", "supergame1"], "round": [1, 1],
            "page_type": ["Contribute", "Results"],
        })

    @pytest.fixture
    def emotion_data(self):
        """Emotion data for only the Contribute page type."""
        return pd.DataFrame({
            "session_code": ["s1"], "label": ["A"],
            "segment": ["supergame1"], "round": [1],
            "page_type": ["Contribute"],
            "emotion_anger": [0.5], "emotion_joy": [0.8],
        })

    def _merge(self, base_panel, emotion_data):
        """Helper: LEFT JOIN emotion onto panel."""
        keys = ["session_code", "label", "segment", "round", "page_type"]
        return base_panel.merge(emotion_data, on=keys, how="left")

    def test_matched_page_has_emotion(self, base_panel, emotion_data):
        """Contribute page should have emotion values after join."""
        merged = self._merge(base_panel, emotion_data)
        contribute = merged[merged["page_type"] == "Contribute"]
        assert contribute["emotion_anger"].iloc[0] == pytest.approx(0.5)

    def test_unmatched_page_has_nan(self, base_panel, emotion_data):
        """Results page should have NaN emotion (no match)."""
        merged = self._merge(base_panel, emotion_data)
        results = merged[merged["page_type"] == "Results"]
        assert pd.isna(results["emotion_anger"].iloc[0])

    def test_emotion_join_preserves_rows(self, base_panel, emotion_data):
        """LEFT JOIN on page_type should not change row count."""
        assert len(self._merge(base_panel, emotion_data)) == len(base_panel)


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
