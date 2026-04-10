"""
Purpose: End-to-end verification that chat→emotion pairing is correct.
    Traces specific player-round observations through the full data pipeline
    to confirm the shift logic aligns chat, projections, and emotions correctly.
Author: Claude Code
Date: 2026-04-09
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
RAW_DIR = Path(__file__).parent.parent / "datastore" / "raw"
IMOTIONS_FILE = Path(__file__).parent.parent / "datastore" / "Rwork" / "all.csv"
MERGED_FILE = DERIVED_DIR / "merged_panel.csv"
EMBEDDING_PROJ_FILE = DERIVED_DIR / "embedding_projections.csv"
PROMISE_CLASS_FILE = DERIVED_DIR / "promise_classifications.csv"

SESSIONS = {
    "irrzlgk2": {"num": 3, "raw_prefix": "03_t2"},
    "j3ki5tli": {"num": 6, "raw_prefix": "06_t2"},
}
FLOAT_TOL = 1e-5
IMOTIONS_EMOTION_COLS = [
    "Anger", "Contempt", "Disgust", "Fear", "Joy", "Sadness", "Surprise",
    "Engagement", "Valence", "Sentimentality", "Confusion", "Neutral", "Attention",
]
MERGE_KEYS = ["session_code", "segment", "round", "label"]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def merged_df():
    """Load merged panel; skip if missing."""
    if not MERGED_FILE.exists():
        pytest.skip("merged_panel.csv not found")
    return pd.read_csv(MERGED_FILE)


@pytest.fixture(scope="module")
def embedding_proj_df():
    """Load embedding projections; skip if missing."""
    if not EMBEDDING_PROJ_FILE.exists():
        pytest.skip("embedding_projections.csv not found")
    return pd.read_csv(EMBEDDING_PROJ_FILE)


@pytest.fixture(scope="module")
def promise_class_df():
    """Load promise classifications; skip if missing."""
    if not PROMISE_CLASS_FILE.exists():
        pytest.skip("promise_classifications.csv not found")
    return pd.read_csv(PROMISE_CLASS_FILE)


@pytest.fixture(scope="module")
def imotions_df():
    """Load raw iMotions data; skip if missing."""
    if not IMOTIONS_FILE.exists():
        pytest.skip("iMotions all.csv not found")
    return pd.read_csv(IMOTIONS_FILE)


# =====
# Helpers
# =====
def _load_raw_chat(session_code):
    """Load raw chat CSV for a session."""
    info = SESSIONS[session_code]
    path = RAW_DIR / f"{info['raw_prefix']}_chat.csv"
    if not path.exists():
        pytest.skip(f"Raw chat file not found: {path}")
    return pd.read_csv(path)


def _get_panel_row(df, session_code, segment, round_num, label):
    """Extract a single Contribute row from merged panel."""
    mask = (
        (df["session_code"] == session_code) & (df["segment"] == segment)
        & (df["round"] == round_num) & (df["label"] == label)
        & (df["page_type"] == "Contribute")
    )
    rows = df[mask]
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    return rows.iloc[0]


def _filter_df(df, session_code, segment, round_num, label):
    """Filter a DataFrame to a specific player-round."""
    return df[
        (df["session_code"] == session_code) & (df["segment"] == segment)
        & (df["round"] == round_num) & (df["label"] == label)
    ]


def _parse_segment_chat(chat_df, segment):
    """Extract segment chat with parsed chatgroup column."""
    seg_num = int(re.search(r"\d+", segment).group())
    seg = chat_df[chat_df["channel"].str.contains(f"supergame{seg_num}")].copy()
    seg["chatgroup"] = seg["channel"].str.extract(r"-(\d+)$").astype(int)
    return seg


def _get_group_chatgroups(seg_chat, label):
    """Find all chatgroups for the game group containing a player."""
    player_cgs = seg_chat[seg_chat["nickname"] == label]["chatgroup"].unique()
    if len(player_cgs) == 0:
        return []
    group_players = set(seg_chat[seg_chat["chatgroup"] == min(player_cgs)]["nickname"])
    return sorted(seg_chat[seg_chat["nickname"].isin(group_players)]["chatgroup"].unique())


def _get_shifted_messages(seg_chat, round_num, label):
    """Get messages from round N-1's chatgroup that influenced round N."""
    group_cgs = _get_group_chatgroups(seg_chat, label)
    source_idx = round_num - 2  # round N-1, 0-indexed
    if source_idx < 0 or source_idx >= len(group_cgs):
        return []
    target_cg = group_cgs[source_idx]
    msgs = seg_chat[
        (seg_chat["chatgroup"] == target_cg) & (seg_chat["nickname"] == label)
    ].sort_values("timestamp")
    return msgs["body"].tolist()


def _dedup_emotion(raw_rows):
    """Apply same dedup logic as load_emotion_data.py."""
    is_zero = (raw_rows[IMOTIONS_EMOTION_COLS] == 0).all(axis=1)
    non_zero = raw_rows[~is_zero]
    if non_zero.empty:
        return raw_rows.iloc[0]
    return non_zero[IMOTIONS_EMOTION_COLS].mean()


def _get_imotions_annotation(session_code, segment, round_num, label):
    """Build iMotions ID and annotation string from player-round identifiers."""
    session_num = SESSIONS[session_code]["num"]
    imotions_id = f"{label}{session_num}"
    annotation = f"s{segment[-1]}r{round_num}Contribute"
    return imotions_id, annotation


# =====
# Test Case 1: Player A, irrzlgk2, supergame2, round 3 (multi-message, dedup)
# =====
@pytest.mark.integration
class TestTraceCase1:
    """Trace A/irrzlgk2/supergame2/round3: 2 messages, iMotions dedup."""

    S, SEG, R, L = "irrzlgk2", "supergame2", 3, "A"
    MSGS = ["Agreed", "Hopefully this is the segment they pick for the payout"]

    def test_raw_chat_matches_promise_classifications(self, promise_class_df):
        """Raw chat round 2 messages match promise_classifications round 3."""
        seg_chat = _parse_segment_chat(_load_raw_chat(self.S), self.SEG)
        assert _get_shifted_messages(seg_chat, self.R, self.L) == self.MSGS
        pc = _filter_df(promise_class_df, self.S, self.SEG, self.R, self.L)
        assert json.loads(pc.iloc[0]["messages"]) == self.MSGS

    def test_projections_aggregate_to_panel(self, embedding_proj_df, merged_df):
        """Message-level projections average to merged_panel values."""
        ep = _filter_df(embedding_proj_df, self.S, self.SEG, self.R, self.L)
        assert len(ep) == 2
        panel = _get_panel_row(merged_df, self.S, self.SEG, self.R, self.L)
        assert abs(panel["proj_pr_dir_small"] - ep["proj_pr_dir_small"].mean()) < FLOAT_TOL
        assert abs(panel["proj_msg_dir_small"] - ep["proj_msg_dir_small"].mean()) < FLOAT_TOL

    def test_embedding_texts_match_raw_chat(self, embedding_proj_df):
        """Embedding message_text fields match raw chat bodies."""
        ep = _filter_df(embedding_proj_df, self.S, self.SEG, self.R, self.L)
        assert ep.sort_values("message_index")["message_text"].tolist() == self.MSGS

    def test_chat_timestamps_precede_next_round(self):
        """Round 1 chat timestamps < round 2 chat timestamps for this group."""
        seg_chat = _parse_segment_chat(_load_raw_chat(self.S), self.SEG)
        cgs = _get_group_chatgroups(seg_chat, self.L)
        assert len(cgs) >= 2
        max_r1 = seg_chat[seg_chat["chatgroup"] == cgs[0]]["timestamp"].max()
        min_r2 = seg_chat[seg_chat["chatgroup"] == cgs[1]]["timestamp"].min()
        assert max_r1 < min_r2

    def test_emotion_from_contribute_annotation(self, imotions_df, merged_df):
        """Panel emotion matches iMotions s2r3Contribute after dedup."""
        imo_id, annot = _get_imotions_annotation(self.S, self.SEG, self.R, self.L)
        raw = imotions_df[(imotions_df["id"] == imo_id) & (imotions_df["Respondent Annotations active"] == annot)]
        assert len(raw) > 0
        expected = _dedup_emotion(raw)
        panel = _get_panel_row(merged_df, self.S, self.SEG, self.R, self.L)
        for imo_col, panel_col in [("Valence", "emotion_valence"), ("Joy", "emotion_joy"), ("Anger", "emotion_anger")]:
            assert abs(panel[panel_col] - expected[imo_col]) < FLOAT_TOL


# =====
# Test Case 2: Player D, j3ki5tli, supergame3, round 2 (single message, no dedup)
# =====
@pytest.mark.integration
class TestTraceCase2:
    """Trace D/j3ki5tli/supergame3/round2: 1 message, single iMotions row."""

    S, SEG, R, L = "j3ki5tli", "supergame3", 2, "D"
    MSGS = ["yeah"]

    def test_raw_chat_matches_promise_classifications(self, promise_class_df):
        """Single raw chat message matches promise_classifications round 2."""
        seg_chat = _parse_segment_chat(_load_raw_chat(self.S), self.SEG)
        assert _get_shifted_messages(seg_chat, self.R, self.L) == self.MSGS
        pc = _filter_df(promise_class_df, self.S, self.SEG, self.R, self.L)
        assert json.loads(pc.iloc[0]["messages"]) == self.MSGS

    def test_single_projection_equals_panel(self, embedding_proj_df, merged_df):
        """Single-message projection equals panel value directly."""
        ep = _filter_df(embedding_proj_df, self.S, self.SEG, self.R, self.L)
        assert len(ep) == 1
        panel = _get_panel_row(merged_df, self.S, self.SEG, self.R, self.L)
        assert abs(panel["proj_pr_dir_small"] - ep.iloc[0]["proj_pr_dir_small"]) < FLOAT_TOL

    def test_emotion_from_contribute_annotation(self, imotions_df, merged_df):
        """Panel emotion matches single iMotions s3r2Contribute row."""
        imo_id, annot = _get_imotions_annotation(self.S, self.SEG, self.R, self.L)
        raw = imotions_df[(imotions_df["id"] == imo_id) & (imotions_df["Respondent Annotations active"] == annot)]
        assert len(raw) == 1
        panel = _get_panel_row(merged_df, self.S, self.SEG, self.R, self.L)
        assert abs(panel["emotion_valence"] - raw.iloc[0]["Valence"]) < FLOAT_TOL


# =====
# Structural shift-logic validation
# =====
@pytest.mark.integration
class TestShiftLogicStructural:
    """Validate shift-logic properties across the full dataset."""

    def test_round_1_has_no_projections(self, merged_df):
        """Round 1 Contribute rows have NaN projections (no prior chat)."""
        r1 = merged_df[(merged_df["round"] == 1) & (merged_df["page_type"] == "Contribute")]
        assert len(r1) > 0
        assert r1["proj_pr_dir_small"].isna().all()

    def test_round_1_has_no_sentiment(self, merged_df):
        """Round 1 Contribute rows have NaN sentiment (no prior chat)."""
        r1 = merged_df[(merged_df["round"] == 1) & (merged_df["page_type"] == "Contribute")]
        assert r1["sentiment_compound_mean"].isna().all()

    def test_round_2_plus_has_some_projections(self, merged_df):
        """Rounds 2+ have at least some non-NaN projections."""
        r2p = merged_df[(merged_df["round"] > 1) & (merged_df["page_type"] == "Contribute")]
        assert r2p["proj_pr_dir_small"].notna().any()

    def test_promise_classifications_round_1_empty(self, promise_class_df):
        """Round 1 promise_classifications have 0 messages."""
        r1 = promise_class_df[promise_class_df["round"] == 1]
        assert (r1["message_count"] == 0).all()

    def test_within_group_chatgroup_timestamps_increase(self):
        """Within each game group, sequential chatgroup timestamps increase."""
        for session_code in SESSIONS:
            chat_df = _load_raw_chat(session_code)
            for seg_num in range(1, 6):
                seg = f"supergame{seg_num}"
                seg_chat = _parse_segment_chat(chat_df, seg)
                if seg_chat.empty:
                    continue
                _assert_group_timestamps_increase(seg_chat, session_code, seg)

    def test_projection_merge_keys_align(self, embedding_proj_df, promise_class_df):
        """Embedding projection keys are a subset of promise classification keys."""
        proj_keys = set(embedding_proj_df.groupby(MERGE_KEYS).groups.keys())
        pc_keys = set(
            promise_class_df[promise_class_df["message_count"] > 0]
            .groupby(MERGE_KEYS).groups.keys()
        )
        missing = proj_keys - pc_keys
        assert len(missing) == 0, f"{len(missing)} projection keys not in promise_classifications"


def _assert_group_timestamps_increase(seg_chat, session_code, segment):
    """Assert timestamps increase across rounds within each game group."""
    cg_players = {
        cg: frozenset(seg_chat[seg_chat["chatgroup"] == cg]["nickname"])
        for cg in seg_chat["chatgroup"].unique()
    }
    visited = set()
    for cg, players in sorted(cg_players.items()):
        if cg in visited:
            continue
        group_cgs = sorted(c for c, p in cg_players.items() if p & players)
        visited.update(group_cgs)
        for i in range(len(group_cgs) - 1):
            max_ts = seg_chat[seg_chat["chatgroup"] == group_cgs[i]]["timestamp"].max()
            min_ts = seg_chat[seg_chat["chatgroup"] == group_cgs[i + 1]]["timestamp"].min()
            assert max_ts < min_ts, (
                f"{session_code}/{segment}: CG {group_cgs[i]} max_ts >= CG {group_cgs[i+1]} min_ts"
            )


# =====
# Emotion annotation verification
# =====
@pytest.mark.integration
class TestEmotionAnnotationAlignment:
    """Verify emotion data comes from correct page annotations."""

    def test_contribute_annotations_have_correct_format(self, imotions_df):
        """All Contribute annotations match sNrMContribute pattern."""
        annots = imotions_df["Respondent Annotations active"].dropna().unique()
        contribute = [a for a in annots if "Contribute" in str(a)]
        assert len(contribute) > 0
        pattern = re.compile(r"^s\d+r\d+Contribute$")
        for annot in contribute:
            assert pattern.match(annot), f"Bad format: {annot}"

    def test_emotion_dedup_produces_unique_keys(self, merged_df):
        """Each (session, label, segment, round, page_type) is unique in game rows."""
        game = merged_df[merged_df["page_type"] != "all_instructions"]
        keys = ["session_code", "label", "segment", "round", "page_type"]
        assert not game.duplicated(subset=keys, keep=False).any()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
