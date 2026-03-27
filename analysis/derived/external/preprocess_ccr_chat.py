"""
Parse and clean CCR (Chen, Chen & Riyanto 2021) raw chat transcripts.

Reads raw chat from four sites via parse_ccr_chat module, aggregates to
group level (116 rows, one per session x color team), validates against
chat_lines.dta, and saves parquet for embedding.

Author: Claude Code
Date: 2026-03-26
"""

import logging
from pathlib import Path

import pandas as pd

from parse_ccr_chat import parse_txt_sites, parse_ztree_sites

logger = logging.getLogger(__name__)

# FILE PATHS
CCR_BASE = Path(
    "analysis/datastore/raw/external_datasets/chen_chen_riyanto_2021"
)
RAW_DATA = CCR_BASE / "2 - Raw Data"
PROCESSED_DATA = CCR_BASE / "3 - Processed Data"
DATA_DTA = PROCESSED_DATA / "data.dta"
CHAT_LINES_DTA = PROCESSED_DATA / "Chat" / "chat_lines.dta"
OUTPUT_DIR = Path("analysis/datastore/derived/external")
OUTPUT_FILE = OUTPUT_DIR / "ccr_chat_clean.parquet"

# Site directories
ORIGINAL_CHATS = RAW_DATA / "0-Original" / "Chats"
SCIENCE_CHATS = RAW_DATA / "1-Science Replication" / "Chats"
NUS_CHAT_EFFORT = RAW_DATA / "2-NUS Replication" / "Chat & Effort"
NTU_CHAT_EFFORT = RAW_DATA / "3-NTU Replication" / "Chat & Effort"


# =====
# Main function
# =====
def main():
    """Parse all chat sources, aggregate, validate, save parquet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _validate_paths()
    session_meta = _load_session_metadata()
    txt_msgs = parse_txt_sites(ORIGINAL_CHATS, SCIENCE_CHATS)
    ztree_msgs = parse_ztree_sites(NUS_CHAT_EFFORT, NTU_CHAT_EFFORT)
    all_messages = pd.concat([txt_msgs, ztree_msgs], ignore_index=True)
    group_df = _aggregate_to_group_level(all_messages, session_meta)
    _validate_output(group_df)
    _save_output(group_df)
    _print_summary(group_df, all_messages)


# =====
# Path validation
# =====
def _validate_paths():
    """Check that required data files and directories exist."""
    required = [
        (DATA_DTA, "data.dta"),
        (CHAT_LINES_DTA, "chat_lines.dta"),
        (ORIGINAL_CHATS, "Original chats"),
        (SCIENCE_CHATS, "Science chats"),
        (NUS_CHAT_EFFORT, "NUS chat & effort"),
        (NTU_CHAT_EFFORT, "NTU chat & effort"),
    ]
    for path, label in required:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}. "
                "Check that the datastore symlink is set up."
            )


# =====
# Session metadata
# =====
def _load_session_metadata():
    """Load session-level treatment indicators from data.dta."""
    data = pd.read_stata(DATA_DTA)
    meta = (
        data[["session", "run", "ingroup", "commonknow"]]
        .drop_duplicates()
        .sort_values("session")
        .astype(int)
        .reset_index(drop=True)
    )
    if len(meta) != 58:
        raise ValueError(f"Expected 58 sessions, got {len(meta)}")
    return meta


# =====
# Group-level aggregation
# =====
def _aggregate_to_group_level(messages, session_meta):
    """Aggregate messages to one row per (session, red).

    Builds a 116-row scaffold so groups with no messages get empty text.
    """
    scaffold = _build_group_scaffold(session_meta)
    group_agg = _compute_group_aggregates(messages)
    merged = scaffold.merge(group_agg, on=["session", "red"], how="left")
    merged["group_chat_text"] = merged["group_chat_text"].fillna("")
    merged["n_messages"] = merged["n_messages"].fillna(0).astype(int)
    merged["n_words"] = merged["group_chat_text"].apply(_count_words)
    merged["n_characters"] = merged["group_chat_text"].apply(len)
    return merged.sort_values(["session", "red"]).reset_index(drop=True)


def _compute_group_aggregates(messages):
    """Aggregate messages to group-level text and counts."""
    if len(messages) == 0:
        return pd.DataFrame(
            columns=["session", "red", "group_chat_text", "n_messages"]
        )
    return (
        messages.groupby(["session", "red"])
        .agg(group_chat_text=("message", "\n".join), n_messages=("message", "count"))
        .reset_index()
    )


def _build_group_scaffold(session_meta):
    """Create a row for every (session, red) combination."""
    rows = []
    for _, row in session_meta.iterrows():
        for red in [0, 1]:
            rows.append({
                "session": row["session"], "red": red,
                "run": row["run"], "ingroup": row["ingroup"],
                "commonknow": row["commonknow"],
            })
    return pd.DataFrame(rows)


def _count_words(text):
    """Count words in text, returning 0 for empty strings."""
    return len(text.split()) if text else 0


# =====
# Validation
# =====
def _validate_output(group_df):
    """Validate output against expected counts and chat_lines.dta."""
    _validate_session_coverage(group_df)
    _validate_group_count(group_df)
    _validate_word_counts(group_df)


def _validate_session_coverage(group_df):
    """Check that all 58 sessions are represented."""
    missing = set(range(1, 59)) - set(group_df["session"].unique())
    if missing:
        raise ValueError(f"Missing sessions: {sorted(missing)}")


def _validate_group_count(group_df):
    """Check that we have 116 groups (58 sessions x 2 colors)."""
    if len(group_df) != 116:
        raise ValueError(f"Expected 116 groups, got {len(group_df)}")


def _validate_word_counts(group_df):
    """Cross-check word counts against chat_lines.dta reference."""
    reference = _build_word_count_reference()
    merged = group_df.merge(reference, on=["session", "red"], how="left")
    merged["word_diff"] = abs(merged["n_words"] - merged["ref_words"])
    max_diff = merged["word_diff"].max()
    logger.info("Word count validation: max_diff=%d, mean_diff=%.1f",
                max_diff, merged["word_diff"].mean())
    if max_diff > 20:
        worst = merged.loc[merged["word_diff"].idxmax()]
        logger.warning("Large word count discrepancy: session=%d, red=%d, diff=%d",
                       worst["session"], worst["red"], worst["word_diff"])


def _build_word_count_reference():
    """Build per-group word count reference from chat_lines.dta."""
    cl = pd.read_stata(CHAT_LINES_DTA)
    ref = (
        cl.drop_duplicates(subset=["session", "red", "line"])
        .groupby(["session", "red"])["count_words"].sum()
        .reset_index()
        .rename(columns={"count_words": "ref_words"})
    )
    ref["session"] = ref["session"].astype(int)
    ref["red"] = ref["red"].astype(int)
    return ref


# =====
# Output and summary
# =====
def _save_output(group_df):
    """Write group-level parquet to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    group_df.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Saved %d rows -> %s", len(group_df), OUTPUT_FILE)


def _print_summary(group_df, all_messages):
    """Print verification statistics."""
    logger.info("--- Chat Parsing Summary ---")
    logger.info("Total messages: %d | Groups: %d | Sessions: %d",
                len(all_messages), len(group_df), group_df["session"].nunique())
    w = group_df["n_words"]
    logger.info("Words/group: mean=%.0f, median=%.0f, min=%d, max=%d",
                w.mean(), w.median(), w.min(), w.max())
    for run_val, mean_w in group_df.groupby("run")["n_words"].mean().items():
        logger.info("  Run %d: mean %.0f words/group", run_val, mean_w)


# %%
if __name__ == "__main__":
    main()
