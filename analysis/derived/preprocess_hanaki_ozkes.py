"""
Preprocess Hanaki & Ozkes (2023) raw zTree session logs into a tidy parquet.

Parses tab-delimited .xls files, extracts chat and subjects tables,
filters to Chat=1 treatment sessions, concatenates chat per player-period,
and merges with investment decisions.

Author: Claude Code
Date: 2026-03-26
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# FILE PATHS
DATA_DIR = Path("analysis/datastore/raw/external_datasets/hanaki_ozkes_2023/Data")
OUTPUT_FILE = Path("analysis/datastore/derived/hanaki_ozkes_chat_decisions.parquet")

# FILES TO SKIP (aggregates / non-session files)
SKIP_FILES = {
    "Data_for_Stata.xls",
    "Top200_Alln.xls",
    "Data_for_RL_MLE.csv",
    "UniformAll.csv",
    "Uniform0108.csv",
}

# Session-to-treatment mapping from BehaviorCode.R lines 73-104
CHAT1_COMP1 = [
    "160503_0820", "160511_0853", "160513_0850", "160519_0934",
    "161018_0903", "161018_1209", "161019_0824", "170510_0938",
    "171018_0829", "171024_1240", "171025_1204", "171122_0835",
]
CHAT1_COMP0 = [
    "160510_0920", "160511_1426", "160512_1420", "161019_1411",
    "170510_1319", "171018_1228", "171024_0842", "171122_1153",
    "171122_1447",
]
CHAT1_2021 = ["211118_1312", "211118_1537"]

# Build lookup: session_id -> (Chat, Comp)
TREATMENT_MAP = {}
for _s in CHAT1_COMP1:
    TREATMENT_MAP[_s] = (1, 1)
for _s in CHAT1_COMP0:
    TREATMENT_MAP[_s] = (1, 0)
for _s in CHAT1_2021:
    TREATMENT_MAP[_s] = (1, 0)


# =====
# Main function
# =====
def main():
    """Parse all Chat=1 sessions and output merged chat-decisions parquet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    session_files = _collect_session_files()
    subjects, chats = _parse_all_sessions(session_files)
    chat_agg = _aggregate_chat(chats)
    merged = _merge_chat_decisions(subjects, chat_agg)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_FILE, index=False)
    logger.info("Wrote %d rows to %s", len(merged), OUTPUT_FILE)


def _parse_all_sessions(session_files):
    """Parse subjects and chat tables from all session files."""
    all_subjects = []
    all_chats = []
    for filepath in session_files:
        session_id = filepath.stem
        chat, comp = TREATMENT_MAP[session_id]
        lines = _read_lines(filepath)
        game_table = _find_game_table(lines)
        col_map = _build_column_map(lines, game_table)
        subjects_df = _parse_subjects(lines, session_id, game_table, col_map)
        subjects_df["Chat"] = chat
        subjects_df["Comp"] = comp
        all_subjects.append(subjects_df)
        chat_df = _parse_chat(lines, session_id, game_table)
        if not chat_df.empty:
            all_chats.append(chat_df)
    subjects = pd.concat(all_subjects, ignore_index=True)
    chats = pd.concat(all_chats, ignore_index=True)
    return subjects, chats


# =====
# File discovery
# =====
def _collect_session_files():
    """Return sorted list of .xls paths for Chat=1 sessions only."""
    chat1_ids = set(TREATMENT_MAP.keys())
    files = []
    for path in sorted(DATA_DIR.glob("*.xls")):
        if path.name in SKIP_FILES or path.stem.endswith("_"):
            continue
        if path.stem in chat1_ids:
            files.append(path)
    logger.info("Found %d Chat=1 session files", len(files))
    return files


# =====
# Parsing helpers
# =====
def _read_lines(filepath):
    """Read all lines from a latin-1 encoded file."""
    return filepath.read_text(encoding="latin-1").splitlines()


def _split_line(line):
    """Split a line on tabs, stripping trailing carriage return."""
    return line.rstrip("\r").split("\t")


def _find_game_table(lines):
    """Find the table_num whose subjects header contains 'Inv'."""
    for line in lines:
        fields = _split_line(line)
        if len(fields) > 3 and fields[2] == "subjects" and fields[3] == "Period":
            if "Inv" in fields:
                return fields[1]
    raise ValueError("No subjects table with 'Inv' column found")


def _build_column_map(lines, game_table):
    """Map column names to indices from the first subjects header row."""
    for line in lines:
        fields = _split_line(line)
        if len(fields) > 3 and fields[1] == game_table and fields[2] == "subjects" and fields[3] == "Period":
            return {name: i for i, name in enumerate(fields)}
    raise ValueError("Subjects header not found for game table")


def _parse_subjects(lines, session_id, game_table, col_map):
    """Extract subjects data rows using header-derived column indices."""
    rows = []
    for line in lines:
        fields = _split_line(line)
        if len(fields) < 3:
            continue
        if fields[1] != game_table or fields[2] != "subjects":
            continue
        if fields[3] == "Period":
            continue
        rows.append(_extract_subject_row(fields, session_id, col_map))
    return pd.DataFrame(rows)


def _extract_subject_row(fields, session_id, col_map):
    """Build a dict from one subjects row using column map."""
    return {
        "session_file": session_id,
        "period": int(fields[col_map["Period"]]),
        "player_id": int(fields[col_map["Subject"]]),
        "group": int(fields[col_map["Group"]]),
        "Inv": _safe_numeric(fields[col_map["Inv"]]),
        "OtherInv": _safe_numeric(fields[col_map["OtherInv"]]),
        "Profit": _safe_numeric(fields[col_map["Profit"]]),
    }


def _parse_chat(lines, session_id, game_table):
    """Extract chat table rows for the game table."""
    rows = []
    for line in lines:
        fields = _split_line(line)
        if len(fields) < 8:
            continue
        if fields[1] != game_table or fields[2] != "chat":
            continue
        if fields[3] == "Period":
            continue
        rows.append(_extract_chat_row(fields, session_id))
    return pd.DataFrame(rows)


def _extract_chat_row(fields, session_id):
    """Build a dict from one chat row's tab-delimited fields."""
    words = fields[7].strip('"')
    return {
        "session_file": session_id,
        "period": int(fields[3]),
        "player_id": int(fields[4]),
        "chat_text": words,
    }


def _safe_numeric(value):
    """Convert a string to float, returning NaN for non-numeric values."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


# =====
# Aggregation and merge
# =====
def _aggregate_chat(chat_df):
    """Concatenate all chat messages per player per period."""
    grouped = chat_df.groupby(
        ["session_file", "period", "player_id"], sort=False
    )["chat_text"].agg(" ".join).reset_index()
    return grouped


def _merge_chat_decisions(subjects_df, chat_df):
    """Left-join chat onto subjects at the player-period level."""
    merged = subjects_df.merge(
        chat_df,
        on=["session_file", "period", "player_id"],
        how="left",
    )
    merged["chat_text"] = merged["chat_text"].fillna("")
    return merged


# %%
if __name__ == "__main__":
    main()
