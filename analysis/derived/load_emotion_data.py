"""
Load and clean iMotions emotion data from raw CSV export.

Reads the combined emotion data file, parses annotations and participant IDs,
maps session numbers to oTree session codes, deduplicates multi-segment
recordings, and returns a clean DataFrame for panel data merging.

Author: Claude Code
Date: 2026-03-11
"""

from pathlib import Path

import pandas as pd

from session_mapping import (
    SESSION_NUM_TO_CODE,
    SESSION_NUM_TO_TREATMENT,
    parse_annotation,
    parse_participant_id,
)

# FILE PATHS
RAW_FILE = Path(__file__).parent.parent / 'datastore' / 'Rwork' / 'all.csv'

# COLUMN DEFINITIONS
RAW_EMOTION_COLS = [
    'Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
    'Engagement', 'Valence', 'Sentimentality', 'Confusion', 'Neutral', 'Attention',
]

EMOTION_RENAME = {col: f'emotion_{col.lower()}' for col in RAW_EMOTION_COLS}

EMOTION_COLS = list(EMOTION_RENAME.values())

DEDUP_KEYS = ['session_code', 'label', 'segment', 'round', 'page_type']


# =====
# Main function
# =====
def main():
    """Load emotion data and print summary."""
    df = load_emotion_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSessions: {sorted(df['session_code'].unique())}")
    print(f"Segments: {sorted(df['segment'].dropna().unique())}")
    print(f"Page types: {sorted(df['page_type'].unique())}")
    print(f"\nNaN counts:\n{df.isna().sum()}")


# =====
# Public API
# =====
def load_emotion_data() -> pd.DataFrame:
    """Load and clean emotion data, returning panel-ready DataFrame."""
    df = pd.read_csv(RAW_FILE)
    df = drop_empty_rows(df)
    df = parse_columns(df)
    df = convert_emotion_columns(df)
    df = deduplicate_recordings(df)
    df = finalize_columns(df)
    return df


# =====
# Cleaning steps
# =====
def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where sESSION is empty or whitespace."""
    df['sESSION'] = df['sESSION'].astype(str).str.strip()
    mask = (df['sESSION'] != '') & (df['sESSION'] != 'nan')
    return df[mask].copy()


def parse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse annotation and ID columns into structured fields."""
    df['sESSION'] = df['sESSION'].astype(float).astype(int)
    parsed = df['Respondent Annotations active'].apply(parse_annotation)
    df['segment'] = parsed.apply(lambda x: x[0])
    df['round'] = parsed.apply(lambda x: x[1])
    df['page_type'] = parsed.apply(lambda x: x[2])
    df['label'] = df.apply(
        lambda row: parse_participant_id(row['id'], row['sESSION']), axis=1
    )
    df['session_code'] = df['sESSION'].map(SESSION_NUM_TO_CODE)
    df['treatment'] = df['sESSION'].map(SESSION_NUM_TO_TREATMENT)
    return df


def convert_emotion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw emotion columns to float."""
    for col in RAW_EMOTION_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# =====
# Deduplication
# =====
def deduplicate_recordings(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate multi-segment recordings by averaging non-zero rows.

    Uses fillna sentinel for NaN groupby keys to avoid pandas KeyError.
    """
    df = _add_dedup_key(df)
    grouped = df.groupby('_dedup_key')
    results = [_resolve_one_group(g, RAW_EMOTION_COLS) for _, g in grouped]
    out = pd.concat(results, ignore_index=True)
    return out.drop(columns=['_dedup_key'])


def _add_dedup_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create composite string key for deduplication, NaN-safe."""
    parts = [df[c].astype(str).fillna('__NA__') for c in DEDUP_KEYS]
    df = df.copy()
    df['_dedup_key'] = parts[0].str.cat(parts[1:], sep='|')
    return df


def _resolve_one_group(group: pd.DataFrame, emotion_cols: list) -> pd.DataFrame:
    """Resolve a single duplicate group to one averaged row."""
    if len(group) == 1:
        return group
    is_zero = (group[emotion_cols] == 0).all(axis=1)
    non_zero = group[~is_zero]
    if non_zero.empty:
        return group.head(1)
    result = non_zero.head(1).copy()
    for col in emotion_cols:
        result[col] = non_zero[col].mean()
    return result


# =====
# Column finalization
# =====
def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename emotion columns and select final output columns."""
    df = df.rename(columns=EMOTION_RENAME)
    output_cols = [
        'session_code', 'treatment', 'label', 'segment', 'round', 'page_type',
    ] + EMOTION_COLS
    return df[output_cols].reset_index(drop=True)


# %%
if __name__ == "__main__":
    main()
