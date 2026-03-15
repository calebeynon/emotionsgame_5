"""
Merge state classification, sentiment, emotion, and embedding data into a single panel.

Builds a panel dataset by cross-joining player-round state data with page types,
appending instruction-phase emotion rows, and LEFT JOINing sentiment, emotion,
and embedding projection scores. Validates row counts and key uniqueness before saving.

Author: Claude Code
Date: 2026-03-11
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_emotion_data import EMOTION_COLS, load_emotion_data

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
STATE_FILE = DERIVED_DIR / 'player_state_classification.csv'
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
PROJECTION_FILE = DERIVED_DIR / 'embedding_projections.csv'
PROMISE_PROJECTION_FILE = DERIVED_DIR / 'promise_embedding_projections.csv'
OUTPUT_FILE = DERIVED_DIR / 'merged_panel.csv'

# MERGE CONFIGURATION
PAGE_TYPES = ['Contribute', 'Results', 'ResultsOnly']
STATE_MERGE_KEYS = ['session_code', 'segment', 'round', 'label']
EMOTION_MERGE_KEYS = ['session_code', 'label', 'segment', 'round', 'page_type']
EXPECTED_GAME_ROWS = 10560

SENTIMENT_COLS = [
    'sentiment_compound_mean', 'sentiment_compound_std',
    'sentiment_compound_min', 'sentiment_compound_max',
    'sentiment_positive_mean', 'sentiment_negative_mean',
    'sentiment_neutral_mean',
]

EMBEDDING_COLS = [
    'proj_msg_dir_small', 'proj_pr_dir_small',
    'proj_msg_dir_large', 'proj_pr_dir_large',
]

PROMISE_EMBEDDING_COLS = [
    'proj_promise_msg_dir_small', 'proj_promise_pr_dir_small',
    'proj_promise_msg_dir_large', 'proj_promise_pr_dir_large',
]

OUTPUT_ORDER = [
    'session_code', 'treatment', 'segment', 'round', 'group', 'label',
    'page_type', 'contribution', 'others_total_contribution', 'player_state',
    'player_behavior', 'made_promise', 'others_threshold', 'player_threshold',
]


# =====
# Main function
# =====
def main():
    """Build merged panel from state, sentiment, and emotion data."""
    state_df = load_state_data()
    sentiment_df = load_sentiment_data()
    emotion_df = load_emotion_data()

    panel = build_game_panel(state_df)
    panel = append_instruction_rows(panel, emotion_df)
    panel = merge_sentiment(panel, sentiment_df)
    projection_df = load_projection_data()
    panel = merge_projections(panel, projection_df)
    promise_proj_df = load_promise_projection_data()
    panel = merge_promise_projections(panel, promise_proj_df)
    panel = merge_emotion(panel, emotion_df)

    validate_panel(panel)
    save_panel(panel)
    print_summary(panel)


# =====
# Data loading
# =====
def load_state_data() -> pd.DataFrame:
    """Load state classification, renaming columns for consistency."""
    df = pd.read_csv(STATE_FILE)
    df = df.rename(columns={'round_num': 'round', 'group_id': 'group'})
    return df


def load_sentiment_data() -> pd.DataFrame:
    """Load sentiment scores, selecting only merge keys and score columns."""
    df = pd.read_csv(SENTIMENT_FILE)
    return df[STATE_MERGE_KEYS + SENTIMENT_COLS]


def load_projection_data() -> pd.DataFrame:
    """Load projections, aggregate message-level to player-round means."""
    df = pd.read_csv(PROJECTION_FILE)
    grouped = df.groupby(STATE_MERGE_KEYS)[EMBEDDING_COLS].mean()
    return grouped.reset_index()


def load_promise_projection_data() -> pd.DataFrame:
    """Load promise projections, aggregate message-level to player-round means."""
    df = pd.read_csv(PROMISE_PROJECTION_FILE)
    grouped = df.groupby(STATE_MERGE_KEYS)[PROMISE_EMBEDDING_COLS].mean()
    return grouped.reset_index()


# =====
# Panel construction
# =====
def build_game_panel(state_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-join state data with page types to create base panel."""
    page_df = pd.DataFrame({'page_type': PAGE_TYPES})
    panel = state_df.merge(page_df, how='cross')
    print(f"Game panel: {len(panel)} rows (expected {EXPECTED_GAME_ROWS})")
    return panel[OUTPUT_ORDER]


def append_instruction_rows(panel: pd.DataFrame, emotion_df: pd.DataFrame) -> pd.DataFrame:
    """Append instruction-phase rows from emotion data."""
    instr = emotion_df[emotion_df['page_type'] == 'all_instructions'].copy()
    instr_rows = instr[['session_code', 'treatment', 'label', 'page_type']]
    combined = pd.concat([panel, instr_rows], ignore_index=True)
    n_instr = len(instr_rows)
    print(f"Added {n_instr} instruction rows -> {len(combined)} total")
    return combined


# =====
# Merge operations
# =====
def merge_sentiment(panel: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN sentiment scores onto panel."""
    merged = panel.merge(sentiment_df, on=STATE_MERGE_KEYS, how='left')
    n_matched = merged[SENTIMENT_COLS[0]].notna().sum()
    print(f"Sentiment merge: {n_matched} rows matched")
    return merged


def merge_projections(panel: pd.DataFrame, projection_df: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN embedding projection scores onto panel."""
    merged = panel.merge(projection_df, on=STATE_MERGE_KEYS, how='left')
    n_matched = merged[EMBEDDING_COLS[0]].notna().sum()
    print(f"Projection merge: {n_matched} rows matched")
    return merged


def merge_promise_projections(
    panel: pd.DataFrame, promise_df: pd.DataFrame
) -> pd.DataFrame:
    """LEFT JOIN promise embedding projection scores onto panel."""
    merged = panel.merge(promise_df, on=STATE_MERGE_KEYS, how='left')
    n_matched = merged[PROMISE_EMBEDDING_COLS[0]].notna().sum()
    print(f"Promise projection merge: {n_matched} rows matched")
    return merged


def merge_emotion(panel: pd.DataFrame, emotion_df: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN emotion scores onto panel."""
    emotion_subset = emotion_df[EMOTION_MERGE_KEYS + EMOTION_COLS]
    merged = panel.merge(emotion_subset, on=EMOTION_MERGE_KEYS, how='left')
    n_matched = merged[EMOTION_COLS[0]].notna().sum()
    print(f"Emotion merge: {n_matched} rows matched")
    return merged


# =====
# Validation
# =====
def validate_panel(panel: pd.DataFrame):
    """Run all validation checks on the merged panel."""
    _validate_game_row_count(panel)
    _validate_no_duplicate_keys(panel)
    _validate_no_suffix_columns(panel)
    _validate_round_1_sentiment(panel)
    _validate_round_1_embeddings(panel)
    _validate_instruction_rows(panel)
    print("All validations passed")


def _validate_game_row_count(panel: pd.DataFrame):
    """Verify expected number of game rows."""
    game = panel[panel['page_type'] != 'all_instructions']
    if len(game) != EXPECTED_GAME_ROWS:
        raise ValueError(f"Expected {EXPECTED_GAME_ROWS} game rows, got {len(game)}")


def _validate_no_duplicate_keys(panel: pd.DataFrame):
    """Verify no duplicate keys in game rows."""
    game = panel[panel['page_type'] != 'all_instructions']
    keys = ['session_code', 'segment', 'round', 'label', 'page_type']
    dups = game.duplicated(subset=keys, keep=False)
    if dups.any():
        raise ValueError(f"Found {dups.sum()} duplicate key rows")


def _validate_no_suffix_columns(panel: pd.DataFrame):
    """Verify no _x or _y columns from merge collisions."""
    bad = [c for c in panel.columns if c.endswith('_x') or c.endswith('_y')]
    if bad:
        raise ValueError(f"Merge collision columns found: {bad}")


def _validate_round_1_sentiment(panel: pd.DataFrame):
    """Verify round 1 has NaN sentiment (no prior chat)."""
    r1 = panel[panel['round'] == 1]
    for col in SENTIMENT_COLS:
        if r1[col].notna().any():
            raise ValueError(f"Round 1 has non-NaN values in {col}")


def _validate_round_1_embeddings(panel: pd.DataFrame):
    """Verify round 1 has NaN embedding projections (no prior chat)."""
    r1 = panel[panel['round'] == 1]
    for col in EMBEDDING_COLS + PROMISE_EMBEDDING_COLS:
        if r1[col].notna().any():
            raise ValueError(f"Round 1 has non-NaN values in {col}")


def _validate_instruction_rows(panel: pd.DataFrame):
    """Verify instruction rows have NaN for state columns."""
    instr = panel[panel['page_type'] == 'all_instructions']
    for col in ['player_state', 'contribution', 'segment']:
        if instr[col].notna().any():
            raise ValueError(f"Instruction rows have non-NaN in {col}")


# =====
# Output
# =====
def save_panel(panel: pd.DataFrame):
    """Save merged panel to CSV with enforced column order."""
    final_order = (
        OUTPUT_ORDER + SENTIMENT_COLS + EMBEDDING_COLS
        + PROMISE_EMBEDDING_COLS + EMOTION_COLS
    )
    panel = panel[final_order]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


def print_summary(panel: pd.DataFrame):
    """Print summary of the merged panel."""
    game = panel[panel['page_type'] != 'all_instructions']
    instr = panel[panel['page_type'] == 'all_instructions']
    print(f"\nPanel: {len(panel)} rows, {len(panel.columns)} columns")
    print(f"  Game rows: {len(game)}, Instruction rows: {len(instr)}")
    print(f"  Sessions: {panel['session_code'].nunique()}")
    print(f"  Columns: {list(panel.columns)}")


# %%
if __name__ == "__main__":
    main()
