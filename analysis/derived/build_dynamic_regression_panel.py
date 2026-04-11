"""
Build regression-ready panel for dynamic GMM estimation.

Merges contributions, behavior classifications, and merged_panel (chat/emotion)
data into a single CSV with derived panel variables matching the R/Stata
dynamic regression specification.

Author: Claude Code
Date: 2026-04-10
"""

from pathlib import Path

import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
CONTRIBUTIONS_FILE = DERIVED_DIR / 'contributions.csv'
BEHAVIOR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
MERGED_PANEL_FILE = DERIVED_DIR / 'merged_panel.csv'
OUTPUT_FILE = DERIVED_DIR / 'dynamic_regression_panel.csv'

# MERGE KEYS
MERGE_KEYS = ['session_code', 'segment', 'round', 'label']

# Period offsets for linearizing round across supergames
PERIOD_OFFSETS = {1: 0, 2: 3, 3: 7, 4: 10, 5: 17}

# Output column order
OUTPUT_COLUMNS = [
    'session_code', 'treatment', 'segment', 'round', 'group', 'label',
    'participant_id', 'contribution', 'payoff',
    'segmentnumber', 'period', 'subject_id',
    'othercont', 'othercontaverage',
    'morethanaverage', 'lessthanaverage', 'diffcont',
    'contmore', 'contless', 'contmore_L1', 'contless_L1',
    'round1', 'round2', 'round3', 'round4', 'round5', 'round6', 'round7',
    'word_count', 'made_promise', 'sentiment_compound_mean', 'emotion_valence',
]


# =====
# Main function (FIRST - shows high-level flow)
# =====
def main():
    """Main execution flow."""
    base_df = pd.read_csv(CONTRIBUTIONS_FILE)
    print(f"Loaded contributions: {len(base_df):,} rows")

    merged_df = merge_all_sources(base_df)
    merged_df = fill_no_message_rounds(merged_df)
    merged_df = convert_made_promise(merged_df)
    merged_df = derive_panel_variables(merged_df)
    merged_df = create_lag_variables(merged_df)
    merged_df = create_round_dummies(merged_df)

    validate(merged_df)

    merged_df[OUTPUT_COLUMNS].to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"Shape: {merged_df.shape[0]:,} rows x {len(OUTPUT_COLUMNS)} columns")


# =====
# Merge data sources
# =====
def merge_all_sources(base_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join behavior classifications and merged_panel onto contributions."""
    behavior_df = pd.read_csv(BEHAVIOR_FILE)[MERGE_KEYS + ['made_promise']]
    panel_df = load_filtered_panel()

    merged = base_df.merge(behavior_df, on=MERGE_KEYS, how='left')
    merged = merged.merge(panel_df, on=MERGE_KEYS, how='left')
    print(f"After merge: {len(merged):,} rows")
    return merged


def load_filtered_panel() -> pd.DataFrame:
    """Load merged_panel.csv filtered to Contribute rows with selected columns."""
    df = pd.read_csv(MERGED_PANEL_FILE)
    df = df[df['page_type'] == 'Contribute'].copy()
    # Round is float in merged_panel; convert to int to match contributions
    df['round'] = df['round'].astype(int)
    return df[MERGE_KEYS + ['word_count', 'sentiment_compound_mean', 'emotion_valence']]


# =====
# Fill NaN for no-message rounds
# =====
def fill_no_message_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """Fill word_count and sentiment NaN with 0 for rounds > 1 (no messages sent)."""
    mask = df['round'] > 1
    df.loc[mask, 'word_count'] = df.loc[mask, 'word_count'].fillna(0)
    df.loc[mask, 'sentiment_compound_mean'] = df.loc[mask, 'sentiment_compound_mean'].fillna(0)
    return df


# =====
# Convert made_promise boolean to integer
# =====
def convert_made_promise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert made_promise from True/False to 1/0 integer."""
    df['made_promise'] = df['made_promise'].astype(int)
    return df


# =====
# Derive panel variables (matching R dynamic_regression.R logic)
# =====
def derive_panel_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create segmentnumber, period, subject_id, and deviation measures."""
    df['segmentnumber'] = df['segment'].str.extract(r'(\d+)').astype(int)
    df['period'] = df.apply(
        lambda r: r['round'] + PERIOD_OFFSETS[r['segmentnumber']], axis=1
    )
    df = assign_session_numbers(df)
    df = derive_deviation_measures(df)
    return df


def assign_session_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Create sessionnumber and subject_id from session ordering."""
    session_order = {s: i + 1 for i, s in enumerate(df['session_code'].unique())}
    df['sessionnumber'] = df['session_code'].map(session_order)
    df['subject_id'] = df['sessionnumber'] * 100 + df['participant_id']
    return df


def derive_deviation_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Compute othercont, othercontaverage, and contribution deviations."""
    # payoff = 25 - contribution + (contribution + othercont) * 0.4
    df['othercont'] = (df['payoff'] - 25 + 0.6 * df['contribution']) / 0.4
    df['othercontaverage'] = df['othercont'] / 3

    df['morethanaverage'] = (df['contribution'] > df['othercontaverage']).astype(int)
    df['lessthanaverage'] = (df['contribution'] < df['othercontaverage']).astype(int)

    df['diffcont'] = df['contribution'] - df['othercontaverage']
    df['contmore'] = df['diffcont'] * df['morethanaverage']
    df['contless'] = -df['diffcont'] * df['lessthanaverage']
    return df


# =====
# Lag variables (within-subject, ordered by period)
# =====
def create_lag_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged deviation variables within each subject."""
    df = df.sort_values(['subject_id', 'period'])
    df['contmore_L1'] = df.groupby('subject_id')['contmore'].shift(1)
    df['contless_L1'] = df.groupby('subject_id')['contless'].shift(1)
    # Period 1 has no valid lag
    df.loc[df['period'] == 1, 'contmore_L1'] = float('nan')
    df.loc[df['period'] == 1, 'contless_L1'] = float('nan')
    return df


# =====
# Round dummies
# =====
def create_round_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Create round indicator variables round1 through round7."""
    for r in range(1, 8):
        df[f'round{r}'] = (df['round'] == r).astype(int)
    return df


# =====
# Validation
# =====
def validate(df: pd.DataFrame):
    """Validate output meets expected structure."""
    assert len(df) == 3520, f"Expected 3,520 rows, got {len(df):,}"

    dupes = df.duplicated(subset=MERGE_KEYS)
    assert not dupes.any(), f"Found {dupes.sum()} duplicate key rows"

    assert df['made_promise'].isna().sum() == 0, "made_promise has NaN values"
    assert set(df['made_promise'].unique()) <= {0, 1}, "made_promise has non-0/1 values"

    # word_count and sentiment NaN only at round 1
    r2plus = df[df['round'] > 1]
    wc_nan = r2plus['word_count'].isna().sum()
    assert wc_nan == 0, f"word_count has {wc_nan} NaN in rounds > 1"
    sc_nan = r2plus['sentiment_compound_mean'].isna().sum()
    assert sc_nan == 0, f"sentiment_compound_mean has {sc_nan} NaN in rounds > 1"

    print("\nValidation passed: 3,520 rows, no duplicates, types correct")


# %%
if __name__ == "__main__":
    main()
