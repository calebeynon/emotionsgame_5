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
    'others_contribution_1', 'others_contribution_2', 'others_contribution_3',
    'othercont', 'othercontaverage',
    'othercontmin', 'othercontmax', 'othercontmed',
    'morethanaverage', 'lessthanaverage', 'diffcont',
    'contmore', 'contless', 'contmore_L1', 'contless_L1',
    'morethanmin', 'lessthanmin', 'diffcontmin',
    'contmoremin', 'contlessmin', 'contmoremin_L1', 'contlessmin_L1',
    'morethanmax', 'lessthanmax', 'diffcontmax',
    'contmoremax', 'contlessmax', 'contmoremax_L1', 'contlessmax_L1',
    'morethanmed', 'lessthanmed', 'diffcontmed',
    'contmoremed', 'contlessmed', 'contmoremed_L1', 'contlessmed_L1',
    # round1 is the only dummy used in regressions (Stata Table DP1 spec).
    # round2..round7 are retained for schema stability: dynamic_regression.R coerces
    # them to int if present (load_and_prepare_data), and the merged-panel test pins
    # the full column set.
    'round1', 'round2', 'round3', 'round4', 'round5', 'round6', 'round7',
    'word_count', 'made_promise', 'sentiment_compound_mean', 'emotion_valence',
]

# Deviation tags for per-peer min/max/med derivations
DEVIATION_TAGS = ['min', 'max', 'med']

# Upper bound on no-message rounds (r>1 rows with empty chat).
# Observed round>1 NaN count is 422; 800 gives headroom.
# Round-1 NaN is filtered out by the mask in fill_no_message_rounds before this
# check (those rows legitimately have no prior chat). R's bound (850) covers
# total NaN across all rows — hence the different magnitude.
MAX_NO_MESSAGE_ROUNDS = 800


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

    merged = safe_left_merge(base_df, behavior_df, 'behavior_classifications')
    merged = safe_left_merge(merged, panel_df, 'merged_panel')
    print(f"After merge: {len(merged):,} rows")
    return merged


def safe_left_merge(left: pd.DataFrame, right: pd.DataFrame,
                    right_name: str) -> pd.DataFrame:
    """Left-merge with one_to_one validation and row-count guard."""
    base_rows = len(left)
    merged = left.merge(right, on=MERGE_KEYS, how='left', validate='one_to_one')
    if len(merged) != base_rows:
        raise ValueError(
            f"Row count changed merging {right_name}: "
            f"{base_rows} -> {len(merged)}. Check key uniqueness in {right_name}."
        )
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
    for col in ('word_count', 'sentiment_compound_mean'):
        nan_count = df.loc[mask, col].isna().sum()
        print(f"  {col} NaN in rounds>1 (no-message rounds): {nan_count}")
        if nan_count > MAX_NO_MESSAGE_ROUNDS:
            raise ValueError(
                f"{col} NaN count {nan_count} exceeds bound {MAX_NO_MESSAGE_ROUNDS}. "
                f"Check upstream chat aggregation for missing keys."
            )
        df.loc[mask, col] = df.loc[mask, col].fillna(0)
    return df


# =====
# Convert made_promise boolean to integer
# =====
def convert_made_promise(df: pd.DataFrame) -> pd.DataFrame:
    """Convert made_promise from True/False to 1/0 integer."""
    nan_mask = df['made_promise'].isna()
    if nan_mask.any():
        offenders = df.loc[nan_mask, MERGE_KEYS].to_dict('records')
        raise ValueError(
            f"made_promise has {nan_mask.sum()} NaN before astype(int). "
            f"Offending keys: {offenders[:5]}{'...' if len(offenders) > 5 else ''}"
        )
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
    # othercont is reverse-engineered from payoff for parity with the Stata do-file;
    # summing others_contribution_{1,2,3} would be numerically equivalent.
    df['othercont'] = (df['payoff'] - 25 + 0.6 * df['contribution']) / 0.4
    df['othercontaverage'] = df['othercont'] / 3

    df = build_deviation('average', df['othercontaverage'], df)
    df = derive_peer_order_stats(df)
    for tag in DEVIATION_TAGS:
        df = build_deviation(tag, df[f'othercont{tag}'], df)
    return df


def derive_peer_order_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute min/max/median across individual peer contributions."""
    peers = df[['others_contribution_1', 'others_contribution_2', 'others_contribution_3']]
    df['othercontmin'] = peers.min(axis=1)
    df['othercontmax'] = peers.max(axis=1)
    df['othercontmed'] = peers.median(axis=1)
    return df


def build_deviation(tag: str, reference: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Create morethan/lessthan/diff/contmore/contless columns against a reference series."""
    # The "average" tag is asymmetric: it produces unsuffixed Stata-compatible names
    # (contmore/contless/diffcont), while min/med/max — added in issue #68 — use the
    # tag as a suffix (e.g. contmoremax, diffcontmax).
    suffix = tag if tag != 'average' else 'average'
    more_col = f'morethan{suffix}'
    less_col = f'lessthan{suffix}'
    diff_col = 'diffcont' if tag == 'average' else f'diffcont{tag}'
    more_val = 'contmore' if tag == 'average' else f'contmore{tag}'
    less_val = 'contless' if tag == 'average' else f'contless{tag}'

    df[more_col] = (df['contribution'] > reference).astype(int)
    df[less_col] = (df['contribution'] < reference).astype(int)
    df[diff_col] = df['contribution'] - reference
    df[more_val] = df[diff_col] * df[more_col]
    df[less_val] = -df[diff_col] * df[less_col]
    return df


# =====
# Lag variables (within-subject, ordered by period)
# =====
def create_lag_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged deviation variables within each subject."""
    df = df.sort_values(['subject_id', 'period'])
    lag_sources = ['contmore', 'contless']
    for tag in DEVIATION_TAGS:
        lag_sources.extend([f'contmore{tag}', f'contless{tag}'])
    for col in lag_sources:
        df[f'{col}_L1'] = df.groupby('subject_id')[col].shift(1)
        df.loc[df['period'] == 1, f'{col}_L1'] = float('nan')
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
    validate_chat_columns(df)
    validate_peer_stats(df)
    validate_lags(df)
    print("\nValidation passed: 3,520 rows, no duplicates, types correct")


def validate_chat_columns(df: pd.DataFrame):
    """word_count and sentiment NaN only allowed at round 1."""
    r2plus = df[df['round'] > 1]
    wc_nan = r2plus['word_count'].isna().sum()
    assert wc_nan == 0, f"word_count has {wc_nan} NaN in rounds > 1"
    sc_nan = r2plus['sentiment_compound_mean'].isna().sum()
    assert sc_nan == 0, f"sentiment_compound_mean has {sc_nan} NaN in rounds > 1"


def validate_peer_stats(df: pd.DataFrame):
    """Ensure min/max/med peer stats are present and consistent."""
    for tag in DEVIATION_TAGS:
        col = f'othercont{tag}'
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"{col} has {nan_count} NaN values"
    assert (df['othercontmin'] <= df['othercontmed']).all(), "min > med in peer stats"
    assert (df['othercontmed'] <= df['othercontmax']).all(), "med > max in peer stats"


def validate_lags(df: pd.DataFrame):
    """Ensure lag columns are NaN iff period==1."""
    lag_cols = ['contmore_L1', 'contless_L1']
    for tag in DEVIATION_TAGS:
        lag_cols.extend([f'contmore{tag}_L1', f'contless{tag}_L1'])
    for col in lag_cols:
        nan_at_p1 = df.loc[df['period'] == 1, col].notna().sum()
        assert nan_at_p1 == 0, f"{col} has non-NaN at period==1"
        nan_after = df.loc[df['period'] > 1, col].isna().sum()
        assert nan_after == 0, f"{col} has {nan_after} NaN at period>1"


# %%
if __name__ == "__main__":
    main()
