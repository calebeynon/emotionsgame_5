"""
Merge sentiment, promise, and behavior data for regression analysis.

Combines behavior_classifications.csv (base with all 3,520 player-rounds) with
sentiment_scores.csv and promise_classifications.csv via LEFT JOIN. Round 1
rows have no prior chat and thus have NaN for sentiment/promise columns.

Author: Claude Code
Date: 2026-01-27
"""

from pathlib import Path

import pandas as pd

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / 'datastore' / 'derived'
BEHAVIOR_FILE = DERIVED_DIR / 'behavior_classifications.csv'
SENTIMENT_FILE = DERIVED_DIR / 'sentiment_scores.csv'
PROMISE_FILE = DERIVED_DIR / 'promise_classifications.csv'
OUTPUT_FILE = DERIVED_DIR / 'issue_17_regression_data.csv'

# MERGE CONFIGURATION
MERGE_KEYS = ['session_code', 'segment', 'round', 'label']

SENTIMENT_COLS = [
    'sentiment_compound_mean', 'sentiment_compound_std',
    'sentiment_compound_min', 'sentiment_compound_max',
    'sentiment_positive_mean', 'sentiment_negative_mean', 'sentiment_neutral_mean',
]

PROMISE_COLS = ['message_count', 'promise_count', 'promise_percentage']

LAGGED_SENTIMENT_COLS = ['sentiment_compound_mean']

# Liar thresholds (contribution below threshold after promise = lie)
THRESHOLD_20 = 20
THRESHOLD_5 = 5


# =====
# Main function
# =====
def main():
    """Main execution flow for merging regression data."""
    behavior_df = load_behavior_data()
    sentiment_df = load_sentiment_data()
    promise_df = load_promise_data()

    print_input_summary(behavior_df, sentiment_df, promise_df)

    merged_df = merge_datasets(behavior_df, sentiment_df, promise_df)
    merged_df = compute_derived_variables(merged_df)
    validate_output(merged_df)

    save_results(merged_df)
    print_output_summary(merged_df)


# =====
# Data loading
# =====
def load_behavior_data() -> pd.DataFrame:
    """Load behavior classifications as base dataset."""
    return pd.read_csv(BEHAVIOR_FILE)


def load_sentiment_data() -> pd.DataFrame:
    """Load sentiment scores, selecting only merge keys and sentiment columns."""
    df = pd.read_csv(SENTIMENT_FILE)
    return df[MERGE_KEYS + SENTIMENT_COLS]


def load_promise_data() -> pd.DataFrame:
    """Load promise classifications, selecting only merge keys and promise columns."""
    df = pd.read_csv(PROMISE_FILE)
    return df[MERGE_KEYS + PROMISE_COLS]


# =====
# Merge and validation
# =====
def merge_datasets(behavior_df: pd.DataFrame, sentiment_df: pd.DataFrame,
                   promise_df: pd.DataFrame) -> pd.DataFrame:
    """Merge behavior, sentiment, and promise data using LEFT JOINs."""
    merged = behavior_df.merge(sentiment_df, on=MERGE_KEYS, how='left')
    merged = merged.merge(promise_df, on=MERGE_KEYS, how='left')
    return merged


def compute_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived variables for regression analysis.

    Creates contemporaneous liar flags and lagged sentiment. The lied_this_period
    flags indicate whether the player lied in the chat immediately preceding THIS
    round's contribution (chat and contribution are already paired in the data).
    """
    df = df.sort_values(MERGE_KEYS)
    group_cols = ['session_code', 'segment', 'label']

    # Contemporaneous liar flags (chat already paired with contribution it influenced)
    df['lied_this_period_20'] = (
        (df['made_promise'] == True) & (df['contribution'] < THRESHOLD_20)
    )
    df['lied_this_period_5'] = (
        (df['made_promise'] == True) & (df['contribution'] < THRESHOLD_5)
    )

    # Lag sentiment (previous round's sentiment)
    for col in LAGGED_SENTIMENT_COLS:
        df[f'{col}_prev'] = df.groupby(group_cols)[col].shift(1)

    return df


def validate_output(df: pd.DataFrame):
    """Validate merged output meets requirements."""
    expected_rows = 3520
    actual_rows = len(df)

    if actual_rows != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {actual_rows}")

    _validate_round_1_nulls(df)
    _validate_lagged_sentiment(df)
    _validate_no_duplicate_columns(df)

    print("\nValidation passed!")


def _validate_round_1_nulls(df: pd.DataFrame):
    """Verify round 1 rows have NaN for sentiment columns."""
    round_1 = df[df['round'] == 1]
    for col in SENTIMENT_COLS:
        non_null_count = round_1[col].notna().sum()
        if non_null_count > 0:
            raise ValueError(f"Round 1 should have NaN for {col}, found {non_null_count} non-null")


def _validate_lagged_sentiment(df: pd.DataFrame):
    """Verify expected NaN structure for lagged sentiment column.

    Round 1: sentiment_compound_mean_prev should be NaN (no previous round).
    Round 2: sentiment_compound_mean_prev should be NaN (round 1 sentiment is NaN).
    """
    col = 'sentiment_compound_mean_prev'

    # Round 1: lagged sentiment must be NaN (no previous round)
    round_1 = df[df['round'] == 1]
    non_null_count = round_1[col].notna().sum()
    if non_null_count > 0:
        raise ValueError(f"Round 1 should have NaN for {col}, found {non_null_count} non-null")

    # Round 2: lagged sentiment must be NaN (because round 1 sentiment is NaN)
    round_2 = df[df['round'] == 2]
    non_null_count = round_2[col].notna().sum()
    if non_null_count > 0:
        raise ValueError(f"Round 2 should have NaN for {col}, found {non_null_count} non-null")


def _validate_no_duplicate_columns(df: pd.DataFrame):
    """Verify no duplicate columns exist in output."""
    cols = df.columns.tolist()
    if len(cols) != len(set(cols)):
        duplicates = [col for col in cols if cols.count(col) > 1]
        raise ValueError(f"Duplicate columns found: {duplicates}")


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save merged DataFrame to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")


def print_input_summary(behavior_df: pd.DataFrame, sentiment_df: pd.DataFrame,
                        promise_df: pd.DataFrame):
    """Print summary of input data."""
    print("=" * 60)
    print("INPUT DATA SUMMARY")
    print("=" * 60)
    print(f"Behavior classifications: {len(behavior_df):,} rows")
    print(f"Sentiment scores:         {len(sentiment_df):,} rows")
    print(f"Promise classifications:  {len(promise_df):,} rows")


def print_output_summary(df: pd.DataFrame):
    """Print summary of merged output."""
    print("\n" + "=" * 60)
    print("OUTPUT DATA SUMMARY")
    print("=" * 60)
    print(f"Total rows:      {len(df):,}")
    print(f"Total columns:   {len(df.columns)}")

    _print_merge_coverage(df)
    _print_column_list(df)


def _print_merge_coverage(df: pd.DataFrame):
    """Print how many rows have sentiment/promise data and derived columns."""
    sentiment_count = df['sentiment_compound_mean'].notna().sum()
    promise_count = df['promise_count'].notna().sum()
    lied_20 = (df['lied_this_period_20'] == True).sum()
    lied_5 = (df['lied_this_period_5'] == True).sum()
    sentiment_prev_count = df['sentiment_compound_mean_prev'].notna().sum()
    total = len(df)

    print(f"\nMerge coverage:")
    print(f"  Rows with sentiment:      {sentiment_count:,} / {total:,} ({100*sentiment_count/total:.1f}%)")
    print(f"  Rows with promises:       {promise_count:,} / {total:,} ({100*promise_count/total:.1f}%)")
    print(f"  Rows with sentiment_prev: {sentiment_prev_count:,} / {total:,} ({100*sentiment_prev_count/total:.1f}%)")
    print(f"\nLiar counts (lied_this_period):")
    print(f"  _20 (contrib < 20):       {lied_20:,} / {total:,} ({100*lied_20/total:.1f}%)")
    print(f"  _5 (contrib < 5):         {lied_5:,} / {total:,} ({100*lied_5/total:.1f}%)")


def _print_column_list(df: pd.DataFrame):
    """Print list of all columns in output."""
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    print("=" * 60)


# %%
if __name__ == "__main__":
    main()
