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
BOX_DERIVED = Path('/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/derived')
BEHAVIOR_FILE = BOX_DERIVED / 'behavior_classifications.csv'
SENTIMENT_FILE = BOX_DERIVED / 'sentiment_scores.csv'
PROMISE_FILE = BOX_DERIVED / 'promise_classifications.csv'
OUTPUT_FILE = BOX_DERIVED / 'issue_17_regression_data.csv'

# MERGE CONFIGURATION
MERGE_KEYS = ['session_code', 'segment', 'round', 'label']

SENTIMENT_COLS = [
    'sentiment_compound_mean', 'sentiment_compound_std',
    'sentiment_compound_min', 'sentiment_compound_max',
    'sentiment_positive_mean', 'sentiment_negative_mean', 'sentiment_neutral_mean',
]

PROMISE_COLS = ['message_count', 'promise_count', 'promise_percentage']

LAGGED_SENTIMENT_COLS = ['sentiment_compound_mean']
LAGGED_LIAR_COLS = ['is_liar_strict', 'is_liar_lenient']


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
    merged_df = compute_lagged_variables(merged_df)
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


def compute_lagged_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged variables for regression analysis.

    Creates period-specific liar flags and lagged sentiment. Unlike cumulative
    is_liar_* flags, lied_prev_period_* is True only if the player lied in
    the immediately preceding round within the same segment.
    """
    df = df.sort_values(MERGE_KEYS)
    group_cols = ['session_code', 'segment', 'label']

    # Lag liar indicators
    for col in LAGGED_LIAR_COLS:
        new_col = col.replace('is_liar', 'lied_prev_period')
        df[new_col] = df.groupby(group_cols)[col].shift(1)

    # Lag sentiment
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
    _validate_lagged_nulls(df)
    _validate_no_duplicate_columns(df)

    print("\nValidation passed!")


def _validate_round_1_nulls(df: pd.DataFrame):
    """Verify round 1 rows have NaN for sentiment columns."""
    round_1 = df[df['round'] == 1]
    for col in SENTIMENT_COLS:
        non_null_count = round_1[col].notna().sum()
        if non_null_count > 0:
            raise ValueError(f"Round 1 should have NaN for {col}, found {non_null_count} non-null")


def _validate_lagged_nulls(df: pd.DataFrame):
    """Verify expected NaN structure for lagged columns.

    Round 1: All lagged columns should be NaN (no previous round exists).
    Round 2: sentiment_compound_mean_prev should be NaN (round 1 sentiment is NaN).
    """
    lagged_cols = ['lied_prev_period_strict', 'lied_prev_period_lenient',
                   'sentiment_compound_mean_prev']

    # Round 1: all lagged columns must be NaN
    round_1 = df[df['round'] == 1]
    for col in lagged_cols:
        non_null_count = round_1[col].notna().sum()
        if non_null_count > 0:
            raise ValueError(
                f"Round 1 should have NaN for {col}, found {non_null_count} non-null"
            )

    # Round 2: lagged sentiment must be NaN (because round 1 sentiment is NaN)
    round_2 = df[df['round'] == 2]
    non_null_count = round_2['sentiment_compound_mean_prev'].notna().sum()
    if non_null_count > 0:
        raise ValueError(
            f"Round 2 should have NaN for sentiment_compound_mean_prev, "
            f"found {non_null_count} non-null"
        )


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
    """Print how many rows have sentiment/promise data and lagged columns."""
    sentiment_count = df['sentiment_compound_mean'].notna().sum()
    promise_count = df['promise_count'].notna().sum()
    lied_prev_count = df['lied_prev_period_strict'].notna().sum()
    sentiment_prev_count = df['sentiment_compound_mean_prev'].notna().sum()
    total = len(df)

    print(f"\nMerge coverage:")
    print(f"  Rows with sentiment:      {sentiment_count:,} / {total:,} ({100*sentiment_count/total:.1f}%)")
    print(f"  Rows with promises:       {promise_count:,} / {total:,} ({100*promise_count/total:.1f}%)")
    print(f"  Rows with lied_prev:      {lied_prev_count:,} / {total:,} ({100*lied_prev_count/total:.1f}%)")
    print(f"  Rows with sentiment_prev: {sentiment_prev_count:,} / {total:,} ({100*sentiment_prev_count/total:.1f}%)")


def _print_column_list(df: pd.DataFrame):
    """Print list of all columns in output."""
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    print("=" * 60)


# %%
if __name__ == "__main__":
    main()
