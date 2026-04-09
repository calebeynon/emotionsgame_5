"""
Liar bucket classification: count lies per participant and assign severity buckets.
Author: Claude Code | Date: 2026-04-09
"""

from pathlib import Path

import pandas as pd

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
INPUT_FILE = DATA_DIR / 'derived' / 'behavior_classifications.csv'
OUTPUT_FILE = DATA_DIR / 'derived' / 'liar_buckets.csv'

# BUCKET THRESHOLDS
BUCKET_ORDER = ['never', 'one_time', 'moderate', 'severe']


# =====
# Main function
# =====
def main():
    """Main execution flow for liar bucket classification."""
    df = load_behavior_data()
    lie_counts = count_lies_per_participant(df)
    buckets = assign_buckets(lie_counts)
    validate_output(buckets)
    save_results(buckets)
    print_summary(buckets)


# =====
# Data loading
# =====
def load_behavior_data() -> pd.DataFrame:
    """Load behavior classifications CSV."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}. Run classify_behavior.py first."
        )
    return pd.read_csv(INPUT_FILE)


# =====
# Lie counting
# =====
def count_lies_per_participant(df: pd.DataFrame) -> pd.DataFrame:
    """Count total rounds where lied_this_round_20 == True per participant."""
    # Treat NaN as False (not a lie)
    df['lied_this_round_20'] = df['lied_this_round_20'].fillna(False)
    participant_lies = (
        df.groupby(['session_code', 'treatment', 'label', 'participant_id'])
        ['lied_this_round_20']
        .sum()
        .reset_index()
        .rename(columns={'lied_this_round_20': 'lie_count'})
    )
    participant_lies['lie_count'] = participant_lies['lie_count'].astype(int)
    return participant_lies


# =====
# Bucket assignment
# =====
def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Assign liar bucket based on lie_count."""
    df['liar_bucket'] = df['lie_count'].apply(_classify_bucket)
    df['liar_bucket'] = pd.Categorical(
        df['liar_bucket'], categories=BUCKET_ORDER, ordered=True
    )
    return df


def _classify_bucket(lie_count: int) -> str:
    """Map lie count to bucket name."""
    if lie_count == 0:
        return 'never'
    if lie_count == 1:
        return 'one_time'
    if lie_count <= 3:
        return 'moderate'
    return 'severe'


# =====
# Validation
# =====
def validate_output(df: pd.DataFrame):
    """Validate output has exactly 160 rows (10 sessions x 16 participants)."""
    n_rows = len(df)
    if n_rows != 160:
        raise ValueError(
            f"Expected 160 participants, got {n_rows}. "
            "Check input data for missing sessions or duplicate participants."
        )
    n_sessions = df['session_code'].nunique()
    if n_sessions != 10:
        raise ValueError(f"Expected 10 sessions, got {n_sessions}.")


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save liar bucket classifications to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total participants: {len(df)}")


def print_summary(df: pd.DataFrame):
    """Print bucket distribution summary."""
    print("\n" + "=" * 50)
    print("LIAR BUCKET DISTRIBUTION")
    print("=" * 50)
    for bucket in BUCKET_ORDER:
        count = (df['liar_bucket'] == bucket).sum()
        pct = count / len(df) * 100
        print(f"  {bucket:>10}: {count:3d} ({pct:5.1f}%)")
    print("=" * 50)
    print(f"\nMean lies per participant: {df['lie_count'].mean():.2f}")
    print(f"Max lies per participant: {df['lie_count'].max()}")


# %%
if __name__ == "__main__":
    main()
