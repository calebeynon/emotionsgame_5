"""
Generate participant-level payoff dataset from raw oTree data.
Author: Claude Code
Date: 2026-01-30
"""

from pathlib import Path

import pandas as pd

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_FILE = DATA_DIR / 'derived' / 'participant_payoffs.csv'

# SUPERGAME ROUND COUNTS
SUPERGAME_ROUNDS = {
    1: 3,
    2: 4,
    3: 3,
    4: 7,
    5: 5,
}


# =====
# Main function
# =====
def main():
    """Main execution flow for participant payoff extraction."""
    all_records = []

    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = extract_treatment(data_file.name)
        df = pd.read_csv(data_file)
        records = process_session(df, treatment)
        all_records.extend(records)

    result_df = pd.DataFrame.from_records(all_records)
    save_results(result_df)
    print_summary(result_df)


# =====
# Treatment extraction
# =====
def extract_treatment(filename: str) -> int:
    """Extract treatment number from filename like '01_t1_data.csv'."""
    if '_t1_' in filename:
        return 1
    return 2 if '_t2_' in filename else 0


# =====
# Session processing
# =====
def process_session(df: pd.DataFrame, treatment: int) -> list:
    """Process single session CSV and return list of participant records."""
    records = []

    for _, row in df.iterrows():
        final_payoff = row.get('finalresults.1.player.final_payoff')
        if pd.isna(final_payoff):
            continue

        record = build_participant_record(row, treatment)
        records.append(record)

    return records


def build_participant_record(row: pd.Series, treatment: int) -> dict:
    """Build single participant record with all payoff columns."""
    return {
        'participant_label': row['participant.label'],
        'session_code': row['session.code'],
        'treatment': treatment,
        'total_payoff': row['finalresults.1.player.final_payoff'],
        'sg1_payoff': compute_supergame_payoff(row, 1),
        'sg2_payoff': compute_supergame_payoff(row, 2),
        'sg3_payoff': compute_supergame_payoff(row, 3),
        'sg4_payoff': compute_supergame_payoff(row, 4),
        'sg5_payoff': compute_supergame_payoff(row, 5),
    }


def compute_supergame_payoff(row: pd.Series, supergame_num: int) -> float:
    """Sum payoffs for all rounds in a supergame."""
    num_rounds = SUPERGAME_ROUNDS[supergame_num]
    total = 0.0

    for round_num in range(1, num_rounds + 1):
        col = f'supergame{supergame_num}.{round_num}.player.payoff'
        payoff = row.get(col, 0)
        if pd.notna(payoff):
            total += payoff

    return total


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save results DataFrame to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total participants: {len(df)}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics for total_payoff."""
    print("\n" + "=" * 60)
    print("PARTICIPANT PAYOFF SUMMARY")
    print("=" * 60)

    print_experiment_wide_stats(df)
    print_treatment_stats(df)

    print("=" * 60)


def print_experiment_wide_stats(df: pd.DataFrame):
    """Print experiment-wide total_payoff statistics."""
    print("\nExperiment-wide (total_payoff):")
    stats = compute_payoff_stats(df['total_payoff'])
    print_stats_line(stats)


def print_treatment_stats(df: pd.DataFrame):
    """Print total_payoff statistics by treatment group."""
    print("\nBy treatment:")
    for treatment in sorted(df['treatment'].unique()):
        subset = df[df['treatment'] == treatment]
        stats = compute_payoff_stats(subset['total_payoff'])
        print(f"  Treatment {treatment} (n={len(subset)}):")
        print_stats_line(stats, indent=4)


def compute_payoff_stats(series: pd.Series) -> dict:
    """Compute mean, min, max, range for a payoff series."""
    return {
        'mean': series.mean(),
        'min': series.min(),
        'max': series.max(),
        'range': series.max() - series.min(),
    }


def print_stats_line(stats: dict, indent: int = 2):
    """Print formatted stats line."""
    prefix = " " * indent
    print(f"{prefix}Mean: {stats['mean']:.2f}, "
          f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, "
          f"Range: {stats['range']:.2f}")


# %%
if __name__ == "__main__":
    main()
