"""
Compute VADER sentiment scores for chat messages.

Reads promise classifications data and computes sentiment scores for each message,
then aggregates to player-round level with mean, std, min, max for compound scores
and mean for positive, negative, neutral components.

Author: Claude Code
Date: 2026-01-26
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# FILE PATHS
INPUT_FILE = Path(__file__).parent.parent / 'datastore' / 'derived' / 'promise_classifications.csv'
OUTPUT_FILE = Path(__file__).parent.parent / 'datastore' / 'derived' / 'sentiment_scores.csv'

# Column definitions
ID_COLS = ['session_code', 'treatment', 'segment', 'round', 'group', 'label', 'participant_id']
PRESERVE_COLS = ['contribution', 'payoff', 'message_count']


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    df = load_data()
    print(f"Loaded {len(df)} player-round records")

    results = compute_all_sentiments(df)
    save_results(results)
    print_summary(results)


# =====
# Data loading
# =====
def load_data() -> pd.DataFrame:
    """Load promise classifications data."""
    return pd.read_csv(INPUT_FILE)


# =====
# Sentiment computation
# =====
def compute_all_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sentiment scores for all player-rounds."""
    sia = SentimentIntensityAnalyzer()
    records = []

    for _, row in df.iterrows():
        messages = json.loads(row['messages'])
        sentiment_record = compute_player_sentiment(row, messages, sia)
        records.append(sentiment_record)

    return pd.DataFrame.from_records(records)


def compute_player_sentiment(row: pd.Series, messages: list, sia: SentimentIntensityAnalyzer) -> dict:
    """Compute aggregated sentiment scores for a single player-round."""
    # Get sentiment scores for each message
    scores = [sia.polarity_scores(msg) for msg in messages]

    # Build result with identifiers and preserved columns
    result = {col: row[col] for col in ID_COLS + PRESERVE_COLS}

    # Add aggregated sentiment scores
    result.update(aggregate_sentiment_scores(scores))
    return result


def aggregate_sentiment_scores(scores: list) -> dict:
    """Aggregate individual message sentiment scores to summary statistics."""
    compounds = [s['compound'] for s in scores]
    positives = [s['pos'] for s in scores]
    negatives = [s['neg'] for s in scores]
    neutrals = [s['neu'] for s in scores]

    return {
        'sentiment_compound_mean': np.mean(compounds),
        'sentiment_compound_std': compute_std(compounds),
        'sentiment_compound_min': np.min(compounds),
        'sentiment_compound_max': np.max(compounds),
        'sentiment_positive_mean': np.mean(positives),
        'sentiment_negative_mean': np.mean(negatives),
        'sentiment_neutral_mean': np.mean(neutrals),
    }


def compute_std(values: list) -> float:
    """Compute standard deviation, returning 0.0 for single values."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save results DataFrame to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to: {OUTPUT_FILE}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics for sentiment scores."""
    print("\n" + "=" * 50)
    print("SENTIMENT SUMMARY")
    print("=" * 50)
    print(f"Total player-rounds processed: {len(df)}")
    print(f"\nCompound sentiment:")
    print(f"  Mean: {df['sentiment_compound_mean'].mean():.3f}")
    print(f"  Range: [{df['sentiment_compound_min'].min():.3f}, {df['sentiment_compound_max'].max():.3f}]")
    print(f"\nComponent means:")
    print(f"  Positive: {df['sentiment_positive_mean'].mean():.3f}")
    print(f"  Negative: {df['sentiment_negative_mean'].mean():.3f}")
    print(f"  Neutral: {df['sentiment_neutral_mean'].mean():.3f}")
    print("=" * 50)


# %%
if __name__ == "__main__":
    main()
