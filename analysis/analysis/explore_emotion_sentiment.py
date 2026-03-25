"""
Purpose: Exploratory analysis of emotion-sentiment data from merged_panel.csv.
         Examines data coverage, distributions, correlations, and co-movement
         patterns between facial emotion metrics and chat sentiment scores.
Author: Claude Code
Date: 2026-03-12
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from sibling packages
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summary_statistics.ss_common import DERIVED_DIR

# FILE PATHS
MERGED_PANEL = DERIVED_DIR / 'merged_panel.csv'

# COLUMN GROUPS
EMOTION_COLS = [
    'emotion_anger', 'emotion_contempt', 'emotion_disgust', 'emotion_fear',
    'emotion_joy', 'emotion_sadness', 'emotion_surprise', 'emotion_engagement',
    'emotion_valence', 'emotion_sentimentality', 'emotion_confusion',
    'emotion_neutral', 'emotion_attention',
]

SENTIMENT_COLS = [
    'sentiment_compound_mean', 'sentiment_compound_std',
    'sentiment_compound_min', 'sentiment_compound_max',
    'sentiment_positive_mean', 'sentiment_negative_mean',
    'sentiment_neutral_mean',
]

# Subset for focused correlation analysis
CORE_SENTIMENT = ['sentiment_compound_mean', 'sentiment_positive_mean', 'sentiment_negative_mean']
CORE_EMOTIONS = [
    'emotion_joy', 'emotion_anger', 'emotion_sadness', 'emotion_fear',
    'emotion_surprise', 'emotion_valence', 'emotion_engagement',
]


# =====
# Main function
# =====

def main():
    """Run exploratory analysis and print results."""
    df = pd.read_csv(MERGED_PANEL)

    print_data_overview(df)
    print_coverage_by_segment(df)
    print_coverage_by_page_type(df)
    print_emotion_descriptives(df)
    print_sentiment_descriptives(df)
    print_emotion_sentiment_correlations(df)
    print_valence_vs_compound(df)
    print_round_level_means(df)


# =====
# Helpers
# =====

def _print_header(title):
    """Print a section header."""
    print("=" * 70)
    print(title)
    print("=" * 70)


# =====
# Data overview
# =====

def print_data_overview(df):
    """Print shape, column counts, and missing data summary."""
    _print_header("DATA OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Sessions: {df['session_code'].nunique()}")
    print(f"Segments: {sorted(df['segment'].dropna().unique())}")
    print(f"Players: {df['label'].nunique()} unique labels")
    print()
    _print_coverage_flags(df)


def _print_coverage_flags(df):
    """Print row-level coverage for emotion, sentiment, and contribution."""
    has_emo = df[EMOTION_COLS].notna().any(axis=1)
    has_sent = df[SENTIMENT_COLS].notna().any(axis=1)
    has_contrib = df['contribution'].notna()

    print("Coverage:")
    print(f"  Emotion data:    {has_emo.sum():>6,} / {len(df):,} ({100*has_emo.mean():.1f}%)")
    print(f"  Sentiment data:  {has_sent.sum():>6,} / {len(df):,} ({100*has_sent.mean():.1f}%)")
    print(f"  Contribution:    {has_contrib.sum():>6,} / {len(df):,} ({100*has_contrib.mean():.1f}%)")
    print(f"  Both emo+sent:   {(has_emo & has_sent).sum():>6,} / {len(df):,}")
    print(f"  All three:       {(has_emo & has_sent & has_contrib).sum():>6,} / {len(df):,}")
    print()


# =====
# Coverage breakdowns
# =====

def print_coverage_by_segment(df):
    """Print emotion/sentiment coverage by supergame segment."""
    _print_header("COVERAGE BY SEGMENT")
    for seg in sorted(df['segment'].dropna().unique()):
        sub = df[df['segment'] == seg]
        n_emo = sub[EMOTION_COLS].notna().any(axis=1).sum()
        n_sent = sub[SENTIMENT_COLS].notna().any(axis=1).sum()
        print(f"  {seg}: {len(sub):>5} rows | emo={n_emo:>5} | sent={n_sent:>5}")
    print()


def print_coverage_by_page_type(df):
    """Print data coverage by page_type."""
    _print_header("COVERAGE BY PAGE TYPE")
    for pt in sorted(df['page_type'].dropna().unique()):
        sub = df[df['page_type'] == pt]
        n_emo = sub[EMOTION_COLS].notna().any(axis=1).sum()
        n_sent = sub[SENTIMENT_COLS].notna().any(axis=1).sum()
        n_cont = sub['contribution'].notna().sum()
        print(f"  {pt:>20}: {len(sub):>5} rows | emo={n_emo:>5} | sent={n_sent:>5} | contrib={n_cont:>5}")
    print()


# =====
# Descriptive statistics
# =====

def print_emotion_descriptives(df):
    """Print descriptive stats for emotion columns."""
    _print_header("EMOTION DESCRIPTIVES (non-null rows only)")
    emo_df = df[EMOTION_COLS].dropna()
    stats = emo_df.describe().T[['mean', 'std', '25%', '50%', '75%', 'max']]
    stats['skew'] = emo_df.skew()
    print(stats.round(3).to_string())
    print()


def print_sentiment_descriptives(df):
    """Print descriptive stats for sentiment columns."""
    _print_header("SENTIMENT DESCRIPTIVES (non-null rows only)")
    sent_df = df[SENTIMENT_COLS].dropna()
    stats = sent_df.describe().T[['mean', 'std', '25%', '50%', '75%', 'max']]
    stats['skew'] = sent_df.skew()
    print(stats.round(4).to_string())
    print()


# =====
# Correlation analysis
# =====

def print_emotion_sentiment_correlations(df):
    """Print Pearson correlations between core emotions and sentiment."""
    _print_header("EMOTION-SENTIMENT CORRELATIONS (Pearson, pairwise complete)")
    both = df[CORE_EMOTIONS + CORE_SENTIMENT].dropna()
    print(f"  N = {len(both):,} rows with complete data\n")

    corr = both[CORE_EMOTIONS].corrwith(both['sentiment_compound_mean'])
    print("  Correlation with sentiment_compound_mean:")
    for col, r in corr.items():
        short = col.replace('emotion_', '')
        print(f"    {short:>15}: r = {r:+.4f}")
    print()


def print_valence_vs_compound(df):
    """Print focused comparison of emotion_valence vs sentiment_compound."""
    _print_header("VALENCE vs COMPOUND SENTIMENT (key convergence indicator)")
    both = df[['emotion_valence', 'sentiment_compound_mean']].dropna()
    r = both['emotion_valence'].corr(both['sentiment_compound_mean'])
    print(f"  N = {len(both):,}")
    print(f"  Pearson r = {r:.4f}")
    print(f"  Valence range: [{both['emotion_valence'].min():.2f}, {both['emotion_valence'].max():.2f}]")
    print(f"  Compound range: [{both['sentiment_compound_mean'].min():.4f}, {both['sentiment_compound_mean'].max():.4f}]")
    print()


# =====
# Round-level temporal patterns
# =====

def print_round_level_means(df):
    """Print mean emotion valence and compound sentiment by segment x round."""
    _print_header("ROUND-LEVEL MEANS: emotion_valence vs sentiment_compound_mean")
    contrib_rows = df[df['page_type'] == 'Contribute']
    grouped = contrib_rows.groupby(['segment', 'round']).agg(
        valence=('emotion_valence', 'mean'),
        compound=('sentiment_compound_mean', 'mean'),
        contribution=('contribution', 'mean'),
        n_emo=('emotion_valence', 'count'),
        n_sent=('sentiment_compound_mean', 'count'),
    ).round(3)
    print(grouped.to_string())
    print()


# %%
if __name__ == "__main__":
    main()
