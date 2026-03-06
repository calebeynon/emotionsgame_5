"""
Purpose: Generate sentiment summary statistics tables for the public goods
         experiment. Outputs descriptive stats, component scores, category
         distributions, intensity breakdowns, and sentiment-contribution
         correlations by treatment and supergame.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import pandas as pd

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    ensure_output_dir,
    load_sentiment,
    write_tex_table,
)

# SENTIMENT THRESHOLDS
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05
_STRONG_THRESHOLD = 0.6
_MODERATE_THRESHOLD = 0.2


# =====
# Main function
# =====

def main():
    """Generate all sentiment summary statistics tables."""
    df = load_sentiment()
    ensure_output_dir()

    write_tex_table(compute_descriptive(df), 'sentiment_descriptive.tex', 'clrrr')
    write_tex_table(compute_components(df), 'sentiment_components.tex', 'clrrr')
    write_tex_table(compute_categories(df), 'sentiment_categories.tex', 'clrrr')
    write_tex_table(compute_intensity(df), 'sentiment_intensity.tex', 'clrrr')
    write_tex_table(compute_correlation(df), 'sentiment_contribution_correlation.tex', 'lrr')


# =====
# Descriptive statistics
# =====

def compute_descriptive(df):
    """Mean compound sentiment and SD, by treatment x supergame.

    Uses an unweighted mean-of-means: each player-round is one observation,
    regardless of how many messages that player sent. The per-player-round
    compound score (sentiment_compound_mean) is already averaged across
    individual messages in the upstream data, so aggregating here gives
    equal weight to every player-round rather than every message.
    """
    grouped = df.groupby(['treatment', 'segment'])['sentiment_compound_mean']
    stats = grouped.agg(['mean', 'std', 'count']).round(4).reset_index()
    stats.columns = ['Treatment', 'Supergame', 'Mean', 'SD', 'N']
    return stats


# =====
# Component scores
# =====

def compute_components(df):
    """Mean positive, negative, neutral scores by treatment x supergame."""
    cols = ['sentiment_positive_mean', 'sentiment_negative_mean', 'sentiment_neutral_mean']
    grouped = df.groupby(['treatment', 'segment'])[cols]
    stats = grouped.mean().round(4).reset_index()
    stats.columns = ['Treatment', 'Supergame', 'Positive', 'Negative', 'Neutral']
    return stats


# =====
# Sentiment categories
# =====

def compute_categories(df):
    """Pct positive, negative, neutral by treatment x supergame."""
    df = _assign_category(df)
    rows = []
    for (treatment, segment), grp in df.groupby(['treatment', 'segment']):
        total = len(grp)
        pct_pos = _pct(grp, 'category', 'Positive', total)
        pct_neg = _pct(grp, 'category', 'Negative', total)
        pct_neu = _pct(grp, 'category', 'Neutral', total)
        rows.append([treatment, segment, pct_pos, pct_neg, pct_neu])
    return pd.DataFrame(rows, columns=[
        'Treatment', 'Supergame', 'Pct Positive', 'Pct Negative', 'Pct Neutral',
    ])


def _assign_category(df):
    """Assign sentiment category based on compound score thresholds."""
    df = df.copy()
    compound = df['sentiment_compound_mean']
    df['category'] = 'Neutral'
    df.loc[compound >= _POS_THRESHOLD, 'category'] = 'Positive'
    df.loc[compound <= _NEG_THRESHOLD, 'category'] = 'Negative'
    return df


# =====
# Sentiment intensity
# =====

def compute_intensity(df):
    """Pct strong, moderate, weak intensity by treatment x supergame."""
    df = _assign_intensity(df)
    rows = []
    for (treatment, segment), grp in df.groupby(['treatment', 'segment']):
        total = len(grp)
        pct_strong = _pct(grp, 'intensity', 'Strong', total)
        pct_moderate = _pct(grp, 'intensity', 'Moderate', total)
        pct_weak = _pct(grp, 'intensity', 'Weak', total)
        rows.append([treatment, segment, pct_strong, pct_moderate, pct_weak])
    return pd.DataFrame(rows, columns=[
        'Treatment', 'Supergame', 'Pct Strong', 'Pct Moderate', 'Pct Weak',
    ])


def _assign_intensity(df):
    """Assign intensity label based on absolute compound score."""
    df = df.copy()
    abs_compound = df['sentiment_compound_mean'].abs()
    df['intensity'] = 'Weak'
    df.loc[abs_compound >= _MODERATE_THRESHOLD, 'intensity'] = 'Moderate'
    df.loc[abs_compound >= _STRONG_THRESHOLD, 'intensity'] = 'Strong'
    return df


# =====
# Sentiment-contribution correlation
# =====

def compute_correlation(df):
    """Pearson correlation between compound sentiment and contribution."""
    rows = []
    for treatment in sorted(df['treatment'].unique()):
        t_data = df[df['treatment'] == treatment]
        r = _pearson_corr(t_data)
        rows.append([f'T{treatment}', r, len(t_data)])
    rows.append(['Overall', _pearson_corr(df), len(df)])
    return pd.DataFrame(rows, columns=['Group', 'Pearson r', 'N'])


def _pearson_corr(df):
    """Pearson r between compound sentiment and contribution, rounded."""
    return round(
        df['sentiment_compound_mean'].corr(df['contribution']), 4,
    )


def _pct(df, column, value, total):
    """Percentage of rows matching a value in a column."""
    return round(100 * (df[column] == value).sum() / total, 1) if total > 0 else 0.0


# %%
if __name__ == "__main__":
    main()
