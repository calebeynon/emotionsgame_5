"""
Purpose: Generate payoff summary statistics tables for the public goods
         experiment. Outputs descriptive stats, per-supergame means,
         inequality metrics, and dollar earnings distributions by treatment.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    PARTICIPATION_FEE,
    POINTS_TO_DOLLARS,
    load_payoffs,
    safe_mean,
    safe_pct,
    write_tex_table,
)

# DOLLAR EARNINGS BIN EDGES
_DOLLAR_BINS = [0, 20, 25, 30, 35, 40, 50]
_DOLLAR_LABELS = ['\\$0--19', '\\$20--24', '\\$25--29', '\\$30--34', '\\$35--39', '\\$40+']

# SUPERGAME PAYOFF COLUMNS
_SG_COLS = [f'sg{i}_payoff' for i in range(1, 6)]
_SG_LABELS = [f'SG{i}' for i in range(1, 6)]


# =====
# Main function
# =====

def main():
    """Generate all payoff summary statistics tables."""
    df = load_payoffs()
    df['dollar_earnings'] = df['total_payoff'] * POINTS_TO_DOLLARS + PARTICIPATION_FEE

    summary = compute_summary(df)
    write_tex_table(summary, 'payoffs_summary.tex', 'lrrrrrr')

    by_sg = compute_by_supergame(df)
    write_tex_table(by_sg, 'payoffs_by_supergame.tex', 'lrrrrr')

    inequality = compute_inequality(df)
    write_tex_table(inequality, 'payoffs_inequality.tex', 'lrrr')

    dollar_dist = compute_dollar_distribution(df)
    write_tex_table(dollar_dist, 'payoffs_dollar_distribution.tex', 'llrr')


# =====
# Summary statistics
# =====

def compute_summary(df):
    """Mean, median, SD, min, max of total payoff and dollar earnings by treatment."""
    rows = []
    for label, subset in _treatment_groups(df):
        rows.append(_summary_row(label, subset, 'total_payoff', 'dollar_earnings'))
    result = pd.DataFrame(rows, columns=[
        'Treatment', 'Mean (pts)', 'Median (pts)', 'SD (pts)',
        'Min (pts)', 'Max (pts)', 'Mean (\\$)',
    ])
    return result


def _summary_row(label, subset, pts_col, dollar_col):
    """Build a single summary row for a treatment group."""
    pts = subset[pts_col]
    dollars = subset[dollar_col]
    if len(pts) == 0:
        return [label] + ['--'] * 6
    return [
        label,
        round(pts.mean(), 2), round(pts.median(), 2), round(pts.std(), 2),
        round(pts.min(), 2), round(pts.max(), 2), round(dollars.mean(), 2),
    ]


# =====
# Per-supergame payoffs
# =====

def compute_by_supergame(df):
    """Mean payoff per supergame by treatment."""
    rows = []
    for label, subset in _treatment_groups(df):
        means = [round(subset[col].mean(), 2) for col in _SG_COLS]
        rows.append([label] + means)
    result = pd.DataFrame(rows, columns=['Treatment'] + _SG_LABELS)
    return result


# =====
# Inequality metrics
# =====

def compute_inequality(df):
    """Gini coefficient, coefficient of variation, and IQR by treatment."""
    rows = []
    for label, subset in _treatment_groups(df):
        payoffs = subset['total_payoff'].values
        gini = gini_coefficient(payoffs)
        cv = round(np.std(payoffs, ddof=1) / np.mean(payoffs), 4) if np.mean(payoffs) > 0 else 0.0
        iqr = round(np.percentile(payoffs, 75) - np.percentile(payoffs, 25), 2)
        rows.append([label, round(gini, 4), cv, iqr])
    result = pd.DataFrame(rows, columns=['Treatment', 'Gini', 'CV', 'IQR'])
    return result


def gini_coefficient(values):
    """Compute the Gini coefficient for an array of values."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    total = np.sum(sorted_vals)
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) - (n + 1) * total) / (n * total)


# =====
# Dollar earnings distribution
# =====

def compute_dollar_distribution(df):
    """Count of participants per dollar earnings bin by treatment."""
    rows = []
    for label, subset in _treatment_groups(df):
        binned = pd.cut(subset['dollar_earnings'], bins=_DOLLAR_BINS, right=False, labels=_DOLLAR_LABELS)
        for bin_label in _DOLLAR_LABELS:
            count = (binned == bin_label).sum()
            pct = safe_pct(count, len(subset))
            rows.append([label, bin_label, count, pct])
    result = pd.DataFrame(rows, columns=['Treatment', 'Bin', 'Count', 'Pct (\\%)'])
    return result


# =====
# Helpers
# =====

_TREATMENT_LABELS = {1: 'IF', 2: 'AF'}


def _treatment_groups(df):
    """Yield (label, subset) for each treatment and the overall dataset."""
    for treatment in sorted(df['treatment'].unique()):
        yield _TREATMENT_LABELS[treatment], df[df['treatment'] == treatment]
    yield 'Overall', df


# %%
if __name__ == "__main__":
    main()
