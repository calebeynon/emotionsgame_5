"""
Purpose: Generate contribution summary statistics tables and plots for the
         public goods experiment. Outputs descriptive stats, frequency tables,
         extreme contribution rates, and histograms by treatment/supergame.
Author: Caleb Eynon
Date: 2026-03-01
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    ENDOWMENT,
    OUTPUT_DIR,
    load_contributions,
    safe_pct,
    write_tex_table,
)

# CONTRIBUTION BIN EDGES for frequency table
_FREQ_BINS = [0, 1, 5, 10, 15, 20, 25, 26]
_FREQ_LABELS = ['0', '1--4', '5--9', '10--14', '15--19', '20--24', '25']


# =====
# Main function
# =====

def main():
    """Generate all contribution summary statistics and plots."""
    df = load_contributions()

    desc = compute_descriptive_stats(df)
    write_tex_table(desc, 'contributions_descriptive.tex', 'clrrrrrr')

    freq = compute_frequency_table(df)
    write_tex_table(freq, 'contributions_frequencies.tex', 'clrr')

    extremes = compute_extreme_rates(df)
    write_tex_table(extremes, 'contributions_extremes.tex', 'clcrr')

    plot_histogram_by_treatment(df)
    plot_histogram_by_supergame(df)


# =====
# Statistics computation
# =====

def compute_descriptive_stats(df):
    """Mean, median, SD, min, max of contributions by treatment x supergame."""
    grouped = df.groupby(['treatment', 'segment'])['contribution']
    stats = grouped.agg(['count', 'mean', 'std', 'min', 'max', 'median'])
    stats = stats.round(2).reset_index()
    stats.columns = [
        'Treatment', 'Supergame', 'N', 'Mean', 'SD', 'Min', 'Max', 'Median',
    ]
    return stats


def compute_frequency_table(df):
    """Count and percentage at each contribution level bin by treatment."""
    rows = []
    for treatment in sorted(df['treatment'].unique()):
        t_data = df[df['treatment'] == treatment]['contribution']
        counts = _bin_contributions(t_data)
        total = len(t_data)
        for label, count in zip(_FREQ_LABELS, counts):
            pct = safe_pct(count, total)
            rows.append([treatment, label, count, pct])
    result = pd.DataFrame(rows, columns=['Treatment', 'Bin', 'Count', 'Pct'])
    return result


def _bin_contributions(series):
    """Bin a contribution series into frequency categories."""
    binned = pd.cut(
        series, bins=_FREQ_BINS, right=False, labels=_FREQ_LABELS,
    )
    return [binned.value_counts().get(label, 0) for label in _FREQ_LABELS]


def compute_extreme_rates(df):
    """Percentage of max (=25) and zero contributions by treatment x supergame x round."""
    grouped = df.groupby(['treatment', 'segment', 'round'])
    rows = []
    for (treatment, segment, rnd), group in grouped:
        total = len(group)
        pct_max = safe_pct((group['contribution'] == ENDOWMENT).sum(), total)
        pct_zero = safe_pct((group['contribution'] == 0).sum(), total)
        rows.append([treatment, segment, rnd, pct_max, pct_zero])
    result = pd.DataFrame(
        rows, columns=['Treatment', 'Supergame', 'Round', 'Pct Max', 'Pct Zero'],
    )
    return result


# =====
# Plotting
# =====

def plot_histogram_by_treatment(df):
    """Side-by-side histograms of contributions for T1 vs T2."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.arange(0, 27, 1)
    for ax, treatment in zip(axes, [1, 2]):
        data = df[df['treatment'] == treatment]['contribution']
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(f'Treatment {treatment}')
        ax.set_xlabel('Contribution')
        ax.set_ylabel('Frequency')
    fig.suptitle('Contribution Distribution by Treatment')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'contributions_histogram_by_treatment.png', dpi=150)
    plt.close(fig)


def plot_histogram_by_supergame(df):
    """Faceted histograms by supergame, colored by treatment."""
    supergames = sorted(df['segment'].unique())
    fig, axes = plt.subplots(1, len(supergames), figsize=(18, 4), sharey=True)
    bins = np.arange(0, 27, 1)
    for ax, sg in zip(axes, supergames):
        sg_data = df[df['segment'] == sg]
        for treatment, color in [(1, '#1f77b4'), (2, '#ff7f0e')]:
            data = sg_data[sg_data['treatment'] == treatment]['contribution']
            ax.hist(data, bins=bins, alpha=0.5, color=color, label=f'T{treatment}')
        ax.set_title(sg.replace('supergame', 'SG'))
        ax.set_xlabel('Contribution')
        if ax == axes[0]:
            ax.set_ylabel('Frequency')
            ax.legend()
    fig.suptitle('Contribution Distribution by Supergame and Treatment')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'contributions_histogram_by_supergame.png', dpi=150)
    plt.close(fig)


# %%
if __name__ == "__main__":
    main()
