"""
Purpose: Generate behavior classification summary statistics for the public
         goods experiment. Outputs promise rates, liar/sucker rates,
         behavioral persistence, and conditional contribution tables.
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
    load_behavior,
    write_tex_table,
)

# SUPERGAME PAIRS for persistence analysis
_SG_PAIRS = [
    ('supergame1', 'supergame2'),
    ('supergame2', 'supergame3'),
    ('supergame3', 'supergame4'),
    ('supergame4', 'supergame5'),
]

# SEGMENT LABELS for display
_SG_DISPLAY = {f'supergame{i}': f'SG{i}' for i in range(1, 6)}


# =====
# Main function
# =====

def main():
    """Generate all behavior classification summary statistics."""
    df = load_behavior()
    ensure_output_dir()

    promise = compute_promise_rates(df)
    write_tex_table(promise, 'behavior_promise_rates.tex', 'llrr')

    liar = compute_liar_rates(df)
    write_tex_table(liar, 'behavior_liar_rates.tex', 'llrrrr')

    sucker = compute_sucker_rates(df)
    write_tex_table(sucker, 'behavior_sucker_rates.tex', 'llrrrr')

    persist = compute_persistence(df)
    write_tex_table(persist, 'behavior_persistence.tex', 'llrrrr')

    cond = compute_conditional_contribution(df)
    write_tex_table(cond, 'behavior_conditional_contribution.tex', 'llrr')


# =====
# Promise rates
# =====

def compute_promise_rates(df):
    """Promise count and rate by treatment x supergame, plus by treatment x round."""
    rows = []
    rows.extend(_promise_by_supergame(df))
    rows.extend(_promise_by_round(df))
    return pd.DataFrame(rows, columns=['Treatment', 'Group', 'Promises', 'Rate (\\%)'])


def _promise_by_supergame(df):
    """Promise rates grouped by treatment and supergame."""
    rows = []
    for (treatment, segment), g in df.groupby(['treatment', 'segment']):
        count = g['made_promise'].sum()
        rate = round(100 * count / len(g), 1)
        rows.append([f'T{treatment}', _SG_DISPLAY[segment], count, rate])
    return rows


def _promise_by_round(df):
    """Promise rates grouped by treatment and round (pooled across supergames)."""
    rows = []
    for (treatment, rnd), g in df.groupby(['treatment', 'round']):
        count = g['made_promise'].sum()
        rate = round(100 * count / len(g), 1)
        rows.append([f'T{treatment}', f'Rd {rnd}', count, rate])
    return rows


# =====
# Liar and sucker rates
# =====

def compute_liar_rates(df):
    """Liar count and rate at 20% and 5% thresholds by treatment x supergame."""
    return _classification_rates(df, 'is_liar_20', 'is_liar_5', 'Liars')


def compute_sucker_rates(df):
    """Sucker count and rate at 20% and 5% thresholds by treatment x supergame."""
    return _classification_rates(df, 'is_sucker_20', 'is_sucker_5', 'Suckers')


def _classification_rates(df, col_20, col_5, label):
    """Count and rate for a classification at both thresholds."""
    rows = []
    for (treatment, segment), g in df.groupby(['treatment', 'segment']):
        n = len(g)
        c20, c5 = g[col_20].sum(), g[col_5].sum()
        r20 = round(100 * c20 / n, 1)
        r5 = round(100 * c5 / n, 1)
        rows.append([f'T{treatment}', _SG_DISPLAY[segment], c20, r20, c5, r5])
    cols = [
        'Treatment', 'Supergame',
        f'{label} (20\\%)', 'Rate (20\\%)', f'{label} (5\\%)', 'Rate (5\\%)',
    ]
    return pd.DataFrame(rows, columns=cols)


# =====
# Behavioral persistence
# =====

def compute_persistence(df):
    """Fraction of players classified in SG N who remain classified in SG N+1."""
    rows = []
    for treatment in sorted(df['treatment'].unique()):
        t_df = df[df['treatment'] == treatment]
        for sg_curr, sg_next in _SG_PAIRS:
            row = _persistence_row(t_df, treatment, sg_curr, sg_next)
            rows.append(row)
    cols = [
        'Treatment', 'Transition',
        'Liar 20\\% Persist', 'Liar 5\\% Persist',
        'Sucker 20\\% Persist', 'Sucker 5\\% Persist',
    ]
    return pd.DataFrame(rows, columns=cols)


def _persistence_row(t_df, treatment, sg_curr, sg_next):
    """Build one persistence row for a supergame transition."""
    curr = _player_flags(t_df[t_df['segment'] == sg_curr])
    nxt = _player_flags(t_df[t_df['segment'] == sg_next])
    transition = f'{_SG_DISPLAY[sg_curr]}--{_SG_DISPLAY[sg_next]}'
    return [
        f'T{treatment}', transition,
        _persist_pct(curr, nxt, 'is_liar_20'),
        _persist_pct(curr, nxt, 'is_liar_5'),
        _persist_pct(curr, nxt, 'is_sucker_20'),
        _persist_pct(curr, nxt, 'is_sucker_5'),
    ]


def _player_flags(sg_df):
    """Aggregate boolean flags to player level (True if ever True in supergame)."""
    flag_cols = ['is_liar_20', 'is_liar_5', 'is_sucker_20', 'is_sucker_5']
    return sg_df.groupby('participant_id')[flag_cols].any()


def _persist_pct(curr_flags, next_flags, col):
    """Percentage of players flagged in current SG who are also flagged in next SG."""
    flagged_curr = set(curr_flags[curr_flags[col]].index)
    if not flagged_curr:
        return '--'
    flagged_next = set(next_flags[next_flags[col]].index)
    persisted = len(flagged_curr & flagged_next)
    return f'{round(100 * persisted / len(flagged_curr), 1)}\\%'


# =====
# Conditional contribution
# =====

def compute_conditional_contribution(df):
    """Mean contribution when promise made vs not, by treatment x supergame."""
    rows = []
    for (treatment, segment), g in df.groupby(['treatment', 'segment']):
        mean_promise = _safe_mean(g[g['made_promise']]['contribution'])
        mean_no = _safe_mean(g[~g['made_promise']]['contribution'])
        rows.append([f'T{treatment}', _SG_DISPLAY[segment], mean_promise, mean_no])
    cols = ['Treatment', 'Supergame', 'Mean (Promise)', 'Mean (No Promise)']
    return pd.DataFrame(rows, columns=cols)


def _safe_mean(series):
    """Return rounded mean or '--' if series is empty."""
    if len(series) == 0:
        return '--'
    return round(series.mean(), 2)


# %%
if __name__ == "__main__":
    main()
