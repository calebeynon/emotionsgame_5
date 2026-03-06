"""
Purpose: Generate group dynamics summary statistics tables for the public
         goods experiment. Outputs cooperation rates, free rider counts,
         within-group variation, and regrouping effects by treatment/supergame.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import pandas as pd

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    ENDOWMENT,
    SUPERGAME_ROUNDS,
    load_contributions,
    safe_mean,
    write_tex_table,
)
_SUPERGAME_ORDER = [f'supergame{i}' for i in range(1, 6)]


# =====
# Main function
# =====

def main():
    """Generate all group dynamics summary statistics tables."""
    df = load_contributions()

    coop = compute_cooperation_rate(df)
    write_tex_table(coop, 'groups_cooperation.tex', 'clrrrrr')

    fr = compute_free_rider_counts(df)
    write_tex_table(fr, 'groups_free_riders.tex', 'clrrrrr')

    sd = compute_within_group_sd(df)
    write_tex_table(sd, 'groups_within_sd.tex', 'clrrrrr')

    regroup = compute_regrouping_effect(df)
    write_tex_table(regroup, 'groups_regrouping_effect.tex', 'clrrrr')


# =====
# Cooperation rate
# =====

def compute_cooperation_rate(df):
    """Mean cooperation rate (contribution / endowment) by treatment x supergame."""
    df = df.copy()
    df['coop_rate'] = df['contribution'] / ENDOWMENT
    grouped = df.groupby(['treatment', 'segment'])['coop_rate']
    stats = grouped.agg(['mean', 'std', 'min', 'max', 'count'])
    stats = stats.round(3).reset_index()
    stats.columns = [
        'Treatment', 'Supergame', 'Mean', 'SD', 'Min', 'Max', 'N',
    ]
    return stats


# =====
# Free rider counts
# =====

def compute_free_rider_counts(df):
    """Mean number of free riders (contribution=0) per group, by treatment x supergame."""
    group_fr = _count_free_riders_per_group(df)
    stats = group_fr.groupby(['treatment', 'segment'])['free_riders']
    stats = stats.agg(['mean', 'std', 'min', 'max', 'count'])
    stats = stats.round(2).reset_index()
    stats.columns = [
        'Treatment', 'Supergame', 'Mean', 'SD', 'Min', 'Max', 'N Groups',
    ]
    return stats


def _count_free_riders_per_group(df):
    """Count free riders (contribution=0) in each group-round."""
    grouped = df.groupby(['treatment', 'segment', 'round', 'session_code', 'group'])
    fr = grouped.apply(
        lambda g: (g['contribution'] == 0).sum(), include_groups=False,
    ).reset_index(name='free_riders')
    return fr


# =====
# Within-group contribution SD
# =====

def compute_within_group_sd(df):
    """Mean within-group contribution SD, by treatment x supergame."""
    group_sd = _compute_group_level_sd(df)
    stats = group_sd.groupby(['treatment', 'segment'])['group_sd']
    stats = stats.agg(['mean', 'std', 'min', 'max', 'count'])
    stats = stats.round(2).reset_index()
    stats.columns = [
        'Treatment', 'Supergame', 'Mean', 'SD', 'Min', 'Max', 'N Groups',
    ]
    return stats


def _compute_group_level_sd(df):
    """Compute contribution SD for each group-round."""
    grouped = df.groupby(['treatment', 'segment', 'round', 'session_code', 'group'])
    sd = grouped['contribution'].std().reset_index(name='group_sd')
    return sd


# =====
# Regrouping effect
# =====

def compute_regrouping_effect(df):
    """Mean contribution in last round of SG N vs first round of SG N+1, by treatment."""
    rows = []
    for sg_num in range(1, 5):
        last_round = SUPERGAME_ROUNDS[sg_num]
        boundary_label = f'SG{sg_num}→SG{sg_num + 1}'
        for treatment in sorted(df['treatment'].unique()):
            last_mean = _mean_contribution_at(df, treatment, sg_num, last_round)
            first_mean = _mean_contribution_at(df, treatment, sg_num + 1, 1)
            diff = round(first_mean - last_mean, 2)
            rows.append([treatment, boundary_label, last_mean, first_mean, diff])
    result = pd.DataFrame(rows, columns=[
        'Treatment', 'Boundary', 'Last Round', 'First Round', 'Difference',
    ])
    return result


def _mean_contribution_at(df, treatment, sg_num, round_num):
    """Mean contribution for a specific treatment, supergame, and round."""
    segment = f'supergame{sg_num}'
    mask = (
        (df['treatment'] == treatment)
        & (df['segment'] == segment)
        & (df['round'] == round_num)
    )
    return safe_mean(df.loc[mask, 'contribution'])


# %%
if __name__ == "__main__":
    main()
