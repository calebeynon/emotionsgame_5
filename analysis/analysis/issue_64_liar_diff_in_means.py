"""
Purpose: Difference-in-means comparisons of liar rates by treatment and by
         gender. Produces a LaTeX table with two Welch t-tests at the
         participant level using the ever-lied indicator.
Author: Caleb Eynon
Date: 2026-04-16
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / 'summary_statistics'))
from ss_common import SESSION_CODE_REMAP, load_behavior, load_raw_data

# FILE PATHS
OUTPUT_DIR = Path('analysis/output/tables')
OUTPUT_TEX = OUTPUT_DIR / 'liar_diff_in_means.tex'

_GENDER_COL = 'finalresults.1.player.q1'


# =====
# Main function (FIRST - shows high-level flow)
# =====
def main():
    """Run diff-in-means tests and export LaTeX table."""
    participants = build_participant_panel()
    print(f"Participants: {len(participants)}")

    treatment_row = compare(
        participants, 'treatment', 1, 2, 'Treatment 1', 'Treatment 2',
    )
    gender_row = compare(
        participants, 'gender', 'Male', 'Female', 'Male', 'Female',
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_tex([treatment_row, gender_row], OUTPUT_TEX)
    print(f"Wrote {OUTPUT_TEX}")


# =====
# Data construction
# =====
def build_participant_panel():
    """Collapse round-level behavior to one ever-lied flag per participant."""
    beh = load_behavior()
    part = (
        beh.groupby(['session_code', 'label', 'treatment'])['lied_this_round_20']
        .max().reset_index(name='ever_lied')
    )
    part['ever_lied'] = part['ever_lied'].astype(int)
    gender = load_gender()
    merged = part.merge(gender, on=['session_code', 'label'], how='left')
    if merged['gender'].isna().any():
        raise ValueError(f"Missing gender for {merged['gender'].isna().sum()} participants")
    return merged


def load_gender():
    """Load participant gender from raw survey data."""
    raw = load_raw_data()
    gender = raw[['session.code', 'participant.label', _GENDER_COL]].rename(
        columns={'session.code': 'session_code', 'participant.label': 'label', _GENDER_COL: 'gender'}
    )
    gender['session_code'] = gender['session_code'].replace(SESSION_CODE_REMAP)
    return gender


# =====
# Statistical comparison
# =====
def compare(df, group_col, a_val, b_val, a_label, b_label):
    """Welch t-test on ever_lied between two groups; returns display dict."""
    a = df.loc[df[group_col] == a_val, 'ever_lied']
    b = df.loc[df[group_col] == b_val, 'ever_lied']
    test = stats.ttest_ind(a, b, equal_var=False)
    return {
        'a_label': a_label, 'a_pct': 100 * a.mean(),
        'b_label': b_label, 'b_pct': 100 * b.mean(),
        'diff_pp': 100 * (a.mean() - b.mean()),
        'p': test.pvalue,
    }


# =====
# LaTeX export
# =====
def write_tex(rows, path):
    """Write table matching the style of other tables in analysis/output."""
    lines = [
        '\\begin{tabular}{lrr}',
        '  \\toprule',
        '  Comparison & Difference (pp) & $p$-value \\\\ ',
        '  \\midrule',
    ]
    for r in rows:
        lines.append(
            f"  {r['a_label']} ({r['a_pct']:.1f}\\%) vs.\\ "
            f"{r['b_label']} ({r['b_pct']:.1f}\\%) & "
            f"{r['diff_pp']:.1f} & {r['p']:.3f} \\\\ "
        )
    lines.extend(['  \\bottomrule', '\\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n')


# %%
if __name__ == '__main__':
    main()
