"""
Purpose: Generate demographic summary statistics tables for the public goods
         experiment. Outputs gender, age, ethnicity, siblings, religion
         distributions, and demographic-contribution correlations by treatment.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import pandas as pd

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    SESSION_CODE_REMAP,
    load_contributions,
    load_raw_data,
    safe_mean,
    safe_pct,
    write_tex_table,
)

# SURVEY COLUMN MAPPING
_Q_COLS = {
    'gender': 'finalresults.1.player.q1',
    'ethnicity': 'finalresults.1.player.q2',
    'age': 'finalresults.1.player.q3',
    'major': 'finalresults.1.player.q4',
    'siblings': 'finalresults.1.player.q5',
    'religion': 'finalresults.1.player.q6',
}

# Religion importance ordered from least to most
_RELIGION_ORDER = [
    'Not at all important',
    'Slightly important',
    'Fairly important',
    'Important',
    'Very important',
]


# =====
# Main function
# =====

def main():
    """Generate all demographic summary statistics tables."""
    survey = extract_survey_data()

    write_tex_table(compute_gender_table(survey), 'demographics_gender.tex', 'lrrr')
    write_tex_table(compute_age_table(survey), 'demographics_age.tex', 'lrrrrr')
    write_tex_table(compute_ethnicity_table(survey), 'demographics_ethnicity.tex', 'lrrr')
    write_tex_table(compute_siblings_table(survey), 'demographics_siblings.tex', 'lrrr')
    write_tex_table(compute_religion_table(survey), 'demographics_religion.tex', 'lrrr')

    corr = compute_demographic_correlations(survey)
    write_tex_table(corr, 'demographics_contribution_correlation.tex', 'llrrr')


# =====
# Data extraction
# =====

def extract_survey_data():
    """Extract survey responses with treatment, label, and session code."""
    raw = load_raw_data()
    survey = raw[['session.code', 'participant.label', 'treatment']].copy()
    for name, col in _Q_COLS.items():
        survey[name] = raw[col]
    survey = survey.rename(columns={
        'session.code': 'session_code', 'participant.label': 'label',
    })
    survey['session_code'] = survey['session_code'].replace(SESSION_CODE_REMAP)
    return survey


# =====
# Categorical distribution tables
# =====

def compute_gender_table(survey):
    """Gender counts and percentages, by treatment and overall."""
    return _categorical_table(survey, 'gender')


def compute_ethnicity_table(survey):
    """Ethnicity counts and percentages, by treatment and overall."""
    return _categorical_table(survey, 'ethnicity')


def compute_religion_table(survey):
    """Religion importance counts and percentages, by treatment and overall."""
    return _categorical_table(survey, 'religion', order=_RELIGION_ORDER)


def _categorical_table(survey, column, order=None):
    """Build count/pct table for a categorical variable by treatment + overall."""
    rows = []
    for treatment in sorted(survey['treatment'].unique()):
        t_rows = _count_and_pct(survey[survey['treatment'] == treatment], column, order)
        rows.extend([[f'T{treatment}'] + r for r in t_rows])
    overall_rows = _count_and_pct(survey, column, order)
    rows.extend([['Overall'] + r for r in overall_rows])
    return pd.DataFrame(rows, columns=['Group', column.title(), 'Count', 'Pct'])


def _count_and_pct(df, column, order=None):
    """Compute count and percentage for each category value."""
    counts = df[column].value_counts()
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    total = len(df)
    return [
        [cat, int(n), safe_pct(n, total)]
        for cat, n in counts.items()
    ]


# =====
# Continuous summary tables
# =====

def compute_age_table(survey):
    """Age mean, median, SD, min, max, by treatment and overall."""
    return _continuous_table(survey, 'age')


def compute_siblings_table(survey):
    """Siblings mean, median, SD, by treatment and overall."""
    return _continuous_table(survey, 'siblings')


def _continuous_table(survey, column):
    """Build descriptive stats table for a numeric variable by treatment + overall."""
    rows = []
    for treatment in sorted(survey['treatment'].unique()):
        t_data = survey[survey['treatment'] == treatment][column]
        rows.append([f'T{treatment}'] + _describe_numeric(t_data))
    rows.append(['Overall'] + _describe_numeric(survey[column]))
    return pd.DataFrame(rows, columns=[
        'Group', 'Mean', 'Median', 'SD', 'Min', 'Max',
    ])


def _describe_numeric(series):
    """Return [mean, median, sd, min, max] rounded to 2 decimal places."""
    if len(series) == 0 or series.isna().all():
        return ['--'] * 5
    return [
        round(series.mean(), 2),
        round(series.median(), 2),
        round(series.std(), 2),
        series.min(),
        series.max(),
    ]


# =====
# Demographic-contribution correlations
# =====

def compute_demographic_correlations(survey):
    """Mean contribution by demographic categories, by treatment."""
    contrib = load_contributions()
    mean_contrib = _participant_mean_contributions(contrib)
    merged = survey.merge(mean_contrib, on=['session_code', 'label'])
    n_dropped = len(survey) - len(merged)
    if n_dropped > 0:
        raise ValueError(f"Demographics merge dropped {n_dropped} participants")
    rows = []
    rows.extend(_gender_correlation(merged))
    rows.extend(_age_correlation(merged))
    rows.extend(_siblings_correlation(merged))
    rows.extend(_religion_correlation(merged))
    return pd.DataFrame(rows, columns=[
        'Category', 'Value', 'T1 Mean', 'T2 Mean', 'Overall Mean',
    ])


def _participant_mean_contributions(contrib):
    """Mean contribution per participant across all rounds."""
    return contrib.groupby(['session_code', 'label'])['contribution'].mean() \
        .reset_index(name='mean_contribution')


def _gender_correlation(merged):
    """Mean contribution by gender x treatment."""
    return _group_means(merged, 'Gender', 'gender')


def _age_correlation(merged):
    """Mean contribution by age group (below/above median) x treatment."""
    median_age = merged['age'].median()
    merged = merged.copy()
    merged['age_group'] = merged['age'].apply(
        lambda x: f'Below median ({int(median_age)})' if x < median_age
        else f'Above median ({int(median_age)})',
    )
    return _group_means(merged, 'Age', 'age_group')


def _siblings_correlation(merged):
    """Mean contribution by sibling count category x treatment."""
    merged = merged.copy()
    merged['sib_group'] = merged['siblings'].apply(_categorize_siblings)
    return _group_means(merged, 'Siblings', 'sib_group')


def _categorize_siblings(n):
    """Categorize sibling count into 0, 1, or 2+."""
    if n == 0:
        return '0'
    if n == 1:
        return '1'
    return '2+'


def _religion_correlation(merged):
    """Mean contribution by religion importance x treatment."""
    return _group_means(merged, 'Religion', 'religion')


def _group_means(merged, category_label, column):
    """Mean contribution for each value of column, by treatment and overall."""
    rows = []
    for value in sorted(merged[column].unique()):
        subset = merged[merged[column] == value]
        t1 = subset[subset['treatment'] == 1]['mean_contribution']
        t2 = subset[subset['treatment'] == 2]['mean_contribution']
        rows.append([
            category_label, value,
            safe_mean(t1),
            safe_mean(t2),
            safe_mean(subset['mean_contribution']),
        ])
    return rows


# %%
if __name__ == "__main__":
    main()
