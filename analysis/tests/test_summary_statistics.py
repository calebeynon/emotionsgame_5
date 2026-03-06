"""
Purpose: Unit and integration tests for the summary statistics pipeline.
         Unit tests verify core computations with synthetic data.
         Integration tests run each module's main() and verify output files.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from the summary_statistics package
_SS_DIR = Path(__file__).resolve().parent.parent / 'analysis' / 'summary_statistics'
sys.path.insert(0, str(_SS_DIR))

from ss_common import OUTPUT_DIR, safe_mean, safe_pct

# EXPECTED OUTPUT FILES per module
_CONTRIBUTIONS_FILES = [
    'contributions_descriptive.tex',
    'contributions_frequencies.tex',
    'contributions_extremes.tex',
    'contributions_histogram_by_treatment.png',
    'contributions_histogram_by_supergame.png',
]
_CHAT_FILES = [
    'chat_volume.tex',
    'chat_length.tex',
    'chat_participation.tex',
    'chat_word_frequency.tex',
    'chat_orphan_volume.tex',
]
_SENTIMENT_FILES = [
    'sentiment_descriptive.tex',
    'sentiment_components.tex',
    'sentiment_categories.tex',
    'sentiment_intensity.tex',
    'sentiment_contribution_correlation.tex',
]
_BEHAVIOR_FILES = [
    'behavior_promise_rates.tex',
    'behavior_liar_rates.tex',
    'behavior_sucker_rates.tex',
    'behavior_persistence.tex',
    'behavior_conditional_contribution.tex',
]
_PAYOFFS_FILES = [
    'payoffs_summary.tex',
    'payoffs_by_supergame.tex',
    'payoffs_inequality.tex',
    'payoffs_dollar_distribution.tex',
]
_GROUPS_FILES = [
    'groups_cooperation.tex',
    'groups_free_riders.tex',
    'groups_within_sd.tex',
    'groups_regrouping_effect.tex',
]
_DEMOGRAPHICS_FILES = [
    'demographics_gender.tex',
    'demographics_age.tex',
    'demographics_ethnicity.tex',
    'demographics_siblings.tex',
    'demographics_religion.tex',
    'demographics_contribution_correlation.tex',
]
_EXPERIMENT_FILES = [
    'experiment_totals.tex',
    'experiment_timing.tex',
]
_EXPECTED_TEX_COUNT = 34
_EXPECTED_PNG_COUNT = 2
_EXPECTED_TOTAL = _EXPECTED_TEX_COUNT + _EXPECTED_PNG_COUNT


# =====
# Helper
# =====

def _verify_tex_file(filepath):
    """Assert a .tex file contains valid tabular environment."""
    content = filepath.read_text()
    assert '\\begin{tabular}' in content
    assert '\\end{tabular}' in content


def _verify_png_file(filepath):
    """Assert a .png file exists with nonzero size."""
    assert filepath.stat().st_size > 0


def _verify_outputs(filenames):
    """Verify all expected files exist and have valid content."""
    for name in filenames:
        path = OUTPUT_DIR / name
        assert path.exists(), f"Missing output: {name}"
        if name.endswith('.tex'):
            _verify_tex_file(path)
        elif name.endswith('.png'):
            _verify_png_file(path)


# =====
# Unit tests — shared utilities
# =====

def test_safe_mean_normal():
    assert safe_mean(pd.Series([10, 20, 30])) == 20.0


def test_safe_mean_empty():
    assert safe_mean(pd.Series([], dtype=float)) == '--'


def test_safe_mean_all_nan():
    assert safe_mean(pd.Series([float('nan'), float('nan')])) == '--'


def test_safe_pct_normal():
    assert safe_pct(25, 100) == 25.0


def test_safe_pct_zero_total():
    assert safe_pct(5, 0) == 0.0


# =====
# Unit tests — gini coefficient
# =====

def test_gini_perfect_equality():
    from ss_payoffs import gini_coefficient
    values = np.array([100, 100, 100, 100])
    assert gini_coefficient(values) == 0.0


def test_gini_known_value():
    """Gini of [1,2,3,4,5] = 0.2667 (known analytical result)."""
    from ss_payoffs import gini_coefficient
    values = np.array([1, 2, 3, 4, 5])
    assert abs(gini_coefficient(values) - 0.2667) < 0.001


def test_gini_all_zero():
    from ss_payoffs import gini_coefficient
    assert gini_coefficient(np.array([0, 0, 0])) == 0.0


def test_gini_single_element():
    from ss_payoffs import gini_coefficient
    assert gini_coefficient(np.array([42])) == 0.0


# =====
# Unit tests — sentiment classification
# =====

def test_assign_category_positive():
    from ss_sentiment import _assign_category
    df = pd.DataFrame({'sentiment_compound_mean': [0.5, 0.1, -0.1, 0.0]})
    result = _assign_category(df)
    assert list(result['category']) == ['Positive', 'Positive', 'Negative', 'Neutral']


def test_assign_category_thresholds():
    """Values exactly at thresholds: >=0.05 is Positive, <=-0.05 is Negative."""
    from ss_sentiment import _assign_category
    df = pd.DataFrame({'sentiment_compound_mean': [0.05, -0.05, 0.04, -0.04]})
    result = _assign_category(df)
    assert list(result['category']) == ['Positive', 'Negative', 'Neutral', 'Neutral']


def test_assign_intensity():
    from ss_sentiment import _assign_intensity
    df = pd.DataFrame({'sentiment_compound_mean': [0.7, -0.3, 0.1, 0.0]})
    result = _assign_intensity(df)
    assert list(result['intensity']) == ['Strong', 'Moderate', 'Weak', 'Weak']


# =====
# Unit tests — extreme contribution rates
# =====

def test_compute_extreme_rates():
    from ss_contributions import compute_extreme_rates
    df = pd.DataFrame({
        'treatment': [1, 1, 1, 1],
        'segment': ['supergame1'] * 4,
        'round': [1, 1, 1, 1],
        'contribution': [0, 25, 25, 10],
    })
    result = compute_extreme_rates(df)
    assert result['Pct Max'].iloc[0] == 50.0
    assert result['Pct Zero'].iloc[0] == 25.0


# =====
# Unit tests — orphan chat tagging
# =====

def test_tag_orphan_messages():
    from ss_chat import _tag_orphan_messages
    # Supergame 1 has 3 rounds, 4 groups per session
    # Pages: base=0, round1=0-3, round2=4-7, round3(orphan)=8-11
    chat = pd.DataFrame({
        'session_code': ['s1'] * 4,
        'supergame': ['supergame1'] * 4,
        'sg_num': [1] * 4,
        'ch_page': [0, 4, 8, 11],
    })
    result = _tag_orphan_messages(chat)
    assert list(result['is_orphan']) == [False, False, True, True]


# =====
# Unit tests — behavioral persistence
# =====

def test_persist_pct_all_persist():
    from ss_behavior import _persist_pct
    # Player (s1, p1) flagged in both supergames
    curr = pd.DataFrame(
        {'is_liar_20': [True]},
        index=pd.MultiIndex.from_tuples([('s1', 'p1')]),
    )
    nxt = pd.DataFrame(
        {'is_liar_20': [True]},
        index=pd.MultiIndex.from_tuples([('s1', 'p1')]),
    )
    assert _persist_pct(curr, nxt, 'is_liar_20') == '100.0\\%'


def test_persist_pct_none_persist():
    from ss_behavior import _persist_pct
    curr = pd.DataFrame(
        {'is_liar_20': [True]},
        index=pd.MultiIndex.from_tuples([('s1', 'p1')]),
    )
    nxt = pd.DataFrame(
        {'is_liar_20': [False]},
        index=pd.MultiIndex.from_tuples([('s1', 'p1')]),
    )
    assert _persist_pct(curr, nxt, 'is_liar_20') == '0.0\\%'


def test_persist_pct_empty_returns_sentinel():
    from ss_behavior import _persist_pct
    curr = pd.DataFrame(
        {'is_liar_20': [False]},
        index=pd.MultiIndex.from_tuples([('s1', 'p1')]),
    )
    nxt = pd.DataFrame(
        {'is_liar_20': [False]},
        index=pd.MultiIndex.from_tuples([('s1', 'p1')]),
    )
    assert _persist_pct(curr, nxt, 'is_liar_20') == '--'


# =====
# Unit tests — extract_treatment
# =====

def test_extract_treatment_valid():
    from ss_common import extract_treatment
    assert extract_treatment('03_t2_data.csv') == 2
    assert extract_treatment('01_t1_chat.csv') == 1


def test_extract_treatment_invalid():
    from ss_common import extract_treatment
    with pytest.raises(ValueError):
        extract_treatment('readme.csv')


# =====
# Integration tests
# =====

@pytest.mark.integration
def test_contributions():
    """Run ss_contributions.py and verify output files."""
    import ss_contributions
    ss_contributions.main()
    _verify_outputs(_CONTRIBUTIONS_FILES)


@pytest.mark.integration
def test_chat():
    """Run ss_chat.py and verify output files."""
    import ss_chat
    ss_chat.main()
    _verify_outputs(_CHAT_FILES)


@pytest.mark.integration
def test_sentiment():
    """Run ss_sentiment.py and verify output files."""
    import ss_sentiment
    ss_sentiment.main()
    _verify_outputs(_SENTIMENT_FILES)


@pytest.mark.integration
def test_behavior():
    """Run ss_behavior.py and verify output files."""
    import ss_behavior
    ss_behavior.main()
    _verify_outputs(_BEHAVIOR_FILES)


@pytest.mark.integration
def test_payoffs():
    """Run ss_payoffs.py and verify output files."""
    import ss_payoffs
    ss_payoffs.main()
    _verify_outputs(_PAYOFFS_FILES)


@pytest.mark.integration
def test_groups():
    """Run ss_groups.py and verify output files."""
    import ss_groups
    ss_groups.main()
    _verify_outputs(_GROUPS_FILES)


@pytest.mark.integration
def test_demographics():
    """Run ss_demographics.py and verify output files."""
    import ss_demographics
    ss_demographics.main()
    _verify_outputs(_DEMOGRAPHICS_FILES)


@pytest.mark.integration
def test_experiment_totals():
    """Run ss_experiment_totals.py and verify output files."""
    import ss_experiment_totals
    ss_experiment_totals.main()
    _verify_outputs(_EXPERIMENT_FILES)


@pytest.mark.integration
def test_total_file_count():
    """Verify pipeline generates 34 .tex and 2 .png files (36 total)."""
    tex_files = [f for f in OUTPUT_DIR.glob('*.tex') if f.name != 'review_all_tables.tex']
    png_files = list(OUTPUT_DIR.glob('*.png'))
    assert len(tex_files) == _EXPECTED_TEX_COUNT, (
        f"Expected {_EXPECTED_TEX_COUNT} .tex files, got {len(tex_files)}"
    )
    assert len(png_files) == _EXPECTED_PNG_COUNT, (
        f"Expected {_EXPECTED_PNG_COUNT} .png files, got {len(png_files)}"
    )
    total = len(tex_files) + len(png_files)
    assert total == _EXPECTED_TOTAL, (
        f"Expected {_EXPECTED_TOTAL} total files, got {total}"
    )
