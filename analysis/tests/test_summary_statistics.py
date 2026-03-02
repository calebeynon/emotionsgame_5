"""
Purpose: Integration tests for the summary statistics pipeline. Runs each
         ss_*.py module's main() and verifies expected output files exist
         with correct LaTeX structure.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import pytest

# Allow imports from the summary_statistics package
_SS_DIR = Path(__file__).resolve().parent.parent / 'analysis' / 'summary_statistics'
sys.path.insert(0, str(_SS_DIR))

from ss_common import OUTPUT_DIR

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
    """Verify total output is 36 files (34 .tex + 2 .png)."""
    tex_files = list(OUTPUT_DIR.glob('*.tex'))
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
