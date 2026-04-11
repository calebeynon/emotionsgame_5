"""
Tests for build_dynamic_regression_panel.py merged output.

Validates dynamic_regression_panel.csv: row count, column presence, key
uniqueness, treatment balance, period linearization, lag correctness,
chat variable coverage, emotion coverage, deviation variables, and subject_id.

Author: Claude Code
Date: 2026-04-10
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# FILE PATHS
PANEL_CSV = (
    Path(__file__).parent.parent / "datastore" / "derived" / "dynamic_regression_panel.csv"
)

# CONSTANTS
EXPECTED_ROW_COUNT = 3520
NUM_SESSIONS = 10
NUM_SUBJECTS = NUM_SESSIONS * 16  # 160
MAX_PERIOD = 22
ENDOWMENT = 25
MULTIPLIER = 0.4
PERIOD_OFFSETS = {1: 0, 2: 3, 3: 7, 4: 10, 5: 17}
EXPECTED_FIRST_PERIODS = {1: 1, 2: 4, 3: 8, 4: 11, 5: 18}
EXPECTED_LAST_PERIODS = {1: 3, 2: 7, 3: 10, 4: 17, 5: 22}
ROUND_1_ROWS = 800  # 10 sessions x 16 players x 5 supergames

EXPECTED_COLUMNS = [
    'session_code', 'treatment', 'segment', 'round', 'group', 'label',
    'participant_id', 'contribution', 'payoff',
    'segmentnumber', 'period', 'subject_id',
    'othercont', 'othercontaverage',
    'morethanaverage', 'lessthanaverage', 'diffcont',
    'contmore', 'contless', 'contmore_L1', 'contless_L1',
    'round1', 'round2', 'round3', 'round4', 'round5', 'round6', 'round7',
    'word_count', 'made_promise', 'sentiment_compound_mean', 'emotion_valence',
]


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def panel_df() -> pd.DataFrame:
    """Load the dynamic_regression_panel.csv output."""
    if not PANEL_CSV.exists():
        pytest.skip(f"dynamic_regression_panel.csv not found: {PANEL_CSV}")
    return pd.read_csv(PANEL_CSV)


# =====
# Test 1: Row count and columns
# =====
class TestStructure:
    """Verify row count, column presence, and key uniqueness."""

    def test_exactly_3520_rows(self, panel_df):
        """10 sessions x 16 players x 22 periods = 3,520 rows."""
        assert len(panel_df) == EXPECTED_ROW_COUNT

    def test_all_expected_columns_present(self, panel_df):
        """Every spec column must appear with correct count."""
        missing = set(EXPECTED_COLUMNS) - set(panel_df.columns)
        assert len(missing) == 0, f"Missing columns: {missing}"
        assert len(panel_df.columns) == len(EXPECTED_COLUMNS)

    def test_no_duplicate_keys(self, panel_df):
        """Each (session_code, segment, round, label) appears exactly once."""
        key = ['session_code', 'segment', 'round', 'label']
        assert panel_df.groupby(key).ngroups == len(panel_df)


# =====
# Test 2: Treatment balance
# =====
class TestTreatmentBalance:
    """Verify equal treatment allocation."""

    def test_1760_rows_per_treatment(self, panel_df):
        """Each treatment should have exactly 1,760 rows."""
        counts = panel_df['treatment'].value_counts()
        assert counts[1] == 1760
        assert counts[2] == 1760

    def test_5_sessions_per_treatment(self, panel_df):
        """Each treatment should have 5 sessions."""
        t_sessions = panel_df.groupby('session_code')['treatment'].first()
        assert t_sessions.value_counts()[1] == 5
        assert t_sessions.value_counts()[2] == 5

    def test_80_subjects_per_treatment(self, panel_df):
        """Each treatment should have 80 unique subjects."""
        for t in [1, 2]:
            n = panel_df[panel_df['treatment'] == t]['subject_id'].nunique()
            assert n == 80, f"Treatment {t}: {n} subjects"


# =====
# Test 3: Period linearization
# =====
class TestPeriodLinearization:
    """Verify period is correctly linearized across supergames."""

    def test_period_range_and_uniqueness(self, panel_df):
        """Period min=1, max=22, 22 unique values."""
        assert panel_df['period'].min() == 1
        assert panel_df['period'].max() == MAX_PERIOD
        assert panel_df['period'].nunique() == MAX_PERIOD

    @pytest.mark.parametrize("supergame,round_num,expected_period", [
        (1, 1, 1), (1, 3, 3),
        (2, 1, 4), (2, 4, 7),
        (3, 1, 8), (3, 3, 10),
        (4, 1, 11), (4, 7, 17),
        (5, 1, 18), (5, 5, 22),
    ])
    def test_specific_period_values(self, panel_df, supergame, round_num, expected_period):
        """Verify (supergame, round) -> period mapping."""
        mask = (panel_df['segmentnumber'] == supergame) & (panel_df['round'] == round_num)
        actual = panel_df.loc[mask, 'period'].unique()
        assert len(actual) == 1 and actual[0] == expected_period

    def test_periods_contiguous_within_supergame(self, panel_df):
        """Within each supergame, periods form a contiguous integer range."""
        for sg in range(1, 6):
            periods = sorted(panel_df.loc[panel_df['segmentnumber'] == sg, 'period'].unique())
            expected = list(range(EXPECTED_FIRST_PERIODS[sg], EXPECTED_LAST_PERIODS[sg] + 1))
            assert periods == expected


# =====
# Test 4: Lag correctness
# =====
class TestLagVariables:
    """Verify contmore_L1 and contless_L1 are correct within-subject lags."""

    def test_lags_nan_at_period_1(self, panel_df):
        """Both lag columns should be NaN at period==1."""
        p1 = panel_df[panel_df['period'] == 1]
        assert p1['contmore_L1'].isna().all()
        assert p1['contless_L1'].isna().all()

    def test_lag_nan_count_is_160(self, panel_df):
        """Lag NaN exactly 160 = one per subject at period 1."""
        assert panel_df['contmore_L1'].isna().sum() == NUM_SUBJECTS
        assert panel_df['contless_L1'].isna().sum() == NUM_SUBJECTS

    def test_contmore_lag_values_correct(self, panel_df):
        """contmore_L1[t] == contmore[t-1] within each subject."""
        errors = _validate_lag(panel_df, 'contmore', 'contmore_L1')
        assert len(errors) == 0, f"contmore_L1 errors:\n" + "\n".join(errors)

    def test_contless_lag_values_correct(self, panel_df):
        """contless_L1[t] == contless[t-1] within each subject."""
        errors = _validate_lag(panel_df, 'contless', 'contless_L1')
        assert len(errors) == 0, f"contless_L1 errors:\n" + "\n".join(errors)

    def test_known_lag_value(self, panel_df):
        """Regression: sa7mprty, sg1, r2, A has contmore_L1 = 8.333..."""
        row = _get_row(panel_df, 'sa7mprty', 'supergame1', 2, 'A')
        assert row['contmore_L1'].values[0] == pytest.approx(8.3333, abs=0.001)


# =====
# Test 5: Chat variable coverage
# =====
class TestChatVariables:
    """Verify made_promise, word_count, and sentiment_compound_mean."""

    def test_made_promise_no_nan_and_binary(self, panel_df):
        """made_promise: zero NaN, values exactly 0 or 1."""
        assert panel_df['made_promise'].isna().sum() == 0
        assert set(panel_df['made_promise'].unique()) <= {0, 1}

    def test_word_count_nan_only_at_round_1(self, panel_df):
        """word_count NaN only at round 1 (800 total), >= 0 elsewhere."""
        assert panel_df['word_count'].isna().sum() == ROUND_1_ROWS
        assert panel_df[panel_df['round'] > 1]['word_count'].isna().sum() == 0
        assert (panel_df['word_count'].dropna() >= 0).all()

    def test_sentiment_nan_only_at_round_1(self, panel_df):
        """sentiment_compound_mean NaN only at round 1 (800 total)."""
        assert panel_df['sentiment_compound_mean'].isna().sum() == ROUND_1_ROWS
        assert panel_df[panel_df['round'] > 1]['sentiment_compound_mean'].isna().sum() == 0


# =====
# Test 6: Emotion variable coverage
# =====
class TestEmotionVariables:
    """Verify emotion_valence coverage and range."""

    def test_emotion_valence_nan_count(self, panel_df):
        """Regression: exactly 824 NaN from real data."""
        assert panel_df['emotion_valence'].isna().sum() == 824

    def test_emotion_valence_range(self, panel_df):
        """Non-NaN emotion_valence should be in [-100, 100]."""
        valid = panel_df['emotion_valence'].dropna()
        assert valid.min() >= -100
        assert valid.max() <= 100

    def test_known_emotion_value(self, panel_df):
        """Regression: irrzlgk2, sg1, r1, A has emotion_valence = 2.4875."""
        row = _get_row(panel_df, 'irrzlgk2', 'supergame1', 1, 'A')
        assert row['emotion_valence'].values[0] == pytest.approx(2.4875, abs=0.001)


# =====
# Test 7: Deviation variable correctness
# =====
class TestDeviationVariables:
    """Verify othercont roundtrip and deviation decomposition."""

    def test_othercont_roundtrip(self, panel_df):
        """othercont = (payoff - 25 + 0.6 * contribution) / 0.4."""
        expected = (panel_df['payoff'] - ENDOWMENT + 0.6 * panel_df['contribution']) / MULTIPLIER
        assert (panel_df['othercont'] - expected).abs().max() < 1e-10

    def test_othercontaverage_is_othercont_div_3(self, panel_df):
        """othercontaverage = othercont / 3."""
        assert (panel_df['othercontaverage'] - panel_df['othercont'] / 3).abs().max() < 1e-12

    def test_flags_mutually_exclusive_and_binary(self, panel_df):
        """Flags are binary and mutually exclusive."""
        for col in ['morethanaverage', 'lessthanaverage']:
            assert set(panel_df[col].unique()) <= {0, 1}
        both = (panel_df['morethanaverage'] == 1) & (panel_df['lessthanaverage'] == 1)
        assert not both.any()

    def test_contmore_contless_decomposition(self, panel_df):
        """contmore + contless = |diffcont|, both >= 0."""
        assert panel_df['contmore'].min() >= -1e-10
        assert panel_df['contless'].min() >= -1e-10
        lhs = panel_df['contmore'] + panel_df['contless']
        assert (lhs - panel_df['diffcont'].abs()).abs().max() < 1e-10

    def test_known_othercont_value(self, panel_df):
        """Regression: sa7mprty, sg1, r1, A -> othercont = 20.0."""
        row = _get_row(panel_df, 'sa7mprty', 'supergame1', 1, 'A')
        assert row['othercont'].values[0] == pytest.approx(20.0)


# =====
# Test 8: Subject ID correctness
# =====
class TestSubjectId:
    """Verify subject_id construction and uniqueness."""

    def test_160_unique_subjects_with_22_rows_each(self, panel_df):
        """160 subjects, each with exactly 22 rows."""
        assert panel_df['subject_id'].nunique() == NUM_SUBJECTS
        counts = panel_df.groupby('subject_id').size()
        assert (counts == MAX_PERIOD).all()

    def test_no_subject_id_collision(self, panel_df):
        """Each (session, participant_id) maps to one subject_id and vice versa."""
        n_ids = panel_df['subject_id'].nunique()
        n_pairs = panel_df.groupby(['session_code', 'participant_id']).ngroups
        assert n_ids == n_pairs

    def test_known_subject_id(self, panel_df):
        """Regression: sa7mprty, label A (pid=1) -> subject_id % 100 == 1."""
        row = _get_row(panel_df, 'sa7mprty', 'supergame1', 1, 'A')
        assert row['participant_id'].values[0] == 1
        assert row['subject_id'].values[0] % 100 == 1


# =====
# Test 9: Round dummies
# =====
class TestRoundDummies:
    """Verify round indicator variables are correct."""

    def test_round_dummies_sum_to_1(self, panel_df):
        """Exactly one round dummy is 1 per row."""
        sums = panel_df[[f'round{r}' for r in range(1, 8)]].sum(axis=1)
        assert (sums == 1).all()

    @pytest.mark.parametrize("r", range(1, 8))
    def test_round_dummy_matches_round(self, panel_df, r):
        """round{r} == 1 iff round == r."""
        assert (panel_df[f'round{r}'] == (panel_df['round'] == r).astype(int)).all()


# =====
# Regression tests from verified outputs
# =====
class TestKnownValues:
    """Validate specific row values from the verified output."""

    def test_sa7mprty_sg1_r1_a(self, panel_df):
        """Full regression test for a known round-1 row."""
        r = _get_row(panel_df, 'sa7mprty', 'supergame1', 1, 'A').iloc[0]
        assert r['contribution'] == pytest.approx(15.0)
        assert r['payoff'] == pytest.approx(24.0)
        assert r['period'] == 1
        assert r['morethanaverage'] == 1
        assert r['round1'] == 1
        assert r['made_promise'] == 0
        assert pd.isna(r['word_count'])
        assert pd.isna(r['contmore_L1'])

    def test_sa7mprty_sg1_r2_a(self, panel_df):
        """Full regression test for a known round-2 row with chat."""
        r = _get_row(panel_df, 'sa7mprty', 'supergame1', 2, 'A').iloc[0]
        assert r['contribution'] == pytest.approx(25.0)
        assert r['othercont'] == pytest.approx(75.0)
        assert r['word_count'] == pytest.approx(18.0)
        assert r['sentiment_compound_mean'] == pytest.approx(0.0)
        assert r['contmore_L1'] == pytest.approx(8.3333, abs=0.001)

    def test_sa7mprty_sg3_r1_a_cross_supergame_lag(self, panel_df):
        """Regression: sg3 r1 has period 8, lag crosses supergame boundary."""
        r = _get_row(panel_df, 'sa7mprty', 'supergame3', 1, 'A').iloc[0]
        assert r['period'] == 8
        assert r['othercont'] == pytest.approx(45.0)
        assert r['lessthanaverage'] == 1
        assert r['contless'] == pytest.approx(5.0)
        assert pd.notna(r['contmore_L1'])


# =====
# Merge integrity
# =====
class TestMergeIntegrity:
    """Verify no rows lost or duplicated, no NaN in core columns."""

    def test_no_nan_in_core_columns(self, panel_df):
        """Core columns (not chat/emotion/lags) should have no NaN."""
        core_cols = [
            'session_code', 'treatment', 'segment', 'round', 'group',
            'label', 'participant_id', 'contribution', 'payoff',
            'segmentnumber', 'period', 'subject_id', 'othercont',
            'othercontaverage', 'morethanaverage', 'lessthanaverage',
            'diffcont', 'contmore', 'contless', 'made_promise',
        ]
        for col in core_cols:
            assert panel_df[col].isna().sum() == 0, f"NaN in '{col}'"

    def test_10_sessions_5_supergames_each(self, panel_df):
        """10 sessions, each with 5 supergames."""
        assert panel_df['session_code'].nunique() == NUM_SESSIONS
        for session in panel_df['session_code'].unique():
            n = panel_df[panel_df['session_code'] == session]['segmentnumber'].nunique()
            assert n == 5, f"Session {session}: {n} supergames"


# =====
# Helper functions
# =====
def _get_row(df, session_code, segment, round_num, label):
    """Fetch a single row by composite key."""
    mask = (
        (df['session_code'] == session_code)
        & (df['segment'] == segment)
        & (df['round'] == round_num)
        & (df['label'] == label)
    )
    result = df[mask]
    assert len(result) == 1, f"Expected 1 row for ({session_code}, {segment}, {round_num}, {label}), got {len(result)}"
    return result


def _validate_lag(df, source_col, lag_col):
    """Check lag_col[t] == source_col[t-1] within each subject."""
    sorted_df = df.sort_values(['subject_id', 'period'])
    expected = sorted_df.groupby('subject_id')[source_col].shift(1)
    expected[sorted_df['period'].values == 1] = np.nan
    actual = sorted_df[lag_col]
    mask = expected.notna()
    mismatches = mask & ((actual - expected).abs() > 1e-10)
    bad = sorted_df[mismatches].head(10)
    return [
        f"subject={r['subject_id']} period={r['period']}: "
        f"expected={expected.iloc[idx]}, got={r[lag_col]}"
        for idx, (_, r) in enumerate(bad.iterrows())
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
