"""
Tests for liar behavioral classification and lied_this_round_20 flag.

Tests the liar flag logic (cumulative promise-breaking) and the non-sticky
lied_this_round_20 flag. Sucker tests are in test_sucker_classification.py.

Author: Claude Code
Date: 2026-01-17
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from behavior_helpers import (
    is_promise_broken_20,
    is_promise_broken_5,
    compute_liar_flags,
    compute_sucker_flags,
    classify_player_behavior,
)
from behavior_helpers_df import compute_lied_this_round_flags

# =====
# Constants
# =====
THRESHOLD_20 = 20
THRESHOLD_5 = 5


# =====
# Fixtures
# =====
@pytest.fixture
def base_promise_df():
    """Create a base promise DataFrame for testing."""
    return pd.DataFrame({
        'session_code': ['abc123'] * 8,
        'treatment': [1] * 8,
        'segment': ['supergame1'] * 8,
        'round': [1, 1, 1, 1, 2, 2, 2, 2],
        'group': [1, 1, 1, 1, 1, 1, 1, 1],
        'label': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
        'participant_id': [1, 2, 3, 4, 1, 2, 3, 4],
        'contribution': [25, 20, 15, 5, 25, 20, 15, 5],
        'made_promise': [True, True, True, True, True, True, True, True],
    })


@pytest.fixture
def multi_segment_df():
    """Create DataFrame spanning multiple segments (supergames)."""
    rows = []
    for segment in ['supergame1', 'supergame2']:
        for round_num in [1, 2]:
            for label, pid, contrib, promise in [
                ('A', 1, 25, True), ('B', 2, 20, True),
                ('C', 3, 15, True), ('D', 4, 5, False),
            ]:
                rows.append({
                    'session_code': 'abc123', 'treatment': 1,
                    'segment': segment, 'round': round_num,
                    'group': 1, 'label': label, 'participant_id': pid,
                    'contribution': contrib, 'made_promise': promise,
                })
    return pd.DataFrame(rows)


def _two_player_df(r1_contribs, r1_promises, r2_contribs=None, r2_promises=None):
    """Helper: two-player, two-round DataFrame with configurable round 1 and 2."""
    r2_contribs = r2_contribs or [25, 25]
    r2_promises = r2_promises or [True, True]
    return pd.DataFrame({
        'session_code': ['abc123'] * 4, 'treatment': [1] * 4,
        'segment': ['supergame1'] * 4,
        'round': [1, 1, 2, 2], 'group': [1] * 4,
        'label': ['A', 'B', 'A', 'B'], 'participant_id': [1, 2, 1, 2],
        'contribution': r1_contribs + r2_contribs,
        'made_promise': r1_promises + r2_promises,
    })


# =====
# Test is_promise_broken helper functions
# =====
class TestIsPromiseBroken:
    """Tests for promise broken threshold functions."""

    def test_threshold_20_broken(self):
        assert is_promise_broken_20(19) is True
        assert is_promise_broken_20(0) is True

    def test_threshold_20_honored(self):
        assert is_promise_broken_20(20) is False
        assert is_promise_broken_20(25) is False

    def test_threshold_5_broken(self):
        assert is_promise_broken_5(4) is True
        assert is_promise_broken_5(0) is True

    def test_threshold_5_honored(self):
        assert is_promise_broken_5(5) is False
        assert is_promise_broken_5(25) is False


# =====
# Test round 1 behavior
# =====
class TestRound1AlwaysFalse:
    """Tests that round 1 always has False for liar/sucker flags."""

    def test_liar_20_false_round_1(self, base_promise_df):
        result = compute_liar_flags(base_promise_df, threshold='20')
        assert (result[result['round'] == 1]['is_liar_20'] == False).all()

    def test_liar_5_false_round_1(self, base_promise_df):
        result = compute_liar_flags(base_promise_df, threshold='5')
        assert (result[result['round'] == 1]['is_liar_5'] == False).all()

    def test_sucker_20_false_round_1(self, base_promise_df):
        result = compute_sucker_flags(base_promise_df, threshold='20')
        assert (result[result['round'] == 1]['is_sucker_20'] == False).all()

    def test_sucker_5_false_round_1(self, base_promise_df):
        result = compute_sucker_flags(base_promise_df, threshold='5')
        assert (result[result['round'] == 1]['is_sucker_5'] == False).all()


# =====
# Test liar classification
# =====
class TestLiarThreshold20:
    """Tests for liar classification with threshold 20."""

    def test_promise_contrib_below_20_becomes_liar(self):
        """Player who promises and contributes < 20 becomes liar next round."""
        df = _two_player_df([19, 25], [True, True])
        result = compute_liar_flags(df, threshold='20')
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r2['is_liar_20'].iloc[0] == True

    def test_no_promise_contrib_below_20_not_liar(self):
        """Player who contributes < 20 but made no promise is not liar."""
        df = _two_player_df([10, 25], [False, True])
        result = compute_liar_flags(df, threshold='20')
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r2['is_liar_20'].iloc[0] == False


class TestLiarThreshold5:
    """Tests for liar classification with threshold 5."""

    def test_promise_contrib_below_5_becomes_liar(self):
        """Player who promises and contributes < 5 becomes liar next round."""
        df = _two_player_df([4, 25], [True, True])
        result = compute_liar_flags(df, threshold='5')
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r2['is_liar_5'].iloc[0] == True


class TestLiarContributionAtThreshold:
    """Tests for boundary conditions at threshold values."""

    def test_contrib_exactly_20_not_liar_20(self):
        """Contribution of exactly 20 is NOT a broken promise (threshold 20)."""
        df = _two_player_df([20, 25], [True, True])
        result = compute_liar_flags(df, threshold='20')
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r2['is_liar_20'].iloc[0] == False

    def test_contrib_exactly_5_not_liar_5(self):
        """Contribution of exactly 5 is NOT a broken promise (threshold 5)."""
        df = _two_player_df([5, 25], [True, True])
        result = compute_liar_flags(df, threshold='5')
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r2['is_liar_5'].iloc[0] == False


class TestLiarPersistence:
    """Tests that liar flag persists across rounds within a segment."""

    def _three_round_df(self):
        """Three-round segment where A breaks promise in round 1."""
        rows = []
        for r in [1, 2, 3]:
            for label, pid, contrib in [('A', 1, 10 if r == 1 else 25), ('B', 2, 25), ('C', 3, 25)]:
                rows.append({
                    'session_code': 'abc123', 'treatment': 1,
                    'segment': 'supergame1', 'round': r,
                    'group': 1, 'label': label, 'participant_id': pid,
                    'contribution': contrib, 'made_promise': True,
                })
        return pd.DataFrame(rows)

    def test_liar_persists_across_rounds(self):
        """Once flagged as liar, stays flagged until segment ends."""
        result = compute_liar_flags(self._three_round_df(), threshold='20')
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        a_r3 = result[(result['label'] == 'A') & (result['round'] == 3)]
        assert a_r2['is_liar_20'].iloc[0] == True
        assert a_r3['is_liar_20'].iloc[0] == True


class TestLiarResetsNewSegment:
    """Tests that liar flag resets at start of new supergame."""

    def test_liar_resets_new_segment(self, multi_segment_df):
        """Liar flag resets at start of new supergame (segment)."""
        df = multi_segment_df.copy()
        mask = (df['label'] == 'C') & (df['segment'] == 'supergame1') & (df['round'] == 1)
        df.loc[mask, 'contribution'] = 10
        df.loc[mask, 'made_promise'] = True

        result = compute_liar_flags(df, threshold='20')

        c_sg1_r2 = result[
            (result['label'] == 'C') & (result['segment'] == 'supergame1') & (result['round'] == 2)
        ]
        c_sg2_r1 = result[
            (result['label'] == 'C') & (result['segment'] == 'supergame2') & (result['round'] == 1)
        ]
        assert c_sg1_r2['is_liar_20'].iloc[0] == True
        assert c_sg2_r1['is_liar_20'].iloc[0] == False


class TestNoPromiseNoLiar:
    """Tests that players who never promise cannot be liars."""

    def test_no_promise_no_liar(self):
        """Player who never makes a promise cannot be a liar."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 6, 'treatment': [1] * 6,
            'segment': ['supergame1'] * 6,
            'round': [1, 1, 2, 2, 3, 3], 'group': [1] * 6,
            'label': ['A', 'B', 'A', 'B', 'A', 'B'], 'participant_id': [1, 2, 1, 2, 1, 2],
            'contribution': [0, 25, 0, 25, 0, 25],
            'made_promise': [False, True, False, True, False, True],
        })
        result = compute_liar_flags(df, threshold='20')
        assert (result[result['label'] == 'A']['is_liar_20'] == False).all()


# =====
# Test lied_this_round_20 (non-cumulative flag)
# =====
class TestLiedThisRound20:
    """Tests for the non-cumulative lied_this_round_20 flag."""

    def _single_row(self, contribution, made_promise):
        return pd.DataFrame({
            'session_code': ['abc123'], 'treatment': [1],
            'segment': ['supergame1'], 'round': [1], 'group': [1],
            'label': ['A'], 'participant_id': [1],
            'contribution': [contribution], 'made_promise': [made_promise],
        })

    def test_lied_when_promise_and_low_contribution(self):
        result = compute_lied_this_round_flags(self._single_row(10, True))
        assert result['lied_this_round_20'].iloc[0] == True

    def test_not_lied_when_no_promise(self):
        result = compute_lied_this_round_flags(self._single_row(10, False))
        assert result['lied_this_round_20'].iloc[0] == False

    def test_not_lied_when_high_contribution(self):
        result = compute_lied_this_round_flags(self._single_row(25, True))
        assert result['lied_this_round_20'].iloc[0] == False

    def test_not_lied_at_threshold(self):
        result = compute_lied_this_round_flags(self._single_row(20, True))
        assert result['lied_this_round_20'].iloc[0] == False

    def test_not_sticky_across_rounds(self):
        """KEY: lied in round 1 but contributes 25 in round 2 -> False in round 2."""
        df = _two_player_df([10, 25], [True, True])
        result = compute_lied_this_round_flags(df)
        a_r1 = result[(result['label'] == 'A') & (result['round'] == 1)]
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r1['lied_this_round_20'].iloc[0] == True
        assert a_r2['lied_this_round_20'].iloc[0] == False

    def test_round_1_can_be_true(self):
        """Unlike is_liar_20, lied_this_round_20 CAN be True in round 1."""
        result = compute_lied_this_round_flags(self._single_row(10, True))
        assert result['lied_this_round_20'].iloc[0] == True


# =====
# Invariant: lied_this_round_20=True implies contribution < 20
# =====
class TestLiedThisRound20Invariant:
    """Invariant: lied_this_round_20=True always implies contribution < 20."""

    @pytest.mark.parametrize("contribution,made_promise", [
        (0, True), (5, True), (10, True), (15, True), (19, True),
        (20, True), (25, True), (0, False), (10, False), (25, False),
    ])
    def test_lied_implies_low_contribution(self, contribution, made_promise):
        """If lied_this_round_20=True then contribution must be < 20."""
        df = pd.DataFrame({
            'session_code': ['abc123'], 'treatment': [1],
            'segment': ['supergame1'], 'round': [1], 'group': [1],
            'label': ['A'], 'participant_id': [1],
            'contribution': [contribution], 'made_promise': [made_promise],
        })
        result = compute_lied_this_round_flags(df)
        if result['lied_this_round_20'].iloc[0]:
            assert contribution < 20


# =====
# Test combined classification function
# =====
class TestClassifyPlayerBehavior:
    """Tests for the combined classification function."""

    def test_classify_adds_all_flags(self, base_promise_df):
        """classify_player_behavior adds all five flag columns."""
        result = classify_player_behavior(base_promise_df)
        for col in ['is_liar_20', 'is_liar_5', 'is_sucker_20', 'is_sucker_5', 'lied_this_round_20']:
            assert col in result.columns

    def test_classify_preserves_original_columns(self, base_promise_df):
        """Original columns are preserved after classification."""
        result = classify_player_behavior(base_promise_df)
        for col in base_promise_df.columns:
            assert col in result.columns

    def test_classify_row_count_unchanged(self, base_promise_df):
        """Row count is unchanged after classification."""
        assert len(classify_player_behavior(base_promise_df)) == len(base_promise_df)


# =====
# Parametrized tests for threshold variations
# =====
class TestThresholdVariations:
    """Parametrized tests for different threshold values."""

    @pytest.mark.parametrize("contribution,expected_broken", [
        (0, True), (10, True), (19, True), (20, False), (25, False),
    ])
    def test_threshold_20_parametrized(self, contribution, expected_broken):
        assert is_promise_broken_20(contribution) == expected_broken

    @pytest.mark.parametrize("contribution,expected_broken", [
        (0, True), (4, True), (5, False), (10, False), (25, False),
    ])
    def test_threshold_5_parametrized(self, contribution, expected_broken):
        assert is_promise_broken_5(contribution) == expected_broken
