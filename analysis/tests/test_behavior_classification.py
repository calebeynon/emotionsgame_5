"""
Tests for behavior classification (liar and sucker flags).

Tests the classification logic for identifying promise-breakers (liars)
and players who trusted promise-breakers (suckers) in the public goods game.

Author: Claude Code
Date: 2026-01-17
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from behavior_helpers import (
    is_promise_broken_strict,
    is_promise_broken_lenient,
    compute_liar_flags,
    compute_sucker_flags,
    classify_player_behavior,
)

# =====
# Constants
# =====
STRICT_THRESHOLD = 20  # Contribution >= 20 honors promise (strict)
LENIENT_THRESHOLD = 5  # Contribution >= 5 honors promise (lenient)


# =====
# Fixtures for mock data
# =====
@pytest.fixture
def base_promise_df():
    """Create a base promise DataFrame for testing."""
    return pd.DataFrame({
        'session_code': ['abc123'] * 8,
        'treatment': [1] * 8,
        'segment': ['supergame1'] * 4 + ['supergame1'] * 4,
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
                ('A', 1, 25, True),
                ('B', 2, 20, True),
                ('C', 3, 15, True),
                ('D', 4, 5, False),
            ]:
                rows.append({
                    'session_code': 'abc123',
                    'treatment': 1,
                    'segment': segment,
                    'round': round_num,
                    'group': 1,
                    'label': label,
                    'participant_id': pid,
                    'contribution': contrib,
                    'made_promise': promise,
                })
    return pd.DataFrame(rows)


# =====
# Test is_promise_broken helper functions
# =====
class TestIsPromiseBroken:
    """Tests for promise broken threshold functions."""

    def test_strict_threshold_broken(self):
        """Contribution < 20 is broken promise (strict)."""
        assert is_promise_broken_strict(19) is True
        assert is_promise_broken_strict(10) is True
        assert is_promise_broken_strict(0) is True

    def test_strict_threshold_honored(self):
        """Contribution >= 20 honors promise (strict)."""
        assert is_promise_broken_strict(20) is False
        assert is_promise_broken_strict(21) is False
        assert is_promise_broken_strict(25) is False

    def test_lenient_threshold_broken(self):
        """Contribution < 5 is broken promise (lenient)."""
        assert is_promise_broken_lenient(4) is True
        assert is_promise_broken_lenient(0) is True

    def test_lenient_threshold_honored(self):
        """Contribution >= 5 honors promise (lenient)."""
        assert is_promise_broken_lenient(5) is False
        assert is_promise_broken_lenient(10) is False
        assert is_promise_broken_lenient(25) is False


# =====
# Test round 1 behavior
# =====
class TestRound1AlwaysFalse:
    """Tests that round 1 always has False for all behavior flags."""

    def test_liar_strict_false_round_1(self, base_promise_df):
        """Round 1 has is_liar_strict=False for all players."""
        result = compute_liar_flags(base_promise_df, threshold='strict')
        round_1 = result[result['round'] == 1]

        assert (round_1['is_liar_strict'] == False).all()

    def test_liar_lenient_false_round_1(self, base_promise_df):
        """Round 1 has is_liar_lenient=False for all players."""
        result = compute_liar_flags(base_promise_df, threshold='lenient')
        round_1 = result[result['round'] == 1]

        assert (round_1['is_liar_lenient'] == False).all()

    def test_sucker_strict_false_round_1(self, base_promise_df):
        """Round 1 has is_sucker_strict=False for all players."""
        result = compute_sucker_flags(base_promise_df, threshold='strict')
        round_1 = result[result['round'] == 1]

        assert (round_1['is_sucker_strict'] == False).all()

    def test_sucker_lenient_false_round_1(self, base_promise_df):
        """Round 1 has is_sucker_lenient=False for all players."""
        result = compute_sucker_flags(base_promise_df, threshold='lenient')
        round_1 = result[result['round'] == 1]

        assert (round_1['is_sucker_lenient'] == False).all()


# =====
# Test liar classification
# =====
class TestLiarStrictThreshold:
    """Tests for liar classification with strict threshold."""

    def test_promise_contrib_below_20_becomes_liar(self):
        """Player who promises and contributes < 20 becomes liar next round."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 4,
            'treatment': [1] * 4,
            'segment': ['supergame1'] * 4,
            'round': [1, 1, 2, 2],
            'group': [1, 1, 1, 1],
            'label': ['A', 'B', 'A', 'B'],
            'participant_id': [1, 2, 1, 2],
            'contribution': [19, 25, 25, 25],  # A contributes 19 in round 1
            'made_promise': [True, True, True, True],
        })

        result = compute_liar_flags(df, threshold='strict')
        a_round_2 = result[(result['label'] == 'A') & (result['round'] == 2)]

        assert a_round_2['is_liar_strict'].iloc[0] == True

    def test_no_promise_contrib_below_20_not_liar(self):
        """Player who contributes < 20 but made no promise is not liar."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 4,
            'treatment': [1] * 4,
            'segment': ['supergame1'] * 4,
            'round': [1, 1, 2, 2],
            'group': [1, 1, 1, 1],
            'label': ['A', 'B', 'A', 'B'],
            'participant_id': [1, 2, 1, 2],
            'contribution': [10, 25, 25, 25],  # A contributes 10 in round 1
            'made_promise': [False, True, False, True],  # A never promises
        })

        result = compute_liar_flags(df, threshold='strict')
        a_round_2 = result[(result['label'] == 'A') & (result['round'] == 2)]

        assert a_round_2['is_liar_strict'].iloc[0] == False


class TestLiarLenientThreshold:
    """Tests for liar classification with lenient threshold."""

    def test_promise_contrib_below_5_becomes_liar(self):
        """Player who promises and contributes < 5 becomes liar next round."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 4,
            'treatment': [1] * 4,
            'segment': ['supergame1'] * 4,
            'round': [1, 1, 2, 2],
            'group': [1, 1, 1, 1],
            'label': ['A', 'B', 'A', 'B'],
            'participant_id': [1, 2, 1, 2],
            'contribution': [4, 25, 25, 25],  # A contributes 4 in round 1
            'made_promise': [True, True, True, True],
        })

        result = compute_liar_flags(df, threshold='lenient')
        a_round_2 = result[(result['label'] == 'A') & (result['round'] == 2)]

        assert a_round_2['is_liar_lenient'].iloc[0] == True


class TestLiarContributionAtThreshold:
    """Tests for boundary conditions at threshold values."""

    def test_contrib_exactly_20_not_liar_strict(self):
        """Contribution of exactly 20 is NOT a broken promise (strict)."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 4,
            'treatment': [1] * 4,
            'segment': ['supergame1'] * 4,
            'round': [1, 1, 2, 2],
            'group': [1, 1, 1, 1],
            'label': ['A', 'B', 'A', 'B'],
            'participant_id': [1, 2, 1, 2],
            'contribution': [20, 25, 25, 25],  # A contributes exactly 20
            'made_promise': [True, True, True, True],
        })

        result = compute_liar_flags(df, threshold='strict')
        a_round_2 = result[(result['label'] == 'A') & (result['round'] == 2)]

        assert a_round_2['is_liar_strict'].iloc[0] == False

    def test_contrib_exactly_5_not_liar_lenient(self):
        """Contribution of exactly 5 is NOT a broken promise (lenient)."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 4,
            'treatment': [1] * 4,
            'segment': ['supergame1'] * 4,
            'round': [1, 1, 2, 2],
            'group': [1, 1, 1, 1],
            'label': ['A', 'B', 'A', 'B'],
            'participant_id': [1, 2, 1, 2],
            'contribution': [5, 25, 25, 25],  # A contributes exactly 5
            'made_promise': [True, True, True, True],
        })

        result = compute_liar_flags(df, threshold='lenient')
        a_round_2 = result[(result['label'] == 'A') & (result['round'] == 2)]

        assert a_round_2['is_liar_lenient'].iloc[0] == False


class TestLiarPersistence:
    """Tests that liar flag persists across rounds within a segment."""

    def test_liar_persists_across_rounds(self):
        """Once flagged as liar, stays flagged until segment ends."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 9,
            'treatment': [1] * 9,
            'segment': ['supergame1'] * 9,
            'round': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'group': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'label': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
            'participant_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            # A breaks promise in round 1, then contributes 25 afterwards
            'contribution': [10, 25, 25, 25, 25, 25, 25, 25, 25],
            'made_promise': [True, True, True, True, True, True, True, True, True],
        })

        result = compute_liar_flags(df, threshold='strict')

        # A should be liar in round 2 and round 3
        a_round_2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        a_round_3 = result[(result['label'] == 'A') & (result['round'] == 3)]

        assert a_round_2['is_liar_strict'].iloc[0] == True
        assert a_round_3['is_liar_strict'].iloc[0] == True


class TestLiarResetsNewSegment:
    """Tests that liar flag resets at start of new supergame."""

    def test_liar_resets_new_segment(self, multi_segment_df):
        """Liar flag resets at start of new supergame (segment)."""
        # Modify player C to break promise in supergame1 round 1
        df = multi_segment_df.copy()
        df.loc[
            (df['label'] == 'C') & (df['segment'] == 'supergame1') & (df['round'] == 1),
            'contribution'
        ] = 10
        df.loc[
            (df['label'] == 'C') & (df['segment'] == 'supergame1') & (df['round'] == 1),
            'made_promise'
        ] = True

        result = compute_liar_flags(df, threshold='strict')

        # C should be liar in supergame1 round 2
        c_sg1_r2 = result[
            (result['label'] == 'C') &
            (result['segment'] == 'supergame1') &
            (result['round'] == 2)
        ]
        assert c_sg1_r2['is_liar_strict'].iloc[0] == True

        # C should NOT be liar in supergame2 round 1 (reset)
        c_sg2_r1 = result[
            (result['label'] == 'C') &
            (result['segment'] == 'supergame2') &
            (result['round'] == 1)
        ]
        assert c_sg2_r1['is_liar_strict'].iloc[0] == False


# =====
# Test sucker classification
# =====
class TestSuckerDefinition:
    """Tests for sucker classification definition."""

    def test_sucker_when_contrib_25_and_groupmate_broke_promise(self):
        """Player who contributes 25 when group member broke promise is sucker."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 8,
            'treatment': [1] * 8,
            'segment': ['supergame1'] * 8,
            'round': [1, 1, 1, 1, 2, 2, 2, 2],
            'group': [1, 1, 1, 1, 1, 1, 1, 1],
            'label': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
            'participant_id': [1, 2, 3, 4, 1, 2, 3, 4],
            # A promises and contributes 10 (breaks promise)
            # B contributes 25 in round 2 (becomes sucker)
            'contribution': [10, 25, 25, 25, 25, 25, 25, 25],
            'made_promise': [True, True, True, True, True, True, True, True],
        })

        result = compute_sucker_flags(df, threshold='strict')
        b_round_2 = result[(result['label'] == 'B') & (result['round'] == 2)]

        assert b_round_2['is_sucker_strict'].iloc[0] == True


class TestSuckerSamePersonRequirement:
    """Tests that the promise-maker must be the one who broke the promise."""

    def test_sucker_requires_same_person_broke_promise(self):
        """Promise-maker must be the one who broke promise (not just low contrib)."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 8,
            'treatment': [1] * 8,
            'segment': ['supergame1'] * 8,
            'round': [1, 1, 1, 1, 2, 2, 2, 2],
            'group': [1, 1, 1, 1, 1, 1, 1, 1],
            'label': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
            'participant_id': [1, 2, 3, 4, 1, 2, 3, 4],
            # A contributes 10 but did NOT make a promise
            # This should NOT create suckers
            'contribution': [10, 25, 25, 25, 25, 25, 25, 25],
            'made_promise': [False, True, True, True, False, True, True, True],
        })

        result = compute_sucker_flags(df, threshold='strict')
        round_2 = result[result['round'] == 2]

        # No one should be a sucker since A didn't make a promise
        assert (round_2['is_sucker_strict'] == False).all()


class TestSuckerPersistence:
    """Tests that sucker flag persists across rounds within a segment."""

    def test_sucker_persists_across_rounds(self):
        """Once flagged as sucker, stays flagged until segment ends."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 9,
            'treatment': [1] * 9,
            'segment': ['supergame1'] * 9,
            'round': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'group': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'label': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
            'participant_id': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            # A breaks promise in round 1
            # B contributes 25 in round 2 (becomes sucker)
            'contribution': [10, 25, 25, 25, 25, 25, 25, 25, 25],
            'made_promise': [True, True, True, True, True, True, True, True, True],
        })

        result = compute_sucker_flags(df, threshold='strict')

        # B should be sucker in round 2 and round 3
        b_round_2 = result[(result['label'] == 'B') & (result['round'] == 2)]
        b_round_3 = result[(result['label'] == 'B') & (result['round'] == 3)]

        assert b_round_2['is_sucker_strict'].iloc[0] == True
        assert b_round_3['is_sucker_strict'].iloc[0] == True


class TestSuckerResetsNewSegment:
    """Tests that sucker flag resets at start of new supergame."""

    def test_sucker_resets_new_segment(self):
        """Sucker flag resets at start of new supergame."""
        rows = []
        # Supergame 1: A breaks promise, B becomes sucker
        for round_num in [1, 2]:
            for label, pid, contrib, promise in [
                ('A', 1, 10 if round_num == 1 else 25, True),
                ('B', 2, 25, True),
            ]:
                rows.append({
                    'session_code': 'abc123',
                    'treatment': 1,
                    'segment': 'supergame1',
                    'round': round_num,
                    'group': 1,
                    'label': label,
                    'participant_id': pid,
                    'contribution': contrib,
                    'made_promise': promise,
                })

        # Supergame 2: Everyone behaves, B should not be sucker
        for round_num in [1, 2]:
            for label, pid, contrib, promise in [
                ('A', 1, 25, True),
                ('B', 2, 25, True),
            ]:
                rows.append({
                    'session_code': 'abc123',
                    'treatment': 1,
                    'segment': 'supergame2',
                    'round': round_num,
                    'group': 1,
                    'label': label,
                    'participant_id': pid,
                    'contribution': contrib,
                    'made_promise': promise,
                })

        df = pd.DataFrame(rows)
        result = compute_sucker_flags(df, threshold='strict')

        # B should be sucker in supergame1 round 2
        b_sg1_r2 = result[
            (result['label'] == 'B') &
            (result['segment'] == 'supergame1') &
            (result['round'] == 2)
        ]
        assert b_sg1_r2['is_sucker_strict'].iloc[0] == True

        # B should NOT be sucker in supergame2 round 1 (reset)
        b_sg2_r1 = result[
            (result['label'] == 'B') &
            (result['segment'] == 'supergame2') &
            (result['round'] == 1)
        ]
        assert b_sg2_r1['is_sucker_strict'].iloc[0] == False


class TestNonChatterCanBeSucker:
    """Tests that non-chatting players can be suckers."""

    def test_non_chatter_can_be_sucker(self):
        """Non-chatting player can be sucker if someone else broke promise."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 8,
            'treatment': [1] * 8,
            'segment': ['supergame1'] * 8,
            'round': [1, 1, 1, 1, 2, 2, 2, 2],
            'group': [1, 1, 1, 1, 1, 1, 1, 1],
            'label': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
            'participant_id': [1, 2, 3, 4, 1, 2, 3, 4],
            # A promises and breaks (contrib 10)
            # D never makes promises but contributes 25
            'contribution': [10, 25, 25, 25, 25, 25, 25, 25],
            'made_promise': [True, True, True, False, True, True, True, False],
        })

        result = compute_sucker_flags(df, threshold='strict')
        d_round_2 = result[(result['label'] == 'D') & (result['round'] == 2)]

        # D contributed 25 when A (a promise-breaker) was in the group
        assert d_round_2['is_sucker_strict'].iloc[0] == True


class TestNoPromiseNoLiar:
    """Tests that players who never promise cannot be liars."""

    def test_no_promise_no_liar(self):
        """Player who never makes a promise cannot be a liar."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 6,
            'treatment': [1] * 6,
            'segment': ['supergame1'] * 6,
            'round': [1, 1, 2, 2, 3, 3],
            'group': [1, 1, 1, 1, 1, 1],
            'label': ['A', 'B', 'A', 'B', 'A', 'B'],
            'participant_id': [1, 2, 1, 2, 1, 2],
            # A never promises but always contributes 0
            'contribution': [0, 25, 0, 25, 0, 25],
            'made_promise': [False, True, False, True, False, True],
        })

        result = compute_liar_flags(df, threshold='strict')
        a_rows = result[result['label'] == 'A']

        # A should never be flagged as liar
        assert (a_rows['is_liar_strict'] == False).all()


# =====
# Test combined classification function
# =====
class TestClassifyPlayerBehavior:
    """Tests for the combined classification function."""

    def test_classify_adds_all_flags(self, base_promise_df):
        """classify_player_behavior adds all four flag columns."""
        result = classify_player_behavior(base_promise_df)

        assert 'is_liar_strict' in result.columns
        assert 'is_liar_lenient' in result.columns
        assert 'is_sucker_strict' in result.columns
        assert 'is_sucker_lenient' in result.columns

    def test_classify_preserves_original_columns(self, base_promise_df):
        """Original columns are preserved after classification."""
        result = classify_player_behavior(base_promise_df)

        for col in base_promise_df.columns:
            assert col in result.columns

    def test_classify_row_count_unchanged(self, base_promise_df):
        """Row count is unchanged after classification."""
        result = classify_player_behavior(base_promise_df)

        assert len(result) == len(base_promise_df)


# =====
# Parametrized tests for threshold variations
# =====
class TestThresholdVariations:
    """Parametrized tests for different threshold values."""

    @pytest.mark.parametrize("contribution,expected_broken", [
        (0, True),
        (10, True),
        (19, True),
        (20, False),
        (25, False),
    ])
    def test_strict_threshold_parametrized(self, contribution, expected_broken):
        """Strict threshold: < 20 is broken."""
        assert is_promise_broken_strict(contribution) == expected_broken

    @pytest.mark.parametrize("contribution,expected_broken", [
        (0, True),
        (4, True),
        (5, False),
        (10, False),
        (25, False),
    ])
    def test_lenient_threshold_parametrized(self, contribution, expected_broken):
        """Lenient threshold: < 5 is broken."""
        assert is_promise_broken_lenient(contribution) == expected_broken
