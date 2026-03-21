"""
Tests for sucker behavioral classification.

Tests the sucker flag logic: a player is a sucker when they contribute the
maximum in a round where a groupmate broke a promise.

Author: Claude Code
Date: 2026-01-17
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'derived'))

from behavior_helpers import compute_sucker_flags, classify_player_behavior


# =====
# Fixtures
# =====
def _make_round_pair(liar_contrib=10, victim_contrib=25, n_players=4):
    """Two-round group where one player breaks promise in round 1."""
    labels = ['A', 'B', 'C', 'D'][:n_players]
    pids = list(range(1, n_players + 1))
    contribs = [liar_contrib] + [victim_contrib] * (n_players - 1)
    rows = []
    for round_num in [1, 2]:
        for label, pid, contrib in zip(labels, pids, contribs):
            rows.append({
                'session_code': 'abc123', 'treatment': 1,
                'segment': 'supergame1', 'round': round_num,
                'group': 1, 'label': label, 'participant_id': pid,
                'contribution': contrib, 'made_promise': True,
            })
    return pd.DataFrame(rows)


def _make_two_segment_df():
    """Supergame1: A breaks promise, B suckered. Supergame2: everyone cooperates."""
    rows = []
    for segment in ['supergame1', 'supergame2']:
        for round_num in [1, 2]:
            a_contrib = 10 if (segment == 'supergame1' and round_num == 1) else 25
            for label, pid, contrib in [('A', 1, a_contrib), ('B', 2, 25)]:
                rows.append({
                    'session_code': 'abc123', 'treatment': 1,
                    'segment': segment, 'round': round_num,
                    'group': 1, 'label': label, 'participant_id': pid,
                    'contribution': contrib, 'made_promise': True,
                })
    return pd.DataFrame(rows)


# =====
# Basic sucker definition
# =====
class TestSuckerDefinition:
    """Tests for sucker classification definition."""

    def test_sucker_when_contrib_25_and_groupmate_broke_promise(self):
        """Player who contributes 25 when group member broke promise is sucker."""
        result = compute_sucker_flags(_make_round_pair(), threshold='20')
        b_r2 = result[(result['label'] == 'B') & (result['round'] == 2)]
        assert b_r2['is_sucker_20'].iloc[0] == True

    def test_sucker_requires_promise_not_just_low_contrib(self):
        """Low contribution without a promise does NOT create suckers."""
        df = _make_round_pair()
        df.loc[(df['label'] == 'A') & (df['round'] == 1), 'made_promise'] = False
        result = compute_sucker_flags(df, threshold='20')
        round_2 = result[result['round'] == 2]
        assert (round_2['is_sucker_20'] == False).all()

    def test_non_chatter_can_be_sucker(self):
        """Non-chatting player can be sucker if groupmate broke promise."""
        df = _make_round_pair()
        df.loc[df['label'] == 'D', 'made_promise'] = False
        result = compute_sucker_flags(df, threshold='20')
        d_r2 = result[(result['label'] == 'D') & (result['round'] == 2)]
        assert d_r2['is_sucker_20'].iloc[0] == True


# =====
# Round 1 always False
# =====
class TestSuckerRound1AlwaysFalse:
    """Sucker flags are always False in round 1."""

    def test_sucker_20_false_round_1(self):
        result = compute_sucker_flags(_make_round_pair(), threshold='20')
        assert (result[result['round'] == 1]['is_sucker_20'] == False).all()

    def test_sucker_5_false_round_1(self):
        result = compute_sucker_flags(_make_round_pair(), threshold='5')
        assert (result[result['round'] == 1]['is_sucker_5'] == False).all()


# =====
# Persistence and reset
# =====
class TestSuckerPersistence:
    """Sucker flag persists within segment, resets across segments."""

    def test_persists_across_rounds(self):
        """Once flagged, stays flagged until segment ends."""
        rows = []
        for r in [1, 2, 3]:
            for label, pid, contrib in [('A', 1, 10 if r == 1 else 25), ('B', 2, 25), ('C', 3, 25)]:
                rows.append({
                    'session_code': 'abc123', 'treatment': 1,
                    'segment': 'supergame1', 'round': r,
                    'group': 1, 'label': label, 'participant_id': pid,
                    'contribution': contrib, 'made_promise': True,
                })
        result = compute_sucker_flags(pd.DataFrame(rows), threshold='20')
        b_r2 = result[(result['label'] == 'B') & (result['round'] == 2)]
        b_r3 = result[(result['label'] == 'B') & (result['round'] == 3)]
        assert b_r2['is_sucker_20'].iloc[0] == True
        assert b_r3['is_sucker_20'].iloc[0] == True

    def test_resets_new_segment(self):
        """Sucker flag resets at start of new supergame."""
        result = compute_sucker_flags(_make_two_segment_df(), threshold='20')
        b_sg1_r2 = result[
            (result['label'] == 'B') & (result['segment'] == 'supergame1') & (result['round'] == 2)
        ]
        b_sg2_r1 = result[
            (result['label'] == 'B') & (result['segment'] == 'supergame2') & (result['round'] == 1)
        ]
        assert b_sg1_r2['is_sucker_20'].iloc[0] == True
        assert b_sg2_r1['is_sucker_20'].iloc[0] == False


# =====
# Same-round logic
# =====
class TestSuckerSameRound:
    """Sucker classification uses same-round promise-breaking, not cumulative."""

    def test_no_sucker_when_promise_broken_prior_round_only(self):
        """No sucker in round 2 if promise was only broken in round 1."""
        df = pd.DataFrame({
            'session_code': ['abc123'] * 4, 'treatment': [1] * 4,
            'segment': ['supergame1'] * 4,
            'round': [1, 1, 2, 2], 'group': [1] * 4,
            'label': ['A', 'B', 'A', 'B'], 'participant_id': [1, 2, 1, 2],
            # Round 1: A breaks promise; Round 2: A honors, B contributes 25
            'contribution': [10, 20, 25, 25],
            'made_promise': [True, True, True, True],
        })
        result = compute_sucker_flags(df, threshold='20')
        b_r2 = result[(result['label'] == 'B') & (result['round'] == 2)]
        b_r1 = result[(result['label'] == 'B') & (result['round'] == 1)]
        assert b_r2['is_sucker_20'].iloc[0] == False
        assert b_r1['is_sucker_20'].iloc[0] == False


# =====
# Combined classification
# =====
class TestCombinedClassifySuckers:
    """classify_player_behavior includes sucker flags."""

    def test_sucker_columns_present(self):
        result = classify_player_behavior(_make_round_pair())
        assert 'is_sucker_20' in result.columns
        assert 'is_sucker_5' in result.columns
