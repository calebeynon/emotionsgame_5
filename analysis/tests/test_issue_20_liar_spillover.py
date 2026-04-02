"""
Tests for liar DiD variables and cross-segment spillover in issue_20_build_did_panel.py.

Verifies per-round lying detection, liar event-study variable computation,
and cross-segment suckering spillover flags.

Author: Claude Code
Date: 2026-02-06
"""

import pytest
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add derived directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from issue_20_build_did_panel import (
    build_liar_flags,
    add_did_variables as compute_did_variables,
    add_cross_segment_spillover,
)

# DEFAULTS for mock data builders
_DEFAULTS = {
    'session_code': 'test_session',
    'treatment': 1,
    'segment': 'supergame_test',
}

_EVENT_DEFAULTS = {
    'suckered_this_round_20': False,
    'suckered_this_round_5': False,
    'lied_this_round_20': False,
    'lied_this_round_5': False,
}


# =====
# Mock data builders
# =====
def _behavior_df(rows):
    """Build behavior DataFrame from row dicts."""
    return pd.DataFrame([{**_DEFAULTS, **r} for r in rows])


def _promise_df(rows):
    """Build promise DataFrame from row dicts."""
    return pd.DataFrame([{**_DEFAULTS, **r} for r in rows])


def _liar_behavior_df(rows):
    """Build behavior DataFrame with suckered cols defaulting to False."""
    defaults = {'suckered_this_round_20': False, 'suckered_this_round_5': False}
    return pd.DataFrame([{**_DEFAULTS, **defaults, **r} for r in rows])


def _lied_df(label, lied_rounds, n_rounds=5):
    """Build rows with lied_this_round columns pre-set."""
    return [
        {
            'label': label, 'round': rnd,
            'contribution': 10 if rnd in lied_rounds else 25,
            'group': 1, **_EVENT_DEFAULTS,
            'lied_this_round_20': rnd in lied_rounds,
            'lied_this_round_5': rnd in lied_rounds,
        }
        for rnd in range(1, n_rounds + 1)
    ]


def _spillover_rows(suckered_segments):
    """Build multi-segment rows (supergame1-3 x 3 rounds)."""
    segments = ['supergame1', 'supergame2', 'supergame3']
    return [
        {
            'session_code': 'test_session', 'treatment': 1,
            'segment': seg, 'label': 'A', 'round': rnd,
            'contribution': 25, 'group': 1, **_EVENT_DEFAULTS,
            'suckered_this_round_20': seg in suckered_segments and rnd == 2,
            'suckered_this_round_5': seg in suckered_segments and rnd == 2,
        }
        for seg in segments for rnd in range(1, 4)
    ]


# =====
# Tests for build_liar_flags (per-round lying detection)
# =====
class TestLiarThisRound:
    """Tests for the build_liar_flags function."""

    def test_liar_flagged_when_promised_and_low_contrib(self):
        """Player who promised and contributed below threshold is flagged."""
        brows = [{'label': 'A', 'round': 2, 'contribution': 10, 'group': 1}]
        prows = [{'label': 'A', 'round': 2, 'promise_count': 1}]
        result = build_liar_flags(_liar_behavior_df(brows), _promise_df(prows))
        assert result['lied_this_round_20'].iloc[0] == True

    def test_not_liar_when_promised_and_high_contrib(self):
        """Player who promised but contributed above threshold is NOT flagged."""
        brows = [{'label': 'A', 'round': 2, 'contribution': 25, 'group': 1}]
        prows = [{'label': 'A', 'round': 2, 'promise_count': 1}]
        result = build_liar_flags(_liar_behavior_df(brows), _promise_df(prows))
        assert result['lied_this_round_20'].iloc[0] == False

    def test_not_liar_when_no_promise(self):
        """Player who didn't promise is NOT flagged even with low contribution."""
        brows = [{'label': 'A', 'round': 2, 'contribution': 5, 'group': 1}]
        prows = [{'label': 'A', 'round': 2, 'promise_count': 0}]
        result = build_liar_flags(_liar_behavior_df(brows), _promise_df(prows))
        assert result['lied_this_round_20'].iloc[0] == False

    def test_round_1_never_lied(self):
        """Round 1 never has lies (no chat/promise data exists)."""
        brows = [{'label': 'A', 'round': 1, 'contribution': 5, 'group': 1}]
        prows = [{'label': 'A', 'round': 1, 'promise_count': 2}]
        result = build_liar_flags(_liar_behavior_df(brows), _promise_df(prows))
        assert result['lied_this_round_20'].iloc[0] == False
        assert result['lied_this_round_5'].iloc[0] == False

    def test_threshold_distinction(self):
        """Contrib=10: lied_20=True (10<20), lied_5=False (10>=5)."""
        brows = [{'label': 'A', 'round': 2, 'contribution': 10, 'group': 1}]
        prows = [{'label': 'A', 'round': 2, 'promise_count': 1}]
        result = build_liar_flags(_liar_behavior_df(brows), _promise_df(prows))
        assert result['lied_this_round_20'].iloc[0] == True
        assert result['lied_this_round_5'].iloc[0] == False

    def test_threshold_5_flags_very_low(self):
        """Contrib=3: both lied_20=True (3<20) and lied_5=True (3<5)."""
        brows = [{'label': 'A', 'round': 2, 'contribution': 3, 'group': 1}]
        prows = [{'label': 'A', 'round': 2, 'promise_count': 1}]
        result = build_liar_flags(_liar_behavior_df(brows), _promise_df(prows))
        assert result['lied_this_round_20'].iloc[0] == True
        assert result['lied_this_round_5'].iloc[0] == True

    def test_output_preserves_all_rows(self):
        """build_liar_flags should not change row count."""
        brows = [
            {'label': 'A', 'round': r, 'contribution': 10, 'group': 1}
            for r in range(1, 4)
        ]
        prows = [
            {'label': 'A', 'round': r, 'promise_count': 1}
            for r in range(1, 4)
        ]
        df = _liar_behavior_df(brows)
        assert len(build_liar_flags(df, _promise_df(prows))) == len(df)


# =====
# Tests for liar DiD variables
# =====
class TestLiarDidVariables:
    """Tests for liar event-study variable computation."""

    def _base_df(self):
        """A lied round 3, B never lied, 5 rounds each."""
        rows = _lied_df('A', {3}) + _lied_df('B', set())
        for r in rows:
            if r['label'] == 'B':
                r['contribution'] = 25
        return _behavior_df(rows)

    def test_first_lied_round(self):
        """A lied round 3 -> first_lied_round_20 = 3 everywhere."""
        a = compute_did_variables(self._base_df()).query("label == 'A'")
        assert (a['first_lied_round_20'] == 3).all()

    def test_is_liar_did_time_invariant(self):
        """is_liar_did_20=True for ALL 5 rows of liar player A."""
        a = compute_did_variables(self._base_df()).query("label == 'A'")
        assert (a['is_liar_did_20'] == True).all()
        assert len(a) == 5

    def test_liar_tau_arithmetic(self):
        """Lied round 3: liar_tau = [-2, -1, 0, 1, 2]."""
        a = compute_did_variables(self._base_df()).query("label == 'A'").sort_values('round')
        assert a['liar_tau_20'].tolist() == [-2, -1, 0, 1, 2]

    def test_liar_post_indicator(self):
        """liar_post = 1 if liar_tau >= 0: [0, 0, 1, 1, 1]."""
        a = compute_did_variables(self._base_df()).query("label == 'A'").sort_values('round')
        assert a['liar_post_20'].tolist() == [0, 0, 1, 1, 1]

    def test_never_lied(self):
        """Never-lied B: is_liar_did=False, tau/post=NaN, did_sample=True."""
        b = compute_did_variables(self._base_df()).query("label == 'B'")
        assert (b['is_liar_did_20'] == False).all()
        assert b['liar_tau_20'].isna().all()
        assert b['liar_post_20'].isna().all()
        assert (b['liar_did_sample_20'] == True).all()

    def test_liar_did_sample_single_event(self):
        """A lied once -> liar_event_count=1, liar_did_sample=True."""
        a = compute_did_variables(self._base_df()).query("label == 'A'")
        assert (a['liar_event_count_20'] == 1).all()
        assert (a['liar_did_sample_20'] == True).all()

    def test_liar_did_sample_multi_event(self):
        """Lied rounds 2 AND 4 -> event_count=2, did_sample=False."""
        result = compute_did_variables(_behavior_df(_lied_df('A', {2, 4})))
        a = result[result['label'] == 'A']
        assert (a['liar_event_count_20'] == 2).all()
        assert (a['liar_did_sample_20'] == False).all()

    def test_segment_boundaries(self):
        """Lying in segment 1 should NOT carry to segment 2."""
        rows = []
        for seg, lie_rnd in [('sg1', {3}), ('sg2', set())]:
            for rnd in range(1, 4):
                rows.append({
                    'session_code': 'test_session', 'treatment': 1,
                    'segment': seg, 'label': 'A', 'round': rnd,
                    'contribution': 10 if rnd in lie_rnd else 25,
                    'group': 1, **_EVENT_DEFAULTS,
                    'lied_this_round_20': rnd in lie_rnd,
                    'lied_this_round_5': rnd in lie_rnd,
                })
        result = compute_did_variables(pd.DataFrame(rows))
        assert (result[result['segment'] == 'sg1']['is_liar_did_20'] == True).all()
        assert (result[result['segment'] == 'sg2']['is_liar_did_20'] == False).all()

    def test_output_preserves_all_rows(self):
        """compute_did_variables should not change row count."""
        df = self._base_df()
        assert len(compute_did_variables(df)) == len(df)


# =====
# Tests for cross-segment spillover
# =====
class TestCrossSegmentSpillover:
    """Tests for the add_cross_segment_spillover function."""

    def _spillover_df(self, suckered_segments):
        """Build multi-segment df with DiD vars, ready for spillover."""
        return compute_did_variables(pd.DataFrame(_spillover_rows(suckered_segments)))

    def test_suckered_prior_segment_true(self):
        """Suckered in supergame1 -> suckered_prior_segment True in sg2, sg3."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame1'}))
        assert (result[result['segment'] == 'supergame2']['suckered_prior_segment_20'] == True).all()
        assert (result[result['segment'] == 'supergame3']['suckered_prior_segment_20'] == True).all()

    def test_suckered_prior_segment_false_for_first(self):
        """suckered_prior_segment is always False for segment 1."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame1'}))
        assert (result[result['segment'] == 'supergame1']['suckered_prior_segment_20'] == False).all()

    def test_first_suckered_segment(self):
        """first_suckered_segment equals segment number of first suckering."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame2'}))
        assert (result['first_suckered_segment_20'] == 2).all()

    def test_segments_since_suckered_arithmetic(self):
        """segments_since = current_segment - first_suckered_segment."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame1'}))
        for seg, expected in [('supergame1', 0), ('supergame2', 1), ('supergame3', 2)]:
            vals = result[result['segment'] == seg]['segments_since_suckered_20']
            assert (vals == expected).all()

    def test_never_suckered_spillover_flags(self):
        """Player not suckered has NaN/False spillover flags."""
        result = add_cross_segment_spillover(self._spillover_df(set()))
        assert (result['suckered_prior_segment_20'] == False).all()
        assert result['first_suckered_segment_20'].isna().all()
        assert result['segments_since_suckered_20'].isna().all()

    def test_suckered_later_no_retroactive_spillover(self):
        """Suckered in sg2 -> sg1 has no spillover, sg3 does."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame2'}))
        assert (result[result['segment'] == 'supergame1']['suckered_prior_segment_20'] == False).all()
        assert (result[result['segment'] == 'supergame3']['suckered_prior_segment_20'] == True).all()

    def test_segments_since_nan_before_first(self):
        """segments_since_suckered is NaN for segments before first suckering."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame2'}))
        assert result[result['segment'] == 'supergame1']['segments_since_suckered_20'].isna().all()

    def test_output_preserves_all_rows(self):
        """add_cross_segment_spillover should not change row count."""
        df = self._spillover_df({'supergame1'})
        assert len(add_cross_segment_spillover(df)) == len(df)

    def test_segment_number_column_dropped(self):
        """Intermediate segment_number column should not remain in output."""
        result = add_cross_segment_spillover(self._spillover_df({'supergame1'}))
        assert 'segment_number' not in result.columns


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
