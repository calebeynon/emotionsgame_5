"""
Tests for issue_20_build_did_panel.py: suckering detection and DiD variables.

Verifies suckering detection and DiD variable computation for the
event-study analysis of being suckered in a public goods game.

Author: Claude Code
Date: 2026-02-05
"""

import pytest
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add derived directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "derived"))

from issue_20_build_did_panel import (
    build_suckered_flags as derive_per_round_suckering,
    add_did_variables as compute_did_variables,
    add_always_cooperator_flags,
)

# DEFAULTS for mock data builders
_DEFAULTS = {
    'session_code': 'test_session',
    'treatment': 1,
    'segment': 'supergame_test',
}


# =====
# Mock data builders
# =====
def _behavior_df(rows):
    """Build behavior DataFrame; rows need label, round, contribution, group."""
    return pd.DataFrame([{**_DEFAULTS, **r} for r in rows])


def _promise_df(rows):
    """Build promise DataFrame; rows need label, round, promise_count."""
    return pd.DataFrame([{**_DEFAULTS, **r} for r in rows])


def _suckered_df(label, suckered_rounds, n_rounds=5):
    """Build rows with suckered_this_round columns pre-set."""
    rows = []
    for rnd in range(1, n_rounds + 1):
        s = rnd in suckered_rounds
        rows.append({
            'label': label, 'round': rnd,
            'contribution': 25, 'group': 1,
            'suckered_this_round_20': s,
            'suckered_this_round_5': s,
        })
    return rows


# =====
# Fixtures
# =====
@pytest.fixture
def base_behavior_df():
    """2 groups x 4 players x 5 rounds. B breaks promise in round 3."""
    g1 = {'A': 25, 'B': 15, 'C': 20, 'D': 15}
    rows = []
    for rnd in range(1, 6):
        for lbl, base in g1.items():
            c = 10 if (lbl == 'B' and rnd == 3) else base
            rows.append({'label': lbl, 'round': rnd,
                         'contribution': c, 'group': 1})
        for lbl in ['E', 'F', 'G', 'H']:
            rows.append({'label': lbl, 'round': rnd,
                         'contribution': 20, 'group': 2})
    return _behavior_df(rows)


@pytest.fixture
def base_promise_df():
    """Promise data: B promises in round 3 only. No round 1 data."""
    rows = []
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for rnd in range(2, 6):
        for lbl in labels:
            pc = 1 if (lbl == 'B' and rnd == 3) else 0
            rows.append({'label': lbl, 'round': rnd,
                         'promise_count': pc})
    return _promise_df(rows)


# =====
# Tests for derive_per_round_suckering
# =====
class TestSuckeredThisRound:
    """Tests for the derive_per_round_suckering function."""

    def test_suckered_this_round_basic(self, base_behavior_df, base_promise_df):
        """A (contrib=25) suckered in round 3 only when B broke promise."""
        result = derive_per_round_suckering(base_behavior_df, base_promise_df)
        a = result[result['label'] == 'A']
        assert a[a['round'] == 3]['suckered_this_round_20'].iloc[0] == True
        other = a[a['round'] != 3]
        assert (other['suckered_this_round_20'] == False).all()

    def test_not_suckered_low_contribution(self, base_behavior_df, base_promise_df):
        """C (contrib=20, not 25) should NOT be flagged as suckered."""
        result = derive_per_round_suckering(base_behavior_df, base_promise_df)
        c = result[result['label'] == 'C']
        assert (c['suckered_this_round_20'] == False).all()

    def test_promise_breaker_not_suckered(self, base_behavior_df, base_promise_df):
        """B (promise breaker, contrib=10) should NOT be suckered."""
        result = derive_per_round_suckering(base_behavior_df, base_promise_df)
        b = result[result['label'] == 'B']
        assert (b['suckered_this_round_20'] == False).all()

    def test_round_1_never_suckered(self, base_behavior_df, base_promise_df):
        """No suckering in round 1 (no promise data exists)."""
        result = derive_per_round_suckering(base_behavior_df, base_promise_df)
        r1 = result[result['round'] == 1]
        assert (r1['suckered_this_round_20'] == False).all()
        assert (r1['suckered_this_round_5'] == False).all()

    def test_threshold_distinction(self):
        """Groupmate contributed 15: strict=True (15<20), lenient=False (15>=5)."""
        brows = []
        for rnd in [1, 2]:
            for lbl, c in [('A', 25), ('B', 15), ('C', 20), ('D', 20)]:
                brows.append({'label': lbl, 'round': rnd,
                              'contribution': c, 'group': 1})
        prows = [{'label': l, 'round': 2,
                  'promise_count': 1 if l == 'B' else 0}
                 for l in ['A', 'B', 'C', 'D']]
        result = derive_per_round_suckering(_behavior_df(brows), _promise_df(prows))
        a_r2 = result[(result['label'] == 'A') & (result['round'] == 2)]
        assert a_r2['suckered_this_round_20'].iloc[0] == True
        assert a_r2['suckered_this_round_5'].iloc[0] == False

    def test_output_preserves_all_rows(self, base_behavior_df, base_promise_df):
        """Output should have same number of rows as input behavior_df."""
        result = derive_per_round_suckering(base_behavior_df, base_promise_df)
        assert len(result) == len(base_behavior_df)


# =====
# Tests for compute_did_variables
# =====
class TestComputeDidVariables:
    """Tests for the compute_did_variables function."""

    def _base_df(self):
        """A suckered round 3, B never suckered, 5 rounds each."""
        rows = _suckered_df('A', {3}) + _suckered_df('B', set())
        for r in rows:
            if r['label'] == 'B':
                r['contribution'] = 15
                r['suckered_this_round_20'] = False
                r['suckered_this_round_5'] = False
        return _behavior_df(rows)

    def test_first_suckered_round(self):
        """A suckered round 3 -> first_suckered_round_20 = 3 everywhere."""
        result = compute_did_variables(self._base_df())
        a = result[result['label'] == 'A']
        assert (a['first_suckered_round_20'] == 3).all()

    def test_got_suckered_time_invariant(self):
        """got_suckered_20=True for ALL 5 rows of suckered player A."""
        result = compute_did_variables(self._base_df())
        a = result[result['label'] == 'A']
        assert (a['got_suckered_20'] == True).all()
        assert len(a) == 5

    def test_tau_arithmetic(self):
        """Suckered round 3: tau = [-2, -1, 0, 1, 2]."""
        result = compute_did_variables(self._base_df())
        a = result[result['label'] == 'A'].sort_values('round')
        assert a['tau_20'].tolist() == [-2, -1, 0, 1, 2]

    def test_post_indicator(self):
        """post = 1 if tau >= 0: [0, 0, 1, 1, 1]."""
        result = compute_did_variables(self._base_df())
        a = result[result['label'] == 'A'].sort_values('round')
        assert a['post_20'].tolist() == [0, 0, 1, 1, 1]

    def test_never_suckered(self):
        """Never-suckered B: got_suckered=False, tau/post=NaN, did_sample=True."""
        result = compute_did_variables(self._base_df())
        b = result[result['label'] == 'B']
        assert (b['got_suckered_20'] == False).all()
        assert b['tau_20'].isna().all()
        assert b['post_20'].isna().all()
        assert (b['did_sample_20'] == True).all()

    def test_did_sample_single_event(self):
        """A suckered once -> event_count=1, did_sample=True."""
        result = compute_did_variables(self._base_df())
        a = result[result['label'] == 'A']
        assert (a['suckered_event_count_20'] == 1).all()
        assert (a['did_sample_20'] == True).all()

    def test_did_sample_multi_event(self):
        """Suckered rounds 2 AND 4 -> event_count=2, did_sample=False."""
        rows = _suckered_df('A', {2, 4})
        result = compute_did_variables(_behavior_df(rows))
        a = result[result['label'] == 'A']
        assert (a['suckered_event_count_20'] == 2).all()
        assert (a['did_sample_20'] == False).all()

    def test_segment_boundaries(self):
        """Suckering in segment 1 should NOT carry to segment 2."""
        rows = []
        for seg, suck_rnd in [('sg1', {3}), ('sg2', set())]:
            for rnd in range(1, 4):
                rows.append({
                    'session_code': 'test_session', 'treatment': 1,
                    'segment': seg, 'label': 'A', 'round': rnd,
                    'contribution': 25, 'group': 1,
                    'suckered_this_round_20': rnd in suck_rnd,
                    'suckered_this_round_5': rnd in suck_rnd,
                })
        result = compute_did_variables(pd.DataFrame(rows))
        assert (result[result['segment'] == 'sg1']['got_suckered_20'] == True).all()
        assert (result[result['segment'] == 'sg2']['got_suckered_20'] == False).all()

    def test_output_preserves_all_rows(self):
        """compute_did_variables should not change row count."""
        df = self._base_df()
        assert len(compute_did_variables(df)) == len(df)


# =====
# Helpers for always-cooperator tests
# =====
PLAYER_SEGMENT_KEYS = ['session_code', 'segment', 'label']

_AC_DEFAULTS = {
    'session_code': 'test_session',
    'segment': 'supergame_test',
}


def _ac_player_rows(label, contributions, got_suckered_20=False,
                    got_suckered_5=False, event_count_20=0,
                    event_count_5=0, segment=None):
    """Build rows for one player across rounds for always-coop tests.

    contributions is a dict {round: contribution_amount}.
    """
    seg = segment or _AC_DEFAULTS['segment']
    rows = []
    for rnd, contrib in sorted(contributions.items()):
        rows.append({
            'session_code': _AC_DEFAULTS['session_code'],
            'segment': seg,
            'label': label,
            'round': rnd,
            'contribution': contrib,
            'got_suckered_20': got_suckered_20,
            'got_suckered_5': got_suckered_5,
            'suckered_event_count_20': event_count_20,
            'suckered_event_count_5': event_count_5,
        })
    return rows


# =====
# Tests for add_always_cooperator_flags
# =====
class TestAlwaysCooperatorFlag:
    """Tests for the add_always_cooperator_flags function."""

    def test_always_25_from_round_2_is_flagged(self):
        """Player contributing 25 in rounds 2-5, never suckered -> always_coop_20."""
        contribs = {1: 25, 2: 25, 3: 25, 4: 25, 5: 25}
        rows = _ac_player_rows('A', contribs)
        df = pd.DataFrame(rows)
        result = add_always_cooperator_flags(df)
        assert (result['always_coop_20'] == True).all()

    def test_round_1_excluded(self):
        """Round 1 contribution is irrelevant; 0 in round 1, 25 in 2-5 -> flagged."""
        contribs = {1: 0, 2: 25, 3: 25, 4: 25, 5: 25}
        rows = _ac_player_rows('A', contribs)
        df = pd.DataFrame(rows)
        result = add_always_cooperator_flags(df)
        assert (result['always_coop_20'] == True).all()

    def test_one_defection_disqualifies(self):
        """One round with contribution < 25 (round 3 = 20) -> not always_coop."""
        contribs = {1: 25, 2: 25, 3: 20, 4: 25, 5: 25}
        rows = _ac_player_rows('A', contribs)
        df = pd.DataFrame(rows)
        result = add_always_cooperator_flags(df)
        assert (result['always_coop_20'] == False).all()

    def test_suckered_player_not_always_coop(self):
        """Player contributes 25 every round but was suckered -> not always_coop."""
        contribs = {1: 25, 2: 25, 3: 25, 4: 25, 5: 25}
        rows = _ac_player_rows('A', contribs, got_suckered_20=True,
                               event_count_20=1)
        df = pd.DataFrame(rows)
        result = add_always_cooperator_flags(df)
        assert (result['always_coop_20'] == False).all()

    def test_segment_boundary(self):
        """Always-coop in segment 1 but not in segment 2 -> flag differs."""
        # Segment 1: all 25 from round 2, never suckered
        rows_s1 = _ac_player_rows('A', {1: 10, 2: 25, 3: 25},
                                  segment='sg1')
        # Segment 2: defects in round 2
        rows_s2 = _ac_player_rows('A', {1: 25, 2: 15, 3: 25},
                                  segment='sg2')
        df = pd.DataFrame(rows_s1 + rows_s2)
        result = add_always_cooperator_flags(df)
        sg1 = result[result['segment'] == 'sg1']
        sg2 = result[result['segment'] == 'sg2']
        assert (sg1['always_coop_20'] == True).all()
        assert (sg2['always_coop_20'] == False).all()

    def test_did_sample_robust_includes_treated(self):
        """Suckered-exactly-once player -> did_sample_robust_20 is True."""
        contribs = {1: 25, 2: 25, 3: 25, 4: 25, 5: 25}
        # Suckered once (event_count=1), so treated; not always_coop
        rows = _ac_player_rows('A', contribs, got_suckered_20=True,
                               event_count_20=1)
        df = pd.DataFrame(rows)
        result = add_always_cooperator_flags(df)
        assert (result['did_sample_robust_20'] == True).all()

    def test_did_sample_robust_excludes_non_coop_controls(self):
        """Never-suckered player who defected in round > 1 -> robust=False."""
        contribs = {1: 25, 2: 25, 3: 15, 4: 25, 5: 25}
        rows = _ac_player_rows('A', contribs)
        df = pd.DataFrame(rows)
        result = add_always_cooperator_flags(df)
        # Not always_coop (defected), not suckered (event_count=0)
        assert (result['always_coop_20'] == False).all()
        assert (result['did_sample_robust_20'] == False).all()


# =====
# Integration: full pipeline
# =====
class TestPipeline:
    """Tests running derive then compute in sequence."""

    def test_full_pipeline_preserves_rows(self, base_behavior_df, base_promise_df):
        """Full pipeline preserves original row count."""
        mid = derive_per_round_suckering(base_behavior_df, base_promise_df)
        assert len(compute_did_variables(mid)) == len(base_behavior_df)

    def test_pipeline_suckered_player(self, base_behavior_df, base_promise_df):
        """A (suckered round 3) has correct DiD variables end-to-end."""
        mid = derive_per_round_suckering(base_behavior_df, base_promise_df)
        result = compute_did_variables(mid)
        a = result[result['label'] == 'A'].sort_values('round')
        assert (a['got_suckered_20'] == True).all()
        assert (a['first_suckered_round_20'] == 3).all()
        assert a['tau_20'].tolist() == [-2, -1, 0, 1, 2]
        assert a['post_20'].tolist() == [0, 0, 1, 1, 1]
        assert (a['did_sample_20'] == True).all()

    def test_pipeline_control_player(self, base_behavior_df, base_promise_df):
        """Unsuckered player E serves as control in DiD sample."""
        mid = derive_per_round_suckering(base_behavior_df, base_promise_df)
        result = compute_did_variables(mid)
        e = result[result['label'] == 'E']
        assert (e['got_suckered_20'] == False).all()
        assert e['tau_20'].isna().all()
        assert (e['did_sample_20'] == True).all()


# %%
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
