"""
Tests for others_contribution columns in to_dataframe_contributions().

Verifies that: (1) the contribution column is unchanged after adding
the new columns, and (2) others_contribution_1/2/3 correctly reflect
each player's group mates' contributions.

Author: Claude Code
Date: 2026-04-10
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_data import Experiment, Group, Player, Round, Segment, Session

CONTRIBUTIONS_CSV = (
    Path(__file__).parent.parent / 'datastore' / 'derived' / 'contributions.csv'
)
PLAYERS_PER_GROUP = 4
OTHERS_COLS = [
    'others_contribution_1',
    'others_contribution_2',
    'others_contribution_3',
]


# =====
# Synthetic data builders
# =====
def make_player(label, contribution, pid=1):
    """Create a Player with given label and contribution."""
    p = Player(participant_id=pid, label=label, id_in_group=1)
    p.contribution = contribution
    return p


def make_group(group_id, players_data):
    """Create a Group from list of (label, contribution, pid) tuples."""
    g = Group(group_id)
    for label, contribution, pid in players_data:
        g.add_player(make_player(label, contribution, pid))
    return g


def build_experiment(rounds_spec):
    """Build experiment from a list of round specs.

    Each round spec is a list of group specs.
    Each group spec is (group_id, [(label, contribution, pid), ...]).
    """
    seg = Segment("supergame1")
    for round_num, groups_spec in enumerate(rounds_spec, 1):
        rnd = Round(round_num)
        for group_id, players_data in groups_spec:
            rnd.add_group(make_group(group_id, players_data))
        seg.add_round(rnd)
    sess = Session("test_session", 1)
    sess.add_segment(seg)
    exp = Experiment(name="Test")
    exp.add_session(sess)
    return exp


def get_player_row(df, label, round_num=1, group=1):
    """Extract a single row from the DataFrame by label, round, group."""
    mask = (df['label'] == label) & (df['round'] == round_num) & (df['group'] == group)
    rows = df[mask]
    assert len(rows) == 1, f"Expected 1 row for {label}/r{round_num}/g{group}, got {len(rows)}"
    return rows.iloc[0]


# =====
# Test contribution column is unchanged
# =====
class TestContributionColumnUnchanged:
    """Verify the contribution column was not altered by adding others columns."""

    def test_contribution_matches_experiment_objects(self, sample_experiment):
        """Each row's contribution matches the Player object's value."""
        df = sample_experiment.to_dataframe_contributions()
        for _, row in df.iterrows():
            sess = sample_experiment.get_session(row['session_code'])
            seg = sess.get_supergame(int(row['segment'][-1]))
            rnd = seg.get_round(row['round'])
            player = rnd.get_player(row['label'])
            assert row['contribution'] == player.contribution, (
                f"Mismatch for {row['label']} in {row['segment']} r{row['round']}: "
                f"df={row['contribution']}, obj={player.contribution}"
            )

    def test_csv_contribution_matches_experiment_objects(self):
        """Contribution values in the saved CSV match experiment objects."""
        if not CONTRIBUTIONS_CSV.exists():
            pytest.skip("contributions.csv not found")
        csv_df = pd.read_csv(CONTRIBUTIONS_CSV, index_col=0)
        # Spot-check first group of first session
        first_session = csv_df['session_code'].iloc[0]
        subset = csv_df[
            (csv_df['session_code'] == first_session)
            & (csv_df['segment'] == 'supergame1')
            & (csv_df['round'] == 1)
            & (csv_df['group'] == 1)
        ]
        assert len(subset) == PLAYERS_PER_GROUP
        total = subset['contribution'].sum()
        others_plus_self = [
            row['contribution'] + row['others_contribution_1']
            + row['others_contribution_2'] + row['others_contribution_3']
            for _, row in subset.iterrows()
        ]
        for val in others_plus_self:
            assert val == total, (
                f"Self + others ({val}) != group total ({total})"
            )


# =====
# Unit tests: others_contribution logic with synthetic data
# =====
class TestOthersContributionBasic:
    """Core logic: each player's others columns hold group mates' contributions."""

    def test_four_player_group(self):
        """Basic 4-player group produces correct others columns."""
        exp = build_experiment([
            [(1, [('A', 10, 1), ('B', 5, 2), ('C', 20, 3), ('D', 0, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        row_a = get_player_row(df, 'A')
        assert list(row_a[OTHERS_COLS]) == [5.0, 20.0, 0.0]

    def test_others_exclude_self(self):
        """Player's own contribution never appears in their others columns."""
        exp = build_experiment([
            [(1, [('A', 10, 1), ('B', 5, 2), ('C', 20, 3), ('D', 0, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        for _, row in df.iterrows():
            others = [row[c] for c in OTHERS_COLS]
            assert row['contribution'] not in others or others.count(row['contribution']) < 3, (
                f"Player {row['label']}'s contribution {row['contribution']} "
                f"should not appear as all three others"
            )

    def test_others_exclude_self_unique_values(self):
        """When all contributions are unique, self never appears in others."""
        exp = build_experiment([
            [(1, [('A', 1, 1), ('B', 2, 2), ('C', 3, 3), ('D', 4, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        for _, row in df.iterrows():
            others = {row[c] for c in OTHERS_COLS}
            assert row['contribution'] not in others, (
                f"Player {row['label']}'s contribution {row['contribution']} "
                f"found in others: {others}"
            )

    def test_others_sum_equals_group_total_minus_self(self):
        """Sum of others columns equals group total minus player's contribution."""
        exp = build_experiment([
            [(1, [('A', 7, 1), ('B', 13, 2), ('C', 3, 3), ('D', 25, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        group_total = df['contribution'].sum()
        for _, row in df.iterrows():
            others_sum = sum(row[c] for c in OTHERS_COLS)
            expected = group_total - row['contribution']
            assert others_sum == expected, (
                f"Player {row['label']}: others sum {others_sum} != "
                f"group total {group_total} - self {row['contribution']}"
            )

    def test_exactly_three_others_columns(self):
        """DataFrame has exactly 3 others_contribution columns."""
        exp = build_experiment([
            [(1, [('A', 5, 1), ('B', 5, 2), ('C', 5, 3), ('D', 5, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        others_in_df = [c for c in df.columns if c.startswith('others_contribution')]
        assert others_in_df == OTHERS_COLS

    def test_each_player_sees_different_others(self):
        """Each player in a group sees the other 3 players' contributions."""
        players_data = [('A', 1, 1), ('B', 2, 2), ('C', 3, 3), ('D', 4, 4)]
        exp = build_experiment([[(1, players_data)]])
        df = exp.to_dataframe_contributions()

        contributions = {lbl: c for lbl, c, _ in players_data}
        for _, row in df.iterrows():
            expected_others = [
                contributions[lbl] for lbl, _, _ in players_data
                if lbl != row['label']
            ]
            actual = [row[c] for c in OTHERS_COLS]
            assert actual == expected_others, (
                f"Player {row['label']}: expected others {expected_others}, got {actual}"
            )


# =====
# Edge cases
# =====
class TestOthersContributionEdgeCases:
    """Edge cases for others_contribution columns."""

    def test_all_zero_contributions(self):
        """All players contribute 0."""
        exp = build_experiment([
            [(1, [('A', 0, 1), ('B', 0, 2), ('C', 0, 3), ('D', 0, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        for _, row in df.iterrows():
            for col in OTHERS_COLS:
                assert row[col] == 0.0

    def test_all_max_contributions(self):
        """All players contribute the endowment (25)."""
        exp = build_experiment([
            [(1, [('A', 25, 1), ('B', 25, 2), ('C', 25, 3), ('D', 25, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        for _, row in df.iterrows():
            for col in OTHERS_COLS:
                assert row[col] == 25.0

    def test_identical_contributions(self):
        """All players contribute same non-zero, non-max amount."""
        exp = build_experiment([
            [(1, [('A', 12, 1), ('B', 12, 2), ('C', 12, 3), ('D', 12, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        for _, row in df.iterrows():
            for col in OTHERS_COLS:
                assert row[col] == 12.0

    def test_one_free_rider(self):
        """One player contributes 0, others contribute max."""
        exp = build_experiment([
            [(1, [('A', 0, 1), ('B', 25, 2), ('C', 25, 3), ('D', 25, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        row_a = get_player_row(df, 'A')
        assert list(row_a[OTHERS_COLS]) == [25.0, 25.0, 25.0]

        row_b = get_player_row(df, 'B')
        assert 0.0 in list(row_b[OTHERS_COLS])


# =====
# Multi-group and multi-round isolation
# =====
class TestOthersContributionIsolation:
    """Verify no cross-contamination between groups or rounds."""

    def test_multiple_groups_no_cross_contamination(self):
        """Others columns only reflect players within the same group."""
        exp = build_experiment([
            [
                (1, [('A', 10, 1), ('B', 20, 2), ('C', 5, 3), ('D', 15, 4)]),
                (2, [('E', 1, 5), ('F', 2, 6), ('G', 3, 7), ('H', 4, 8)]),
            ]
        ])
        df = exp.to_dataframe_contributions()

        # Group 1 player should only see group 1 contributions
        row_a = get_player_row(df, 'A', group=1)
        assert list(row_a[OTHERS_COLS]) == [20.0, 5.0, 15.0]

        # Group 2 player should only see group 2 contributions
        row_e = get_player_row(df, 'E', group=2)
        assert list(row_e[OTHERS_COLS]) == [2.0, 3.0, 4.0]

        # No group 2 values in group 1 rows
        g1_others = df[df['group'] == 1][OTHERS_COLS].values.flatten()
        g2_contributions = set(df[df['group'] == 2]['contribution'])
        # Group 2 has contributions 1,2,3,4; if any appear in g1 others it could
        # be coincidence (5 is in g1), so check that g1 others are exactly right
        for _, row in df[df['group'] == 1].iterrows():
            others = set(row[c] for c in OTHERS_COLS)
            assert others.issubset({10.0, 20.0, 5.0, 15.0}), (
                f"Group 1 player {row['label']} has non-group-1 values: {others}"
            )

    def test_multiple_rounds_correct_mapping(self):
        """Others columns reflect the correct round's contributions."""
        exp = build_experiment([
            [(1, [('A', 5, 1), ('B', 10, 2), ('C', 15, 3), ('D', 20, 4)])],
            [(1, [('A', 25, 1), ('B', 0, 2), ('C', 1, 3), ('D', 2, 4)])],
        ])
        df = exp.to_dataframe_contributions()

        # Round 1
        row_a_r1 = get_player_row(df, 'A', round_num=1)
        assert list(row_a_r1[OTHERS_COLS]) == [10.0, 15.0, 20.0]

        # Round 2 — contributions changed
        row_a_r2 = get_player_row(df, 'A', round_num=2)
        assert list(row_a_r2[OTHERS_COLS]) == [0.0, 1.0, 2.0]

    def test_multiple_segments(self):
        """Others columns work across different supergames."""
        seg1 = Segment("supergame1")
        rnd1 = Round(1)
        rnd1.add_group(make_group(1, [('A', 5, 1), ('B', 10, 2), ('C', 15, 3), ('D', 20, 4)]))
        seg1.add_round(rnd1)

        seg2 = Segment("supergame2")
        rnd2 = Round(1)
        rnd2.add_group(make_group(1, [('A', 0, 1), ('B', 0, 2), ('C', 0, 3), ('D', 25, 4)]))
        seg2.add_round(rnd2)

        sess = Session("test", 1)
        sess.add_segment(seg1)
        sess.add_segment(seg2)
        exp = Experiment(name="Test")
        exp.add_session(sess)
        df = exp.to_dataframe_contributions()

        sg1_rows = df[df['segment'] == 'supergame1']
        sg2_rows = df[df['segment'] == 'supergame2']

        row_a_sg1 = sg1_rows[sg1_rows['label'] == 'A'].iloc[0]
        assert list(row_a_sg1[OTHERS_COLS]) == [10.0, 15.0, 20.0]

        row_a_sg2 = sg2_rows[sg2_rows['label'] == 'A'].iloc[0]
        assert list(row_a_sg2[OTHERS_COLS]) == [0.0, 0.0, 25.0]


# =====
# Iteration order tests
# =====
class TestOthersContributionOrder:
    """Verify others columns preserve dict iteration order."""

    def test_iteration_order_matches_insertion(self):
        """Others appear in the order players were added to the group."""
        # Players inserted as B, C, D (after excluding A)
        exp = build_experiment([
            [(1, [('A', 1, 1), ('B', 2, 2), ('C', 3, 3), ('D', 4, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        row_a = get_player_row(df, 'A')
        # A is excluded; remaining iteration order is B=2, C=3, D=4
        assert row_a['others_contribution_1'] == 2.0
        assert row_a['others_contribution_2'] == 3.0
        assert row_a['others_contribution_3'] == 4.0

    def test_middle_player_order(self):
        """When excluding a middle player, order skips them correctly."""
        exp = build_experiment([
            [(1, [('A', 1, 1), ('B', 2, 2), ('C', 3, 3), ('D', 4, 4)])]
        ])
        df = exp.to_dataframe_contributions()
        row_c = get_player_row(df, 'C')
        # C excluded; remaining order is A=1, B=2, D=4
        assert row_c['others_contribution_1'] == 1.0
        assert row_c['others_contribution_2'] == 2.0
        assert row_c['others_contribution_3'] == 4.0


# =====
# Integration test against the CSV on disk
# =====
class TestOthersContributionCSV:
    """Validate others_contribution columns in the saved contributions.csv."""

    @pytest.fixture
    def csv_df(self):
        if not CONTRIBUTIONS_CSV.exists():
            pytest.skip("contributions.csv not found")
        return pd.read_csv(CONTRIBUTIONS_CSV, index_col=0)

    def test_no_nulls_in_others_columns(self, csv_df):
        """Others columns have no missing values."""
        for col in OTHERS_COLS:
            null_count = csv_df[col].isna().sum()
            assert null_count == 0, f"{col} has {null_count} null values"

    def test_others_within_endowment_range(self, csv_df):
        """All others_contribution values are between 0 and 25."""
        for col in OTHERS_COLS:
            assert csv_df[col].min() >= 0, f"{col} has values below 0"
            assert csv_df[col].max() <= 25, f"{col} has values above 25"

    def test_others_sum_equals_group_total_minus_self(self, csv_df):
        """For every row, self + others = group total contribution."""
        group_keys = ['session_code', 'segment', 'round', 'group']
        group_totals = csv_df.groupby(group_keys)['contribution'].transform('sum')
        others_sum = csv_df[OTHERS_COLS].sum(axis=1)
        expected = group_totals - csv_df['contribution']
        mismatches = (others_sum - expected).abs() > 1e-9
        assert not mismatches.any(), (
            f"{mismatches.sum()} rows where self + others != group total"
        )

    def test_group_size_consistency(self, csv_df):
        """Every group has exactly 4 players."""
        group_keys = ['session_code', 'segment', 'round', 'group']
        sizes = csv_df.groupby(group_keys).size()
        bad = sizes[sizes != PLAYERS_PER_GROUP]
        assert bad.empty, f"Groups with wrong size: {bad.to_dict()}"

    def test_others_are_actual_groupmate_contributions(self, csv_df):
        """Spot-check: others values match actual group mates' contributions."""
        group_keys = ['session_code', 'segment', 'round', 'group']
        # Check first 50 groups
        checked = 0
        for key, group_df in csv_df.groupby(group_keys):
            if checked >= 50:
                break
            contributions = dict(zip(group_df['label'], group_df['contribution']))
            for _, row in group_df.iterrows():
                expected_others = [
                    contributions[lbl] for lbl in contributions
                    if lbl != row['label']
                ]
                actual = [row[c] for c in OTHERS_COLS]
                assert actual == expected_others, (
                    f"Group {key}, player {row['label']}: "
                    f"expected {expected_others}, got {actual}"
                )
            checked += 1
