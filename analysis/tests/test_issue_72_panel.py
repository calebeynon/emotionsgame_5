"""
Tests for issue_72_panel.csv (lying-contagion panel).

Validates schema, row count, binary domains, monotonicity, and manually
traced participant rows. All expected values verified by inspecting
behavior_classifications.csv round-by-round for specific
(session_code, segment, round, group, label) tuples.

Author: Claude Code
Date: 2026-04-19
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
BEHAVIOR_CSV = DERIVED_DIR / "behavior_classifications.csv"
PANEL_CSV = DERIVED_DIR / "issue_72_panel.csv"

# MANUALLY VERIFIED CONSTANTS
EXPECTED_ROW_COUNT = 2720
REQUIRED_COLUMNS = [
    "session_code", "treatment", "segment", "round", "group", "label",
    "lied", "self_lied_lag", "group_lied_lag",
    "any_self_lied_prior", "any_group_lied_prior",
    "cluster_group", "label_session",
]
BINARY_COLUMNS = [
    "lied", "group_lied_lag", "self_lied_lag",
    "any_group_lied_prior", "any_self_lied_prior",
]

# =====
# Manually traced cases
# Each case was verified by reading lied_this_round_20 values in
# behavior_classifications.csv for the exact (session, segment, group, label).
# =====

# CASE A — sole liar in prior round, tests self-exclusion.
# 6sdkxl2q / supergame2 / round 2 / group 4: only D lied in round 2.
# In round 3, D's group_lied_lag should be 0 (self-excluded);
# G, K, N's group_lied_lag should be 1 (D, who is not them, lied).
SOLE_LIAR_CASE_A = {
    "session_code": "6sdkxl2q",
    "segment": "supergame2",
    "round": 3,
    "group": 4,
    "sole_liar_label": "D",
    "other_labels": ("G", "K", "N"),
}

# CASE B — second sole-liar case for robustness.
# iiu3xixz / supergame1 / round 2 / group 3: only L lied in round 2.
SOLE_LIAR_CASE_B = {
    "session_code": "iiu3xixz",
    "segment": "supergame1",
    "round": 3,
    "group": 3,
    "sole_liar_label": "L",
    "other_labels": ("C", "G", "Q"),
}

# CASE C — never-liar: B in 6sdkxl2q lied 0 times across all 22 rounds.
# Every panel row for B in 6sdkxl2q must have any_self_lied_prior == 0.
NEVER_LIAR_CASE = {
    "session_code": "6sdkxl2q",
    "label": "B",
}

# CASE D — bug-regression case for self-exclusion in any_group_lied_prior.
# sa7mprty / supergame4 / group 4: R lied in round 4; N lied in round 5.
# For R in round 6, any_group_lied_prior must be 1 (N lied prior and is NOT R).
# With the original max-minus-self bug this was 0 because R had also lied.
BUG_REGRESSION_CASE = {
    "session_code": "sa7mprty",
    "segment": "supergame4",
    "group": 4,
    "round": 6,
    "label": "R",
    "expected_any_self_lied_prior": 1,
    "expected_any_group_lied_prior": 1,
    "expected_group_lied_lag": 1,
    "expected_self_lied_lag": 0,
    "expected_lied": 0,
}

# CASE E — same-round group_lied_lag bug-regression.
# iiu3xixz / supergame2 / group 1: A AND L both lied in round 2.
# For L in round 3: self_lied_lag=1 AND group_lied_lag=1 (A also lied).
# With the original max-minus-self bug this collapsed to 0 (group_max=1, self_lag=1).
SAME_ROUND_BOTH_LIED_CASE = {
    "session_code": "iiu3xixz",
    "segment": "supergame2",
    "group": 1,
    "round": 3,
    "label": "L",
    "expected_lied": 1,
    "expected_self_lied_lag": 1,
    "expected_group_lied_lag": 1,
    "expected_any_self_lied_prior": 1,
    "expected_any_group_lied_prior": 1,
}

# CASE F — all four groupmates lied together.
# sa7mprty / supergame1 / group 3 round 2: C, G, L, Q all lied simultaneously.
# In round 3, each of the four must have:
#   self_lied_lag=1, group_lied_lag=1, any_self_lied_prior=1, any_group_lied_prior=1.
# Strongest test of sum-minus-self arithmetic: even when every group member lied
# (including self), each player must still see a groupmate's lie.
ALL_GROUP_LIED_CASE = {
    "session_code": "sa7mprty",
    "segment": "supergame1",
    "group": 3,
    "round": 3,
    "all_labels": ("C", "G", "L", "Q"),
}


# =====
# Fixtures
# =====
@pytest.fixture
def behavior_df():
    if not BEHAVIOR_CSV.exists():
        pytest.skip(f"Source data not found: {BEHAVIOR_CSV}")
    return pd.read_csv(BEHAVIOR_CSV)


@pytest.fixture
def panel_df():
    if not PANEL_CSV.exists():
        pytest.skip(
            f"Panel not found: {PANEL_CSV}. "
            "Run analysis/derived/build_issue_72_panel.py first."
        )
    return pd.read_csv(PANEL_CSV)


def _get_row(panel_df, *, session_code, segment, round_, group, label):
    """Return the single row matching the full composite key."""
    row = panel_df[
        (panel_df["session_code"] == session_code)
        & (panel_df["segment"] == segment)
        & (panel_df["round"] == round_)
        & (panel_df["group"] == group)
        & (panel_df["label"] == label)
    ]
    assert len(row) == 1, (
        f"Expected exactly 1 row for "
        f"{session_code}/{segment}/r{round_}/g{group}/{label}, got {len(row)}"
    )
    return row.iloc[0]


# =====
# Schema
# =====
class TestSchema:
    def test_required_columns_present(self, panel_df):
        for col in REQUIRED_COLUMNS:
            assert col in panel_df.columns, f"Missing column: {col}"

    def test_no_nan_in_constructed_cols(self, panel_df):
        for col in BINARY_COLUMNS:
            n_nan = panel_df[col].isna().sum()
            assert n_nan == 0, f"{col} has {n_nan} NaN values"

    def test_no_nan_in_identifier_cols(self, panel_df):
        for col in ["session_code", "treatment", "segment", "round",
                    "group", "label", "cluster_group", "label_session"]:
            n_nan = panel_df[col].isna().sum()
            assert n_nan == 0, f"{col} has {n_nan} NaN values"


# =====
# Row count
# =====
class TestRowCount:
    def test_exactly_2720_rows(self, panel_df):
        assert len(panel_df) == EXPECTED_ROW_COUNT

    def test_no_round_one_rows(self, panel_df):
        assert (panel_df["round"] >= 2).all(), (
            "Panel should drop round==1 rows (no self_lied_lag available)"
        )

    def test_min_round_is_two(self, panel_df):
        assert panel_df["round"].min() == 2


# =====
# Binary domain
# =====
class TestBinaryDomain:
    @pytest.mark.parametrize("col", BINARY_COLUMNS)
    def test_values_are_zero_or_one(self, panel_df, col):
        unique = set(panel_df[col].unique().tolist())
        assert unique <= {0, 1}, f"{col} has non-binary values: {unique}"


# =====
# Monotonicity
# Within (session, segment, label), cumulative "any prior" flags must be
# monotone non-decreasing across rounds (once flipped to 1, never returns to 0).
# =====
class TestMonotonicity:
    def test_any_self_lied_prior_monotone(self, panel_df):
        violations = []
        grouped = panel_df.groupby(["session_code", "segment", "label"])
        for key, group in grouped:
            vals = group.sort_values("round")["any_self_lied_prior"].values
            for i in range(1, len(vals)):
                if vals[i] < vals[i - 1]:
                    violations.append((key, vals.tolist()))
                    break
        assert not violations, (
            f"any_self_lied_prior non-monotone in {len(violations)} series: "
            f"{violations[:3]}"
        )

    def test_any_group_lied_prior_monotone(self, panel_df):
        violations = []
        grouped = panel_df.groupby(["session_code", "segment", "label"])
        for key, group in grouped:
            vals = group.sort_values("round")["any_group_lied_prior"].values
            for i in range(1, len(vals)):
                if vals[i] < vals[i - 1]:
                    violations.append((key, vals.tolist()))
                    break
        assert not violations, (
            f"any_group_lied_prior non-monotone in {len(violations)} series: "
            f"{violations[:3]}"
        )


# =====
# Manually traced rows
# =====
class TestManuallyTraced:
    def test_sole_liar_case_a_self_row(self, panel_df):
        """D (sole liar in r2) has self_lied_lag=1, group_lied_lag=0."""
        case = SOLE_LIAR_CASE_A
        row = _get_row(
            panel_df,
            session_code=case["session_code"],
            segment=case["segment"],
            round_=case["round"],
            group=case["group"],
            label=case["sole_liar_label"],
        )
        assert row["self_lied_lag"] == 1
        assert row["group_lied_lag"] == 0, (
            "Self-exclusion: D's own lie in r2 must not appear in D's group_lied_lag for r3"
        )
        assert row["any_self_lied_prior"] == 1
        assert row["any_group_lied_prior"] == 0

    def test_sole_liar_case_b_self_row(self, panel_df):
        """L (sole liar in r2) has self_lied_lag=1, group_lied_lag=0."""
        case = SOLE_LIAR_CASE_B
        row = _get_row(
            panel_df,
            session_code=case["session_code"],
            segment=case["segment"],
            round_=case["round"],
            group=case["group"],
            label=case["sole_liar_label"],
        )
        assert row["self_lied_lag"] == 1
        assert row["group_lied_lag"] == 0
        assert row["any_self_lied_prior"] == 1
        assert row["any_group_lied_prior"] == 0

    def test_never_liar_any_self_lied_prior_always_zero(self, panel_df):
        """B in 6sdkxl2q never lies; any_self_lied_prior must be 0 everywhere."""
        case = NEVER_LIAR_CASE
        sub = panel_df[
            (panel_df["session_code"] == case["session_code"])
            & (panel_df["label"] == case["label"])
        ]
        assert len(sub) > 0, "Never-liar case found no rows"
        assert (sub["any_self_lied_prior"] == 0).all(), (
            f"B in 6sdkxl2q has any_self_lied_prior>0 somewhere: "
            f"{sub[sub['any_self_lied_prior'] != 0][['segment','round']].values.tolist()}"
        )
        assert (sub["self_lied_lag"] == 0).all()
        assert (sub["lied"] == 0).all()

    def test_bug_regression_self_exclusion_with_own_prior_lie(self, panel_df):
        """Regression: R lied in r4, N lied in r5 → R's r6 any_group_lied_prior=1.

        The original max-minus-self arithmetic produced 0 here because
        self==1 AND group_max==1 subtracted to 0. Must be 1 after fix.
        """
        case = BUG_REGRESSION_CASE
        row = _get_row(
            panel_df,
            session_code=case["session_code"],
            segment=case["segment"],
            round_=case["round"],
            group=case["group"],
            label=case["label"],
        )
        assert row["lied"] == case["expected_lied"]
        assert row["self_lied_lag"] == case["expected_self_lied_lag"]
        assert row["group_lied_lag"] == case["expected_group_lied_lag"]
        assert row["any_self_lied_prior"] == case["expected_any_self_lied_prior"]
        assert row["any_group_lied_prior"] == case["expected_any_group_lied_prior"], (
            "Self-exclusion bug regression: R (who lied in r4) must still see "
            "N's r5 lie in any_group_lied_prior for r6."
        )

    def test_bug_regression_same_round_both_lied(self, panel_df):
        """Regression: A and L both lied in r2 → L's r3 group_lied_lag=1.

        The original max-minus-self arithmetic produced 0 here because both
        group_max_prev and self_lied_lag were 1. Must be 1 after fix since
        another player (A) lied in the prior round.
        """
        case = SAME_ROUND_BOTH_LIED_CASE
        row = _get_row(
            panel_df,
            session_code=case["session_code"],
            segment=case["segment"],
            round_=case["round"],
            group=case["group"],
            label=case["label"],
        )
        assert row["lied"] == case["expected_lied"]
        assert row["self_lied_lag"] == case["expected_self_lied_lag"]
        assert row["group_lied_lag"] == case["expected_group_lied_lag"], (
            "Same-round both-lied bug regression: L (self_lied_lag=1) must "
            "still see A's r2 lie in group_lied_lag for r3."
        )
        assert row["any_self_lied_prior"] == case["expected_any_self_lied_prior"]
        assert row["any_group_lied_prior"] == case["expected_any_group_lied_prior"]

    def test_all_group_lied_together(self, panel_df):
        """All 4 groupmates lied simultaneously in r2 → each has group_lied_lag=1 in r3.

        Strongest self-exclusion test: in the buggy max-minus-self formula this
        would fail identically for every player (group_max=1 − self=1 = 0 for
        every row). Sum-based formula must produce 1 for all four since each
        player still has 3 other groupmates who lied.
        """
        case = ALL_GROUP_LIED_CASE
        for label in case["all_labels"]:
            row = _get_row(
                panel_df,
                session_code=case["session_code"],
                segment=case["segment"],
                round_=case["round"],
                group=case["group"],
                label=label,
            )
            assert row["self_lied_lag"] == 1, f"{label} lied in r2 → self_lied_lag=1"
            assert row["group_lied_lag"] == 1, (
                f"{label}: all 3 other groupmates lied in r2 → group_lied_lag "
                "must be 1 (self-excluded sum-minus-self)"
            )
            assert row["any_self_lied_prior"] == 1
            assert row["any_group_lied_prior"] == 1


# =====
# Self-exclusion (sole liar → self=0, groupmates=1)
# =====
class TestSelfExclusion:
    @pytest.mark.parametrize(
        "case",
        [SOLE_LIAR_CASE_A, SOLE_LIAR_CASE_B],
        ids=["6sdkxl2q_sg2_g4_D", "iiu3xixz_sg1_g3_L"],
    )
    def test_sole_liar_self_row_has_zero_group_lied_lag(self, panel_df, case):
        row = _get_row(
            panel_df,
            session_code=case["session_code"],
            segment=case["segment"],
            round_=case["round"],
            group=case["group"],
            label=case["sole_liar_label"],
        )
        assert row["group_lied_lag"] == 0, (
            f"Sole liar {case['sole_liar_label']} must have group_lied_lag=0 "
            "(self-excluded)"
        )

    @pytest.mark.parametrize(
        "case",
        [SOLE_LIAR_CASE_A, SOLE_LIAR_CASE_B],
        ids=["6sdkxl2q_sg2_g4", "iiu3xixz_sg1_g3"],
    )
    def test_sole_liar_groupmates_have_group_lied_lag_one(self, panel_df, case):
        for other_label in case["other_labels"]:
            row = _get_row(
                panel_df,
                session_code=case["session_code"],
                segment=case["segment"],
                round_=case["round"],
                group=case["group"],
                label=other_label,
            )
            assert row["group_lied_lag"] == 1, (
                f"Groupmate {other_label} of sole liar "
                f"{case['sole_liar_label']} must have group_lied_lag=1"
            )
            assert row["self_lied_lag"] == 0, (
                f"Groupmate {other_label} did not lie in prior round"
            )
