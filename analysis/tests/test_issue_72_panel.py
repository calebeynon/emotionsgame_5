"""
Tests for issue_72_panel.csv (lying-contagion panel).

Validates schema, row count, binary domains, monotonicity, and manually
traced participant rows. Case constants live in
tests/fixtures/issue_72_cases.py; expected values were verified by
inspecting behavior_classifications.csv round-by-round for specific
(session_code, segment, round, group, label) tuples.

Author: Claude Code
Date: 2026-04-19
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.issue_72_cases import (
    SOLE_LIAR_CASE_A, SOLE_LIAR_CASE_B, NEVER_LIAR_CASE,
    BUG_REGRESSION_CASE, SAME_ROUND_BOTH_LIED_CASE, ALL_GROUP_LIED_CASE,
    REQUIRED_COLUMNS, BINARY_COLUMNS,
)

# FILE PATHS
DERIVED_DIR = Path(__file__).parent.parent / "datastore" / "derived"
BEHAVIOR_CSV = DERIVED_DIR / "behavior_classifications.csv"
PANEL_CSV = DERIVED_DIR / "issue_72_panel.csv"

# Tightly coupled to TestRowCount below; kept in-file.
EXPECTED_ROW_COUNT = 2720


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


def _find_segment_reset_case(panel_df):
    """Return (session, label, s1, s2) where s1 ended with flag=1, s2 starts at 0."""
    for (session, label), sub in panel_df.groupby(["session_code", "label"]):
        segs = sorted(sub["segment"].unique())
        for i in range(len(segs) - 1):
            s1, s2 = segs[i], segs[i + 1]
            s1_max = sub[sub["segment"] == s1]["any_group_lied_prior"].max()
            s2_first = sub[sub["segment"] == s2].sort_values("round").iloc[0]
            if s1_max == 1 and s2_first["any_group_lied_prior"] == 0:
                return [(session, label, s1, s2)]
    return []


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


def _row_from_case(panel_df, case, label_key="label"):
    """Look up the row for a case dict using case[label_key] as the label."""
    return _get_row(
        panel_df,
        session_code=case["session_code"],
        segment=case["segment"],
        round_=case["round"],
        group=case["group"],
        label=case[label_key],
    )


def _assert_row_matches(panel_df, case, fields):
    """Assert row[field] == case[f'expected_{field}'] for each field in fields."""
    row = _row_from_case(panel_df, case)
    for field in fields:
        expected = case[f"expected_{field}"]
        assert row[field] == expected, (
            f"{field}: expected {expected}, got {row[field]}"
        )
    return row


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

    def test_binary_columns_are_integer_dtype(self, panel_df):
        for col in BINARY_COLUMNS:
            assert pd.api.types.is_integer_dtype(panel_df[col]), (
                f"{col} has dtype {panel_df[col].dtype}; NaN contamination "
                f"would silently coerce to float."
            )

    def test_cluster_group_stable_within_session_segment_group(self, panel_df):
        nunique = (
            panel_df.groupby(["session_code", "segment", "group"])["cluster_group"]
            .nunique()
        )
        assert (nunique == 1).all(), (
            f"cluster_group must be stable across rounds within "
            f"(session, segment, group). Violating groups: "
            f"{nunique[nunique != 1].head().to_dict()}"
        )


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
    @pytest.mark.parametrize(
        "col", ["any_self_lied_prior", "any_group_lied_prior"],
    )
    def test_cumulative_flag_monotone(self, panel_df, col):
        violations = []
        grouped = panel_df.groupby(["session_code", "segment", "label"])
        for key, group in grouped:
            vals = group.sort_values("round")[col].values
            for i in range(1, len(vals)):
                if vals[i] < vals[i - 1]:
                    violations.append((key, vals.tolist()))
                    break
        assert not violations, (
            f"{col} non-monotone in {len(violations)} series: {violations[:3]}"
        )


# =====
# Manually traced rows
# =====
_BUG_REGRESSION_FIELDS = ["lied", "self_lied_lag", "group_lied_lag",
                          "any_self_lied_prior", "any_group_lied_prior"]


class TestManuallyTraced:
    @pytest.mark.parametrize(
        "case", [SOLE_LIAR_CASE_A, SOLE_LIAR_CASE_B],
        ids=["6sdkxl2q_sg2_g4_D", "iiu3xixz_sg1_g3_L"],
    )
    def test_sole_liar_self_row_flags(self, panel_df, case):
        """Sole liar has self_lied_lag=1, group_lied_lag=0 (self-excluded)."""
        row = _row_from_case(panel_df, case, label_key="sole_liar_label")
        assert row["self_lied_lag"] == 1
        assert row["group_lied_lag"] == 0, (
            f"Self-exclusion: {case['sole_liar_label']}'s own prior lie must "
            "not appear in their own group_lied_lag"
        )
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

    @pytest.mark.parametrize(
        "case", [BUG_REGRESSION_CASE, SAME_ROUND_BOTH_LIED_CASE],
        ids=["self_exclusion_own_prior_lie", "same_round_both_lied"],
    )
    def test_bug_regression_cases(self, panel_df, case):
        """Regressions for original max-minus-self self-exclusion bug."""
        _assert_row_matches(panel_df, case, _BUG_REGRESSION_FIELDS)

    def test_any_group_lied_prior_resets_across_segments(self, panel_df):
        """Segment boundary resets any_group_lied_prior to 0 for next segment's r2."""
        candidates = _find_segment_reset_case(panel_df)
        assert candidates, (
            "No test case found — expected at least one player with prior-segment "
            "any_group_lied_prior=1 and next-segment first-row=0 (segment reset)."
        )

    def test_all_group_lied_together(self, panel_df):
        """All 4 groupmates lied in r2 → each has all four flags=1 in r3."""
        c = ALL_GROUP_LIED_CASE
        for label in c["all_labels"]:
            row = _get_row(
                panel_df, session_code=c["session_code"], segment=c["segment"],
                round_=c["round"], group=c["group"], label=label,
            )
            assert row["self_lied_lag"] == 1, f"{label}: self_lied_lag=1"
            assert row["group_lied_lag"] == 1, (
                f"{label}: 3 others lied → group_lied_lag=1"
            )
            assert row["any_self_lied_prior"] == 1
            assert row["any_group_lied_prior"] == 1


# =====
# Self-exclusion (sole liar → self=0, groupmates=1)
# =====
class TestSelfExclusion:
    @pytest.mark.parametrize(
        "case", [SOLE_LIAR_CASE_A, SOLE_LIAR_CASE_B],
        ids=["6sdkxl2q_sg2_g4_D", "iiu3xixz_sg1_g3_L"],
    )
    def test_sole_liar_self_row_has_zero_group_lied_lag(self, panel_df, case):
        row = _row_from_case(panel_df, case, label_key="sole_liar_label")
        assert row["group_lied_lag"] == 0, (
            f"Sole liar {case['sole_liar_label']} must have group_lied_lag=0 "
            "(self-excluded)"
        )

    @pytest.mark.parametrize(
        "case", [SOLE_LIAR_CASE_A, SOLE_LIAR_CASE_B],
        ids=["6sdkxl2q_sg2_g4", "iiu3xixz_sg1_g3"],
    )
    def test_sole_liar_groupmates_have_group_lied_lag_one(self, panel_df, case):
        for other_label in case["other_labels"]:
            row = _get_row(
                panel_df, session_code=case["session_code"],
                segment=case["segment"], round_=case["round"],
                group=case["group"], label=other_label,
            )
            assert row["group_lied_lag"] == 1, (
                f"Groupmate {other_label} of sole liar "
                f"{case['sole_liar_label']} must have group_lied_lag=1"
            )
            assert row["self_lied_lag"] == 0, (
                f"Groupmate {other_label} did not lie in prior round"
            )
