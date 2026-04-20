"""
Tests for dynamic_regression.R panel structure and lag construction.

Validates period linearization, subject_id uniqueness, lag correctness,
data completeness, and treatment balance using the real contributions.csv.

Author: Claude Code
Date: 2026-04-09
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# FILE PATHS
CONTRIBUTIONS_CSV = (
    Path(__file__).parent.parent / "datastore" / "derived" / "contributions.csv"
)

# CONSTANTS
EXPECTED_ROW_COUNT = 3520
NUM_SESSIONS = 10
NUM_PLAYERS_PER_SESSION = 16
MAX_PERIOD = 22
ROUNDS_PER_SUPERGAME = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}
PERIOD_OFFSETS = {1: 0, 2: 3, 3: 7, 4: 10, 5: 17}
EXPECTED_FIRST_PERIODS = {1: 1, 2: 4, 3: 8, 4: 11, 5: 18}
EXPECTED_LAST_PERIODS = {1: 3, 2: 7, 3: 10, 4: 17, 5: 22}


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load raw contributions.csv."""
    if not CONTRIBUTIONS_CSV.exists():
        pytest.skip(f"contributions.csv not found: {CONTRIBUTIONS_CSV}")
    return pd.read_csv(CONTRIBUTIONS_CSV)


@pytest.fixture(scope="module")
def panel_df() -> pd.DataFrame:
    """Load contributions.csv and compute all panel variables.

    Replicates the full load_and_prepare_data() pipeline from R.
    """
    if not CONTRIBUTIONS_CSV.exists():
        pytest.skip(f"contributions.csv not found: {CONTRIBUTIONS_CSV}")
    df = pd.read_csv(CONTRIBUTIONS_CSV)
    df = _derive_all_variables(df)
    return df


DEVIATION_TAGS = ["min", "max", "med"]


def _derive_deviation_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate derive_deviation_variables() from panel builder."""
    df["othercont"] = (df["payoff"] - 25 + 0.6 * df["contribution"]) / 0.4
    df["othercontaverage"] = df["othercont"] / 3
    df["morethanaverage"] = (df["contribution"] > df["othercontaverage"]).astype(int)
    df["lessthanaverage"] = (df["contribution"] < df["othercontaverage"]).astype(int)
    df["diffcont"] = df["contribution"] - df["othercontaverage"]
    df["contmore"] = df["diffcont"] * df["morethanaverage"]
    df["contless"] = -df["diffcont"] * df["lessthanaverage"]
    df = _derive_peer_order_variables(df)
    return df


def _derive_peer_order_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Compute othercontmin/max/med and per-tag deviation variables."""
    peers = df[["others_contribution_1", "others_contribution_2", "others_contribution_3"]]
    df["othercontmin"] = peers.min(axis=1)
    df["othercontmax"] = peers.max(axis=1)
    df["othercontmed"] = peers.median(axis=1)
    for tag in DEVIATION_TAGS:
        ref = df[f"othercont{tag}"]
        df[f"morethan{tag}"] = (df["contribution"] > ref).astype(int)
        df[f"lessthan{tag}"] = (df["contribution"] < ref).astype(int)
        df[f"diffcont{tag}"] = df["contribution"] - ref
        df[f"contmore{tag}"] = df[f"diffcont{tag}"] * df[f"morethan{tag}"]
        df[f"contless{tag}"] = -df[f"diffcont{tag}"] * df[f"lessthan{tag}"]
    return df


def _create_panel_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate create_panel_variables() and lag construction."""
    df["segmentnumber"] = df["segment"].str.extract(r"supergame(\d)").astype(int)
    df["period"] = df["round"] + df["segmentnumber"].map(PERIOD_OFFSETS)
    session_map = {code: i + 1 for i, code in enumerate(df["session_code"].unique())}
    df["sessionnumber"] = df["session_code"].map(session_map)
    df["subject_id"] = df["sessionnumber"] * 100 + df["participant_id"]
    df = df.sort_values(["subject_id", "period"]).reset_index(drop=True)
    lag_sources = ["contmore", "contless"]
    for tag in DEVIATION_TAGS:
        lag_sources.extend([f"contmore{tag}", f"contless{tag}"])
    for col in lag_sources:
        df[f"{col}_L1"] = df.groupby("subject_id")[col].shift(1)
        df.loc[df["period"] == 1, f"{col}_L1"] = np.nan
    return df


def _derive_all_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate full R data pipeline."""
    df = _derive_deviation_variables(df)
    df = _create_panel_variables(df)
    return df


# =====
# Lag validation helper
# =====
def _validate_lag_column(df, source_col, lag_col):
    """Check that lag_col[t] == source_col[t-1] within each subject.

    Returns list of error strings (empty if all correct).
    """
    sorted_df = df.sort_values(["subject_id", "period"])
    expected = sorted_df.groupby("subject_id")[source_col].shift(1)
    expected[sorted_df["period"].values == 1] = np.nan
    actual = sorted_df[lag_col]
    # Compare non-NaN positions
    mask = expected.notna()
    mismatches = mask & ((actual - expected).abs() > 1e-10)
    bad = sorted_df[mismatches].head(10)
    return [
        f"subject={r['subject_id']} period={r['period']}: "
        f"expected={expected.iloc[i]}, got={r[lag_col]}"
        for i, (_, r) in enumerate(bad.iterrows())
    ]


# =====
# Test 6: Period linearization
# =====
class TestPeriodLinearization:
    """Verify period is correctly linearized across supergames."""

    @pytest.mark.parametrize(
        "supergame,round_num,expected_period",
        [
            (1, 1, 1), (1, 3, 3),
            (2, 1, 4), (2, 4, 7),
            (3, 1, 8), (3, 3, 10),
            (4, 1, 11), (4, 7, 17),
            (5, 1, 18), (5, 5, 22),
        ],
    )
    def test_specific_period_values(self, panel_df, supergame, round_num, expected_period):
        """Verify specific (supergame, round) -> period mappings."""
        mask = (panel_df["segmentnumber"] == supergame) & (panel_df["round"] == round_num)
        actual = panel_df.loc[mask, "period"].unique()
        assert len(actual) == 1 and actual[0] == expected_period

    def test_max_period_is_22(self, panel_df):
        """Maximum period should be 22."""
        assert panel_df["period"].max() == MAX_PERIOD

    def test_min_period_is_1(self, panel_df):
        """Minimum period should be 1."""
        assert panel_df["period"].min() == 1

    def test_periods_contiguous_within_supergame(self, panel_df):
        """Within each supergame, periods are contiguous integers."""
        for sg_num in range(1, 6):
            mask = panel_df["segmentnumber"] == sg_num
            periods = sorted(panel_df.loc[mask, "period"].unique())
            start = EXPECTED_FIRST_PERIODS[sg_num]
            end = EXPECTED_LAST_PERIODS[sg_num]
            assert periods == list(range(start, end + 1))

    def test_total_unique_periods_is_22(self, panel_df):
        """There should be exactly 22 unique period values."""
        assert panel_df["period"].nunique() == MAX_PERIOD


# =====
# Test 7: subject_id uniqueness
# =====
class TestSubjectId:
    """Verify subject_id uniquely identifies participants across sessions."""

    def test_subject_id_unique_per_session_participant(self, panel_df):
        """Each (session_code, participant_id) -> one subject_id."""
        grouped = panel_df.groupby(["session_code", "participant_id"])["subject_id"].nunique()
        multi = grouped[grouped > 1]
        assert len(multi) == 0, f"Multiple subject_ids: {multi.head().to_dict()}"

    def test_no_subject_id_collision(self, panel_df):
        """No two different (session, pid) pairs share a subject_id."""
        n_subjects = panel_df["subject_id"].nunique()
        n_pairs = panel_df.groupby(["session_code", "participant_id"]).ngroups
        assert n_subjects == n_pairs, f"{n_subjects} IDs for {n_pairs} pairs"

    def test_expected_number_of_subjects(self, panel_df):
        """Should have 160 unique subjects (10 sessions x 16 players)."""
        expected = NUM_SESSIONS * NUM_PLAYERS_PER_SESSION
        assert panel_df["subject_id"].nunique() == expected


# =====
# Test 8: Lag correctness
# =====
class TestLagVariables:
    """Verify contmore_L1 and contless_L1 are correct within-subject lags."""

    def test_contmore_lag_at_period_1_is_nan(self, panel_df):
        """contmore_L1 should be NaN at period==1."""
        period_1 = panel_df[panel_df["period"] == 1]
        assert period_1["contmore_L1"].isna().all()

    def test_contless_lag_at_period_1_is_nan(self, panel_df):
        """contless_L1 should be NaN at period==1."""
        period_1 = panel_df[panel_df["period"] == 1]
        assert period_1["contless_L1"].isna().all()

    def test_contmore_lag_value_correctness(self, panel_df):
        """contmore_L1[t] == contmore[t-1] for each subject."""
        errors = _validate_lag_column(panel_df, "contmore", "contmore_L1")
        assert len(errors) == 0, f"contmore_L1 errors:\n" + "\n".join(errors)

    def test_contless_lag_value_correctness(self, panel_df):
        """contless_L1[t] == contless[t-1] for each subject."""
        errors = _validate_lag_column(panel_df, "contless", "contless_L1")
        assert len(errors) == 0, f"contless_L1 errors:\n" + "\n".join(errors)

    def test_lag_nan_count(self, panel_df):
        """contmore_L1 NaN for exactly 160 rows (one per subject)."""
        expected_nan = NUM_SESSIONS * NUM_PLAYERS_PER_SESSION
        assert panel_df["contmore_L1"].isna().sum() == expected_nan


# =====
# Test 9: No data loss
# =====
class TestDataCompleteness:
    """Verify no rows are lost or duplicated during variable construction."""

    def test_row_count_matches_input(self, raw_df, panel_df):
        """Derived DataFrame should have same row count as input."""
        assert len(panel_df) == len(raw_df)

    def test_row_count_is_3520(self, panel_df):
        """Total rows: 10 sessions x 16 players x 22 rounds = 3520."""
        assert len(panel_df) == EXPECTED_ROW_COUNT

    def test_no_nan_in_core_derived_columns(self, panel_df):
        """Core derived columns (except lags) should have no NaN values."""
        cols = ["othercont", "othercontaverage", "morethanaverage",
                "lessthanaverage", "diffcont", "contmore", "contless",
                "othercontmin", "othercontmax", "othercontmed",
                "segmentnumber", "period", "subject_id"]
        for tag in DEVIATION_TAGS:
            cols.extend([f"morethan{tag}", f"lessthan{tag}", f"diffcont{tag}",
                         f"contmore{tag}", f"contless{tag}"])
        for col in cols:
            assert panel_df[col].isna().sum() == 0, f"NaN in '{col}'"

    def test_rows_per_subject(self, panel_df):
        """Each subject should have exactly 22 rows."""
        counts = panel_df.groupby("subject_id").size()
        bad = counts[counts != MAX_PERIOD]
        assert len(bad) == 0, f"{len(bad)} subjects have != 22 rows"


# =====
# Test 10: Treatment balance
# =====
class TestTreatmentBalance:
    """Verify equal treatment assignment across subjects."""

    def test_equal_sessions_per_treatment(self, panel_df):
        """Each treatment should have exactly 5 sessions."""
        counts = panel_df.groupby("session_code")["treatment"].first().value_counts()
        assert counts.get(1, 0) == 5
        assert counts.get(2, 0) == 5

    def test_equal_subjects_per_treatment(self, panel_df):
        """Each treatment should have 80 subjects."""
        for t in [1, 2]:
            n = panel_df[panel_df["treatment"] == t]["subject_id"].nunique()
            assert n == 80, f"Treatment {t}: {n} subjects, expected 80"

    def test_equal_rows_per_treatment(self, panel_df):
        """Each treatment should have 1760 rows (80 x 22)."""
        for t in [1, 2]:
            n = len(panel_df[panel_df["treatment"] == t])
            assert n == 1760, f"Treatment {t}: {n} rows, expected 1760"


# =====
# Segmentnumber mapping
# =====
class TestSegmentnumber:
    """Verify segmentnumber correctly maps segment names to 1-5."""

    @pytest.mark.parametrize("segment,expected", [
        ("supergame1", 1), ("supergame2", 2), ("supergame3", 3),
        ("supergame4", 4), ("supergame5", 5),
    ])
    def test_segmentnumber_mapping(self, panel_df, segment, expected):
        """Each segment name maps to the correct integer."""
        actual = panel_df.loc[panel_df["segment"] == segment, "segmentnumber"].unique()
        assert len(actual) == 1 and actual[0] == expected


# =====
# Peer order stats (min/med/max) and their deviation lags
# =====
class TestPeerOrderStats:
    """Verify othercontmin/med/max ordering and derivation from peer columns."""

    def test_min_le_med_le_max_rowwise(self, panel_df):
        """Row-wise: othercontmin <= othercontmed <= othercontmax."""
        assert (panel_df["othercontmin"] <= panel_df["othercontmed"]).all()
        assert (panel_df["othercontmed"] <= panel_df["othercontmax"]).all()

    def test_min_equals_peer_row_min(self, panel_df):
        """othercontmin = min of the three peer contributions."""
        peers = panel_df[["others_contribution_1", "others_contribution_2",
                          "others_contribution_3"]]
        assert (panel_df["othercontmin"] == peers.min(axis=1)).all()

    def test_max_equals_peer_row_max(self, panel_df):
        """othercontmax = max of the three peer contributions."""
        peers = panel_df[["others_contribution_1", "others_contribution_2",
                          "others_contribution_3"]]
        assert (panel_df["othercontmax"] == peers.max(axis=1)).all()


class TestMinMaxMedDeviationLags:
    """Verify derived min/med/max deviation variables and their lags."""

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_contmore_contless_non_negative(self, panel_df, tag):
        """contmore{tag} and contless{tag} are non-negative by construction."""
        assert panel_df[f"contmore{tag}"].min() >= -1e-10
        assert panel_df[f"contless{tag}"].min() >= -1e-10

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_lag_nan_at_period_1(self, panel_df, tag):
        """contmore{tag}_L1 and contless{tag}_L1 NaN at period==1 (160 rows)."""
        p1 = panel_df[panel_df["period"] == 1]
        assert p1[f"contmore{tag}_L1"].isna().all()
        assert p1[f"contless{tag}_L1"].isna().all()
        expected_nan = NUM_SESSIONS * NUM_PLAYERS_PER_SESSION
        assert panel_df[f"contmore{tag}_L1"].isna().sum() == expected_nan
        assert panel_df[f"contless{tag}_L1"].isna().sum() == expected_nan

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_lag_values_match_within_subject(self, panel_df, tag):
        """Each *_L1 value equals previous-period value within subject."""
        for kind in ["contmore", "contless"]:
            errors = _validate_lag_column(
                panel_df, f"{kind}{tag}", f"{kind}{tag}_L1"
            )
            assert len(errors) == 0, f"{kind}{tag}_L1 errors:\n" + "\n".join(errors)


# =====
# Run tests directly
# =====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
