"""
Min/med/max deviation variable tests for build_dynamic_regression_panel.py.

Verifies contmore/less{min,max,med} derivation and _L1 lag correctness,
using both property-based checks and hand-verified known rows.

Author: Claude Code
Date: 2026-04-19
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _helpers import get_row, validate_lag_column

# FILE PATHS
PANEL_CSV = (
    Path(__file__).parent.parent / "datastore" / "derived" / "dynamic_regression_panel.csv"
)

# CONSTANTS
NUM_SESSIONS = 10
NUM_SUBJECTS = NUM_SESSIONS * 16  # 160
DEVIATION_TAGS = ['min', 'max', 'med']


# =====
# Fixtures
# =====
@pytest.fixture(scope="module")
def panel_df():
    """Load panel CSV once per module."""
    import pandas as pd
    if not PANEL_CSV.exists():
        pytest.fail(
            f"dynamic_regression_panel.csv not found at {PANEL_CSV}. "
            f"Run: uv run python analysis/derived/build_dynamic_regression_panel.py"
        )
    return pd.read_csv(PANEL_CSV)


# =====
# Helper functions
# =====
def _assert_minmedmax(r, othermin, othermed, othermax, diffmin, diffmed, diffmax,
                      moremin, moremed, moremax, lessmin, lessmed, lessmax,
                      mtmin, mtmed, mtmax, ltmin, ltmed, ltmax):
    """Assert all min/med/max deviation fields against expected values."""
    assert r['othercontmin'] == pytest.approx(othermin)
    assert r['othercontmed'] == pytest.approx(othermed)
    assert r['othercontmax'] == pytest.approx(othermax)
    assert r['diffcontmin'] == pytest.approx(diffmin)
    assert r['diffcontmed'] == pytest.approx(diffmed)
    assert r['diffcontmax'] == pytest.approx(diffmax)
    assert r['contmoremin'] == pytest.approx(moremin)
    assert r['contmoremed'] == pytest.approx(moremed)
    assert r['contmoremax'] == pytest.approx(moremax)
    assert r['contlessmin'] == pytest.approx(lessmin)
    assert r['contlessmed'] == pytest.approx(lessmed)
    assert r['contlessmax'] == pytest.approx(lessmax)
    assert r['morethanmin'] == mtmin and r['lessthanmin'] == ltmin
    assert r['morethanmed'] == mtmed and r['lessthanmed'] == ltmed
    assert r['morethanmax'] == mtmax and r['lessthanmax'] == ltmax


# =====
# Property-based deviation variable checks
# =====
class TestMinMaxMedDeviationVariables:
    """Verify contmore/less{min,max,med} derivation and _L1 lag correctness."""

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_contmore_contless_non_negative(self, panel_df, tag):
        """contmore{tag} and contless{tag} must be >= 0."""
        more = panel_df[f'contmore{tag}']
        less = panel_df[f'contless{tag}']
        assert more.min() >= -1e-10, f"contmore{tag} has negative min: {more.min()}"
        assert less.min() >= -1e-10, f"contless{tag} has negative min: {less.min()}"

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_flags_binary_and_mutually_exclusive(self, panel_df, tag):
        """morethan{tag}/lessthan{tag} are 0/1 and never both 1 on same row."""
        more = panel_df[f'morethan{tag}']
        less = panel_df[f'lessthan{tag}']
        assert set(more.unique()) <= {0, 1}
        assert set(less.unique()) <= {0, 1}
        both = (more == 1) & (less == 1)
        assert not both.any(), f"{both.sum()} rows have both flags set for tag={tag}"

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_decomposition_matches_abs_diffcont(self, panel_df, tag):
        """contmore{tag} + contless{tag} equals |diffcont{tag}|."""
        lhs = panel_df[f'contmore{tag}'] + panel_df[f'contless{tag}']
        rhs = panel_df[f'diffcont{tag}'].abs()
        assert (lhs - rhs).abs().max() < 1e-10

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_lag_nan_at_period_1_only(self, panel_df, tag):
        """contmore{tag}_L1 and contless{tag}_L1 NaN iff period==1 (160 rows each)."""
        for kind in ['contmore', 'contless']:
            col = f'{kind}{tag}_L1'
            assert panel_df.loc[panel_df['period'] == 1, col].notna().sum() == 0
            assert panel_df.loc[panel_df['period'] > 1, col].isna().sum() == 0
            assert panel_df[col].isna().sum() == NUM_SUBJECTS

    @pytest.mark.parametrize("tag", DEVIATION_TAGS)
    def test_lag_values_match_within_subject(self, panel_df, tag):
        """contmore{tag}_L1[t] == contmore{tag}[t-1] within each subject."""
        for kind in ['contmore', 'contless']:
            errors = validate_lag_column(panel_df, f'{kind}{tag}', f'{kind}{tag}_L1')
            assert len(errors) == 0, (
                f"{kind}{tag}_L1 errors:\n" + "\n".join(errors)
            )


# =====
# Hand-verified known rows
# =====
class TestMinMaxMedKnownRows:
    """Verify derivation on specific rows whose values were manually checked."""

    def test_sa7mprty_sg1_r1_a_min_med_max(self, panel_df):
        """sa7mprty sg1 r1 A: contribution=15, others={6,5,10} -> min=5 med=6 max=10.

        All three references below contribution -> contmore*=diff, contless*=0.
        """
        r = get_row(panel_df, 'sa7mprty', 'supergame1', 1, 'A').iloc[0]
        assert r['others_contribution_1'] == 6.0
        assert r['others_contribution_2'] == 5.0
        assert r['others_contribution_3'] == 10.0
        _assert_minmedmax(r, 5, 6, 10, 10, 9, 5, 10, 9, 5, 0, 0, 0, 1, 1, 1, 0, 0, 0)

    def test_irrzlgk2_sg1_r1_a_mixed_min_med_max(self, panel_df):
        """irrzlgk2 sg1 r1 A: contribution=8, others={25,25,0} -> min=0 med=25 max=25.

        contribution above min (more), below med and max (less).
        """
        r = get_row(panel_df, 'irrzlgk2', 'supergame1', 1, 'A').iloc[0]
        _assert_minmedmax(r, 0, 25, 25, 8, -17, -17, 8, 0, 0, 0, 17, 17, 1, 0, 0, 0, 1, 1)

    def test_sa7mprty_sg3_r1_a_tied_min_med(self, panel_df):
        """sa7mprty sg3 r1 A: contribution=10, others={10,25,10} -> min=10 med=10 max=25."""
        r = get_row(panel_df, 'sa7mprty', 'supergame3', 1, 'A').iloc[0]
        assert r['othercontmin'] == pytest.approx(10.0)
        assert r['othercontmed'] == pytest.approx(10.0)
        assert r['othercontmax'] == pytest.approx(25.0)
        assert r['contmoremin'] == pytest.approx(0.0)
        assert r['contlessmin'] == pytest.approx(0.0)
        assert r['contlessmax'] == pytest.approx(15.0)

    def test_sa7mprty_sg1_r2_a_lags_carry_r1_values(self, panel_df):
        """sa7mprty sg1 r2 A: lag columns should equal r1 derivations (10, 9, 5, 0s)."""
        r = get_row(panel_df, 'sa7mprty', 'supergame1', 2, 'A').iloc[0]
        assert r['period'] == 2
        assert r['contmoremin_L1'] == pytest.approx(10.0)
        assert r['contmoremed_L1'] == pytest.approx(9.0)
        assert r['contmoremax_L1'] == pytest.approx(5.0)
        assert r['contlessmin_L1'] == pytest.approx(0.0)
        assert r['contlessmed_L1'] == pytest.approx(0.0)
        assert r['contlessmax_L1'] == pytest.approx(0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
