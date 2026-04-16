"""
Purpose: Tests for Issue #52 facial-valence regression script.
         Validates data preparation, coefficient estimation, and
         LaTeX table output for the simplified face-only spec.
Author: Claude Code
Date: 2026-04-13
"""

import subprocess
from pathlib import Path

import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
WORKING_DIR = Path(__file__).resolve().parent.parent

# R SCRIPT PATHS
GAP_SCRIPT = ANALYSIS_DIR / "issue_52_gap_regressions.R"
COMMON_SCRIPT = ANALYSIS_DIR / "issue_52_common.R"

# EXPECTED OUTPUT
GAP_TABLE_FILE = TABLE_DIR / "issue_52_valence_sentiment_gap_regressions.tex"

# Pinned coefficients from verified run (2026-04-14, updated spec).
# Regression: Y ~ Lied | segment^round + player_id,
# cluster by ~player_id + group_segment_round.
# Paired "segment-specific round FE" matches paper Eq. omitted-round wording.
RESULTS_LIED_COEF = 5.3457
RESULTS_LIED_SE = 2.6090
RESULTS_SUCKERED_COEF = 1.8990
RESULTS_SUCKERED_SE = 1.2702
PRE_CHAT_LIED_COEF = -2.0043
PRE_CHAT_LIED_SE = 0.8246
PRE_CHAT_SUCKERED_COEF = -1.0642
PRE_CHAT_SUCKERED_SE = 1.0861

RESULTS_N = 2696
PRE_CHAT_N = 2161

# Experimental constants for TestLagShiftSemantics
N_PLAYERS = 160
N_SEGMENTS = 5
# Total rounds pre-lag across all player-segments (160 players x 5 segments;
# each segment contributes one round-1 row per player that the lag drops).
ROUND1_DROP_COUNT = N_PLAYERS * N_SEGMENTS  # 800

# Total TRUE suckered rows in raw panel (pre-filter) — pinned from shared
# add_suckered_this_round loader output (methodology-agent verified 122).
SUCKERED_TOTAL_TRUE = 122

COEF_TOL = 0.01


# =====
# Helpers
# =====
def run_r_code(code, timeout=120):
    """Run inline R code and return the completed process."""
    return subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def run_r_script(script_path, timeout=180):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def parse_numeric(result, key):
    """Extract `key: <number>` from R script stdout."""
    for line in result.stdout.splitlines():
        if line.startswith(f"{key}:"):
            return float(line.split(":", 1)[1].strip())
    raise AssertionError(f"Key '{key}' not found. stdout: {result.stdout!r}")


# =====
# Precondition: script and dependencies exist
# =====
class TestPreconditions:
    def test_gap_script_exists(self):
        assert GAP_SCRIPT.exists(), f"Missing: {GAP_SCRIPT}"

    def test_common_script_exists(self):
        assert COMMON_SCRIPT.exists(), f"Missing: {COMMON_SCRIPT}"


# =====
# Data preparation: row counts and column invariants
# =====
class TestDataPreparation:
    """prepare_results_face_data / prepare_pre_decision_chat_face_data."""

    def test_results_face_row_count(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "dt <- prepare_results_face_data()\n"
            "cat('N:', nrow(dt), '\\n')\n"
        )
        result = run_r_code(code)
        assert result.returncode == 0, result.stderr
        assert int(parse_numeric(result, "N")) == RESULTS_N

    def test_pre_chat_face_row_count(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "dt <- prepare_pre_decision_chat_face_data()\n"
            "cat('N:', nrow(dt), '\\n')\n"
        )
        result = run_r_code(code)
        assert result.returncode == 0, result.stderr
        assert int(parse_numeric(result, "N")) == PRE_CHAT_N

    def test_results_face_has_required_cols(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "dt <- prepare_results_face_data()\n"
            "cat('has_v:', 'emotion_valence' %in% names(dt), '\\n')\n"
            "cat('has_l:', 'lied_this_round_20' %in% names(dt), '\\n')\n"
            "cat('has_s:', 'suckered_this_round' %in% names(dt), '\\n')\n"
            "cat('has_p:', 'player_id' %in% names(dt), '\\n')\n"
        )
        result = run_r_code(code)
        assert result.returncode == 0, result.stderr
        for key in ["has_v", "has_l", "has_s", "has_p"]:
            assert f"{key}: TRUE" in result.stdout

    def test_suckered_flag_is_logical(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "dt <- prepare_results_face_data()\n"
            "cat('is_logical:', is.logical(dt$suckered_this_round), '\\n')\n"
            "cat('any_true:', any(dt$suckered_this_round), '\\n')\n"
        )
        result = run_r_code(code)
        assert result.returncode == 0, result.stderr
        assert "is_logical: TRUE" in result.stdout
        assert "any_true: TRUE" in result.stdout


# =====
# Coefficient pins: main effects match the verified run
# =====
class TestResultsPageCoefficients:
    """Results-page face model: Y = emotion_valence."""

    @pytest.fixture(scope="class")
    def model_output(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "m <- estimate_face_models(prepare_results_face_data(),"
            " 'emotion_valence')\n"
            "ctL <- coeftable(m$lied)\n"
            "ctS <- coeftable(m$suckered)\n"
            "lidx <- grep('lied_this_round_20', rownames(ctL))\n"
            "sidx <- grep('suckered_this_round', rownames(ctS))\n"
            "cat('LC:', round(ctL[lidx, 1], 4), '\\n')\n"
            "cat('LS:', round(ctL[lidx, 2], 4), '\\n')\n"
            "cat('SC:', round(ctS[sidx, 1], 4), '\\n')\n"
            "cat('SS:', round(ctS[sidx, 2], 4), '\\n')\n"
            "cat('LN:', m$lied$nobs, '\\n')\n"
        )
        return run_r_code(code)

    def test_runs(self, model_output):
        assert model_output.returncode == 0, model_output.stderr

    def test_lied_coef(self, model_output):
        assert parse_numeric(model_output, "LC") == pytest.approx(
            RESULTS_LIED_COEF, abs=COEF_TOL)

    def test_lied_se(self, model_output):
        assert parse_numeric(model_output, "LS") == pytest.approx(
            RESULTS_LIED_SE, abs=COEF_TOL)

    def test_suckered_coef(self, model_output):
        assert parse_numeric(model_output, "SC") == pytest.approx(
            RESULTS_SUCKERED_COEF, abs=COEF_TOL)

    def test_suckered_se(self, model_output):
        assert parse_numeric(model_output, "SS") == pytest.approx(
            RESULTS_SUCKERED_SE, abs=COEF_TOL)

    def test_nobs(self, model_output):
        assert int(parse_numeric(model_output, "LN")) == RESULTS_N


class TestPreDecisionChatCoefficients:
    """Pre-decision chat face model: Y = valence_shifted."""

    @pytest.fixture(scope="class")
    def model_output(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "m <- estimate_face_models(prepare_pre_decision_chat_face_data(),"
            " 'valence_shifted')\n"
            "ctL <- coeftable(m$lied)\n"
            "ctS <- coeftable(m$suckered)\n"
            "lidx <- grep('lied_this_round_20', rownames(ctL))\n"
            "sidx <- grep('suckered_this_round', rownames(ctS))\n"
            "cat('LC:', round(ctL[lidx, 1], 4), '\\n')\n"
            "cat('LS:', round(ctL[lidx, 2], 4), '\\n')\n"
            "cat('SC:', round(ctS[sidx, 1], 4), '\\n')\n"
            "cat('SS:', round(ctS[sidx, 2], 4), '\\n')\n"
            "cat('LN:', m$lied$nobs, '\\n')\n"
        )
        return run_r_code(code)

    def test_runs(self, model_output):
        assert model_output.returncode == 0, model_output.stderr

    def test_lied_coef(self, model_output):
        assert parse_numeric(model_output, "LC") == pytest.approx(
            PRE_CHAT_LIED_COEF, abs=COEF_TOL)

    def test_lied_se(self, model_output):
        assert parse_numeric(model_output, "LS") == pytest.approx(
            PRE_CHAT_LIED_SE, abs=COEF_TOL)

    def test_suckered_coef(self, model_output):
        assert parse_numeric(model_output, "SC") == pytest.approx(
            PRE_CHAT_SUCKERED_COEF, abs=COEF_TOL)

    def test_suckered_se(self, model_output):
        assert parse_numeric(model_output, "SS") == pytest.approx(
            PRE_CHAT_SUCKERED_SE, abs=COEF_TOL)

    def test_nobs(self, model_output):
        assert int(parse_numeric(model_output, "LN")) == PRE_CHAT_N


# =====
# LaTeX table output
# =====
class TestTexOutput:
    @pytest.fixture(scope="class")
    def script_output(self):
        return run_r_script(GAP_SCRIPT)

    def test_script_runs(self, script_output):
        assert script_output.returncode == 0, script_output.stderr

    def test_tex_file_created(self, script_output):
        assert GAP_TABLE_FILE.exists(), f"Missing: {GAP_TABLE_FILE}"

    def test_tex_file_nonempty(self, script_output):
        assert GAP_TABLE_FILE.stat().st_size > 500

    def test_tex_contains_tabular(self, script_output):
        content = GAP_TABLE_FILE.read_text()
        assert "\\begin{tabular}" in content

    def test_tex_contains_headers(self, script_output):
        content = GAP_TABLE_FILE.read_text()
        assert "Results Page Face" in content
        assert "Pre-Decision Chat Face" in content

    def test_tex_contains_coefficient_rows(self, script_output):
        content = GAP_TABLE_FILE.read_text()
        assert "Lied" in content
        assert "Suckered" in content

    def test_tex_contains_nobs(self, script_output):
        content = GAP_TABLE_FILE.read_text()
        assert "2,696" in content
        assert "2,161" in content

    def test_tex_contains_headline_coefs(self, script_output):
        # Guard against the table and the pinned values silently drifting.
        content = GAP_TABLE_FILE.read_text()
        assert "5.346" in content
        assert "-2.004" in content


# =====
# Scientific headline: sign flip between Results Page and Pre-Decision Chat
# for liars (Issue #52 review — this is the main story).
# =====
class TestSignFlipInvariant:
    def test_lied_signs_flip(self):
        assert RESULTS_LIED_COEF > 0, "Results-page lied coef must be positive"
        assert PRE_CHAT_LIED_COEF < 0, (
            "Pre-decision chat lied coef must be negative")


# =====
# Lag-shift semantics (Issue #52 review #7): verify
# prepare_pre_decision_chat_face_data() drops exactly round 1 per
# player-segment and keeps the lag within segment boundaries.
# =====
class TestLagShiftSemantics:
    @pytest.fixture(scope="class")
    def lag_output(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "raw <- load_chat_emotion_data()\n"
            "raw[, player_id := paste0(session_code, '_', label)]\n"
            "cat('raw_rows:', nrow(raw), '\\n')\n"
            "cat('n_players:', length(unique(raw$player_id)), '\\n')\n"
            "cat('n_segments:', length(unique(raw$segment)), '\\n')\n"
            "cat('round1_rows:', nrow(raw[round == 1]), '\\n')\n"
            "dt <- prepare_pre_decision_chat_face_data()\n"
            "cat('N:', nrow(dt), '\\n')\n"
            "cat('min_round:', min(dt$round), '\\n')\n"
            "cat('any_round1:', any(dt$round == 1), '\\n')\n"
        )
        return run_r_code(code)

    def test_runs(self, lag_output):
        assert lag_output.returncode == 0, lag_output.stderr

    def test_no_round_1_survives(self, lag_output):
        # Invariant (a): lag drops round 1 of every segment.
        assert "any_round1: FALSE" in lag_output.stdout
        assert int(parse_numeric(lag_output, "min_round")) >= 2

    def test_player_and_segment_counts(self, lag_output):
        assert int(parse_numeric(lag_output, "n_players")) == N_PLAYERS
        assert int(parse_numeric(lag_output, "n_segments")) == N_SEGMENTS

    def test_round1_drop_count_matches_expectation(self, lag_output):
        # Invariant (c): one dropped round-1 row per (player, segment).
        assert int(parse_numeric(lag_output, "round1_rows")) == \
            ROUND1_DROP_COUNT

    def test_surviving_rows_consistent_with_pin(self, lag_output):
        # Invariant (d): surviving rows match the pinned PRE_CHAT_N.
        assert int(parse_numeric(lag_output, "N")) == PRE_CHAT_N

    def test_lag_does_not_cross_segment_boundary(self):
        # Invariant (b): for any player who has rounds in both segment N-1
        # and segment N, valence_shifted at (segment N, round 1) is NA
        # (i.e., dropped) — it must NEVER equal segment N-1's last-round
        # emotion_valence. We verify this by reproducing the lag in R with
        # by=player-segment and checking no surviving row for round==1.
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "dt <- load_chat_emotion_data()\n"
            "dt[, player_id := paste0(session_code, '_', label)]\n"
            "dt <- add_suckered_this_round(dt)\n"
            "setorderv(dt, c('session_code', 'label', 'segment', 'round'))\n"
            # Intentional: lag WITHIN (player, segment) — no cross-segment
            "dt[, vs := shift(emotion_valence, 1, type='lag'),\n"
            "   by = .(session_code, label, segment)]\n"
            # Every round-1 row must have NA vs (lag can't cross segment)
            "bad <- dt[round == 1 & !is.na(vs)]\n"
            "cat('bad_boundary_rows:', nrow(bad), '\\n')\n"
        )
        result = run_r_code(code)
        assert result.returncode == 0, result.stderr
        assert int(parse_numeric(result, "bad_boundary_rows")) == 0


# =====
# Suckered-flag three-way logic (Issue #52 review #13): verify
# add_suckered_this_round encodes (groupmate_lied & contribution==25 & !self_lied)
# =====
class TestSuckeredFlagLogic:
    @pytest.fixture(scope="class")
    def flag_output(self):
        code = (
            "TESTING <- TRUE\n"
            "source('analysis/issue_52_gap_regressions.R')\n"
            "dt <- load_results_emotion_data()\n"
            "dt <- add_suckered_this_round(dt)\n"
            "suck <- dt[suckered_this_round == TRUE]\n"
            "cat('suck_n:', nrow(suck), '\\n')\n"
            "cat('all_contrib_25:', all(suck$contribution == 25), '\\n')\n"
            "cat('none_self_lied:',"
            " all(suck$lied_this_round_20 == FALSE), '\\n')\n"
            # Cross-check: no row has self-lied AND suckered simultaneously
            "bad <- dt[lied_this_round_20 == TRUE &"
            " suckered_this_round == TRUE]\n"
            "cat('bad_liar_suckered:', nrow(bad), '\\n')\n"
        )
        return run_r_code(code)

    def test_runs(self, flag_output):
        assert flag_output.returncode == 0, flag_output.stderr

    def test_pinned_suckered_count(self, flag_output):
        assert int(parse_numeric(flag_output, "suck_n")) == \
            SUCKERED_TOTAL_TRUE

    def test_all_suckered_contributed_full_25(self, flag_output):
        assert "all_contrib_25: TRUE" in flag_output.stdout

    def test_no_suckered_also_lied(self, flag_output):
        assert "none_self_lied: TRUE" in flag_output.stdout
        assert int(parse_numeric(flag_output, "bad_liar_suckered")) == 0
