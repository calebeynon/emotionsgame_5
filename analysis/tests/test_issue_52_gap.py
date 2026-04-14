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

# Pinned coefficients from verified run (2026-04-13, simplified spec)
# Regression: Y ~ Lied + i(round) | segment + player_id, cluster by player
RESULTS_LIED_COEF = 5.4866
RESULTS_LIED_SE = 2.306
RESULTS_SUCKERED_COEF = 1.8846
RESULTS_SUCKERED_SE = 1.5202
PRE_CHAT_LIED_COEF = -2.1404
PRE_CHAT_LIED_SE = 0.8983
PRE_CHAT_SUCKERED_COEF = -1.2348
PRE_CHAT_SUCKERED_SE = 1.2136

RESULTS_N = 2696
PRE_CHAT_N = 2161

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
