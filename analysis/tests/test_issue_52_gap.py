"""
Purpose: Tests for Issue #52 valence-sentiment gap regression script.
         Validates stacked data construction, model estimation, and table output.
Author: Claude Code
Date: 2026-04-10
"""

import re
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

# EXPECTED COUNTS (from real data run 2026-04-10)
EXPECTED_COMPLETE_CASES = 1791
EXPECTED_STACKED_ROWS = 3582


def run_r_code(code, timeout=120):
    """Run inline R code and return the completed process."""
    return subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def run_r_script(script_path, timeout=120):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


# =====
# Precondition checks
# =====
REQUIRED_FILES = [COMMON_SCRIPT, GAP_SCRIPT]


class TestPreconditions:
    """Verify required scripts and data files exist."""

    @pytest.mark.parametrize("path", REQUIRED_FILES, ids=lambda p: p.name)
    def test_required_file_exists(self, path):
        assert path.exists(), f"Required file missing: {path}"


# =====
# Script sourcing
# =====
class TestGapScriptSourcing:
    """Test that the gap regression script sources without error."""

    def test_sources_without_error(self):
        result = run_r_code(
            'TESTING <- TRUE; source("analysis/issue_52_gap_regressions.R")'
        )
        assert result.returncode == 0, f"Source failed: {result.stderr}"


# =====
# Stacked data construction tests
# =====
class TestStackedDataConstruction:
    """Test prepare_stacked_data() creates correct stacked dataset."""

    def test_stacked_has_exactly_2x_complete_case_rows(self):
        """Stacked data should have exactly 2x the complete-case input."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
cat("STACKED_ROWS:", nrow(stacked), "\\n")
cat("HALF_ROWS:", nrow(stacked) / 2, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"STACKED_ROWS: {EXPECTED_STACKED_ROWS}" in result.stdout
        assert f"HALF_ROWS: {EXPECTED_COMPLETE_CASES}" in result.stdout

    def test_channel_has_exactly_two_levels(self):
        """Channel factor should have exactly 'chat' and 'face' levels."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
cat("LEVELS:", paste(levels(stacked$channel), collapse="|"), "\\n")
cat("N_LEVELS:", nlevels(stacked$channel), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LEVELS: chat|face" in result.stdout
        assert "N_LEVELS: 2" in result.stdout

    def test_equal_rows_per_channel(self):
        """Each channel should have exactly half the rows."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
cat("FACE_N:", nrow(stacked[channel == "face"]), "\\n")
cat("CHAT_N:", nrow(stacked[channel == "chat"]), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"FACE_N: {EXPECTED_COMPLETE_CASES}" in result.stdout
        assert f"CHAT_N: {EXPECTED_COMPLETE_CASES}" in result.stdout

    def test_no_na_in_y_column(self):
        """Y column should have zero NAs (complete-case filter applied)."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
cat("Y_NA:", sum(is.na(stacked$Y)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "Y_NA: 0" in result.stdout

    def test_each_player_round_has_exactly_2_rows(self):
        """Each unique player-round should appear exactly twice."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
counts <- stacked[, .N, by = .(session_code, segment, round, label)]
cat("MIN_PER:", min(counts$N), "\\n")
cat("MAX_PER:", max(counts$N), "\\n")
cat("UNIQUE_PLAYER_ROUNDS:", nrow(counts), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MIN_PER: 2" in result.stdout
        assert "MAX_PER: 2" in result.stdout
        assert f"UNIQUE_PLAYER_ROUNDS: {EXPECTED_COMPLETE_CASES}" in result.stdout

    def test_required_columns_present(self):
        """Stacked data should contain all expected columns."""
        expected_cols = [
            "Y", "channel", "session_code", "segment", "round",
            "group", "label", "player_id", "contribution",
            "lied_this_round_20", "suckered_this_round",
        ]
        checks = "; ".join(
            f'cat("{col}:", "{col}" %in% names(stacked), "\\n")'
            for col in expected_cols
        )
        result = run_r_code(f"""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
{checks}
""")
        assert result.returncode == 0, result.stderr
        for col in expected_cols:
            assert f"{col}: TRUE" in result.stdout, (
                f"Column {col} missing from stacked data"
            )


# =====
# Y-value correctness: face rows = valence, chat rows = sentiment
# =====
# Verification (2026-04-10):
# Player D (6sdkxl2q) supergame2 round 3:
#   Raw emotion_valence = -8.5728
#   Raw sentiment_compound_mean = -0.0515
# Traced through merged_panel.csv -> keep_complete_cases -> stack_channels


class TestYValueCorrectness:
    """Verify face rows hold valence and chat rows hold sentiment."""

    def test_face_y_matches_emotion_valence(self):
        """Face channel Y must equal emotion_valence for Player D s2r3."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
d <- stacked[channel == "face" & session_code == "6sdkxl2q" &
             label == "D" & segment == "supergame2" & round == 3]
cat("FACE_Y:", round(d$Y, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "FACE_Y: -8.5728" in result.stdout

    def test_chat_y_matches_sentiment_compound(self):
        """Chat channel Y must equal sentiment_compound_mean for Player D s2r3."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
d <- stacked[channel == "chat" & session_code == "6sdkxl2q" &
             label == "D" & segment == "supergame2" & round == 3]
cat("CHAT_Y:", round(d$Y, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "CHAT_Y: -0.0515" in result.stdout

    def test_face_and_chat_y_differ(self):
        """Face and chat Y for same player-round should differ."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
face_y <- stacked[channel == "face" & session_code == "6sdkxl2q" &
                   label == "D" & segment == "supergame2" & round == 3]$Y
chat_y <- stacked[channel == "chat" & session_code == "6sdkxl2q" &
                   label == "D" & segment == "supergame2" & round == 3]$Y
cat("DIFFER:", face_y != chat_y, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "DIFFER: TRUE" in result.stdout


# =====
# Complete-case filtering tests
# =====
class TestCompleteCaseFiltering:
    """Test keep_complete_cases() drops rows missing valence or sentiment."""

    def test_complete_case_count(self):
        """Should retain exactly 1791 complete cases."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_results_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- add_suckered_this_round(dt)
dt <- keep_complete_cases(dt)
cat("N:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"N: {EXPECTED_COMPLETE_CASES}" in result.stdout

    def test_no_na_valence_after_filter(self):
        """No NA emotion_valence after complete-case filter."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_results_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- add_suckered_this_round(dt)
dt <- keep_complete_cases(dt)
cat("VALENCE_NA:", sum(is.na(dt$emotion_valence)), "\\n")
cat("SENTIMENT_NA:", sum(is.na(dt$sentiment_compound_mean)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VALENCE_NA: 0" in result.stdout
        assert "SENTIMENT_NA: 0" in result.stdout


# =====
# Suckered-this-round derivation tests
# =====
class TestSuckeredThisRound:
    """Test add_suckered_this_round() logic."""

    def test_suckered_column_exists(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
cat("HAS_COL:", "suckered_this_round" %in% names(stacked), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_COL: TRUE" in result.stdout

    def test_suckered_requires_full_contribution(self):
        """Suckered players must have contributed 25 (full endowment)."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
suckered <- stacked[suckered_this_round == TRUE]
cat("MIN_CONTRIB:", min(suckered$contribution), "\\n")
cat("MAX_CONTRIB:", max(suckered$contribution), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MIN_CONTRIB: 25" in result.stdout
        assert "MAX_CONTRIB: 25" in result.stdout

    def test_suckered_not_liar(self):
        """Suckered players must NOT be liars themselves."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
bad <- stacked[suckered_this_round == TRUE & lied_this_round_20 == TRUE]
cat("VIOLATIONS:", nrow(bad), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VIOLATIONS: 0" in result.stdout


# =====
# Model estimation tests
# =====
# Verification (2026-04-10):
# Lied gap interaction: coef=7.5962, p=0.0106 (significant)
# Suckered gap interaction: coef=1.7648, p=0.2610 (not significant)
# Both models use n=3582 observations


class TestModelEstimation:
    """Test that both gap models estimate without error."""

    def test_lied_gap_model_estimates(self):
        """Lied gap model should estimate without error."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
cat("LIED_CLASS:", class(models$lied_gap)[1], "\\n")
cat("LIED_NOBS:", models$lied_gap$nobs, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIED_CLASS: fixest" in result.stdout
        assert f"LIED_NOBS: {EXPECTED_STACKED_ROWS}" in result.stdout

    def test_suckered_gap_model_estimates(self):
        """Suckered gap model should estimate without error."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
cat("SUCK_CLASS:", class(models$suckered_gap)[1], "\\n")
cat("SUCK_NOBS:", models$suckered_gap$nobs, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "SUCK_CLASS: fixest" in result.stdout
        assert f"SUCK_NOBS: {EXPECTED_STACKED_ROWS}" in result.stdout

    def test_lied_gap_interaction_term_present(self):
        """Lied model must include the channel x lied interaction term."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$lied_gap)
has_interaction <- any(grepl("channel.*lied", rownames(ct)))
cat("HAS_INTERACTION:", has_interaction, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_INTERACTION: TRUE" in result.stdout

    def test_suckered_gap_interaction_term_present(self):
        """Suckered model must include the channel x suckered interaction."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$suckered_gap)
has_interaction <- any(grepl("channel.*suckered", rownames(ct)))
cat("HAS_INTERACTION:", has_interaction, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_INTERACTION: TRUE" in result.stdout


# =====
# Regression coefficient values (from real data run 2026-04-10)
# =====
# Verification method:
# 1. Ran full pipeline: load data -> complete cases -> stack -> estimate
# 2. Lied interaction (face x lied): coef=7.5962, se=2.9250, p=0.0106
# 3. Suckered interaction (face x suckered): coef=1.7648, se=1.5628, p=0.2610
# 4. Lied main effect: coef=-1.1398, se=0.7803, p=0.1466
# 5. Suckered main effect: coef=-0.3268, se=0.5055, p=0.5191


class TestRegressionCoefficients:
    """Pin regression coefficients from verified run."""

    def test_lied_interaction_coefficient(self):
        """Face x Lied interaction: coef=7.5962."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$lied_gap)
idx <- grep("channel.*lied", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
cat("SE:", round(ct[idx, 2], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "COEF: 7.5962" in result.stdout
        assert "SE: 2.925" in result.stdout

    def test_lied_main_effect(self):
        """Lied main effect: coef=-1.1398."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$lied_gap)
idx <- grep("^lied_this_round_20TRUE$", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "COEF: -1.1398" in result.stdout

    def test_suckered_interaction_coefficient(self):
        """Face x Suckered interaction: coef=1.7648."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$suckered_gap)
idx <- grep("channel.*suckered", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
cat("SE:", round(ct[idx, 2], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "COEF: 1.7648" in result.stdout
        assert "SE: 1.5628" in result.stdout

    def test_suckered_main_effect(self):
        """Suckered main effect: coef=-0.3268."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$suckered_gap)
idx <- grep("^suckered_this_roundTRUE$", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "COEF: -0.3268" in result.stdout


# =====
# LaTeX table output tests
# =====
class TestTexOutput:
    """Test that the script produces the expected .tex file."""

    def test_full_script_runs_successfully(self):
        result = run_r_script(GAP_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_tex_file_created(self):
        """The .tex output file should exist after running the script."""
        assert GAP_TABLE_FILE.exists(), (
            f"Table file missing: {GAP_TABLE_FILE}"
        )

    def test_tex_file_nonempty(self):
        """The .tex file should have substantial content."""
        assert GAP_TABLE_FILE.stat().st_size > 500, (
            f"Table file suspiciously small: {GAP_TABLE_FILE.stat().st_size}"
        )

    def test_tex_file_contains_tabular(self):
        """The .tex file should contain a tabular environment."""
        content = GAP_TABLE_FILE.read_text()
        assert "\\begin{tabular}" in content or "tabular" in content, (
            "No tabular environment found in .tex file"
        )
