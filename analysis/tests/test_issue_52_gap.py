"""
Purpose: Tests for Issue #52 valence-sentiment gap regression script.
         Validates stacked data construction, model estimation, and table output.
Author: Claude Code
Date: 2026-04-10
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

# EXPECTED COUNTS (from real data run 2026-04-10, with valence shift)
EXPECTED_COMPLETE_CASES = 1845
EXPECTED_STACKED_ROWS = 3690


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
    """Test prepare_chat_stacked_data() creates correct stacked dataset."""

    def test_stacked_has_exactly_2x_complete_case_rows(self):
        """Stacked data should have exactly 2x the complete-case input."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
cat("Y_NA:", sum(is.na(stacked$Y)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "Y_NA: 0" in result.stdout

    def test_each_player_round_has_exactly_2_rows(self):
        """Each unique player-round should appear exactly twice."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
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
            "Y", "channel", "session_code", "segment", "round", "group",
            "label", "player_id", "contribution", "lied_this_round_20",
            "suckered_this_round"]
        checks = "; ".join(
            f'cat("{c}:", "{c}" %in% names(s), "\\n")' for c in expected_cols
        )
        result = run_r_code(
            f'TESTING <- TRUE; source("analysis/issue_52_gap_regressions.R")'
            f"\ns <- prepare_chat_stacked_data()\n{checks}"
        )
        assert result.returncode == 0, result.stderr
        for col in expected_cols:
            assert f"{col}: TRUE" in result.stdout, f"{col} missing"


# =====
# Y-value correctness: face rows = shifted valence (lagged), chat rows = sentiment
# =====
# After the valence shift, face Y uses round N-1's emotion_valence (not round N's).
# Chat Y still uses the current round's sentiment_compound_mean.


class TestYValueCorrectness:
    """Verify face rows hold shifted valence and chat rows hold sentiment."""

    def test_face_y_matches_shifted_valence(self):
        """Face channel Y must equal valence_shifted (lagged from prior round)."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- add_suckered_this_round(dt)
dt <- shift_valence_to_influenced_round(dt)
shifted_val <- dt[session_code == "6sdkxl2q" & label == "D" &
                  segment == "supergame2" & round == 3]$valence_shifted
stacked <- prepare_chat_stacked_data()
face_y <- stacked[channel == "face" & session_code == "6sdkxl2q" &
                   label == "D" & segment == "supergame2" & round == 3]$Y
cat("FACE_Y:", round(face_y, 4), "\\n")
cat("SHIFTED:", round(shifted_val, 4), "\\n")
cat("MATCH:", all.equal(face_y, shifted_val), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MATCH: TRUE" in result.stdout

    def test_face_y_is_not_current_round_valence(self):
        """Face Y should NOT equal the current round's own emotion_valence."""
        result = run_r_code("""
TESTING <- TRUE; source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
raw <- dt[session_code == "6sdkxl2q" & label == "D" &
          segment == "supergame2" & round == 3]$emotion_valence
stacked <- prepare_chat_stacked_data()
face_y <- stacked[channel == "face" & session_code == "6sdkxl2q" &
                   label == "D" & segment == "supergame2" & round == 3]$Y
cat("DIFFERENT:", raw != face_y, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "DIFFERENT: TRUE" in result.stdout

    def test_chat_y_matches_sentiment_compound(self):
        """Chat channel Y must equal sentiment_compound_mean for Player D s2r3."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
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
        """Complete cases count should match expected (filters NA shifted valence)."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- add_suckered_this_round(dt)
dt <- shift_valence_to_influenced_round(dt)
dt <- dt[!is.na(valence_shifted) & !is.na(sentiment_compound_mean)]
cat("N:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"N: {EXPECTED_COMPLETE_CASES}" in result.stdout

    def test_no_na_shifted_valence_after_filter(self):
        """No NA valence_shifted after complete-case filter."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- add_suckered_this_round(dt)
dt <- shift_valence_to_influenced_round(dt)
dt <- dt[!is.na(valence_shifted) & !is.na(sentiment_compound_mean)]
cat("SHIFTED_NA:", sum(is.na(dt$valence_shifted)), "\\n")
cat("SENTIMENT_NA:", sum(is.na(dt$sentiment_compound_mean)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "SHIFTED_NA: 0" in result.stdout
        assert "SENTIMENT_NA: 0" in result.stdout


# =====
# Valence shift (lag) tests
# =====
class TestValenceShift:
    """Test that valence_shifted correctly lags emotion_valence by one round."""

    def test_round_1_has_no_shifted_valence(self):
        """All round-1 rows should have valence_shifted = NA."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- shift_valence_to_influenced_round(dt)
r1 <- dt[round == 1]
cat("R1_ALL_NA:", all(is.na(r1$valence_shifted)), "\\n")
cat("R1_COUNT:", nrow(r1), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "R1_ALL_NA: TRUE" in result.stdout

    def test_shift_does_not_cross_segments(self):
        """First round of each supergame should have valence_shifted = NA."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- shift_valence_to_influenced_round(dt)
first_rounds <- dt[round == 1]
n_na <- sum(is.na(first_rounds$valence_shifted))
cat("FIRST_ROUND_NA:", n_na, "\\n")
cat("FIRST_ROUND_TOTAL:", nrow(first_rounds), "\\n")
cat("ALL_NA:", n_na == nrow(first_rounds), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "ALL_NA: TRUE" in result.stdout

    def test_shift_is_correct_lag(self):
        """Shifted value at round N should equal emotion_valence at round N-1."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
dt <- shift_valence_to_influenced_round(dt)
# Pick Player D, supergame2: check round 3 shifted = round 2 valence
r2 <- dt[session_code == "6sdkxl2q" & label == "D" &
         segment == "supergame2" & round == 2]$emotion_valence
r3_shifted <- dt[session_code == "6sdkxl2q" & label == "D" &
                 segment == "supergame2" & round == 3]$valence_shifted
cat("R2_VALENCE:", round(r2, 4), "\\n")
cat("R3_SHIFTED:", round(r3_shifted, 4), "\\n")
cat("MATCH:", all.equal(r2, r3_shifted), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MATCH: TRUE" in result.stdout

    def test_original_valence_preserved(self):
        """emotion_valence column should still exist and be unchanged."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
dt <- load_chat_emotion_data()
dt[, player_id := paste0(session_code, "_", label)]
# Save original valence before shift
orig <- dt[session_code == "6sdkxl2q" & label == "D" &
           segment == "supergame2" & round == 3]$emotion_valence
dt <- shift_valence_to_influenced_round(dt)
after <- dt[session_code == "6sdkxl2q" & label == "D" &
            segment == "supergame2" & round == 3]$emotion_valence
cat("HAS_COL:", "emotion_valence" %in% names(dt), "\\n")
cat("PRESERVED:", all.equal(orig, after), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_COL: TRUE" in result.stdout
        assert "PRESERVED: TRUE" in result.stdout


# =====
# Suckered-this-round derivation tests
# =====
class TestSuckeredThisRound:
    """Test add_suckered_this_round() logic."""

    def test_suckered_column_exists(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
cat("HAS_COL:", "suckered_this_round" %in% names(stacked), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_COL: TRUE" in result.stdout

    def test_suckered_requires_full_contribution(self):
        """Suckered players must have contributed 25 (full endowment)."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
bad <- stacked[suckered_this_round == TRUE & lied_this_round_20 == TRUE]
cat("VIOLATIONS:", nrow(bad), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VIOLATIONS: 0" in result.stdout


# =====
# Model estimation tests
# =====
# Note: n-obs will change after valence shift (round-1 rows dropped).
# PLACEHOLDER counts will be pinned in Task #3.


class TestModelEstimation:
    """Test that both gap models estimate without error."""

    def test_lied_gap_model_estimates(self):
        """Lied gap model should estimate without error."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
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
stacked <- prepare_chat_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$suckered_gap)
has_interaction <- any(grepl("channel.*suckered", rownames(ct)))
cat("HAS_INTERACTION:", has_interaction, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_INTERACTION: TRUE" in result.stdout


# =====
# Regression coefficient values
# =====
# Pinned coefficients from verified run (2026-04-10, with valence shift)
LIED_INTERACTION_COEF = 1.3816
LIED_INTERACTION_SE = 2.8247
LIED_MAIN_COEF = -1.971
SUCKERED_INTERACTION_COEF = -0.7439
SUCKERED_INTERACTION_SE = 1.5848
SUCKERED_MAIN_COEF = -0.3583


class TestRegressionCoefficients:
    """Pin regression coefficients from verified run (with valence shift)."""

    def test_lied_interaction_coefficient(self):
        """Face x Lied interaction coefficient."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$lied_gap)
idx <- grep("channel.*lied", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
cat("SE:", round(ct[idx, 2], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"COEF: {LIED_INTERACTION_COEF}" in result.stdout
        assert f"SE: {LIED_INTERACTION_SE}" in result.stdout

    def test_lied_main_effect(self):
        """Lied main effect coefficient."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$lied_gap)
idx <- grep("^lied_this_round_20TRUE$", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"COEF: {LIED_MAIN_COEF}" in result.stdout

    def test_suckered_interaction_coefficient(self):
        """Face x Suckered interaction coefficient."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$suckered_gap)
idx <- grep("channel.*suckered", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
cat("SE:", round(ct[idx, 2], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"COEF: {SUCKERED_INTERACTION_COEF}" in result.stdout
        assert f"SE: {SUCKERED_INTERACTION_SE}" in result.stdout

    def test_suckered_main_effect(self):
        """Suckered main effect coefficient."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_gap_regressions.R")
stacked <- prepare_chat_stacked_data()
models <- estimate_gap_models(stacked)
ct <- coeftable(models$suckered_gap)
idx <- grep("^suckered_this_roundTRUE$", rownames(ct))
cat("COEF:", round(ct[idx, 1], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"COEF: {SUCKERED_MAIN_COEF}" in result.stdout


# =====
# LaTeX table output tests
# =====
class TestTexOutput:
    """Test that the script produces the combined .tex table."""

    def test_full_script_runs_successfully(self):
        result = run_r_script(GAP_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_tex_file_created(self):
        assert GAP_TABLE_FILE.exists(), f"Missing: {GAP_TABLE_FILE}"

    def test_tex_file_nonempty(self):
        assert GAP_TABLE_FILE.stat().st_size > 500

    def test_tex_file_contains_tabular(self):
        content = GAP_TABLE_FILE.read_text()
        assert "\\begin{tabular}" in content
