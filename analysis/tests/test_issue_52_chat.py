"""
Purpose: Tests for Issue #52 chat-period emotion analysis R scripts and outputs.
         Mirrors test_issue_52.py but validates the Chat (page_type=="Results")
         data loading, plotting, and regression scripts.
Author: Claude Code
Date: 2026-04-09
"""

import re
import subprocess
from pathlib import Path

import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
DATA_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "merged_panel.csv"
BEHAVIOR_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "behavior_classifications.csv"
WORKING_DIR = Path(__file__).resolve().parent.parent

# R SCRIPT PATHS
COMMON_SCRIPT = ANALYSIS_DIR / "issue_52_common.R"
CHAT_VALENCE_PLOTS_SCRIPT = ANALYSIS_DIR / "issue_52_chat_valence_plots.R"
CHAT_ALL_EMOTIONS_SCRIPT = ANALYSIS_DIR / "issue_52_chat_all_emotions_plots.R"
CHAT_WITHIN_PERSON_SCRIPT = ANALYSIS_DIR / "issue_52_chat_within_person_plots.R"
CHAT_DETRENDED_SCRIPT = ANALYSIS_DIR / "issue_52_chat_detrended_plots.R"
CHAT_REGRESSIONS_SCRIPT = ANALYSIS_DIR / "issue_52_chat_regressions.R"

# OUTPUT DIRECTORIES
CHAT_WP_PLOT_DIR = PLOT_DIR / "within_person_chat"
CHAT_DETRENDED_PLOT_DIR = PLOT_DIR / "within_person_detrended_chat"
CHAT_REG_TABLE = TABLE_DIR / "issue_52_chat_round_regressions.tex"

# EXPECTED COUNTS (from real data — verified 2026-04-09)
EXPECTED_CHAT_ROWS = 3520
EXPECTED_CHAT_VALENCE_NON_NA = 2787
EXPECTED_CHAT_LIAR_TRUE = 123
EXPECTED_CHAT_SUCKER_TRUE = 205
EXPECTED_CHAT_FIRST_TIME_LIAR = 46
EXPECTED_CHAT_FIRST_TIME_SUCKER = 79

# EXPECTED PLOT FILES
EXPECTED_CHAT_VALENCE_PLOTS = [
    "chat_valence_by_liar_status.png",
    "chat_valence_by_sucker_status.png",
    "chat_valence_by_first_time_liar.png",
    "chat_valence_by_first_time_sucker.png",
]

EMOTION_SHORT_NAMES = [
    "anger", "contempt", "disgust", "fear", "joy", "sadness",
    "surprise", "engagement", "valence", "sentimentality",
    "confusion", "neutral", "attention",
]

EXPECTED_CHAT_WP_PLOT_FILES = [
    f"chat_{emo}_by_{var}.png"
    for emo in EMOTION_SHORT_NAMES
    for var in ["liar_status", "sucker_status",
                "first_time_liar", "first_time_sucker"]
]

EXPECTED_CHAT_DETRENDED_PLOTS = [
    "valence_segment_mean_by_liar_status.png",
    "valence_segment_mean_by_sucker_status.png",
    "valence_segment_mean_by_first_time_liar.png",
    "valence_segment_mean_by_first_time_sucker.png",
    "valence_reverse_by_liar_status.png",
    "valence_reverse_by_sucker_status.png",
    "valence_reverse_by_first_time_liar.png",
    "valence_reverse_by_first_time_sucker.png",
]


def run_r_script(script_path, timeout=120):
    """Run an R script and return the completed process."""
    return subprocess.run(
        ["Rscript", str(script_path)],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


def run_r_code(code, timeout=60):
    """Run inline R code and return the completed process."""
    return subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True, text=True,
        cwd=str(WORKING_DIR), timeout=timeout,
    )


# =====
# Precondition checks
# =====
REQUIRED_FILES = [
    DATA_FILE, BEHAVIOR_FILE, COMMON_SCRIPT,
    CHAT_VALENCE_PLOTS_SCRIPT, CHAT_ALL_EMOTIONS_SCRIPT,
    CHAT_WITHIN_PERSON_SCRIPT, CHAT_DETRENDED_SCRIPT,
    CHAT_REGRESSIONS_SCRIPT,
]


class TestChatPreconditions:
    """Verify data files and chat R scripts exist."""

    @pytest.mark.parametrize("path", REQUIRED_FILES, ids=lambda p: p.name)
    def test_required_file_exists(self, path):
        assert path.exists(), f"Required file missing: {path}"


# =====
# load_chat_emotion_data() tests
# =====
class TestChatDataLoading:
    """Test chat data loading: row count, page_type, valence, behavior flags."""

    def test_returns_expected_row_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"ROWS: {EXPECTED_CHAT_ROWS}" in result.stdout

    def test_page_type_always_results(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("UNIQUE_PAGE:", paste(unique(dt$page_type), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "UNIQUE_PAGE: Results" in result.stdout

    def test_valence_non_na_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("VALENCE_NON_NA:", sum(!is.na(dt$emotion_valence)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"VALENCE_NON_NA: {EXPECTED_CHAT_VALENCE_NON_NA}" in result.stdout

    def test_has_expected_columns(self):
        expected_cols = [
            "emotion_valence", "is_liar_20", "is_sucker_20",
            "first_time_liar", "first_time_sucker", "valence_z",
            "liar_label", "sucker_label",
            "first_time_liar_label", "first_time_sucker_label",
        ]
        checks = "; ".join(
            f'cat("{col}:", "{col}" %in% names(dt), "\\n")'
            for col in expected_cols
        )
        result = run_r_code(f"""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
{checks}
""")
        assert result.returncode == 0, result.stderr
        for col in expected_cols:
            assert f"{col}: TRUE" in result.stdout, (
                f"Column {col} missing from chat output"
            )

    def test_zscore_parameters(self):
        """Z-score computed on N=2787, mean=5.1414, SD=14.1212."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
v <- dt$emotion_valence[!is.na(dt$emotion_valence)]
cat("N:", length(v), "\\n")
cat("MEAN:", round(mean(v), 4), "\\n")
cat("SD:", round(sd(v), 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N: 2787" in result.stdout
        assert "MEAN: 5.1414" in result.stdout
        assert "SD: 14.1212" in result.stdout

    def test_valence_z_mean_near_zero(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
complete <- dt[!is.na(valence_z)]
cat("MEAN_Z:", round(mean(complete$valence_z), 6), "\\n")
""")
        assert result.returncode == 0, result.stderr
        match = re.search(r"MEAN_Z:\s+([-\d.]+)", result.stdout)
        assert match is not None, f"MEAN_Z not found: {result.stdout}"
        assert abs(float(match.group(1))) < 0.01

    def test_valence_z_sd_near_one(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
complete <- dt[!is.na(valence_z)]
cat("SD_Z:", round(sd(complete$valence_z), 6), "\\n")
""")
        assert result.returncode == 0, result.stderr
        match = re.search(r"SD_Z:\s+([-\d.]+)", result.stdout)
        assert match is not None, f"SD_Z not found: {result.stdout}"
        assert abs(float(match.group(1)) - 1.0) < 0.01


# =====
# Behavior classification merge tests (chat)
# =====
class TestChatBehaviorMerge:
    """Test behavior flag counts match expectations on chat data."""

    def test_liar_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("LIAR_TRUE:", sum(dt$is_liar_20 == TRUE, na.rm=TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"LIAR_TRUE: {EXPECTED_CHAT_LIAR_TRUE}" in result.stdout

    def test_sucker_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("SUCKER_TRUE:", sum(dt$is_sucker_20 == TRUE, na.rm=TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"SUCKER_TRUE: {EXPECTED_CHAT_SUCKER_TRUE}" in result.stdout

    def test_no_na_in_behavior_flags(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("LIAR_NA:", sum(is.na(dt$is_liar_20)), "\\n")
cat("SUCKER_NA:", sum(is.na(dt$is_sucker_20)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIAR_NA: 0" in result.stdout
        assert "SUCKER_NA: 0" in result.stdout

    def test_first_time_liar_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("FIRST_LIAR:", sum(dt$first_time_liar == TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"FIRST_LIAR: {EXPECTED_CHAT_FIRST_TIME_LIAR}" in result.stdout

    def test_first_time_sucker_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("FIRST_SUCKER:", sum(dt$first_time_sucker == TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"FIRST_SUCKER: {EXPECTED_CHAT_FIRST_TIME_SUCKER}" in result.stdout

    def test_readable_labels(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("LIAR_LABELS:", paste(sort(unique(dt$liar_label)), collapse="|"), "\\n")
cat("SUCKER_LABELS:", paste(sort(unique(dt$sucker_label)), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIAR_LABELS: Honest|Liar" in result.stdout
        assert "SUCKER_LABELS: Non-sucker|Sucker" in result.stdout

    def test_first_time_labels_categories(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
cat("LIAR_CATS:", paste(sort(unique(dt$first_time_liar_label[
    !is.na(dt$first_time_liar_label)])), collapse="|"), "\\n")
cat("SUCKER_CATS:", paste(sort(unique(dt$first_time_sucker_label[
    !is.na(dt$first_time_sucker_label)])), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIAR_CATS: First-time Liar|Honest|Repeat Liar" in result.stdout
        assert "SUCKER_CATS: First-time Sucker|Non-sucker|Repeat Sucker" in result.stdout


# =====
# Regression values traced from raw data (chat period)
# =====
# Verification method (2026-04-09):
# 1. Chat data uses page_type=="Results" (the chat/results page, not ResultsOnly)
# 2. Behavior flags are identical counts — same round-level classifications
# 3. Player D (6sdkxl2q) s2r3: valence=-7.0259 (different from ResultsOnly -8.5728)
# 4. Player K (6sdkxl2q) s2r3: valence=65.8873 (different from ResultsOnly 18.2767)
# 5. Z-score params: N=2787, mean=5.1414, SD=14.1212
# 6. Group means: Liar z-mean=0.2718 (n=90), Honest z-mean=-0.0091 (n=2697)
#    Sucker z-mean=0.125 (n=157), Non-sucker z-mean=-0.0075 (n=2630)


class TestChatRegressionValues:
    """Pin down values verified from the chat-period data pipeline."""

    def test_traced_player_d_chat_valence(self):
        """Player D s2r3 chat-period valence=-7.0259."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame2" & round==3]
cat("VALENCE:", round(d$emotion_valence, 4), "\\n")
cat("IS_LIAR:", d$is_liar_20, "\\n")
cat("FIRST_TIME:", d$first_time_liar, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VALENCE: -7.0259" in result.stdout
        assert "IS_LIAR: TRUE" in result.stdout
        assert "FIRST_TIME: TRUE" in result.stdout

    def test_traced_player_k_chat_valence(self):
        """Player K s2r3 chat-period valence=65.8873."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data()
k <- dt[session_code=="6sdkxl2q" & label=="K" & segment=="supergame2" & round==3]
cat("VALENCE:", round(k$emotion_valence, 4), "\\n")
cat("IS_SUCKER:", k$is_sucker_20, "\\n")
cat("FIRST_TIME:", k$first_time_sucker, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VALENCE: 65.8873" in result.stdout
        assert "IS_SUCKER: TRUE" in result.stdout
        assert "FIRST_TIME: TRUE" in result.stdout

    def test_group_means_match(self):
        """Group means verified from load_chat_emotion_data()."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_chat_valence_plots.R")
dt <- load_chat_emotion_data()
dt <- dt[!is.na(valence_z)]
s <- summarize_valence(dt[!is.na(is_liar_20)], "liar_label")
for (i in seq_len(nrow(s))) {
    cat(s$group[i], "MEAN:", round(s$mean[i], 4), "N:", s$n[i], "\\n")
}
s2 <- summarize_valence(dt[!is.na(is_sucker_20)], "sucker_label")
for (i in seq_len(nrow(s2))) {
    cat(s2$group[i], "MEAN:", round(s2$mean[i], 4), "N:", s2$n[i], "\\n")
}
""")
        assert result.returncode == 0, result.stderr
        assert "Liar MEAN: 0.2718 N: 90" in result.stdout
        assert "Honest MEAN: -0.0091 N: 2697" in result.stdout
        assert "Sucker MEAN: 0.125 N: 157" in result.stdout
        assert "Non-sucker MEAN: -0.0075 N: 2630" in result.stdout


# =====
# issue_52_chat_valence_plots.R tests
# =====
class TestChatValencePlots:
    """Test chat valence dot plot script execution and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(CHAT_VALENCE_PLOTS_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.parametrize("filename", EXPECTED_CHAT_VALENCE_PLOTS)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = PLOT_DIR / filename
        assert path.exists(), f"Plot file missing: {path}"
        assert path.stat().st_size > 10_000, (
            f"Plot file suspiciously small: {path.stat().st_size} bytes"
        )

    def test_liar_summary_has_two_groups(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_chat_valence_plots.R")
dt <- load_chat_emotion_data()
dt <- dt[!is.na(valence_z) & !is.na(is_liar_20)]
s <- summarize_valence(dt, "liar_label")
cat("N_GROUPS:", nrow(s), "\\n")
cat("GROUPS:", paste(sort(s$group), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N_GROUPS: 2" in result.stdout
        assert "GROUPS: Honest|Liar" in result.stdout

    def test_summary_n_sums_to_total(self):
        """Sum of group n values should equal total filtered rows."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_chat_valence_plots.R")
dt <- load_chat_emotion_data()
dt <- dt[!is.na(valence_z) & !is.na(is_liar_20)]
s <- summarize_valence(dt, "liar_label")
cat("SUM_N:", sum(s$n), "\\n")
cat("TOTAL:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        sum_n = re.search(r"SUM_N:\s+(\d+)", result.stdout)
        total = re.search(r"TOTAL:\s+(\d+)", result.stdout)
        assert sum_n and total
        assert sum_n.group(1) == total.group(1)


# =====
# issue_52_chat_all_emotions_plots.R tests
# =====
class TestChatAllEmotionsPlots:
    """Test chat all-emotions plot script execution and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(CHAT_ALL_EMOTIONS_SCRIPT, timeout=300)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_52_plots(self):
        """Should produce 4 plots x 13 emotions = 52 PNG files."""
        pngs = list(PLOT_DIR.glob("chat_*_by_*.png"))
        assert len(pngs) == 52, (
            f"Expected 52 chat emotion plots, found {len(pngs)}: "
            f"{sorted(p.name for p in pngs)}"
        )

    @pytest.mark.parametrize("emotion", ["anger", "joy", "valence"])
    def test_spot_check_emotion_plots(self, emotion):
        """Spot-check that a few key emotions have all 4 variants."""
        for suffix in ["liar_status", "sucker_status",
                       "first_time_liar", "first_time_sucker"]:
            path = PLOT_DIR / f"chat_{emotion}_by_{suffix}.png"
            assert path.exists(), f"Missing: {path}"
            assert path.stat().st_size > 10_000


# =====
# issue_52_chat_within_person_plots.R tests
# =====
class TestChatWithinPersonPlots:
    """Test chat within-person deviation plot script and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(CHAT_WITHIN_PERSON_SCRIPT, timeout=300)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_52_plots(self):
        """Should produce 4 plots x 13 emotions = 52 PNG files."""
        if not CHAT_WP_PLOT_DIR.exists():
            pytest.skip("Plot dir missing")
        pngs = list(CHAT_WP_PLOT_DIR.glob("*.png"))
        assert len(pngs) == 52, (
            f"Expected 52 plots, found {len(pngs)}: "
            f"{sorted(p.name for p in pngs)}"
        )

    @pytest.mark.parametrize("filename", EXPECTED_CHAT_WP_PLOT_FILES)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = CHAT_WP_PLOT_DIR / filename
        assert path.exists(), f"Plot file missing: {path}"
        assert path.stat().st_size > 10_000, (
            f"Plot file suspiciously small: {path.stat().st_size} bytes"
        )


# =====
# load_chat_emotion_data_wp() tests
# =====
class TestChatWithinPersonData:
    """Test the chat within-person data loader."""

    def test_returns_results_page_rows(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_wp()
cat("PAGE_TYPES:", paste(unique(dt$page_type), collapse="|"), "\\n")
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "PAGE_TYPES: Results" in result.stdout
        assert f"ROWS: {EXPECTED_CHAT_ROWS}" in result.stdout

    def test_has_all_14_wpd_columns(self):
        """13 emotion _wpd columns + valence_wpd alias = 14."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_wp()
wpd <- grep("_wpd$", names(dt), value=TRUE)
cat("N_WPD:", length(wpd), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N_WPD: 14" in result.stdout

    def test_valence_wpd_non_na_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_wp()
cat("NON_NA:", sum(!is.na(dt$emotion_valence_wpd)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "NON_NA: 2774" in result.stdout

    def test_behavior_columns_present(self):
        """Behavior classification columns survive wp pipeline."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_wp()
cat("LIAR:", "is_liar_20" %in% names(dt), "\\n")
cat("SUCKER:", "is_sucker_20" %in% names(dt), "\\n")
cat("FT_LIAR:", "first_time_liar" %in% names(dt), "\\n")
cat("FT_SUCKER:", "first_time_sucker" %in% names(dt), "\\n")
cat("LIAR_LABEL:", "liar_label" %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        for tag in ["LIAR:", "SUCKER:", "FT_LIAR:", "FT_SUCKER:", "LIAR_LABEL:"]:
            assert f"{tag} TRUE" in result.stdout

    def test_no_instruction_rows_in_output(self):
        """Instruction baseline used for deviation but filtered out."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_wp()
cat("INSTR_ROWS:", sum(dt$page_type == "all_instructions"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "INSTR_ROWS: 0" in result.stdout


# =====
# issue_52_chat_detrended_plots.R tests
# =====
class TestChatDetrendedPlots:
    """Test chat detrended valence plot script and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(CHAT_DETRENDED_SCRIPT, timeout=300)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_8_plots(self):
        """Should produce 4 plots x 2 methods = 8 PNG files."""
        if not CHAT_DETRENDED_PLOT_DIR.exists():
            pytest.skip("Plot dir missing")
        pngs = list(CHAT_DETRENDED_PLOT_DIR.glob("*.png"))
        assert len(pngs) == 8, (
            f"Expected 8 plots, found {len(pngs)}: "
            f"{sorted(p.name for p in pngs)}"
        )

    @pytest.mark.parametrize("filename", EXPECTED_CHAT_DETRENDED_PLOTS)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = CHAT_DETRENDED_PLOT_DIR / filename
        assert path.exists(), f"Plot file missing: {path}"
        assert path.stat().st_size > 10_000, (
            f"Plot file suspiciously small: {path.stat().st_size} bytes"
        )


# =====
# load_chat_emotion_data_detrended() tests
# =====
class TestChatDetrendedData:
    """Test the chat detrended data loader."""

    def test_returns_expected_row_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_detrended()
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"ROWS: {EXPECTED_CHAT_ROWS}" in result.stdout

    def test_returns_results_page(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_detrended()
cat("PAGE_TYPES:", paste(unique(dt$page_type), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "PAGE_TYPES: Results" in result.stdout

    def test_has_both_detrended_columns(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_detrended()
cat("HAS_SEGMEAN:", "valence_segmean_detrended" %in% names(dt), "\\n")
cat("HAS_REVERSE:", "valence_reverse_detrended" %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_SEGMEAN: TRUE" in result.stdout
        assert "HAS_REVERSE: TRUE" in result.stdout

    def test_detrended_non_na_counts(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_detrended()
cat("SEGMEAN_NON_NA:", sum(!is.na(dt$valence_segmean_detrended)), "\\n")
cat("REVERSE_NON_NA:", sum(!is.na(dt$valence_reverse_detrended)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "SEGMEAN_NON_NA: 2696" in result.stdout
        assert "REVERSE_NON_NA: 2787" in result.stdout

    def test_behavior_columns_present(self):
        """Behavior classification columns survive detrended pipeline."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_chat_emotion_data_detrended()
cols <- c("is_liar_20", "is_sucker_20", "first_time_liar",
          "first_time_sucker", "liar_label", "sucker_label")
for (col in cols) cat(col, ":", col %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        for col in ["is_liar_20", "is_sucker_20", "first_time_liar",
                     "first_time_sucker", "liar_label", "sucker_label"]:
            assert f"{col} : TRUE" in result.stdout, (
                f"Column {col} missing from detrended output"
            )


# =====
# issue_52_chat_regressions.R tests
# =====
class TestChatRegressions:
    """Test chat regression script: execution, table output, obs count."""

    def test_script_runs_successfully(self):
        result = run_r_script(CHAT_REGRESSIONS_SCRIPT, timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_latex_table_exists_and_nonempty(self):
        assert CHAT_REG_TABLE.exists(), (
            f"Regression table missing: {CHAT_REG_TABLE}"
        )
        assert CHAT_REG_TABLE.stat().st_size > 500, (
            f"Table file too small: {CHAT_REG_TABLE.stat().st_size} bytes"
        )

    def test_regression_obs_count(self):
        """Valence regressions should have n=2787 observations."""
        result = run_r_code("""
library(fixest)
TESTING <- TRUE
source("analysis/issue_52_chat_regressions.R")
dt <- prepare_regression_data()
dt_val <- dt[!is.na(emotion_valence)]
cat("N_OBS:", nrow(dt_val), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"N_OBS: {EXPECTED_CHAT_VALENCE_NON_NA}" in result.stdout

    def test_model_produces_coefficients(self):
        """Regression on lied_this_round_20 produces a valid coefficient."""
        result = run_r_code("""
library(fixest)
TESTING <- TRUE
source("analysis/issue_52_chat_regressions.R")
dt <- prepare_regression_data()
dt_val <- dt[!is.na(emotion_valence)]
m <- feols(emotion_valence ~ lied_this_round_20 + round | segment + player_id,
           cluster=~player_id, data=dt_val)
ct <- coeftable(m)
cat("NOBS:", m$nobs, "\\n")
cat("COEF:", round(ct["lied_this_round_20TRUE", 1], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "NOBS: 2787" in result.stdout
        assert "COEF: 3.1989" in result.stdout

    def test_suckered_this_round_flag(self):
        """suckered_this_round should be derived correctly."""
        result = run_r_code("""
library(fixest)
TESTING <- TRUE
source("analysis/issue_52_chat_regressions.R")
dt <- prepare_regression_data()
cat("LIED_TRUE:", sum(dt$lied_this_round_20 == TRUE, na.rm=TRUE), "\\n")
cat("SUCKERED_TRUE:", sum(dt$suckered_this_round == TRUE, na.rm=TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIED_TRUE: 94" in result.stdout
        assert "SUCKERED_TRUE: 122" in result.stdout

    def test_latex_table_contains_expected_content(self):
        """The LaTeX table should reference key variables."""
        content = CHAT_REG_TABLE.read_text()
        assert "lied" in content.lower() or "Lied" in content
        assert "suckered" in content.lower() or "Suckered" in content
