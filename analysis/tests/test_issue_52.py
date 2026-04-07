"""
Purpose: Tests for Issue #52 valence of liars/suckers R scripts and plot outputs.
Author: Claude Code
Date: 2026-04-06
"""

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

# FILE PATHS
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
DATA_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "merged_panel.csv"
BEHAVIOR_FILE = Path(__file__).resolve().parent.parent / "datastore" / "derived" / "behavior_classifications.csv"
WORKING_DIR = Path(__file__).resolve().parent.parent

# R SCRIPT PATHS
COMMON_SCRIPT = ANALYSIS_DIR / "issue_52_common.R"
VALENCE_PLOTS_SCRIPT = ANALYSIS_DIR / "issue_52_valence_plots.R"
WITHIN_PERSON_PLOTS_SCRIPT = ANALYSIS_DIR / "issue_52_within_person_plots.R"
WITHIN_PERSON_PLOT_DIR = PLOT_DIR / "within_person"

# EXPECTED COUNTS (from real data)
EXPECTED_RESULTS_ROWS = 3520
EXPECTED_VALENCE_NON_NA = 2696
EXPECTED_LIAR_TRUE = 123
EXPECTED_SUCKER_TRUE = 205
EXPECTED_FIRST_TIME_LIAR = 46
EXPECTED_FIRST_TIME_SUCKER = 79


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
    VALENCE_PLOTS_SCRIPT, WITHIN_PERSON_PLOTS_SCRIPT,
]

# EXPECTED PLOT OUTPUT FILES
EXPECTED_PLOT_FILES = [
    "results_valence_by_liar_status.png",
    "results_valence_by_sucker_status.png",
    "results_valence_by_first_time_liar.png",
    "results_valence_by_first_time_sucker.png",
]


class TestPreconditions:
    """Verify data files and R scripts exist."""

    @pytest.mark.parametrize("path", REQUIRED_FILES, ids=lambda p: p.name)
    def test_required_file_exists(self, path):
        assert path.exists(), f"Required file missing: {path}"


# =====
# issue_52_common.R sourcing tests
# =====
class TestCommonSourcing:
    """Test that the common script sources without error."""

    def test_sources_without_error(self):
        result = run_r_code('TESTING <- TRUE; source("analysis/issue_52_common.R")')
        assert result.returncode == 0, f"Source failed: {result.stderr}"


# =====
# load_results_emotion_data() tests
# =====
class TestLoadResultsEmotionData:
    """Test the main data loading function."""

    def test_returns_expected_row_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"ROWS: {EXPECTED_RESULTS_ROWS}" in result.stdout

    def test_page_type_always_results_only(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("UNIQUE_PAGE:", paste(unique(dt$page_type), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "UNIQUE_PAGE: ResultsOnly" in result.stdout

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
dt <- load_results_emotion_data()
{checks}
""")
        assert result.returncode == 0, result.stderr
        for col in expected_cols:
            assert f"{col}: TRUE" in result.stdout, (
                f"Column {col} missing from output"
            )

    def test_valence_non_na_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("VALENCE_NON_NA:", sum(!is.na(dt$emotion_valence)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"VALENCE_NON_NA: {EXPECTED_VALENCE_NON_NA}" in result.stdout


# =====
# Behavior classification merge tests
# =====
class TestBehaviorMerge:
    """Test behavior classification columns after merge."""

    def test_liar_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("LIAR_TRUE:", sum(dt$is_liar_20 == TRUE, na.rm=TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"LIAR_TRUE: {EXPECTED_LIAR_TRUE}" in result.stdout

    def test_sucker_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("SUCKER_TRUE:", sum(dt$is_sucker_20 == TRUE, na.rm=TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"SUCKER_TRUE: {EXPECTED_SUCKER_TRUE}" in result.stdout

    def test_no_na_in_behavior_flags(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("LIAR_NA:", sum(is.na(dt$is_liar_20)), "\\n")
cat("SUCKER_NA:", sum(is.na(dt$is_sucker_20)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIAR_NA: 0" in result.stdout
        assert "SUCKER_NA: 0" in result.stdout

    def test_readable_labels(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("LIAR_LABELS:", paste(sort(unique(dt$liar_label)), collapse="|"), "\\n")
cat("SUCKER_LABELS:", paste(sort(unique(dt$sucker_label)), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIAR_LABELS: Honest|Liar" in result.stdout
        assert "SUCKER_LABELS: Non-sucker|Sucker" in result.stdout


# =====
# First-time flag tests
# =====
class TestFirstTimeFlags:
    """Test first-time liar/sucker flag computation."""

    def test_first_time_liar_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("FIRST_LIAR:", sum(dt$first_time_liar == TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"FIRST_LIAR: {EXPECTED_FIRST_TIME_LIAR}" in result.stdout

    def test_first_time_sucker_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("FIRST_SUCKER:", sum(dt$first_time_sucker == TRUE), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert f"FIRST_SUCKER: {EXPECTED_FIRST_TIME_SUCKER}" in result.stdout

    def test_first_time_liar_implies_is_liar(self):
        """first_time_liar==TRUE must imply is_liar_20==TRUE."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
violators <- dt[first_time_liar == TRUE & is_liar_20 != TRUE]
cat("VIOLATIONS:", nrow(violators), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VIOLATIONS: 0" in result.stdout

    def test_first_time_sucker_implies_is_sucker(self):
        """first_time_sucker==TRUE must imply is_sucker_20==TRUE."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
violators <- dt[first_time_sucker == TRUE & is_sucker_20 != TRUE]
cat("VIOLATIONS:", nrow(violators), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VIOLATIONS: 0" in result.stdout

    def test_at_most_one_first_time_liar_per_player(self):
        """Each player should have at most one first_time_liar==TRUE row."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
ftl <- dt[first_time_liar == TRUE, .N, by = .(session_code, label)]
cat("MAX_PER_PLAYER:", max(ftl$N), "\\n")
cat("N_PLAYERS:", nrow(ftl), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MAX_PER_PLAYER: 1" in result.stdout

    def test_at_most_one_first_time_sucker_per_player(self):
        """Each player should have at most one first_time_sucker==TRUE row."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
fts <- dt[first_time_sucker == TRUE, .N, by = .(session_code, label)]
cat("MAX_PER_PLAYER:", max(fts$N), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MAX_PER_PLAYER: 1" in result.stdout

    def test_first_time_labels_categories(self):
        """first_time_liar_label should have 3 categories."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("LIAR_CATS:", paste(sort(unique(dt$first_time_liar_label[!is.na(dt$first_time_liar_label)])),
    collapse="|"), "\\n")
cat("SUCKER_CATS:", paste(sort(unique(dt$first_time_sucker_label[!is.na(dt$first_time_sucker_label)])),
    collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LIAR_CATS: First-time Liar|Honest|Repeat Liar" in result.stdout
        assert "SUCKER_CATS: First-time Sucker|Non-sucker|Repeat Sucker" in result.stdout


# =====
# Z-score tests
# =====
class TestZscoreValence:
    """Test z-scored valence computation."""

    def test_valence_z_mean_near_zero(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
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
dt <- load_results_emotion_data()
complete <- dt[!is.na(valence_z)]
cat("SD_Z:", round(sd(complete$valence_z), 6), "\\n")
""")
        assert result.returncode == 0, result.stderr
        match = re.search(r"SD_Z:\s+([-\d.]+)", result.stdout)
        assert match is not None, f"SD_Z not found: {result.stdout}"
        assert abs(float(match.group(1)) - 1.0) < 0.01

    def test_valence_z_na_where_valence_na(self):
        """valence_z should be NA wherever emotion_valence is NA."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
cat("VALENCE_NA:", sum(is.na(dt$emotion_valence)), "\\n")
cat("Z_NA:", sum(is.na(dt$valence_z)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        # Both NA counts should be equal
        val_match = re.search(r"VALENCE_NA:\s+(\d+)", result.stdout)
        z_match = re.search(r"Z_NA:\s+(\d+)", result.stdout)
        assert val_match and z_match
        assert val_match.group(1) == z_match.group(1)


# =====
# Cross-validation with Python
# =====
class TestCrossValidation:
    """Cross-validate R results against Python computation on same data."""

    def test_results_only_row_count_matches_python(self):
        df = pd.read_csv(DATA_FILE)
        ro = df[df["page_type"] == "ResultsOnly"]
        assert len(ro) == EXPECTED_RESULTS_ROWS

    def test_liar_count_matches_python(self):
        df = pd.read_csv(DATA_FILE)
        ro = df[df["page_type"] == "ResultsOnly"]
        bc = pd.read_csv(BEHAVIOR_FILE)
        keys = ["session_code", "segment", "round", "group", "label"]
        merged = ro.merge(
            bc[keys + ["is_liar_20", "is_sucker_20"]],
            on=keys, how="left",
        )
        assert (merged["is_liar_20"] == True).sum() == EXPECTED_LIAR_TRUE  # noqa: E712
        assert (merged["is_sucker_20"] == True).sum() == EXPECTED_SUCKER_TRUE  # noqa: E712


# =====
# issue_52_valence_plots.R tests
# =====
class TestValencePlots:
    """Test valence dot plot script execution and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(VALENCE_PLOTS_SCRIPT)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.parametrize("filename", EXPECTED_PLOT_FILES)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = PLOT_DIR / filename
        assert path.exists(), f"Plot file missing: {path}"
        assert path.stat().st_size > 10_000, (
            f"Plot file suspiciously small: {path.stat().st_size} bytes"
        )


# =====
# Valence summary statistics tests
# =====
class TestValenceSummary:
    """Test summarize_valence helper function from plots script."""

    def test_liar_summary_has_two_groups(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_valence_plots.R")
dt <- load_results_emotion_data()
dt <- dt[!is.na(valence_z) & !is.na(is_liar_20)]
s <- summarize_valence(dt, "liar_label")
cat("N_GROUPS:", nrow(s), "\\n")
cat("GROUPS:", paste(sort(s$group), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N_GROUPS: 2" in result.stdout
        assert "GROUPS: Honest|Liar" in result.stdout

    def test_sucker_summary_has_two_groups(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_valence_plots.R")
dt <- load_results_emotion_data()
dt <- dt[!is.na(valence_z) & !is.na(is_sucker_20)]
s <- summarize_valence(dt, "sucker_label")
cat("N_GROUPS:", nrow(s), "\\n")
cat("GROUPS:", paste(sort(s$group), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N_GROUPS: 2" in result.stdout
        assert "GROUPS: Non-sucker|Sucker" in result.stdout

    def test_first_time_liar_summary_has_three_groups(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_valence_plots.R")
dt <- load_results_emotion_data()
dt <- dt[!is.na(valence_z) & !is.na(is_liar_20)]
s <- summarize_valence(dt, "first_time_liar_label")
cat("N_GROUPS:", nrow(s), "\\n")
cat("GROUPS:", paste(sort(s$group), collapse="|"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N_GROUPS: 3" in result.stdout
        assert "GROUPS: First-time Liar|Honest|Repeat Liar" in result.stdout

    def test_summary_n_sums_to_total(self):
        """Sum of group n values should equal total filtered rows."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_valence_plots.R")
dt <- load_results_emotion_data()
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

    def test_ci_bounds_contain_mean(self):
        """95% CI lower should be <= mean and upper should be >= mean."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_valence_plots.R")
dt <- load_results_emotion_data()
dt <- dt[!is.na(valence_z) & !is.na(is_liar_20)]
s <- summarize_valence(dt, "liar_label")
cat("LOWER_OK:", all(s$lower <= s$mean), "\\n")
cat("UPPER_OK:", all(s$upper >= s$mean), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "LOWER_OK: TRUE" in result.stdout
        assert "UPPER_OK: TRUE" in result.stdout


# =====
# Regression tests: hand-traced through raw data sources
# =====
# Verification method (2026-04-07):
# 1. Traced Player D (session 6sdkxl2q/session 11, raw id D11) end-to-end:
#    - behavior_classifications.csv: is_liar_20=True at supergame2 r3,r4 only
#    - Rwork/all.csv: 22 ResultsOnly rows, all Valence values match merged_panel 1:1
#    - First-time liar flag correctly placed at supergame2 r3 (earliest occurrence)
# 2. Traced Player K (session 6sdkxl2q, raw id K11):
#    - is_sucker_20=True at supergame2 r3,r4 and supergame4 r7
#    - Raw valence at sucker rounds: 18.28 (s2r3), 35.65 (s2r4), 29.13 (s4r7)
#    - First-time sucker at supergame2 r3
# 3. Traced Player C (session 6uv359rf/session 4):
#    - Both liar AND sucker — 0 raw emotion rows (session 4 missing C,E from iMotions)
#    - Confirms NaN valence is from missing collection, not a merge bug
# 4. Z-score parameters: mean=3.162542, SD=12.551225, N=2696
# 5. All group means independently computed in Python from raw CSVs match R output


class TestRegressionValues:
    """Pin down values verified by tracing raw data through the full pipeline.

    Group means were independently computed in Python directly from
    merged_panel.csv + behavior_classifications.csv and cross-validated
    against the R script output. Individual observations were traced
    back to Rwork/all.csv raw Valence values.
    """

    def test_zscore_parameters(self):
        """Z-score computed on N=2696, mean=3.1625, SD=12.5512."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
v <- dt$emotion_valence[!is.na(dt$emotion_valence)]
cat("N:", length(v), "\\n")
cat("MEAN:", round(mean(v), 4), "\\n")
cat("SD:", round(sd(v), 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "N: 2696" in result.stdout
        assert "MEAN: 3.1625" in result.stdout
        assert "SD: 12.5512" in result.stdout

    def test_traced_player_d_valence(self):
        """Player D s2r3: raw valence=-8.5728 from Rwork/all.csv row D11."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame2" & round==3]
cat("VALENCE:", round(d$emotion_valence, 4), "\\n")
cat("IS_LIAR:", d$is_liar_20, "\\n")
cat("FIRST_TIME:", d$first_time_liar, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VALENCE: -8.5728" in result.stdout
        assert "IS_LIAR: TRUE" in result.stdout
        assert "FIRST_TIME: TRUE" in result.stdout

    def test_traced_player_k_valence(self):
        """Player K s2r3: raw valence=18.2767 from Rwork/all.csv row K11."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
k <- dt[session_code=="6sdkxl2q" & label=="K" & segment=="supergame2" & round==3]
cat("VALENCE:", round(k$emotion_valence, 4), "\\n")
cat("IS_SUCKER:", k$is_sucker_20, "\\n")
cat("FIRST_TIME:", k$first_time_sucker, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "VALENCE: 18.2767" in result.stdout
        assert "IS_SUCKER: TRUE" in result.stdout
        assert "FIRST_TIME: TRUE" in result.stdout

    def test_traced_player_d_repeat_liar(self):
        """Player D s2r4: second liar round, should NOT be first-time."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame2" & round==4]
cat("IS_LIAR:", d$is_liar_20, "\\n")
cat("FIRST_TIME:", d$first_time_liar, "\\n")
cat("LABEL:", d$first_time_liar_label, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "IS_LIAR: TRUE" in result.stdout
        assert "FIRST_TIME: FALSE" in result.stdout
        assert "LABEL: Repeat Liar" in result.stdout

    def test_group_means_match_python(self):
        """Group means verified independently in Python from raw CSVs."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_valence_plots.R")
dt <- load_results_emotion_data()
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
        # These match independent Python computation on the same CSVs
        assert "Liar MEAN: 0.2298 N: 88" in result.stdout
        assert "Honest MEAN: -0.0078 N: 2608" in result.stdout
        assert "Sucker MEAN: 0.3855 N: 151" in result.stdout
        assert "Non-sucker MEAN: -0.0229 N: 2545" in result.stdout


# =====
# Within-person deviation function tests
# =====
class TestWithinPersonDeviation:
    """Test the within_person_deviation expanding-window function."""

    def test_cold_start_first_obs_is_na(self):
        """First obs (instruction) has no prior — NA."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- fread(INPUT_CSV)
dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
dt <- within_person_deviation(dt, "emotion_valence")
p <- dt[session_code=="6sdkxl2q" & label=="D"]
setorderv(p, c("segment", "round"))
cat("INSTR_WPD:", is.na(p$emotion_valence_wpd[1]), "\\n")
cat("S1R1_WPD:", round(p$emotion_valence_wpd[2], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "INSTR_WPD: TRUE" in result.stdout
        assert "S1R1_WPD: 1.4428" in result.stdout

    def test_second_results_obs_has_deviation(self):
        """Second ResultsOnly obs (3rd overall) has 2 prior — valid deviation."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- fread(INPUT_CSV)
dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
dt <- within_person_deviation(dt, "emotion_valence")
p <- dt[session_code=="6sdkxl2q" & label=="D"]
setorderv(p, c("segment", "round"))
cat("S1R2_WPD_NA:", is.na(p$emotion_valence_wpd[3]), "\\n")
cat("S1R2_WPD:", round(p$emotion_valence_wpd[3], 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "S1R2_WPD_NA: FALSE" in result.stdout
        assert "S1R2_WPD: 2.5798" in result.stdout

    def test_cross_supergame_accumulation(self):
        """History accumulates across supergame boundaries.

        Player D s2r1 uses instruction + s1r1,s1r2,s1r3 as prior (4 obs).
        Manual: prior=c(-2.1264,-0.6837,1.1748,2.6747), val=-0.8607 -> d=-1.1206
        """
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- fread(INPUT_CSV)
dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
dt <- within_person_deviation(dt, "emotion_valence")
p <- dt[session_code=="6sdkxl2q" & label=="D"]
setorderv(p, c("segment", "round"))
# Row 5 = supergame2 round 1
s2r1 <- p[segment=="supergame2" & round==1]
cat("S2R1_WPD:", round(s2r1$emotion_valence_wpd, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "S2R1_WPD: -1.1206" in result.stdout

    def test_expanding_window_grows(self):
        """Later rounds use more prior observations."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- fread(INPUT_CSV)
dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
dt <- within_person_deviation(dt, "emotion_valence")
p <- dt[session_code=="6sdkxl2q" & label=="D"]
setorderv(p, c("segment", "round"))
# Row 9 = supergame3 round 1 (8 prior obs)
s3r1 <- p[segment=="supergame3" & round==1]
cat("S3R1_WPD:", round(s3r1$emotion_valence_wpd, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "S3R1_WPD: 10.206" in result.stdout

    def test_na_valence_produces_na_deviation(self):
        """Players with all-NA valence get all-NA wpd values."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- fread(INPUT_CSV)
dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
dt <- within_person_deviation(dt, "emotion_valence")
# Player E in session sa7mprty has all NA valence
p <- dt[session_code=="sa7mprty" & label=="E"]
cat("ALL_NA:", all(is.na(p$emotion_valence_wpd)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "ALL_NA: TRUE" in result.stdout

    def test_output_column_naming(self):
        """Output column should be {col}_wpd."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- fread(INPUT_CSV)
dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
dt <- within_person_deviation(dt, "emotion_anger")
cat("HAS_COL:", "emotion_anger_wpd" %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "HAS_COL: TRUE" in result.stdout


# =====
# load_results_emotion_data_wp() tests
# =====
class TestLoadResultsEmotionDataWP:
    """Test the within-person data loader."""

    def test_returns_results_only_rows(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
cat("PAGE_TYPES:", paste(unique(dt$page_type), collapse="|"), "\\n")
cat("ROWS:", nrow(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "PAGE_TYPES: ResultsOnly" in result.stdout
        assert f"ROWS: {EXPECTED_RESULTS_ROWS}" in result.stdout

    def test_has_all_13_wpd_columns(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
wpd <- grep("_wpd$", names(dt), value=TRUE)
cat("N_WPD:", length(wpd), "\\n")
""")
        assert result.returncode == 0, result.stderr
        # 13 emotion columns + valence_wpd alias = 14
        assert "N_WPD: 14" in result.stdout

    @pytest.mark.parametrize("col", [
        "emotion_anger_wpd", "emotion_contempt_wpd", "emotion_disgust_wpd",
        "emotion_fear_wpd", "emotion_joy_wpd", "emotion_sadness_wpd",
        "emotion_surprise_wpd", "emotion_engagement_wpd",
        "emotion_valence_wpd", "emotion_sentimentality_wpd",
        "emotion_confusion_wpd", "emotion_neutral_wpd",
        "emotion_attention_wpd", "valence_wpd",
    ])
    def test_wpd_column_exists(self, col):
        result = run_r_code(f"""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
cat("EXISTS:", "{col}" %in% names(dt), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "EXISTS: TRUE" in result.stdout

    def test_valence_wpd_alias_matches(self):
        """valence_wpd should equal emotion_valence_wpd."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
both_valid <- !is.na(dt$valence_wpd) & !is.na(dt$emotion_valence_wpd)
cat("MATCH:", all(dt$valence_wpd[both_valid] == dt$emotion_valence_wpd[both_valid]), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "MATCH: TRUE" in result.stdout

    def test_valence_wpd_non_na_count(self):
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
cat("NON_NA:", sum(!is.na(dt$emotion_valence_wpd)), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "NON_NA: 2696" in result.stdout

    def test_behavior_columns_present(self):
        """Behavior classification columns survive wp pipeline."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
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
        """Instruction baseline used for deviation computation but filtered out."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
cat("INSTR_ROWS:", sum(dt$page_type == "all_instructions"), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "INSTR_ROWS: 0" in result.stdout


# =====
# Within-person deviation regression values (traced from raw data)
# =====
# Verification method (2026-04-07):
# 1. Player D (6sdkxl2q): instruction valence=-2.1264, followed by 22 ResultsOnly
# 2. Manual deviation at s1r2: prior=[-2.1264, -0.6837], val=1.1748
#    mean(prior)=-1.40505, dev=1.1748-(-1.40505)=2.5798
# 3. Cross-supergame s2r1: prior=[-2.1264,-0.6837,1.1748,2.6747], val=-0.8607
#    mean(prior)=0.2599, dev=-0.8607-0.2599=-1.1206
# 4. Full pipeline: 3520 ResultsOnly rows, 2696 with non-NA valence_wpd


class TestWithinPersonRegressionValues:
    """Pin within-person deviation values verified by manual computation."""

    def test_player_d_s1r2_deviation(self):
        """Manual: prior=[-2.1264,-0.6837], val=1.1748, dev=2.5798."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame1" & round==2]
cat("WPD:", round(d$emotion_valence_wpd, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "WPD: 2.5798" in result.stdout

    def test_player_d_s2r1_cross_supergame(self):
        """Cross-supergame: 4 prior obs, dev=-1.1206."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame2" & round==1]
cat("WPD:", round(d$emotion_valence_wpd, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "WPD: -1.1206" in result.stdout

    def test_player_d_s2r3_liar_round(self):
        """Player D at first-time liar round: dev=-10.1273."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame2" & round==3]
cat("WPD:", round(d$emotion_valence_wpd, 4), "\\n")
cat("IS_LIAR:", d$is_liar_20, "\\n")
cat("FIRST_TIME:", d$first_time_liar, "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "WPD: -10.1273" in result.stdout
        assert "IS_LIAR: TRUE" in result.stdout
        assert "FIRST_TIME: TRUE" in result.stdout

    def test_player_d_last_round(self):
        """Player D s5r5 (22nd obs, 21 prior): dev=-15.7187."""
        result = run_r_code("""
TESTING <- TRUE
source("analysis/issue_52_common.R")
dt <- load_results_emotion_data_wp()
d <- dt[session_code=="6sdkxl2q" & label=="D" & segment=="supergame5" & round==5]
cat("WPD:", round(d$emotion_valence_wpd, 4), "\\n")
""")
        assert result.returncode == 0, result.stderr
        assert "WPD: -15.7187" in result.stdout


# =====
# issue_52_within_person_plots.R tests
# =====

# All 13 emotions x 4 plot types = 52 expected files
EMOTION_SHORT_NAMES = [
    "anger", "contempt", "disgust", "fear", "joy", "sadness",
    "surprise", "engagement", "valence", "sentimentality",
    "confusion", "neutral", "attention",
]
EXPECTED_WP_PLOT_FILES = [
    f"results_{emo}_by_{var}.png"
    for emo in EMOTION_SHORT_NAMES
    for var in ["liar_status", "sucker_status",
                "first_time_liar", "first_time_sucker"]
]


class TestWithinPersonPlots:
    """Test within-person deviation plot script and outputs."""

    def test_script_runs_successfully(self):
        result = run_r_script(WITHIN_PERSON_PLOTS_SCRIPT, timeout=300)
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_generates_52_plots(self):
        """Should produce 4 plots x 13 emotions = 52 PNG files."""
        if not WITHIN_PERSON_PLOT_DIR.exists():
            pytest.skip("Plot dir missing — run test_script_runs_successfully first")
        pngs = list(WITHIN_PERSON_PLOT_DIR.glob("*.png"))
        assert len(pngs) == 52, (
            f"Expected 52 plots, found {len(pngs)}: {sorted(p.name for p in pngs)}"
        )

    @pytest.mark.parametrize("filename", EXPECTED_WP_PLOT_FILES)
    def test_plot_file_exists_and_nonempty(self, filename):
        path = WITHIN_PERSON_PLOT_DIR / filename
        assert path.exists(), f"Plot file missing: {path}"
        assert path.stat().st_size > 10_000, (
            f"Plot file suspiciously small: {path.stat().st_size} bytes"
        )
