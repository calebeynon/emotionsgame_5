# Purpose: Diff-in-diff event study regression of contribution behavior after being suckered
# Author: Claude Code
# Date: 2026-02-05
#
# Model: contribution ~ i(tau, got_suckered, ref = 0) + treatment | round + segment
# Clustering: session_code + segment + group (concatenated as cluster_id)
#
# Two threshold models:
#   < 20 (tau_20): groupmate broke promise by contributing < 20 after promising
#   < 5  (tau_5):  groupmate broke promise by contributing < 5 after promising
#
# The i() interaction creates event-time dummies interacted with treatment status.
# Control players (got_suckered == FALSE) have NA tau values; we set these to a
# sentinel value (999) excluded via ref = c(0, 999) so all interaction terms are zero.

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_20_did_contribution.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    model_strict <- run_did_regression(dt, "20")
    model_lenient <- run_did_regression(dt, "5")

    print_summary_stats(dt)
    cat("\n--- < 20 Model ---\n")
    print(summary(model_strict))
    cat("\n--- < 5 Model ---\n")
    print(summary(model_lenient))

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(model_strict, model_lenient, OUTPUT_TEX)
    cat("\nRegression table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- fread(filepath)
    convert_bool_cols(dt)
    prepare_tau_cols(dt)
    return(dt)
}

convert_bool_cols <- function(dt) {
    bool_cols <- c(
        "got_suckered_20", "got_suckered_5",
        "did_sample_20", "did_sample_5"
    )
    for (col in bool_cols) {
        dt[, (col) := as.integer(as.logical(get(col)))]
    }
}

prepare_tau_cols <- function(dt) {
    # Sentinel value (999) for control players' NA tau, excluded via ref
    dt[is.na(tau_20), tau_20 := 999]
    dt[is.na(tau_5), tau_5 := 999]
    dt[, tau_20 := as.integer(tau_20)]
    dt[, tau_5 := as.integer(tau_5)]
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required_cols <- c(
        "contribution", "got_suckered_20", "got_suckered_5",
        "tau_20", "tau_5", "did_sample_20", "did_sample_5",
        "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }
}

# =====
# DiD regression estimation
# =====
run_did_regression <- function(dt, threshold) {
    tau_col <- paste0("tau_", threshold)
    suckered_col <- paste0("got_suckered_", threshold)
    sample_col <- paste0("did_sample_", threshold)

    dt_sub <- dt[get(sample_col) == 1]

    formula_str <- sprintf(
        "contribution ~ i(%s, %s, ref = c(0, 999)) + treatment | round + segment",
        tau_col, suckered_col
    )

    model <- feols(
        as.formula(formula_str),
        data = dt_sub,
        cluster = ~cluster_id
    )

    return(model)
}

# =====
# Summary statistics
# =====
print_summary_stats <- function(dt) {
    cat("=== Summary Statistics ===\n")
    cat("Total observations:", nrow(dt), "\n")
    for (thresh in c("20", "5")) {
        sample_col <- paste0("did_sample_", thresh)
        suckered_col <- paste0("got_suckered_", thresh)
        n_sample <- sum(dt[[sample_col]] == 1)
        n_treated <- sum(dt[[sample_col]] == 1 & dt[[suckered_col]] == 1)
        n_control <- sum(dt[[sample_col]] == 1 & dt[[suckered_col]] == 0)
        cat(sprintf(
            "Threshold %s: %d obs (%d treated, %d control)\n",
            thresh, n_sample, n_treated, n_control
        ))
    }
}

# =====
# LaTeX output
# =====
export_latex_table <- function(model_strict, model_lenient, filepath) {
    etable(
        model_strict, model_lenient,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        headers = c("< 20 Threshold", "< 5 Threshold"),
        title = "Diff-in-Diff: Effect of Being Suckered on Contributions"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
