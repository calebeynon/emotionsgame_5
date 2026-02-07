# Purpose: Diff-in-diff event study regression of contribution behavior after being suckered
# Author: Claude Code
# Date: 2026-02-05
#
# Model: contribution ~ i(tau, got_suckered, ref = 0) + treatment | round + segment
# Clustering: session_code + segment + group (concatenated as cluster_id)
#
# Four models (two thresholds x two sample definitions):
#   < 20 (tau_20): groupmate broke promise by contributing < 20 after promising
#   < 5  (tau_5):  groupmate broke promise by contributing < 5 after promising
#   Main: controls are non-suckered players
#   Robust: controls restricted to always-cooperators (never broke a promise)
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

    model_20_main   <- run_did_regression(dt, "20", "did_sample_20")
    model_20_robust <- run_did_regression(dt, "20", "did_sample_robust_20")
    model_5_main    <- run_did_regression(dt, "5", "did_sample_5")
    model_5_robust  <- run_did_regression(dt, "5", "did_sample_robust_5")

    print_summary_stats(dt)
    print_all_models(model_20_main, model_20_robust, model_5_main, model_5_robust)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    models <- list(model_20_main, model_20_robust, model_5_main, model_5_robust)
    export_latex_table(models, OUTPUT_TEX)
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
        "did_sample_20", "did_sample_5",
        "did_sample_robust_20", "did_sample_robust_5"
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
        "did_sample_robust_20", "did_sample_robust_5",
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
run_did_regression <- function(dt, threshold, sample_col) {
    tau_col <- paste0("tau_", threshold)
    suckered_col <- paste0("got_suckered_", threshold)
    dt_sub <- dt[get(sample_col) == 1]

    formula_str <- sprintf(
        "contribution ~ i(%s, %s, ref = c(0, 999)) + treatment | round + segment",
        tau_col, suckered_col
    )

    feols(as.formula(formula_str), data = dt_sub, cluster = ~cluster_id)
}

# =====
# Summary statistics
# =====
print_summary_stats <- function(dt) {
    cat("=== Summary Statistics ===\n")
    cat("Total observations:", nrow(dt), "\n")
    for (thresh in c("20", "5")) {
        print_threshold_stats(dt, thresh, paste0("did_sample_", thresh), "Main")
        print_threshold_stats(dt, thresh, paste0("did_sample_robust_", thresh), "Robust")
    }
}

print_threshold_stats <- function(dt, thresh, sample_col, label) {
    suckered_col <- paste0("got_suckered_", thresh)
    n_sample <- sum(dt[[sample_col]] == 1)
    n_treated <- sum(dt[[sample_col]] == 1 & dt[[suckered_col]] == 1)
    n_control <- sum(dt[[sample_col]] == 1 & dt[[suckered_col]] == 0)
    cat(sprintf(
        "Threshold %s (%s): %d obs (%d treated, %d control)\n",
        thresh, label, n_sample, n_treated, n_control
    ))
}

# =====
# LaTeX output
# =====
print_all_models <- function(m20_main, m20_robust, m5_main, m5_robust) {
    labels <- c("< 20 (Main)", "< 20 (Robust)", "< 5 (Main)", "< 5 (Robust)")
    models <- list(m20_main, m20_robust, m5_main, m5_robust)
    for (i in seq_along(models)) {
        cat(sprintf("\n--- %s Model ---\n", labels[i]))
        print(summary(models[[i]]))
    }
}

export_latex_table <- function(models, filepath) {
    etable(
        models[[1]], models[[2]], models[[3]], models[[4]],
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        headers = c("< 20 (Main)", "< 20 (Robust)", "< 5 (Main)", "< 5 (Robust)"),
        title = "Diff-in-Diff: Effect of Being Suckered on Contributions"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
