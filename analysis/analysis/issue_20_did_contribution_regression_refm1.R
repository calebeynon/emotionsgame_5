# Purpose: Robustness check — DiD event study with tau=-1 reference period
# Author: Claude Code
# Date: 2026-04-01
#
# Identical to issue_20_did_contribution_regression.R except ref = c(-1, 999)
# instead of ref = c(0, 999). This makes pre-treatment coefficients relative to
# the last clean pre-event period, not the mechanically-constrained suckering round.

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "issue_20_did_contribution_refm1.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)

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
    bool_cols <- c(
        "got_suckered_20", "got_suckered_5",
        "did_sample_20", "did_sample_5",
        "did_sample_robust_20", "did_sample_robust_5"
    )
    for (col in bool_cols) {
        dt[, (col) := as.integer(as.logical(get(col)))]
    }
    dt[is.na(tau_20), tau_20 := 999]
    dt[is.na(tau_5), tau_5 := 999]
    dt[, tau_20 := as.integer(tau_20)]
    dt[, tau_5 := as.integer(tau_5)]
    return(dt)
}

# =====
# DiD regression estimation (ref = -1 instead of 0)
# =====
run_did_regression <- function(dt, threshold, sample_col) {
    tau_col <- paste0("tau_", threshold)
    suckered_col <- paste0("got_suckered_", threshold)
    dt_sub <- dt[get(sample_col) == 1]

    formula_str <- sprintf(
        "contribution ~ i(%s, %s, ref = c(-1, 999)) + treatment | round + segment",
        tau_col, suckered_col
    )

    feols(as.formula(formula_str), data = dt_sub, cluster = ~cluster_id)
}

# =====
# Summary statistics
# =====
print_summary_stats <- function(dt) {
    cat("=== Summary Statistics (ref = -1) ===\n")
    cat("Total observations:", nrow(dt), "\n")
    for (thresh in c("20", "5")) {
        for (info in list(
            list(col = paste0("did_sample_", thresh), label = "Main"),
            list(col = paste0("did_sample_robust_", thresh), label = "Robust")
        )) {
            suckered_col <- paste0("got_suckered_", thresh)
            n_sample <- sum(dt[[info$col]] == 1)
            n_treated <- sum(dt[[info$col]] == 1 & dt[[suckered_col]] == 1)
            cat(sprintf(
                "Threshold %s (%s): %d obs (%d treated, %d control)\n",
                thresh, info$label, n_sample, n_treated, n_sample - n_treated
            ))
        }
    }
}

# =====
# Print and export
# =====
print_all_models <- function(m20_main, m20_robust, m5_main, m5_robust) {
    labels <- c("< 20 (Main)", "< 20 (Robust)", "< 5 (Main)", "< 5 (Robust)")
    models <- list(m20_main, m20_robust, m5_main, m5_robust)
    for (i in seq_along(models)) {
        cat(sprintf("\n--- %s Model (ref = -1) ---\n", labels[i]))
        print(summary(models[[i]]))
    }
}

export_latex_table <- function(models, filepath) {
    etable(
        models[[1]], models[[2]], models[[3]], models[[4]],
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        headers = c("$<$ 20 (Main)", "$<$ 20 (Robust)", "$<$ 5 (Main)", "$<$ 5 (Robust)"),
        title = "Diff-in-Diff: Effect of Being Suckered on Contributions (ref = $\\tau = -1$)"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
