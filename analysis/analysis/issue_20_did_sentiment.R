# Purpose: DiD event study of being suckered on chat sentiment (compound VADER score)
# Author: Claude Code
# Date: 2026-02-05
#
# Models: sentiment_compound_mean ~ i(tau, got_suckered, ref = c(0, 999)) + treatment | round + segment
# Clustering: cluster_id (session-segment-group)
#
# Two thresholds for "suckered":
#   < 20 (tau_20): Groupmate broke promise by contributing < 20
#   < 5  (tau_5):  Groupmate broke promise by contributing < 5
#
# Control players (got_suckered == FALSE) have NA tau. We set tau = 999 for these
# players so i() produces zero coefficients, with ref = c(0, 999).

# nolint start
library(data.table)
library(fixest)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR_TABLES <- "output/tables"
OUTPUT_DIR_PLOTS <- "output/plots"
OUTPUT_TEX <- file.path(OUTPUT_DIR_TABLES, "issue_20_did_sentiment.tex")
OUTPUT_PLOT <- file.path(OUTPUT_DIR_PLOTS, "issue_20_sentiment_trajectory.png")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    model_strict <- run_regression(dt, "tau_20", "got_suckered_20", "did_sample_20")
    model_lenient <- run_regression(dt, "tau_5", "got_suckered_5", "did_sample_5")

    report_sample_sizes(model_strict, model_lenient)

    dir.create(OUTPUT_DIR_TABLES, recursive = TRUE, showWarnings = FALSE)
    dir.create(OUTPUT_DIR_PLOTS, recursive = TRUE, showWarnings = FALSE)

    export_latex_table(model_strict, model_lenient, OUTPUT_TEX)
    cat("Regression table exported to:", OUTPUT_TEX, "\n")

    coef_df <- build_coefficient_df(model_strict, model_lenient)
    create_trajectory_plot(coef_df, OUTPUT_PLOT)
    cat("Trajectory plot exported to:", OUTPUT_PLOT, "\n")
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))

    bool_cols <- c("got_suckered_20", "got_suckered_5", "did_sample_20", "did_sample_5")
    for (col in bool_cols) {
        dt[, (col) := (get(col) == "True")]
    }

    # Control players have NA tau; set to 999 so i() treats them as a reference bin
    dt[is.na(tau_20), tau_20 := 999]
    dt[is.na(tau_5), tau_5 := 999]

    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required_cols <- c(
        "sentiment_compound_mean", "got_suckered_20", "got_suckered_5",
        "tau_20", "tau_5", "did_sample_20", "did_sample_5",
        "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }

    cat("\n=== Data Summary ===\n")
    cat(sprintf("  Total rows: %d\n", nrow(dt)))
    cat(sprintf("  Sentiment NA: %d\n", sum(is.na(dt$sentiment_compound_mean))))
    cat(sprintf("  did_sample_20 TRUE: %d\n", sum(dt$did_sample_20)))
    cat(sprintf("  did_sample_5 TRUE: %d\n", sum(dt$did_sample_5)))
    cat("\n")
}

# =====
# Regression estimation
# =====
run_regression <- function(dt, tau_var, treat_var, sample_var) {
    sub <- dt[get(sample_var) == TRUE & !is.na(sentiment_compound_mean)]

    formula_str <- sprintf(
        "sentiment_compound_mean ~ i(%s, %s, ref = c(0, 999)) + treatment | round + segment",
        tau_var, treat_var
    )

    model <- feols(as.formula(formula_str), data = sub, cluster = ~cluster_id)
    return(model)
}

# =====
# Sample accountability
# =====
report_sample_sizes <- function(model_strict, model_lenient) {
    cat("=== Sample Sizes ===\n")
    cat(sprintf("  < 20 model: N = %d\n", model_strict$nobs))
    cat(sprintf("  < 5 model: N = %d\n", model_lenient$nobs))
    cat("\n")
}

# =====
# Coefficient extraction for plotting
# =====
extract_coefficients <- function(model, threshold_label) {
    ct <- coeftable(model)
    coef_names <- rownames(ct)

    # i() coefficients look like "tau_20::0:got_suckered_20"; exclude control bin 999
    is_tau <- grepl("^tau_", coef_names) & !grepl("::999:", coef_names)
    ct_sub <- ct[is_tau, , drop = FALSE]
    tau_vals <- as.numeric(gsub(".*::(-?[0-9]+):.*", "\\1", rownames(ct_sub)))

    data.frame(
        tau = tau_vals,
        estimate = ct_sub[, "Estimate"],
        se = ct_sub[, "Std. Error"],
        ci_lower = ct_sub[, "Estimate"] - 1.96 * ct_sub[, "Std. Error"],
        ci_upper = ct_sub[, "Estimate"] + 1.96 * ct_sub[, "Std. Error"],
        threshold = threshold_label,
        row.names = NULL
    )
}

build_coefficient_df <- function(model_strict, model_lenient) {
    df_strict <- extract_coefficients(model_strict, "< 20 Threshold")
    df_lenient <- extract_coefficients(model_lenient, "< 5 Threshold")
    rbind(df_strict, df_lenient)
}

# =====
# Trajectory plot
# =====
create_trajectory_plot <- function(coef_df, filepath) {
    p <- ggplot(coef_df, aes(x = tau, y = estimate)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
        geom_point(size = 2) +
        facet_wrap(~ threshold, ncol = 1) +
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Effect on Sentiment (Compound)"
        ) +
        theme_minimal(base_size = 12)

    ggsave(filepath, plot = p, width = 6.5, height = 4, dpi = 300)
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
        title = "Diff-in-Diff: Effect of Being Suckered on Chat Sentiment"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
