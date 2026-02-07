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
# Two sample definitions per threshold:
#   Main:   suckered-once treated vs never-suckered control
#   Robust: suckered-once treated vs always-cooperator control (contributed 25 every round > 1)
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
OUTPUT_COEFPLOT_20 <- file.path(OUTPUT_DIR_PLOTS, "issue_20_sentiment_coefplot_20.png")
OUTPUT_COEFPLOT_5 <- file.path(OUTPUT_DIR_PLOTS, "issue_20_sentiment_coefplot_5.png")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    models <- run_all_models(dt)
    report_sample_sizes(models)

    dir.create(OUTPUT_DIR_TABLES, recursive = TRUE, showWarnings = FALSE)
    dir.create(OUTPUT_DIR_PLOTS, recursive = TRUE, showWarnings = FALSE)

    export_latex_table(models, OUTPUT_TEX)
    cat("Regression table exported to:", OUTPUT_TEX, "\n")

    coef_df <- build_coefficient_df(models$main_20, models$main_5)
    create_trajectory_plot(coef_df, OUTPUT_PLOT)
    cat("Trajectory plot exported to:", OUTPUT_PLOT, "\n")

    save_comparison_plot(models, "20", "< 20 Threshold", OUTPUT_COEFPLOT_20)
    save_comparison_plot(models, "5", "< 5 Threshold", OUTPUT_COEFPLOT_5)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))

    bool_cols <- c(
        "got_suckered_20", "got_suckered_5",
        "did_sample_20", "did_sample_5",
        "did_sample_robust_20", "did_sample_robust_5"
    )
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
        "did_sample_robust_20", "did_sample_robust_5",
        "treatment", "round", "segment", "cluster_id"
    )
    missing <- setdiff(required_cols, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }

    cat("\n=== Data Summary ===\n")
    cat(sprintf("  Total rows: %d\n", nrow(dt)))
    cat(sprintf("  Sentiment NA: %d\n", sum(is.na(dt$sentiment_compound_mean))))
    print_sample_counts(dt)
    cat("\n")
}

print_sample_counts <- function(dt) {
    samples <- c("did_sample_20", "did_sample_robust_20",
                 "did_sample_5", "did_sample_robust_5")
    for (s in samples) cat(sprintf("  %s TRUE: %d\n", s, sum(dt[[s]])))
}

# =====
# Regression estimation
# =====
run_all_models <- function(dt) {
    list(
        main_20   = run_regression(dt, "tau_20", "got_suckered_20", "did_sample_20"),
        robust_20 = run_regression(dt, "tau_20", "got_suckered_20", "did_sample_robust_20"),
        main_5    = run_regression(dt, "tau_5", "got_suckered_5", "did_sample_5"),
        robust_5  = run_regression(dt, "tau_5", "got_suckered_5", "did_sample_robust_5")
    )
}

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
report_sample_sizes <- function(models) {
    cat("=== Sample Sizes ===\n")
    for (name in names(models)) {
        cat(sprintf("  %s: N = %d\n", name, models[[name]]$nobs))
    }
    cat("\n")
}

# =====
# Coefficient extraction for plotting
# =====
extract_coefficients <- function(model, threshold_label, spec_label = "Main") {
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
        spec = spec_label,
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
# Comparison plots (main vs robustness spec)
# =====
save_comparison_plot <- function(models, threshold, label, output_path) {
    main_key <- paste0("main_", threshold)
    robust_key <- paste0("robust_", threshold)
    coef_df <- rbind(
        extract_coefficients(models[[main_key]], label, "Main"),
        extract_coefficients(models[[robust_key]], label, "Always-Cooperator Controls")
    )
    create_comparison_plot(coef_df, output_path)
    cat("Comparison plot exported to:", output_path, "\n")
}

create_comparison_plot <- function(coef_df, filepath) {
    dodge <- position_dodge(width = 0.4)
    spec_colors <- c("Main" = "black", "Always-Cooperator Controls" = "#377EB8")
    p <- ggplot(coef_df, aes(x = tau, y = estimate, color = spec)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                      width = 0.2, position = dodge) +
        geom_point(size = 2, position = dodge) +
        scale_color_manual(values = spec_colors) +
        labs(x = expression("Rounds Since Suckered (" * tau * ")"),
             y = "Effect on Sentiment (Compound)", color = "Specification") +
        theme_minimal(base_size = 12) +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
    ggsave(filepath, plot = p, width = 6.5, height = 4, dpi = 300)
}

# =====
# LaTeX output
# =====
export_latex_table <- function(models, filepath) {
    etable(
        models$main_20, models$robust_20, models$main_5, models$robust_5,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        headers = c("< 20 (Main)", "< 20 (Robust)", "< 5 (Main)", "< 5 (Robust)"),
        title = "Diff-in-Diff: Effect of Being Suckered on Chat Sentiment"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
