# Purpose: Event study coefficient plots for DiD analysis of being suckered
# Author: Claude Code
# Date: 2026-02-05
#
# Plots estimated coefficients from i(tau, got_suckered) interaction terms
# with 95% confidence intervals. Produces:
#   1. Original faceted plot (2 thresholds, main spec only)
#   2. Per-threshold comparison plots overlaying main vs robustness spec

# nolint start
library(data.table)
library(fixest)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/plots"
OUTPUT_PLOT <- file.path(OUTPUT_DIR, "issue_20_did_coefplot.png")
OUTPUT_PLOT_20 <- file.path(OUTPUT_DIR, "issue_20_did_coefplot_20.png")
OUTPUT_PLOT_5 <- file.path(OUTPUT_DIR, "issue_20_did_coefplot_5.png")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

    models <- run_all_models(dt)
    save_original_plot(models)
    save_comparison_plot(models, "20", "< 20 Threshold", OUTPUT_PLOT_20)
    save_comparison_plot(models, "5", "< 5 Threshold", OUTPUT_PLOT_5)
}

run_all_models <- function(dt) {
    list(
        m20_main   = run_did_regression(dt, "20", "did_sample_20"),
        m20_robust = run_did_regression(dt, "20", "did_sample_robust_20"),
        m5_main    = run_did_regression(dt, "5", "did_sample_5"),
        m5_robust  = run_did_regression(dt, "5", "did_sample_robust_5")
    )
}

# =====
# Data loading and preparation (mirrors regression script)
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
    dt[is.na(tau_20), tau_20 := 999]
    dt[is.na(tau_5), tau_5 := 999]
    dt[, tau_20 := as.integer(tau_20)]
    dt[, tau_5 := as.integer(tau_5)]
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
# Coefficient extraction
# =====
extract_coefs <- function(model, threshold_label, spec_label = "Main") {
    ct <- as.data.frame(coeftable(model))
    ct$name <- rownames(ct)

    # Keep only interaction terms (contain "::")
    ct <- ct[grepl("::", ct$name), ]

    # Parse tau value from names like "tau_20::-6:got_suckered_20"
    ct$tau <- as.integer(gsub(".*::([-0-9]+):.*", "\\1", ct$name))

    ref_row <- data.frame(
        Estimate = 0, `Std. Error` = 0, `t value` = NA,
        `Pr(>|t|)` = NA, name = "reference", tau = 0L,
        check.names = FALSE
    )

    ct <- rbind(ct, ref_row)
    build_coef_df(ct, threshold_label, spec_label)
}

build_coef_df <- function(ct, threshold_label, spec_label) {
    data.frame(
        tau = ct$tau,
        estimate = ct$Estimate,
        ci_lower = ct$Estimate - 1.96 * ct$`Std. Error`,
        ci_upper = ct$Estimate + 1.96 * ct$`Std. Error`,
        threshold = threshold_label,
        spec = spec_label
    )
}

# =====
# Plot saving helpers
# =====
save_original_plot <- function(models) {
    coef_df <- rbind(
        extract_coefs(models$m20_main, "< 20 Threshold"),
        extract_coefs(models$m5_main, "< 5 Threshold")
    )
    p <- create_coefplot(coef_df)
    ggsave(OUTPUT_PLOT, p, width = 6.5, height = 4, dpi = 300)
    message("Plot saved to: ", OUTPUT_PLOT)
}

save_comparison_plot <- function(models, threshold, label, output_path) {
    main_key <- paste0("m", threshold, "_main")
    robust_key <- paste0("m", threshold, "_robust")
    coef_df <- rbind(
        extract_coefs(models[[main_key]], label, "Main"),
        extract_coefs(models[[robust_key]], label, "Always-Cooperator Controls")
    )
    p <- create_comparison_coefplot(coef_df)
    ggsave(output_path, p, width = 6.5, height = 4, dpi = 300)
    message("Plot saved to: ", output_path)
}

# =====
# Plotting
# =====
create_coefplot <- function(coef_df) {
    ggplot(coef_df, aes(x = tau, y = estimate)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
        geom_point(size = 2) +
        facet_wrap(~ threshold, ncol = 1) +
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Effect on Contribution"
        ) +
        theme_minimal(base_size = 12) +
        theme(panel.grid.minor = element_blank())
}

create_comparison_coefplot <- function(coef_df) {
    dodge <- position_dodge(width = 0.4)
    ggplot(coef_df, aes(x = tau, y = estimate, color = spec)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
        geom_errorbar(
            aes(ymin = ci_lower, ymax = ci_upper),
            width = 0.2, position = dodge
        ) +
        geom_point(size = 2, position = dodge) +
        scale_color_manual(values = c("Main" = "black", "Always-Cooperator Controls" = "#377EB8")) +
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Effect on Contribution",
            color = NULL
        ) +
        theme_minimal(base_size = 12) +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
