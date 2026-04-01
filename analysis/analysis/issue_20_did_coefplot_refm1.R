# Purpose: Event study coefficient plots with tau=-1 reference period
# Author: Claude Code
# Date: 2026-04-01
#
# Identical to issue_20_did_coefplot.R except ref = c(-1, 999).
# Reference row plotted at tau=-1 instead of tau=0.

# nolint start
library(data.table)
library(fixest)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/plots"
OUTPUT_PLOT <- file.path(OUTPUT_DIR, "issue_20_did_coefplot_refm1.png")
OUTPUT_PLOT_20 <- file.path(OUTPUT_DIR, "issue_20_did_coefplot_refm1_20.png")
OUTPUT_PLOT_5 <- file.path(OUTPUT_DIR, "issue_20_did_coefplot_refm1_5.png")

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
# DiD regression estimation (ref = -1)
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

run_all_models <- function(dt) {
    list(
        m20_main   = run_did_regression(dt, "20", "did_sample_20"),
        m20_robust = run_did_regression(dt, "20", "did_sample_robust_20"),
        m5_main    = run_did_regression(dt, "5", "did_sample_5"),
        m5_robust  = run_did_regression(dt, "5", "did_sample_robust_5")
    )
}

# =====
# Coefficient extraction (reference row at tau=-1)
# =====
extract_coefs <- function(model, threshold_label, spec_label = "Main") {
    ct <- as.data.frame(coeftable(model))
    ct$name <- rownames(ct)
    ct <- ct[grepl("::", ct$name), ]
    ct$tau <- as.integer(gsub(".*::([-0-9]+):.*", "\\1", ct$name))

    # Reference row at tau = -1 (not 0)
    ref_row <- data.frame(
        Estimate = 0, `Std. Error` = 0, `t value` = NA,
        `Pr(>|t|)` = NA, name = "reference", tau = -1L,
        check.names = FALSE
    )

    ct <- rbind(ct, ref_row)
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
# Plot saving
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
# Plotting (vertical line at tau=-1 reference)
# =====
create_coefplot <- function(coef_df) {
    ggplot(coef_df, aes(x = tau, y = estimate)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = -0.5, linetype = "dashed", color = "red",
                   alpha = 0.5) +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2) +
        geom_point(size = 2) +
        facet_wrap(~ threshold, ncol = 1) +
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Effect on Contribution (ref = -1)"
        ) +
        theme_minimal(base_size = 12) +
        theme(panel.grid.minor = element_blank())
}

create_comparison_coefplot <- function(coef_df) {
    dodge <- position_dodge(width = 0.4)
    ggplot(coef_df, aes(x = tau, y = estimate, color = spec)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = -0.5, linetype = "dashed", color = "gray50",
                   alpha = 0.5) +
        geom_errorbar(
            aes(ymin = ci_lower, ymax = ci_upper),
            width = 0.2, position = dodge
        ) +
        geom_point(size = 2, position = dodge) +
        scale_color_manual(values = c("Main" = "black",
                                      "Always-Cooperator Controls" = "#377EB8")) +
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Effect on Contribution (ref = -1)",
            color = NULL
        ) +
        theme_minimal(base_size = 12) +
        theme(panel.grid.minor = element_blank(), legend.position = "bottom")
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
