# Purpose: Heterogeneous treatment DiD coefficient plots for being suckered
# Author: Claude Code
# Date: 2026-04-10
#
# Uses the same fully interacted model as issue_59_het_did_regression.R:
#   contribution ~ i(tau, suckered_t1, ref) + i(tau, suckered_t2, ref) + treatment
#                  | round + segment
# Extracts IF and AF coefficients from the single model and overlays them.
# Treatment coding: 1 = IF (Individual Feedback), 2 = AF (Aggregate Feedback).
# Produces 4 plots: 2 thresholds (20, 5) x 2 sample definitions (main, robust).

# nolint start
library(data.table)
library(fixest)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/plots"
OUTPUT_PLOT_20_MAIN <- file.path(OUTPUT_DIR, "issue_59_het_did_coefplot_20_main.pdf")
OUTPUT_PLOT_20_ROBUST <- file.path(OUTPUT_DIR, "issue_59_het_did_coefplot_20_robust.pdf")
OUTPUT_PLOT_5_MAIN <- file.path(OUTPUT_DIR, "issue_59_het_did_coefplot_5_main.pdf")
OUTPUT_PLOT_5_ROBUST <- file.path(OUTPUT_DIR, "issue_59_het_did_coefplot_5_robust.pdf")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

    save_het_plot(dt, "20", "did_sample_20", OUTPUT_PLOT_20_MAIN)
    save_het_plot(dt, "20", "did_sample_robust_20", OUTPUT_PLOT_20_ROBUST)
    save_het_plot(dt, "5", "did_sample_5", OUTPUT_PLOT_5_MAIN)
    save_het_plot(dt, "5", "did_sample_robust_5", OUTPUT_PLOT_5_ROBUST)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- fread(filepath)
    convert_bool_cols(dt)
    prepare_tau_cols(dt)
    dt[, treatment := factor(treatment)]
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
# Treatment-specific suckered indicators (matches regression script)
# =====
create_treatment_indicators <- function(dt_sub, threshold) {
    suckered_col <- paste0("got_suckered_", threshold)
    t1_col <- paste0("suckered_t1_", threshold)
    t2_col <- paste0("suckered_t2_", threshold)
    dt_sub[, (t1_col) := as.integer(get(suckered_col) == 1 & treatment == "1")]
    dt_sub[, (t2_col) := as.integer(get(suckered_col) == 1 & treatment == "2")]
}

# =====
# Fully interacted DiD regression (matches regression script)
# =====
run_het_did_regression <- function(dt, threshold, sample_col) {
    tau_col <- paste0("tau_", threshold)
    t1_col <- paste0("suckered_t1_", threshold)
    t2_col <- paste0("suckered_t2_", threshold)
    dt_sub <- dt[get(sample_col) == 1]
    create_treatment_indicators(dt_sub, threshold)

    formula_str <- sprintf(
        "contribution ~ i(%s, %s, ref = c(0, 999)) + i(%s, %s, ref = c(0, 999)) + treatment | round + segment",
        tau_col, t1_col, tau_col, t2_col
    )

    feols(as.formula(formula_str), data = dt_sub, cluster = ~cluster_id)
}

# =====
# Coefficient extraction
# =====
extract_treatment_coefs <- function(model, threshold, treatment_num) {
    ct <- as.data.frame(coeftable(model))
    ct$name <- rownames(ct)
    # Filter for this treatment's coefficients (contain suckered_t1 or suckered_t2)
    trt_pattern <- paste0("suckered_t", treatment_num, "_", threshold)
    ct <- ct[grepl(trt_pattern, ct$name, fixed = TRUE), ]
    # Parse tau value from names like "tau_20::-6:suckered_t1_20"
    ct$tau <- as.integer(sapply(strsplit(ct$name, "::"), function(p) sub(":.*", "", p[2])))
    ref_row <- data.frame(
        Estimate = 0, `Std. Error` = 0, `t value` = NA,
        `Pr(>|t|)` = NA, name = "reference", tau = 0L,
        check.names = FALSE
    )
    ct <- rbind(ct, ref_row)
    treatment_label <- if (treatment_num == "1") "IF" else "AF"
    build_coef_df(ct, treatment_label)
}

build_coef_df <- function(ct, treatment_label) {
    data.frame(
        tau = ct$tau,
        estimate = ct$Estimate,
        ci_lower = ct$Estimate - 1.96 * ct$`Std. Error`,
        ci_upper = ct$Estimate + 1.96 * ct$`Std. Error`,
        treatment = treatment_label
    )
}

# =====
# Plot saving
# =====
save_het_plot <- function(dt, threshold, sample_col, output_path) {
    model <- run_het_did_regression(dt, threshold, sample_col)

    coef_df <- rbind(
        extract_treatment_coefs(model, threshold, "1"),
        extract_treatment_coefs(model, threshold, "2")
    )

    p <- create_het_coefplot(coef_df)
    ggsave(output_path, p, width = 7, height = 5)
    message("Plot saved to: ", output_path)
}

# =====
# Theme
# =====
theme_econ <- function() {
    theme_minimal() +
        theme(
            panel.grid.minor = element_blank(),
            panel.grid.major = element_line(color = "gray90"),
            text = element_text(family = "serif"),
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 11),
            legend.position = "bottom",
            legend.text = element_text(size = 10)
        )
}

# =====
# Plotting
# =====
create_het_coefplot <- function(coef_df) {
    dodge <- position_dodge(width = 0.4)
    ggplot(coef_df, aes(x = tau, y = estimate, shape = treatment)) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
        geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper, linetype = treatment),
                      width = 0.2, position = dodge, color = "black") +
        geom_point(size = 2.5, position = dodge, color = "black") +
        scale_shape_manual(values = c("IF" = 16, "AF" = 17)) +
        scale_linetype_manual(values = c("IF" = "solid", "AF" = "dashed")) +
        labs(x = expression("Rounds Since Suckered (" * tau * ")"),
             y = "Effect on Contribution", shape = NULL, linetype = NULL) +
        theme_econ()
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
