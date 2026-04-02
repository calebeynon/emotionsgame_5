# Purpose: Raw mean contribution plot at each event time (tau) for suckered players
# Author: Claude Code
# Date: 2026-02-06
#
# Plots mean contribution levels with SE bars for treated (suckered) players,
# with dashed reference line at control group grand mean, faceted by threshold.

# nolint start
library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/plots"
OUTPUT_PLOT <- file.path(OUTPUT_DIR, "issue_20_raw_means.png")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)

    means_20 <- compute_treated_means(dt, "20", "< 20 Threshold")
    means_5 <- compute_treated_means(dt, "5", "< 5 Threshold")
    treated_df <- rbind(means_20, means_5)

    ctrl_20 <- compute_control_mean(dt, "20", "< 20 Threshold")
    ctrl_5 <- compute_control_mean(dt, "5", "< 5 Threshold")
    ctrl_df <- rbind(ctrl_20, ctrl_5)

    p <- create_raw_means_plot(treated_df, ctrl_df)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    ggsave(OUTPUT_PLOT, p, width = 6.5, height = 4, dpi = 300)
    message("Plot saved to: ", OUTPUT_PLOT)
}

# =====
# Data loading and preparation (mirrors coefplot script)
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
    dt[is.na(tau_20), tau_20 := 999]
    dt[is.na(tau_5), tau_5 := 999]
    dt[, tau_20 := as.integer(tau_20)]
    dt[, tau_5 := as.integer(tau_5)]
}

# =====
# Summary statistics computation
# =====
compute_treated_means <- function(dt, threshold, label) {
    sample_col <- paste0("did_sample_", threshold)
    suckered_col <- paste0("got_suckered_", threshold)
    tau_col <- paste0("tau_", threshold)

    dt_sub <- dt[get(sample_col) == 1 & get(suckered_col) == 1]
    dt_sub <- dt_sub[get(tau_col) != 999]

    agg <- dt_sub[, .(
        mean_contrib = mean(contribution),
        se = sd(contribution) / sqrt(.N)
    ), by = tau_col]

    setnames(agg, tau_col, "tau")
    agg[, threshold := label]
    return(agg)
}

compute_control_mean <- function(dt, threshold, label) {
    sample_col <- paste0("did_sample_", threshold)
    suckered_col <- paste0("got_suckered_", threshold)

    dt_ctrl <- dt[get(sample_col) == 1 & get(suckered_col) == 0]

    data.frame(
        grand_mean = mean(dt_ctrl$contribution),
        threshold = label
    )
}

# =====
# Plotting
# =====
create_raw_means_plot <- function(treated_df, ctrl_df) {
    p <- build_base_plot(treated_df, ctrl_df)
    p + style_raw_means_plot()
}

build_base_plot <- function(treated_df, ctrl_df) {
    ggplot(treated_df, aes(x = tau, y = mean_contrib)) +
        geom_hline(
            data = ctrl_df, aes(yintercept = grand_mean),
            linetype = "dashed", color = "gray50"
        ) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
        geom_errorbar(
            aes(ymin = mean_contrib - se, ymax = mean_contrib + se),
            width = 0.2
        ) +
        geom_line() +
        geom_point(size = 2) +
        facet_wrap(~ threshold, ncol = 1)
}

style_raw_means_plot <- function() {
    list(
        labs(
            x = expression("Rounds Since Suckered (" * tau * ")"),
            y = "Mean Contribution"
        ),
        theme_minimal(base_size = 12),
        theme(panel.grid.minor = element_blank())
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
