# Purpose: 2x2 faceted trajectory plot comparing sucker vs liar mean contributions
# Author: Claude Code
# Date: 2026-02-06
#
# Plots mean contribution at each event time (tau) for treated players,
# with dashed reference lines for control grand mean, faceted by role x threshold.

# nolint start
library(data.table)
library(ggplot2)

# FILE PATHS
INPUT_CSV <- "datastore/derived/issue_20_did_panel.csv"
OUTPUT_DIR <- "output/plots"
OUTPUT_PLOT <- file.path(OUTPUT_DIR, "issue_20_sucker_liar_trajectories.png")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)

    treated_df <- build_treated_df(dt)
    ctrl_df <- build_control_df(dt)

    p <- create_trajectory_plot(treated_df, ctrl_df)
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    ggsave(OUTPUT_PLOT, p, width = 8, height = 6, dpi = 300)
    message("Plot saved to: ", OUTPUT_PLOT)
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
        "is_liar_did_20", "is_liar_did_5",
        "liar_did_sample_20", "liar_did_sample_5"
    )
    for (col in bool_cols) {
        dt[, (col) := as.integer(as.logical(get(col)))]
    }
}

prepare_tau_cols <- function(dt) {
    for (col in c("tau_20", "tau_5", "liar_tau_20", "liar_tau_5")) {
        dt[is.na(get(col)), (col) := 999]
        dt[, (col) := as.integer(get(col))]
    }
}

# =====
# Treated means computation
# =====
build_treated_df <- function(dt) {
    rbind(
        compute_treated_means(dt, "sucker", "20"),
        compute_treated_means(dt, "sucker", "5"),
        compute_treated_means(dt, "liar", "20"),
        compute_treated_means(dt, "liar", "5")
    )
}

compute_treated_means <- function(dt, role, threshold) {
    cols <- get_column_names(role, threshold)
    tau_col <- cols$tau
    dt_sub <- dt[get(cols$sample) == 1 & get(cols$treated) == 1]
    dt_sub <- dt_sub[get(tau_col) != 999]

    agg <- dt_sub[, .(
        mean_contrib = mean(contribution),
        se = sd(contribution) / sqrt(.N)
    ), by = tau_col]

    setnames(agg, tau_col, "tau")
    agg[, role := format_role(role)]
    agg[, threshold := format_threshold(threshold)]
    return(agg)
}

# =====
# Control means computation
# =====
build_control_df <- function(dt) {
    rbind(
        compute_control_mean(dt, "sucker", "20"),
        compute_control_mean(dt, "sucker", "5"),
        compute_control_mean(dt, "liar", "20"),
        compute_control_mean(dt, "liar", "5")
    )
}

compute_control_mean <- function(dt, role, threshold) {
    cols <- get_column_names(role, threshold)
    dt_ctrl <- dt[get(cols$sample) == 1 & get(cols$treated) == 0]

    data.frame(
        grand_mean = mean(dt_ctrl$contribution),
        role = format_role(role),
        threshold = format_threshold(threshold)
    )
}

# =====
# Column name helpers
# =====
get_column_names <- function(role, threshold) {
    if (role == "sucker") {
        list(
            sample = paste0("did_sample_", threshold),
            treated = paste0("got_suckered_", threshold),
            tau = paste0("tau_", threshold)
        )
    } else {
        list(
            sample = paste0("liar_did_sample_", threshold),
            treated = paste0("is_liar_did_", threshold),
            tau = paste0("liar_tau_", threshold)
        )
    }
}

format_role <- function(role) {
    ifelse(role == "sucker", "Sucker", "Liar")
}

format_threshold <- function(threshold) {
    paste0("< ", threshold, " Threshold")
}

# =====
# Plotting
# =====
create_trajectory_plot <- function(treated_df, ctrl_df) {
    treated_df$role <- factor(treated_df$role, levels = c("Sucker", "Liar"))
    ctrl_df$role <- factor(ctrl_df$role, levels = c("Sucker", "Liar"))

    build_trajectory_layers(treated_df, ctrl_df) +
        style_trajectory_plot()
}

build_trajectory_layers <- function(treated_df, ctrl_df) {
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
        facet_grid(role ~ threshold)
}

style_trajectory_plot <- function() {
    list(
        labs(
            x = expression("Event Time (" * tau * ")"),
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
