# Purpose: Generate dot plots of within-person deviation emotions on the
#          Results page by liar/sucker status for all 13 AFFDEX emotions
# Author: Claude Code
# Date: 2026-04-07

# nolint start
source("analysis/issue_52_common.R")
# nolint end

# OUTPUT DIRECTORY
WP_PLOT_DIR <- file.path(PLOT_DIR, "within_person")

# COLORS (matching population deviation plots)
WP_COLOR <- "#2166AC"

# =====
# Main function
# =====
main <- function() {
    dt <- load_results_emotion_data_wp()
    results <- lapply(EMOTION_COLS, function(col) {
        generate_wp_emotion_plots(dt, col)
    })
    results_dt <- rbindlist(results)
    print_wp_summary(results_dt)
}

# =====
# Generate all 4 plots for one emotion (within-person deviation)
# =====
generate_wp_emotion_plots <- function(dt, emotion_col) {
    d_col <- paste0(emotion_col, "_wpd")
    display <- EMOTION_DISPLAY_NAMES[[emotion_col]]
    short_name <- sub("^emotion_", "", emotion_col)
    dt_valid <- dt[!is.na(get(d_col))]

    plot_specs <- build_wp_plot_specs(short_name)
    summaries <- mapply(
        function(filter_col, group_col, filename) {
            save_wp_plot(dt_valid, d_col, display,
                         filter_col, group_col, filename)
        },
        plot_specs$filter, plot_specs$group,
        plot_specs$file, SIMPLIFY = FALSE
    )
    rbindlist(summaries)
}

# =====
# Define the 4 plot specifications per emotion
# =====
build_wp_plot_specs <- function(short_name) {
    data.table(
        filter = c("is_liar_20", "is_sucker_20",
                    "is_liar_20", "is_sucker_20"),
        group  = c("liar_label", "sucker_label",
                    "first_time_liar_label", "first_time_sucker_label"),
        file   = c(
            sprintf("results_%s_by_liar_status.png", short_name),
            sprintf("results_%s_by_sucker_status.png", short_name),
            sprintf("results_%s_by_first_time_liar.png", short_name),
            sprintf("results_%s_by_first_time_sucker.png", short_name)
        )
    )
}

# =====
# Save one plot and return summary row
# =====
save_wp_plot <- function(dt, d_col, display,
                         filter_col, group_col, filename) {
    dt_sub <- dt[!is.na(get(filter_col))]
    p <- build_wp_dotplot(dt_sub, d_col, group_col, display)
    save_plot(p, filename, subdir = WP_PLOT_DIR)
    summarize_wp_emotion(dt_sub, d_col, group_col, display, filename)
}

# =====
# Summary statistics by group
# =====
summarize_wp_emotion <- function(dt, d_col, group_col,
                                 display, filename) {
    s <- dt[, .(mean = mean(get(d_col)),
                se = sd(get(d_col)) / sqrt(.N),
                n = .N), by = group_col]
    setnames(s, group_col, "group")
    s[, lower := mean - 1.96 * se]
    s[, upper := mean + 1.96 * se]
    sig <- any(s$lower > 0 | s$upper < 0, na.rm = TRUE)
    data.table(emotion = display, plot = filename, significant = sig)
}

# =====
# Build horizontal dot plot with 95% CIs (within-person deviation)
# =====
build_wp_dotplot <- function(dt, d_col, group_col, display) {
    s <- dt[, .(mean = mean(get(d_col)),
                se = sd(get(d_col)) / sqrt(.N),
                n = .N), by = group_col]
    setnames(s, group_col, "group")
    s[, lower := mean - 1.96 * se]
    s[, upper := mean + 1.96 * se]
    s[, group_n := paste0(group, " (n=", n, ")")]

    ggplot(s, aes(x = mean, y = group_n)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.15,
                       color = WP_COLOR) +
        geom_point(size = 2.5, color = WP_COLOR) +
        labs(x = paste0(display, " (Within-Person Deviation)"), y = NULL) +
        PLOT_THEME
}

# =====
# Print final summary
# =====
print_wp_summary <- function(results_dt) {
    n_plots <- nrow(results_dt)
    n_emotions <- length(unique(results_dt$emotion))
    sig_rows <- results_dt[significant == TRUE]

    message(sprintf("\n=== Summary: %d plots generated for %d emotions ===",
                    n_plots, n_emotions))
    message(sprintf("Output directory: %s", WP_PLOT_DIR))
    if (nrow(sig_rows) > 0) {
        message("Plots with CI excluding zero:")
        for (i in seq_len(nrow(sig_rows))) {
            message(sprintf("  - %s: %s", sig_rows$emotion[i],
                            sig_rows$plot[i]))
        }
    } else {
        message("No plots had group CIs excluding zero.")
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
