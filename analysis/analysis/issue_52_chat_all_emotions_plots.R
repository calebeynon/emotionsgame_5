# Purpose: Generate dot plots of z-scored emotions during the Chat period
#          by liar/sucker status for all 13 AFFDEX emotions
# Author: Claude Code
# Date: 2026-04-09

# nolint start
source("analysis/issue_52_common.R")
# nolint end

# =====
# Main function
# =====
main <- function() {
    dt <- load_chat_emotion_data()
    results <- lapply(EMOTION_COLS, function(col) {
        generate_emotion_plots(dt, col)
    })
    results_dt <- rbindlist(results)
    print_summary(results_dt)
}

# =====
# Generate all 4 plots for one emotion
# =====
generate_emotion_plots <- function(dt, emotion_col) {
    dt <- zscore_emotion(dt, emotion_col)
    z_col <- paste0(emotion_col, "_z")
    display <- EMOTION_DISPLAY_NAMES[[emotion_col]]
    short_name <- sub("^emotion_", "", emotion_col)
    dt_valid <- dt[!is.na(get(z_col))]

    plot_specs <- build_plot_specs(short_name)
    summaries <- mapply(
        function(filter_col, group_col, filename) {
            save_single_plot(dt_valid, z_col, display,
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
build_plot_specs <- function(short_name) {
    data.table(
        filter = c("is_liar_20", "is_sucker_20",
                    "is_liar_20", "is_sucker_20"),
        group  = c("liar_label", "sucker_label",
                    "first_time_liar_label", "first_time_sucker_label"),
        file   = c(
            sprintf("chat_%s_by_liar_status.png", short_name),
            sprintf("chat_%s_by_sucker_status.png", short_name),
            sprintf("chat_%s_by_first_time_liar.png", short_name),
            sprintf("chat_%s_by_first_time_sucker.png", short_name)
        )
    )
}

# =====
# Save one plot and return summary row
# =====
save_single_plot <- function(dt, z_col, display,
                             filter_col, group_col, filename) {
    dt_sub <- dt[!is.na(get(filter_col))]
    p <- build_emotion_dotplot(dt_sub, z_col, group_col, display)
    save_plot(p, filename)
    summarize_emotion(dt_sub, z_col, group_col, display, filename)
}

# =====
# Summary statistics by group (parameterized z column)
# =====
summarize_emotion <- function(dt, z_col, group_col,
                              display, filename) {
    s <- dt[, .(mean = mean(get(z_col)),
                se = sd(get(z_col)) / sqrt(.N),
                n = .N), by = group_col]
    setnames(s, group_col, "group")
    s[, lower := mean - 1.96 * se]
    s[, upper := mean + 1.96 * se]
    sig <- any(s$lower > 0 | s$upper < 0)
    data.table(emotion = display, plot = filename, significant = sig)
}

# =====
# Build horizontal dot plot with 95% CIs (parameterized)
# =====
build_emotion_dotplot <- function(dt, z_col, group_col, display) {
    s <- dt[, .(mean = mean(get(z_col)),
                se = sd(get(z_col)) / sqrt(.N),
                n = .N), by = group_col]
    setnames(s, group_col, "group")
    s[, lower := mean - 1.96 * se]
    s[, upper := mean + 1.96 * se]
    s[, group_n := paste0(group, " (n=", n, ")")]

    ggplot(s, aes(x = mean, y = group_n)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.15,
                       color = VALENCE_COLOR) +
        geom_point(size = 2.5, color = VALENCE_COLOR) +
        labs(x = paste0(display, " (Z-Score)"), y = NULL) +
        PLOT_THEME
}

# =====
# Print final summary
# =====
print_summary <- function(results_dt) {
    n_plots <- nrow(results_dt)
    n_emotions <- length(unique(results_dt$emotion))
    sig_rows <- results_dt[significant == TRUE]

    message(sprintf("\n=== Summary: %d plots generated for %d emotions ===",
                    n_plots, n_emotions))
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

# COLORS (matching valence plots)
VALENCE_COLOR <- "#2166AC"

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
