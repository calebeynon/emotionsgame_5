# Purpose: Dot plots of z-scored valence during the Chat period by liar/sucker
#          status, including first-time vs repeat breakdowns
# Author: Claude Code
# Date: 2026-04-09

# nolint start
source("analysis/issue_52_common.R")
# nolint end

# COLORS
VALENCE_COLOR <- "#2166AC"

# OUTPUT FILES
LIAR_PLOT <- "chat_valence_by_liar_status.png"
SUCKER_PLOT <- "chat_valence_by_sucker_status.png"
FIRST_LIAR_PLOT <- "chat_valence_by_first_time_liar.png"
FIRST_SUCKER_PLOT <- "chat_valence_by_first_time_sucker.png"

# =====
# Main function
# =====
main <- function() {
    dt <- load_chat_emotion_data()
    dt <- dt[!is.na(valence_z)]

    save_plot(build_valence_dotplot(dt[!is.na(is_liar_20)], "liar_label"),
              LIAR_PLOT)
    save_plot(build_valence_dotplot(dt[!is.na(is_sucker_20)], "sucker_label"),
              SUCKER_PLOT)
    save_plot(build_valence_dotplot(dt[!is.na(is_liar_20)],
                                   "first_time_liar_label"),
              FIRST_LIAR_PLOT)
    save_plot(build_valence_dotplot(dt[!is.na(is_sucker_20)],
                                   "first_time_sucker_label"),
              FIRST_SUCKER_PLOT)
}

# =====
# Summary statistics by group
# =====
summarize_valence <- function(dt, group_col) {
    summary_dt <- dt[, .(
        mean = mean(valence_z),
        se = sd(valence_z) / sqrt(.N),
        n = .N
    ), by = group_col]
    setnames(summary_dt, group_col, "group")
    summary_dt[, lower := mean - 1.96 * se]
    summary_dt[, upper := mean + 1.96 * se]
    return(summary_dt)
}

# =====
# Append sample size to group labels
# =====
add_n_labels <- function(summary_dt) {
    summary_dt[, group_n := paste0(group, " (n=", n, ")")]
    return(summary_dt)
}

# =====
# Build horizontal dot plot with 95% CIs
# =====
build_valence_dotplot <- function(dt, group_col) {
    summary_dt <- add_n_labels(summarize_valence(dt, group_col))
    ggplot(summary_dt, aes(x = mean, y = group_n)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.15,
                       color = VALENCE_COLOR) +
        geom_point(size = 2.5, color = VALENCE_COLOR) +
        labs(x = "Valence (Z-Score)", y = NULL) +
        PLOT_THEME
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
