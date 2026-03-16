# Purpose: Dot plots comparing z-scored emotion valence and sentiment compound
#          across player types (cooperative state, liar status, sucker status)
# Author: Claude Code
# Date: 2026-03-14

# nolint start
source("analysis/issue_39_common.R")
# nolint end

# OUTPUT FILES
STATE_PLOT <- "emotion_sentiment_gap_by_cooperative_state.png"
LIAR_PLOT <- "emotion_sentiment_gap_by_liar_status.png"
SUCKER_PLOT <- "emotion_sentiment_gap_by_sucker_status.png"
LIAR_STATE_PLOT <- "emotion_sentiment_gap_by_liar_x_state.png"

# COLORS
EMOTION_COLOR <- "#2166AC"
SENTIMENT_COLOR <- "#B2182B"

# =====
# Main function
# =====
main <- function() {
    dt <- prepare_plot_data()
    save_plot(build_dotplot(dt, "state_label", "Emotion vs Sentiment by Cooperative State"), STATE_PLOT)
    save_plot(build_dotplot(dt[!is.na(is_liar_20)], "liar_label", "Emotion vs Sentiment by Liar Status"), LIAR_PLOT)
    save_plot(build_dotplot(dt[!is.na(is_sucker_20)], "sucker_label", "Emotion vs Sentiment by Sucker Status"), SUCKER_PLOT)
    liar_state_dt <- dt[!is.na(is_liar_20)]
    liar_state_dt[, liar_state := paste(liar_label, state_label, sep = " / ")]
    save_plot(build_dotplot(liar_state_dt, "liar_state", "Emotion vs Sentiment by Liar Status x Cooperative State"), LIAR_STATE_PLOT)
}

# =====
# Data preparation
# =====
prepare_plot_data <- function() {
    dt <- load_contribute_data()
    dt <- merge_behavior_classifications(dt)
    dt <- compute_zscores(dt)
    dt <- dt[!is.na(valence_z) & !is.na(compound_z)]

    dt[, state_label := ifelse(
        player_state == "cooperative", "Cooperative", "Noncooperative"
    )]
    dt[, liar_label := ifelse(is_liar_20 == TRUE, "Liar", "Honest")]
    dt[, sucker_label := ifelse(is_sucker_20 == TRUE, "Sucker", "Non-sucker")]
    return(dt)
}

# =====
# Summary statistics by group
# =====
summarize_by_group <- function(dt, group_col) {
    val_summary <- dt[, .(
        mean = mean(valence_z), se = sd(valence_z) / sqrt(.N), n = .N
    ), by = group_col]
    val_summary[, measure := "Emotion (Valence)"]

    cmp_summary <- dt[, .(
        mean = mean(compound_z), se = sd(compound_z) / sqrt(.N), n = .N
    ), by = group_col]
    cmp_summary[, measure := "Sentiment (Compound)"]

    out <- rbindlist(list(val_summary, cmp_summary))
    setnames(out, group_col, "group")
    out[, lower := mean - 1.96 * se]
    out[, upper := mean + 1.96 * se]
    return(out)
}

# =====
# Dot plot construction
# =====
build_dotplot <- function(dt, group_col, title) {
    summary_dt <- add_n_labels(summarize_by_group(dt, group_col))
    ggplot(summary_dt, aes(x = mean, y = group_n, color = measure)) +
        geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
        geom_errorbarh(
            aes(xmin = lower, xmax = upper),
            position = position_dodge(width = 0.4), height = 0.15
        ) +
        geom_point(
            position = position_dodge(width = 0.4), size = 2.5
        ) +
        scale_color_manual(values = c(
            "Emotion (Valence)" = EMOTION_COLOR,
            "Sentiment (Compound)" = SENTIMENT_COLOR
        )) +
        labs(x = "Z-Score", y = NULL, title = title, color = NULL) +
        PLOT_THEME + theme(legend.position = "bottom")
}

add_n_labels <- function(summary_dt) {
    n_labels <- summary_dt[measure == "Emotion (Valence)", paste0(group, " (n=", n, ")")]
    name_map <- setNames(n_labels, summary_dt[measure == "Emotion (Valence)", group])
    summary_dt[, group_n := name_map[group]]
    return(summary_dt)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
