# Purpose: Dot plots comparing z-scored anger, fear, and sentiment compound
#          across player types (cooperative state, liar status, sucker status)
# Author: Claude Code
# Date: 2026-03-17

# nolint start
source("analysis/issue_39_common.R")
# nolint end

# OUTPUT FILES (saved to _sandbox_data/ for review)
SANDBOX_DIR <- "_sandbox_data"
SUCKER_PLOT <- "negative_emotion_by_sucker_status.png"
LIAR_PLOT <- "negative_emotion_by_liar_status.png"
STATE_PLOT <- "negative_emotion_by_cooperative_state.png"
LIAR_STATE_PLOT <- "negative_emotion_by_liar_x_state.png"
LIAR_ROUND_PLOT <- "negative_emotion_by_liar_round_status.png"
LIAR_ROUND_STATE_PLOT <- "negative_emotion_by_liar_round_x_state.png"

# COLORS
ANGER_COLOR <- "#B2182B"
FEAR_COLOR <- "#E66101"
SENTIMENT_COLOR <- "#2166AC"

# =====
# Main function
# =====
main <- function() {
    dt <- prepare_plot_data()
    save_plot(build_dotplot(dt, "state_label", "Anger, Fear & Sentiment by Cooperative State"), STATE_PLOT, subdir = SANDBOX_DIR)
    save_plot(build_dotplot(dt[!is.na(is_liar_20)], "liar_label", "Anger, Fear & Sentiment by Liar Status"), LIAR_PLOT, subdir = SANDBOX_DIR)
    save_plot(build_dotplot(dt[!is.na(is_sucker_20)], "sucker_label", "Anger, Fear & Sentiment by Sucker Status"), SUCKER_PLOT, subdir = SANDBOX_DIR)
    liar_state_dt <- dt[!is.na(is_liar_20)]
    liar_state_dt[, liar_state := paste(liar_label, state_label, sep = " / ")]
    save_plot(build_dotplot(liar_state_dt, "liar_state", "Anger, Fear & Sentiment by Liar Status x Cooperative State"), LIAR_STATE_PLOT, subdir = SANDBOX_DIR)

    # Round-specific liar plots
    round_dt <- dt[!is.na(lied_this_round_20)]
    save_plot(build_dotplot(round_dt, "liar_round_label", "Anger, Fear & Sentiment by Round Liar Status"), LIAR_ROUND_PLOT, subdir = SANDBOX_DIR)
    round_dt[, liar_round_state := paste(liar_round_label, state_label, sep = " / ")]
    save_plot(build_dotplot(round_dt, "liar_round_state", "Anger, Fear & Sentiment by Round Liar x Cooperative State"), LIAR_ROUND_STATE_PLOT, subdir = SANDBOX_DIR)
}

# =====
# Data preparation
# =====
prepare_plot_data <- function() {
    dt <- load_contribute_data()
    dt <- merge_behavior_classifications(dt)
    dt <- compute_zscores(dt)
    dt <- compute_negative_emotion_zscores(dt)
    dt <- dt[!is.na(anger_z) & !is.na(fear_z) & !is.na(compound_z)]
    dt[, state_label := ifelse(player_state == "cooperative", "Cooperative", "Noncooperative")]
    dt[, liar_label := ifelse(is_liar_20 == TRUE, "Liar", "Honest")]
    dt[, sucker_label := ifelse(is_sucker_20 == TRUE, "Sucker", "Non-sucker")]
    dt <- add_liar_round_label(dt)
    return(dt)
}

compute_negative_emotion_zscores <- function(dt) {
    complete <- dt[!is.na(emotion_anger) & !is.na(emotion_fear)]
    anger_mean <- mean(complete$emotion_anger)
    anger_sd <- sd(complete$emotion_anger)
    fear_mean <- mean(complete$emotion_fear)
    fear_sd <- sd(complete$emotion_fear)
    if (anger_sd == 0) stop("Zero variance in emotion_anger -- cannot z-score")
    if (fear_sd == 0) stop("Zero variance in emotion_fear -- cannot z-score")
    dt[, anger_z := (emotion_anger - anger_mean) / anger_sd]
    dt[, fear_z := (emotion_fear - fear_mean) / fear_sd]
    return(dt)
}

# =====
# Summary statistics by group
# =====
summarize_by_group <- function(dt, group_col) {
    out <- rbindlist(list(
        summarize_measure(dt, group_col, "anger_z", "Anger"),
        summarize_measure(dt, group_col, "fear_z", "Fear"),
        summarize_measure(dt, group_col, "compound_z", "Sentiment (Compound)")
    ))
    setnames(out, group_col, "group")
    out[, lower := mean - 1.96 * se]
    out[, upper := mean + 1.96 * se]
    return(out)
}

summarize_measure <- function(dt, group_col, value_col, label) {
    s <- dt[, .(mean = mean(get(value_col)), se = sd(get(value_col)) / sqrt(.N), n = .N), by = group_col]
    s[, measure := label]
    return(s)
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
            position = position_dodge(width = 0.5), height = 0.15
        ) +
        geom_point(
            position = position_dodge(width = 0.5), size = 2.5
        ) +
        scale_color_manual(values = c(
            "Anger" = ANGER_COLOR,
            "Fear" = FEAR_COLOR,
            "Sentiment (Compound)" = SENTIMENT_COLOR
        )) +
        labs(x = "Z-Score", y = NULL, title = title, color = NULL) +
        PLOT_THEME + theme(legend.position = "bottom")
}

add_n_labels <- function(summary_dt) {
    n_labels <- summary_dt[measure == "Anger", paste0(group, " (n=", n, ")")]
    name_map <- setNames(n_labels, summary_dt[measure == "Anger", group])
    summary_dt[, group_n := name_map[group]]
    return(summary_dt)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
