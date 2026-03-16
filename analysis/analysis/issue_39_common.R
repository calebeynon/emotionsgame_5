# Purpose: Shared utilities for Issue #39 emotion-sentiment analysis scripts
# Author: Claude Code
# Date: 2026-03-14

library(data.table)
library(ggplot2)
library(scales)
library(RColorBrewer)

# FILE PATHS
INPUT_CSV <- "datastore/derived/merged_panel.csv"
BEHAVIOR_CSV <- "datastore/derived/behavior_classifications.csv"
PLOT_DIR <- file.path("output", "plots")
TABLE_DIR <- file.path("output", "tables")

# COLUMN GROUPS - all emotion columns from merged_panel.csv
EMOTION_COLS <- c(
    "emotion_anger", "emotion_contempt", "emotion_disgust", "emotion_fear",
    "emotion_joy", "emotion_sadness", "emotion_surprise", "emotion_engagement",
    "emotion_valence", "emotion_sentimentality", "emotion_confusion",
    "emotion_neutral", "emotion_attention"
)

# All sentiment columns from merged_panel.csv
SENTIMENT_COLS <- c(
    "sentiment_compound_mean", "sentiment_compound_std",
    "sentiment_compound_min", "sentiment_compound_max",
    "sentiment_positive_mean", "sentiment_negative_mean",
    "sentiment_neutral_mean"
)

# Focused subsets for core analysis
CORE_EMOTIONS <- c(
    "emotion_joy", "emotion_anger", "emotion_sadness", "emotion_fear",
    "emotion_surprise", "emotion_valence", "emotion_engagement"
)

CORE_SENTIMENT <- c(
    "sentiment_compound_mean", "sentiment_positive_mean",
    "sentiment_negative_mean"
)

# SHARED THEME
PLOT_THEME <- theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank())

# =====
# Data loading
# =====
load_contribute_data <- function(filepath = INPUT_CSV) {
    dt <- fread(filepath)
    dt <- dt[page_type == "Contribute"]

    # Cluster ID for multi-way clustering in regressions
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]

    # Convert logical/string boolean to 0/1
    dt[, made_promise := as.integer(as.logical(made_promise))]

    return(dt)
}

# =====
# Plot saving
# =====
save_plot <- function(plot, filename, subdir = PLOT_DIR,
                      width = 6.5, height = 4) {
    dir.create(subdir, recursive = TRUE, showWarnings = FALSE)
    outpath <- file.path(subdir, filename)
    ggsave(outpath, plot = plot,
           width = width, height = height,
           units = "in", dpi = 300)
    message("Saved: ", outpath)
}

# =====
# Behavior classification merge
# =====
merge_behavior_classifications <- function(dt) {
    bc <- fread(BEHAVIOR_CSV)
    merge_keys <- c("session_code", "segment", "round", "group", "label")
    merge_cols <- c(merge_keys, "is_liar_20", "is_sucker_20")
    bc_subset <- bc[, ..merge_cols]
    merged <- merge(dt, bc_subset, by = merge_keys, all.x = TRUE)
    na_count <- sum(is.na(merged$is_liar_20))
    if (na_count > 0) {
        message(sprintf("Note: %d of %d rows unmatched in behavior merge (NA values)",
                        na_count, nrow(merged)))
    }
    return(merged)
}

# =====
# Z-score computation
# =====
compute_zscores <- function(dt) {
    complete <- dt[!is.na(emotion_valence) & !is.na(sentiment_compound_mean)]
    val_mean <- mean(complete$emotion_valence)
    val_sd <- sd(complete$emotion_valence)
    cmp_mean <- mean(complete$sentiment_compound_mean)
    cmp_sd <- sd(complete$sentiment_compound_mean)
    if (val_sd == 0) stop("Zero variance in emotion_valence — cannot z-score")
    if (cmp_sd == 0) stop("Zero variance in sentiment_compound_mean — cannot z-score")
    dt[, valence_z := (emotion_valence - val_mean) / val_sd]
    dt[, compound_z := (sentiment_compound_mean - cmp_mean) / cmp_sd]
    dt[, zscore_gap := valence_z - compound_z]
    return(dt)
}
