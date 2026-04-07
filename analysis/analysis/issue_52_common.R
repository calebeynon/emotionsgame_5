# Purpose: Shared data prep for Issue #52 — valence of liars and suckers
#          Loads ResultsOnly emotion data, merges behavior classifications,
#          and computes first-time liar/sucker flags
# Author: Claude Code
# Date: 2026-04-06

# nolint start
source("analysis/issue_39_common.R")
# nolint end

# DISPLAY NAMES for emotion columns (maps column name -> pretty label)
EMOTION_DISPLAY_NAMES <- c(
    emotion_anger = "Anger", emotion_contempt = "Contempt",
    emotion_disgust = "Disgust", emotion_fear = "Fear",
    emotion_joy = "Joy", emotion_sadness = "Sadness",
    emotion_surprise = "Surprise", emotion_engagement = "Engagement",
    emotion_valence = "Valence", emotion_sentimentality = "Sentimentality",
    emotion_confusion = "Confusion", emotion_neutral = "Neutral",
    emotion_attention = "Attention"
)

# =====
# Main entry point (demonstrates usage)
# =====
main <- function() {
    dt <- load_results_emotion_data()
    message(sprintf("Loaded %d ResultsOnly rows (%d with valence data)",
                    nrow(dt), sum(!is.na(dt$emotion_valence))))
    message(sprintf("First-time liars: %d, First-time suckers: %d",
                    sum(dt$first_time_liar, na.rm = TRUE),
                    sum(dt$first_time_sucker, na.rm = TRUE)))
}

# =====
# Load and prepare ResultsOnly emotion data
# =====
load_results_emotion_data <- function(filepath = INPUT_CSV) {
    dt <- fread(filepath)
    dt <- dt[page_type == "ResultsOnly"]
    dt <- merge_behavior_classifications(dt)
    dt <- add_readable_labels(dt)
    dt <- add_first_time_flags(dt)
    dt <- add_first_time_labels(dt)
    dt <- zscore_valence(dt)
    return(dt)
}

# =====
# Readable liar/sucker labels
# =====
add_readable_labels <- function(dt) {
    dt[, liar_label := ifelse(is_liar_20 == TRUE, "Liar", "Honest")]
    dt[, sucker_label := ifelse(is_sucker_20 == TRUE, "Sucker", "Non-sucker")]
    return(dt)
}

# =====
# First-time flags: earliest liar/sucker round per player
# =====
add_first_time_flags <- function(dt) {
    dt <- add_first_time_flag(dt, "is_liar_20", "first_time_liar")
    dt <- add_first_time_flag(dt, "is_sucker_20", "first_time_sucker")
    return(dt)
}

add_first_time_flag <- function(dt, source_col, flag_col) {
    # Sort by segment then round to find earliest occurrence
    setorderv(dt, c("session_code", "label", "segment", "round"))
    dt[, (flag_col) := FALSE]

    # Mark first TRUE occurrence per player
    positive <- dt[get(source_col) == TRUE]
    first_idx <- positive[, .I[1],
                          by = .(session_code, label)]$V1
    first_rows <- positive[first_idx]

    # Merge flag back via row matching
    merge_keys <- c("session_code", "label", "segment", "round")
    dt[first_rows, (flag_col) := TRUE, on = merge_keys]
    return(dt)
}

# =====
# Combined first-time labels
# =====
add_first_time_labels <- function(dt) {
    dt[, first_time_liar_label := fifelse(
        is.na(is_liar_20), NA_character_,
        fifelse(first_time_liar, "First-time Liar",
                fifelse(is_liar_20, "Repeat Liar", "Honest"))
    )]
    dt[, first_time_sucker_label := fifelse(
        is.na(is_sucker_20), NA_character_,
        fifelse(first_time_sucker, "First-time Sucker",
                fifelse(is_sucker_20, "Repeat Sucker", "Non-sucker"))
    )]
    return(dt)
}

# =====
# Within-person expanding-window deviation from mean
# =====
within_person_deviation <- function(dt, col) {
    d_col <- paste0(col, "_wpd")
    setorderv(dt, c("session_code", "label", "segment", "round"))
    dt[, (d_col) := compute_expanding_deviation(.SD[[col]]),
       by = .(session_code, label)]
    return(dt)
}

# Helper: expanding-window deviation from prior mean (min k=1)
compute_expanding_deviation <- function(x) {
    n <- length(x)
    d <- rep(NA_real_, n)
    for (t in seq_len(n)) {
        if (t < 2) next
        prior <- x[seq_len(t - 1)]
        prior <- prior[!is.na(prior)]
        if (length(prior) < 1) next
        d[t] <- x[t] - mean(prior)
    }
    return(d)
}

# =====
# Load ResultsOnly data with within-person deviations
# =====
load_results_emotion_data_wp <- function(filepath = INPUT_CSV) {
    dt <- fread(filepath)
    dt <- dt[page_type %in% c("ResultsOnly", "all_instructions")]
    dt <- apply_within_person_deviations(dt)
    dt <- dt[page_type == "ResultsOnly"]
    dt <- merge_behavior_classifications(dt)
    dt <- add_readable_labels(dt)
    dt <- add_first_time_flags(dt)
    dt <- add_first_time_labels(dt)
    return(dt)
}

# Apply within-person deviations to all emotion columns + valence
apply_within_person_deviations <- function(dt) {
    for (col in EMOTION_COLS) {
        dt <- within_person_deviation(dt, col)
    }
    # Also create valence_wpd alias for convenience
    if (!"valence_wpd" %in% names(dt)) {
        dt[, valence_wpd := emotion_valence_wpd]
    }
    return(dt)
}

# =====
# Z-score emotion_valence within ResultsOnly subset
# =====
zscore_valence <- function(dt) {
    complete <- dt[!is.na(emotion_valence)]
    if (nrow(complete) == 0) stop("No non-NA valence values in ResultsOnly data")
    val_mean <- mean(complete$emotion_valence)
    val_sd <- sd(complete$emotion_valence)
    if (val_sd == 0) stop("Zero variance in emotion_valence — cannot z-score")
    dt[, valence_z := (emotion_valence - val_mean) / val_sd]
    return(dt)
}

# =====
# Generic z-score for any emotion column -> {col}_z
# =====
zscore_emotion <- function(dt, col) {
    z_col <- paste0(col, "_z")
    complete <- dt[!is.na(get(col))]
    if (nrow(complete) == 0) stop(sprintf("No non-NA values in %s", col))
    col_mean <- mean(complete[[col]])
    col_sd <- sd(complete[[col]])
    if (col_sd == 0) stop(sprintf("Zero variance in %s — cannot z-score", col))
    dt[, (z_col) := (get(col) - col_mean) / col_sd]
    return(dt)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
