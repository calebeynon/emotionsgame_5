# Purpose: Shared data prep for Issue #52 — valence of liars and suckers
#          Loads ResultsOnly and Chat emotion data, merges behavior
#          classifications, and computes first-time liar/sucker flags
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
    print_data_summary(dt, "ResultsOnly")
    chat_dt <- load_chat_emotion_data()
    print_data_summary(chat_dt, "Chat")
}

# =====
# Summary printer
# =====
print_data_summary <- function(dt, label) {
    message(sprintf("Loaded %d %s rows (%d with valence data)",
                    nrow(dt), label, sum(!is.na(dt$emotion_valence))))
    message(sprintf("  First-time liars: %d, First-time suckers: %d",
                    sum(dt$first_time_liar, na.rm = TRUE),
                    sum(dt$first_time_sucker, na.rm = TRUE)))
}

# =====
# Private helper: load emotion data filtered by page_type
# =====
load_emotion_data_by_page <- function(target_page, filepath = INPUT_CSV) {
    dt <- fread(filepath)
    dt <- dt[page_type == target_page]
    dt <- merge_behavior_classifications(dt)
    dt <- add_readable_labels(dt)
    dt <- add_first_time_flags(dt)
    dt <- add_first_time_labels(dt)
    # NOTE: zscore_valence() is NOT applied here. Earlier versions z-scored on
    # the raw panel before per-spec filtering (e.g. the pre-decision chat spec
    # drops round 1 after a lag shift), which leaked dropped rows into the
    # denominator (Issue #52 review #15). Specs that need valence_z should
    # call zscore_valence() after their own filtering.
    return(dt)
}

# =====
# Public loaders: ResultsOnly and Chat
# =====
load_results_emotion_data <- function(filepath = INPUT_CSV) {
    return(load_emotion_data_by_page("ResultsOnly", filepath))
}

load_chat_emotion_data <- function(filepath = INPUT_CSV) {
    return(load_emotion_data_by_page("Results", filepath))
}

# =====
# Loads neutral-face baseline from introduction/all_instructions page,
# used as counterfactual channel in the stacked gap DiD
# =====
load_baseline_emotion_data <- function(filepath = INPUT_CSV) {
    dt <- fread(filepath)
    dt <- dt[page_type == "all_instructions"]
    keep_cols <- intersect(c("session_code", "treatment", "label",
                             "emotion_valence"), names(dt))
    dt <- dt[, ..keep_cols]
    dt[, player_id := paste0(session_code, "_", label)]
    dup_check <- dt[, .N, by = .(session_code, label)]
    if (any(dup_check$N != 1)) {
        stop("load_baseline_emotion_data: expected one row per ",
             "(session_code, label) on all_instructions page; found duplicates")
    }
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
# Merge-integrity assertions (Issue #52 review #5)
# Fail loudly when a left join silently drops rows or produces empty output
# =====
assert_merge_integrity <- function(dt_before, dt_after, label,
                                   min_retention = 0.99) {
    n_before <- nrow(dt_before)
    n_after <- nrow(dt_after)
    retention <- n_after / n_before
    if (retention < min_retention) {
        stop(sprintf(
            "%s: merge retained %d/%d rows (%.1f%%), below %.1f%% threshold",
            label, n_after, n_before, 100 * retention,
            100 * min_retention))
    }
    if (n_after == 0) {
        stop(sprintf("%s: merge produced zero rows", label))
    }
}

assert_flag_nonzero <- function(dt, flag_col, label) {
    n_true <- sum(dt[[flag_col]], na.rm = TRUE)
    if (n_true == 0) {
        stop(sprintf(
            "%s: flag %s has zero TRUE rows — likely a merge-key mismatch",
            label, flag_col))
    }
    message(sprintf("%s: %s has %d TRUE / %d total rows",
                    label, flag_col, n_true, nrow(dt)))
}

# =====
# Derive suckered-this-round flag from behavior CSV (shared across specs).
# Also constructs group_segment_round for two-way clustering: suckered is a
# group-level event (one groupmate lying flips the flag for all non-liar
# groupmates), so residuals correlate within the group-round.
# =====
add_suckered_this_round <- function(dt, label = "add_suckered_this_round") {
    bc <- fread(BEHAVIOR_CSV)
    gh <- bc[lied_this_round_20 == TRUE,
             .(groupmate_lied = TRUE),
             by = .(session_code, segment, round, group)]
    dt_before <- dt
    dt <- merge(dt, gh,
                by = c("session_code", "segment", "round", "group"),
                all.x = TRUE)
    assert_merge_integrity(dt_before, dt, label)
    dt[is.na(groupmate_lied), groupmate_lied := FALSE]
    dt[, suckered_this_round := groupmate_lied &
           (contribution == 25) & (lied_this_round_20 == FALSE)]
    dt[, groupmate_lied := NULL]
    dt[, group_segment_round := paste(session_code, segment, round, group,
                                      sep = "_")]
    assert_flag_nonzero(dt, "suckered_this_round", label)
    return(dt)
}

# =====
# Z-score emotion_valence within the filtered subset
# =====
zscore_valence <- function(dt) {
    complete <- dt[!is.na(emotion_valence)]
    if (nrow(complete) == 0) stop("No non-NA valence values in data")
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
