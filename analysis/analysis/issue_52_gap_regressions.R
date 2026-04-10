# Purpose: Stacked valence-sentiment gap regressions — tests whether the
#          face-words gap shifts when a player lies or gets suckered
# Author: Claude Code
# Date: 2026-04-10

# nolint start
source("analysis/issue_52_common.R")
# nolint end

library(fixest)

# OUTPUT
GAP_TABLE_FILE <- file.path(TABLE_DIR, "issue_52_valence_sentiment_gap_regressions.tex")

# =====
# Main function
# =====
main <- function() {
    stacked <- prepare_stacked_data()
    models <- estimate_gap_models(stacked)
    export_gap_table(models)
    print_gap_summary(models)
}

# =====
# Prepare stacked data: two rows per player-round (face + chat)
# =====
prepare_stacked_data <- function() {
    dt <- load_results_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt <- keep_complete_cases(dt)
    stacked <- stack_channels(dt)
    return(stacked)
}

# =====
# Keep only rows with both valence and sentiment
# =====
keep_complete_cases <- function(dt) {
    dt <- dt[!is.na(emotion_valence) & !is.na(sentiment_compound_mean)]
    message(sprintf("Complete cases (both valence & sentiment): %d", nrow(dt)))
    return(dt)
}

# =====
# Stack face and chat channels into long format
# =====
stack_channels <- function(dt) {
    id_cols <- c("session_code", "segment", "round", "group",
                 "label", "player_id", "contribution",
                 "lied_this_round_20", "suckered_this_round")
    face <- dt[, c(..id_cols, "emotion_valence")]
    face[, `:=`(Y = emotion_valence, channel = "face")]
    face[, emotion_valence := NULL]

    chat <- dt[, c(..id_cols, "sentiment_compound_mean")]
    chat[, `:=`(Y = sentiment_compound_mean, channel = "chat")]
    chat[, sentiment_compound_mean := NULL]

    stacked <- rbind(face, chat)
    stacked[, channel := factor(channel, levels = c("chat", "face"))]
    stacked[, segment := factor(segment)]
    stacked[, round := factor(round)]
    message(sprintf("Stacked rows: %d (face + chat)", nrow(stacked)))
    return(stacked)
}

# =====
# Derive suckered-this-round from round-level conditions
# =====
add_suckered_this_round <- function(dt) {
    bc <- fread(BEHAVIOR_CSV)
    group_has_liar <- bc[lied_this_round_20 == TRUE,
                         .(groupmate_lied = TRUE),
                         by = .(session_code, segment, round, group)]
    dt <- merge(dt, group_has_liar,
                by = c("session_code", "segment", "round", "group"),
                all.x = TRUE)
    dt[is.na(groupmate_lied), groupmate_lied := FALSE]
    dt[, suckered_this_round := groupmate_lied &
           (contribution == 25) & (lied_this_round_20 == FALSE)]
    dt[, groupmate_lied := NULL]
    return(dt)
}

# =====
# Estimate gap models: channel x flag interactions
# =====
estimate_gap_models <- function(stacked) {
    list(
        lied_gap = feols(
            Y ~ lied_this_round_20 +
                i(channel, lied_this_round_20, ref = "chat") + round |
                segment + player_id + channel,
            cluster = ~player_id, data = stacked),
        suckered_gap = feols(
            Y ~ suckered_this_round +
                i(channel, suckered_this_round, ref = "chat") + round |
                segment + player_id + channel,
            cluster = ~player_id, data = stacked)
    )
}

# =====
# Export LaTeX table
# =====
export_gap_table <- function(models) {
    dir.create(dirname(GAP_TABLE_FILE), recursive = TRUE,
               showWarnings = FALSE)
    coef_names <- build_coef_dict()
    etable(models$lied_gap, models$suckered_gap,
           headers = c("Lied (Face-Chat Gap)",
                       "Suckered (Face-Chat Gap)"),
           se.below = TRUE, dict = coef_names,
           signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
           tex = TRUE, file = GAP_TABLE_FILE)
    message(sprintf("Table saved: %s", GAP_TABLE_FILE))
}

# =====
# Readable coefficient names for etable
# =====
build_coef_dict <- function() {
    lied_i <- 'i(factor_var = channel, var = lied_this_round_20, ref = "chat")'
    suck_i <- 'i(factor_var = channel, var = suckered_this_round, ref = "chat")'
    dict <- c("Lied This Round", "Suckered This Round",
              "Face $\\times$ Lied", "Face $\\times$ Suckered")
    names(dict) <- c("lied_this_round_20TRUE",
                     "suckered_this_roundTRUE", lied_i, suck_i)
    return(dict)
}

# =====
# Print coefficient summary (interaction terms only)
# =====
print_gap_summary <- function(models) {
    for (name in names(models)) {
        m <- models[[name]]
        ct <- coeftable(m)
        for (r in seq_len(nrow(ct))) {
            rn <- rownames(ct)[r]
            if (grepl("^round", rn)) next
            message(sprintf("  %s | %s: coef=%.4f se=%.4f p=%.4f",
                            name, rn, ct[r, 1], ct[r, 2], ct[r, 4]))
        }
        message(sprintf("  %s n=%d", name, m$nobs))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
