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
# Main function — estimates both specs, exports combined table
# =====
main <- function() {
    results_models <- estimate_gap_models(prepare_results_stacked_data())
    chat_models <- estimate_gap_models(prepare_chat_stacked_data())
    export_combined_table(results_models, chat_models)
    print_gap_summary(results_models, "results-page")
    print_gap_summary(chat_models, "chat-period")
}

# =====
# Prepare stacked data: results-page face (no shift needed)
# =====
prepare_results_stacked_data <- function() {
    dt <- load_results_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt <- dt[!is.na(emotion_valence) & !is.na(sentiment_compound_mean)]
    message(sprintf("Results-page complete cases: %d", nrow(dt)))
    return(stack_channels(dt, face_col = "emotion_valence"))
}

# =====
# Prepare stacked data: chat-period face (shifted by 1 round)
# =====
prepare_chat_stacked_data <- function() {
    dt <- load_chat_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt <- shift_valence_to_influenced_round(dt)
    dt <- dt[!is.na(valence_shifted) & !is.na(sentiment_compound_mean)]
    message(sprintf("Chat-period complete cases: %d", nrow(dt)))
    return(stack_channels(dt, face_col = "valence_shifted"))
}

# =====
# Shift valence to the round it influenced (lag by 1 within player-segment)
# =====
shift_valence_to_influenced_round <- function(dt) {
    setorderv(dt, c("session_code", "label", "segment", "round"))
    dt[, valence_shifted := shift(emotion_valence, n = 1, type = "lag"),
       by = .(session_code, label, segment)]
    n_dropped <- sum(is.na(dt$valence_shifted) & !is.na(dt$emotion_valence))
    message(sprintf("Valence shifted: %d round-1 rows become NA", n_dropped))
    return(dt)
}

# =====
# Stack face and chat channels into long format
# =====
stack_channels <- function(dt, face_col) {
    id_cols <- c("session_code", "segment", "round", "group",
                 "label", "player_id", "contribution",
                 "lied_this_round_20", "suckered_this_round")
    keep_face <- c(id_cols, face_col)
    face <- dt[, ..keep_face]
    face[, `:=`(Y = get(face_col), channel = "face")]
    face[, (face_col) := NULL]

    keep_chat <- c(id_cols, "sentiment_compound_mean")
    chat <- dt[, ..keep_chat]
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
# Export combined LaTeX table (4 columns: results-page + chat-period)
# =====
export_combined_table <- function(results_models, chat_models) {
    dir.create(dirname(GAP_TABLE_FILE), recursive = TRUE,
               showWarnings = FALSE)
    coef_names <- build_coef_dict()
    etable(results_models$lied_gap, results_models$suckered_gap,
           chat_models$lied_gap, chat_models$suckered_gap,
           headers = list("Results Page Face" = 2, "Chat Period Face" = 2),
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
print_gap_summary <- function(models, label = "") {
    message(sprintf("--- %s ---", label))
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
