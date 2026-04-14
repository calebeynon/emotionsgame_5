# Purpose: Facial-valence regressions for liars and suckers. Two face
#          windows: ResultsOnly (post-outcome) and pre-decision chat
#          (Results/chat page lagged 1 round).
# Author: Claude Code
# Date: 2026-04-13
library(fixest)
# nolint start
source("analysis/issue_52_common.R")
# nolint end

# OUTPUT
GAP_TABLE_FILE <- file.path(
    TABLE_DIR, "issue_52_valence_sentiment_gap_regressions.tex")

# =====
# Main function — estimates both specs, exports combined table
# =====
main <- function() {
    res <- estimate_face_models(prepare_results_face_data(),
                                "emotion_valence")
    pre <- estimate_face_models(prepare_pre_decision_chat_face_data(),
                                "valence_shifted")
    export_combined_table(res, pre)
    print_summary(res, "Results Page Face")
    print_summary(pre, "Pre-Decision Chat Face")
}

# =====
# Results-page face rows (AFFDEX ResultsOnly page)
# =====
prepare_results_face_data <- function() {
    dt <- load_results_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt <- dt[!is.na(emotion_valence)]
    message(sprintf("Results-page face rows: %d", nrow(dt)))
    return(dt)
}

# =====
# Pre-decision chat face rows (AFFDEX Results/chat page, lagged 1 round
# within player-segment so chat at end of round t-1 aligns with round t)
# =====
prepare_pre_decision_chat_face_data <- function() {
    dt <- load_chat_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    setorderv(dt, c("session_code", "label", "segment", "round"))
    dt[, valence_shifted := shift(emotion_valence, n = 1, type = "lag"),
       by = .(session_code, label, segment)]
    dt <- dt[!is.na(valence_shifted)]
    message(sprintf("Pre-decision chat face rows: %d", nrow(dt)))
    return(dt)
}

# =====
# Derive suckered-this-round from behavior CSV round-level conditions
# =====
add_suckered_this_round <- function(dt) {
    bc <- fread(BEHAVIOR_CSV)
    gh <- bc[lied_this_round_20 == TRUE,
             .(groupmate_lied = TRUE),
             by = .(session_code, segment, round, group)]
    dt <- merge(dt, gh,
                by = c("session_code", "segment", "round", "group"),
                all.x = TRUE)
    dt[is.na(groupmate_lied), groupmate_lied := FALSE]
    dt[, suckered_this_round := groupmate_lied &
           (contribution == 25) & (lied_this_round_20 == FALSE)]
    dt[, groupmate_lied := NULL]
    return(dt)
}

# =====
# Estimate lied and suckered models on a face-only panel
# =====
estimate_face_models <- function(dt, y_col) {
    dt <- copy(dt)
    dt[, round := factor(round)]
    fml_lied <- as.formula(paste0(
        y_col, " ~ lied_this_round_20 + i(round)",
        " | segment + player_id"))
    fml_suck <- as.formula(paste0(
        y_col, " ~ suckered_this_round + i(round)",
        " | segment + player_id"))
    list(
        lied = feols(fml_lied, cluster = ~player_id, data = dt),
        suckered = feols(fml_suck, cluster = ~player_id, data = dt)
    )
}

# =====
# Export 4-column LaTeX table (Results Page + Pre-Decision Chat)
# =====
export_combined_table <- function(res, pre) {
    dir.create(dirname(GAP_TABLE_FILE), recursive = TRUE,
               showWarnings = FALSE)
    etable(res$lied, res$suckered, pre$lied, pre$suckered,
           headers = list("Results Page Face" = 2,
                          "Pre-Decision Chat Face" = 2),
           se.below = TRUE,
           dict = build_coef_dict(),
           order = c("Lied", "Suckered", "round"),
           signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
           notes = build_table_note(), tex = TRUE, file = GAP_TABLE_FILE)
    message(sprintf("Saved: %s", GAP_TABLE_FILE))
}

# =====
# Caption note
# =====
build_coef_dict <- function() {
    c(lied_this_round_20TRUE = "Lied",
      suckered_this_roundTRUE = "Suckered",
      emotion_valence = "Emotion Valence",
      valence_shifted = "Emotion Valence")
}

build_table_note <- function() {
    paste("Dependent variable: AFFDEX facial valence. Results Page Face",
          "columns measure valence on the post-outcome ResultsOnly page.",
          "Pre-Decision Chat Face columns measure valence on the Results",
          "chat page at the end of round t-1, shifted forward to round",
          "t (the contribution it influenced). Lied = 1 if a player's",
          "contribution falls at least 20 tokens below their most recent",
          "chat promise; Suckered = 1 if a groupmate lied, the player",
          "contributed the full 25-token endowment, and did not lie.",
          "The same regression was also estimated for the 12 other",
          "AFFDEX emotions (anger, contempt, disgust, fear, joy, sadness,",
          "surprise, engagement, sentimentality, confusion, neutral,",
          "attention); headline results for joy and engagement appear",
          "in the appendix.")
}

# =====
# Print coefficient summary (non-round terms only)
# =====
print_summary <- function(models, label) {
    message(sprintf("--- %s ---", label))
    for (name in names(models)) {
        ct <- coeftable(models[[name]])
        for (r in seq_len(nrow(ct))) {
            rn <- rownames(ct)[r]
            if (grepl("^round", rn)) next
            message(sprintf("  %s | %s: coef=%.4f se=%.4f p=%.4f",
                            name, rn, ct[r, 1], ct[r, 2], ct[r, 4]))
        }
        message(sprintf("  %s n=%d", name, models[[name]]$nobs))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
