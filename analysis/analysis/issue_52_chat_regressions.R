# Purpose: Round-specific liar/sucker emotion regressions on the Chat page
#          Player + segment FE, round controls, clustered SEs by player
# Author: Claude Code
# Date: 2026-04-09

library(fixest)

# nolint start
source("analysis/issue_52_common.R")
# nolint end

# OUTPUT
REG_TABLE_FILE <- file.path(TABLE_DIR, "issue_52_chat_round_regressions.tex")

# =====
# Main function
# =====
main <- function() {
    dt <- prepare_regression_data()
    models <- estimate_models(dt)
    export_table(models)
    print_summary(models)
}

# =====
# Prepare data with round-specific flags
# =====
prepare_regression_data <- function() {
    dt <- load_chat_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt[, segment := factor(segment)]
    dt[, round := factor(round)]
    return(dt)
}

# =====
# Derive suckered-this-round from round-level conditions
# =====
add_suckered_this_round <- function(dt) {
    # A player is suckered in round R if: a groupmate lied this round
    # AND the player cooperated (contributed max = 25)
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
# Estimate 3 models with player + segment FE
# =====
estimate_models <- function(dt) {
    dt_val <- dt[!is.na(emotion_valence)]
    dt_joy <- dt[!is.na(emotion_joy)]

    list(
        lied_joy = feols(emotion_joy ~ lied_this_round_20 +
            round | segment + player_id,
            cluster = ~player_id, data = dt_joy),
        lied_valence = feols(emotion_valence ~ lied_this_round_20 +
            round | segment + player_id,
            cluster = ~player_id, data = dt_val),
        suckered_valence = feols(emotion_valence ~ suckered_this_round +
            round | segment + player_id,
            cluster = ~player_id, data = dt_val)
    )
}

# =====
# Export LaTeX table
# =====
export_table <- function(models) {
    dir.create(dirname(REG_TABLE_FILE), recursive = TRUE,
               showWarnings = FALSE)
    etable(models$lied_joy, models$lied_valence, models$suckered_valence,
           headers = c("Joy (Lied, Chat)", "Valence (Lied, Chat)",
                       "Valence (Suckered, Chat)"),
           se.below = TRUE,
           signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10),
           tex = TRUE, file = REG_TABLE_FILE)
    message(sprintf("Table saved: %s", REG_TABLE_FILE))
}

# =====
# Print coefficient summary
# =====
print_summary <- function(models) {
    for (name in names(models)) {
        m <- models[[name]]
        ct <- coeftable(m)
        flag_row <- grep("lied_this|suckered_this", rownames(ct))
        message(sprintf("  %s: coef=%.3f se=%.3f p=%.4f n=%d",
                        name, ct[flag_row, 1], ct[flag_row, 2],
                        ct[flag_row, 4], m$nobs))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
