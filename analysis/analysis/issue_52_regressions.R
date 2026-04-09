# Purpose: Round-specific liar/sucker emotion regressions on the Results page
#          Lied-this-round → Joy, Valence; Suckered-this-round → Valence
# Author: Claude Code
# Date: 2026-04-08

# nolint start
source("analysis/issue_52_common.R")
# nolint end

library(fixest)

# OUTPUT
REG_TABLE_FILE <- file.path(TABLE_DIR, "issue_52_round_regressions.tex")

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
    dt <- load_results_emotion_data()
    dt[, player_id := paste0(session_code, "_", label)]
    dt <- add_suckered_this_round(dt)
    dt[, segment := factor(segment)]
    dt[, round := factor(round)]
    return(dt)
}

# =====
# Derive suckered-this-round from cumulative flag
# =====
add_suckered_this_round <- function(dt) {
    setorderv(dt, c("session_code", "label", "segment", "round"))
    dt[, prev_sucker := shift(is_sucker_20, 1, fill = FALSE),
       by = .(session_code, label, segment)]
    dt[, suckered_this_round := (is_sucker_20 == TRUE) &
           (prev_sucker == FALSE)]
    dt[, prev_sucker := NULL]
    return(dt)
}

# =====
# Estimate the 3 key models
# =====
estimate_models <- function(dt) {
    dt_val <- dt[!is.na(emotion_valence)]
    dt_joy <- dt[!is.na(emotion_joy)]

    list(
        lied_joy = feols(emotion_joy ~ lied_this_round_20 +
            segment + round, cluster = ~player_id, data = dt_joy),
        lied_valence = feols(emotion_valence ~ lied_this_round_20 +
            segment + round, cluster = ~player_id, data = dt_val),
        suckered_valence = feols(emotion_valence ~ suckered_this_round +
            segment + round, cluster = ~player_id, data = dt_val)
    )
}

# =====
# Export LaTeX table
# =====
export_table <- function(models) {
    dir.create(dirname(REG_TABLE_FILE), recursive = TRUE,
               showWarnings = FALSE)
    etable(models$lied_joy, models$lied_valence, models$suckered_valence,
           headers = c("Joy (Lied)", "Valence (Lied)",
                       "Valence (Suckered)"),
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
        flag_idx <- grep("lied_this|suckered_this", names(coef(m)))
        coef_val <- coef(m)[flag_idx]
        se_val <- sqrt(vcov(m)[flag_idx, flag_idx])
        p_val <- 2 * pt(abs(coef_val / se_val),
                        m$nobs - length(coef(m)), lower.tail = FALSE)
        message(sprintf("  %s: coef=%.3f se=%.3f p=%.4f n=%d",
                        name, coef_val, se_val, p_val, m$nobs))
    }
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
