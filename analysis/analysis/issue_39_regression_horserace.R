# Purpose: Horse race regression comparing emotion vs sentiment predictors of contribution
# Author: Claude Code
# Date: 2026-03-14
#
# 5 models progressively add predictors:
#   m1: emotion_valence only
#   m2: sentiment_compound_mean only
#   m3: both predictors
#   m4: both + interaction
#   m5: full model (+ made_promise + treatment)
#
# All models use round + segment fixed effects, clustered SEs at session-segment-group

# nolint start
library(data.table)
library(fixest)

source("analysis/issue_39_common.R")

# FILE PATHS
OUTPUT_TEX <- file.path(TABLE_DIR, "emotion_sentiment_horserace.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_filter_data()
    print_sample_info(dt)

    models <- estimate_models(dt)

    dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
    export_latex_table(models, OUTPUT_TEX)

    cat("Horse race table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Data loading and filtering
# =====
load_and_filter_data <- function() {
    dt <- load_contribute_data()

    key_vars <- c("contribution", "emotion_valence", "sentiment_compound_mean")
    dt <- dt[complete.cases(dt[, ..key_vars])]

    # Interaction term
    dt[, emo_sent_interact := emotion_valence * sentiment_compound_mean]

    return(dt)
}

# =====
# Diagnostics
# =====
print_sample_info <- function(dt) {
    cat("Sample size:", nrow(dt), "\n")
    cat("Unique clusters:", uniqueN(dt$cluster_id), "\n")
    cat("Unique players:", uniqueN(dt$label), "\n")
}

# =====
# Model estimation
# =====
run_feols <- function(formula_str, dt) {
    feols(as.formula(formula_str), data = dt, cluster = ~cluster_id)
}

estimate_models <- function(dt) {
    m1 <- run_feols("contribution ~ emotion_valence | round + segment", dt)
    m2 <- run_feols("contribution ~ sentiment_compound_mean | round + segment", dt)
    m3 <- run_feols("contribution ~ emotion_valence + sentiment_compound_mean | round + segment", dt)
    m4 <- run_feols("contribution ~ emotion_valence * sentiment_compound_mean | round + segment", dt)
    m5 <- run_feols(paste(
        "contribution ~ emotion_valence + sentiment_compound_mean +",
        "emo_sent_interact + made_promise + treatment | round + segment"
    ), dt)

    return(list(m1 = m1, m2 = m2, m3 = m3, m4 = m4, m5 = m5))
}

# =====
# LaTeX output
# =====
export_latex_table <- function(models, filepath) {
    etable(
        models$m1, models$m2, models$m3, models$m4, models$m5,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        dict = c(
            emotion_valence = "Emotion Valence",
            sentiment_compound_mean = "Sentiment (Compound)",
            emo_sent_interact = "Valence $\\times$ Sentiment",
            "emotion_valence:sentiment_compound_mean" = "Valence $\\times$ Sentiment",
            made_promise = "Made Promise",
            treatment = "Treatment"
        ),
        headers = c("Emotion", "Sentiment", "Both", "Interaction", "Full"),
        title = "Horse Race: Emotion vs.\\ Sentiment as Predictors of Contribution"
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
