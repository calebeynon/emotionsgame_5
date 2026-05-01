# Purpose: Combine behavior and sentiment contribution regressions into one table
# Author: Claude Code
# Date: 2026-04-01
#
# Model 1: contribution ~ made_promise + is_sucker + treatment | round + segment
# Model 2: contribution ~ sentiment + message_count + treatment | round + segment

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
BEHAVIOR_CSV <- "datastore/derived/behavior_classifications.csv"
SENTIMENT_CSV <- "datastore/derived/sentiment_scores.csv"
OUTPUT_DIR <- "output/tables"
OUTPUT_TEX <- file.path(OUTPUT_DIR, "contribution_regression_combined.tex")

# =====
# Main function
# =====
main <- function() {
    model_behavior <- run_behavior_model()
    model_sentiment <- run_sentiment_model()

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
    export_combined_table(model_behavior, model_sentiment, OUTPUT_TEX)
    cat("Combined table exported to:", OUTPUT_TEX, "\n")
}

# =====
# Behavior model
# =====
run_behavior_model <- function() {
    dt <- as.data.table(read.csv(BEHAVIOR_CSV))
    for (col in c("made_promise", "is_sucker_20")) {
        dt[, (col) := as.integer(get(col) == "True")]
    }
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]
    feols(
        contribution ~ made_promise + is_sucker_20 + treatment | round + segment,
        data = dt, cluster = ~cluster_id
    )
}

# =====
# Sentiment model
# =====
run_sentiment_model <- function() {
    dt <- as.data.table(read.csv(SENTIMENT_CSV))
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]
    feols(
        contribution ~ sentiment_compound_mean + message_count + treatment | round + segment,
        data = dt, cluster = ~cluster_id
    )
}

# =====
# Combined LaTeX output
# =====
export_combined_table <- function(m_behavior, m_sentiment, filepath) {
    etable(
        m_sentiment, m_behavior,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        dict = c(
            made_promise = "Made Promise",
            is_sucker_20 = "Is Sucker",
            sentiment_compound_mean = "Sentiment",
            message_count = "Message Count",
            treatment = "AF (vs IF)"
        )
    )
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
