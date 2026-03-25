# Purpose: Orthogonal decomposition of sentiment and strategic deception regression
# Author: Claude Code
# Date: 2026-03-14
#
# Part A: Decompose sentiment into emotion-aligned and orthogonal components,
#   then test whether orthogonal sentiment predicts contribution beyond emotion.
# Part B: Test whether emotion-sentiment gap predicts noncooperative behavior
#   and promise-breaking (lying).

# nolint start
library(fixest)

source("analysis/issue_39_common.R")
# nolint end

# FILE PATHS
ORTHOGONAL_TEX <- file.path(TABLE_DIR, "emotion_sentiment_orthogonal.tex")
DECEPTION_TEX <- file.path(TABLE_DIR, "emotion_sentiment_deception.tex")
DESCRIPTIVE_TEX <- file.path(TABLE_DIR, "emotion_sentiment_deception_descriptive.tex")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_filter_data()
    print_sample_info(dt)

    dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

    run_orthogonal_decomposition(dt)
    run_deception_analysis(dt)

    cat("All decomposition/deception tables exported.\n")
}

# =====
# Data loading and filtering
# =====
load_and_filter_data <- function() {
    dt <- load_contribute_data()
    n_before <- nrow(dt)

    key_vars <- c("contribution", "emotion_valence",
                  "sentiment_compound_mean", "player_state")
    dt <- dt[complete.cases(dt[, ..key_vars])]
    cat(sprintf("Complete-case filter: %d -> %d rows (%d dropped)\n",
                n_before, nrow(dt), n_before - nrow(dt)))

    dt[, player_id := paste(label, session_code, sep = "_")]
    dt[, noncooperative := as.integer(player_state != "cooperative")]

    return(dt)
}

print_sample_info <- function(dt) {
    cat("Complete-case sample:", nrow(dt), "\n")
    cat("Noncooperative obs:", sum(dt$noncooperative), "\n")
    cat("Promise obs:", sum(dt$made_promise, na.rm = TRUE), "\n")
}

# =====
# Part A: Orthogonal decomposition
# =====
run_orthogonal_decomposition <- function(dt) {
    dt <- decompose_sentiment(dt)
    models <- estimate_orthogonal_models(dt)
    export_orthogonal_table(models, ORTHOGONAL_TEX)
    cat("Orthogonal table exported to:", ORTHOGONAL_TEX, "\n")
}

decompose_sentiment <- function(dt) {
    # Regress sentiment on emotion_valence with player FE
    first_stage <- feols(
        sentiment_compound_mean ~ emotion_valence | player_id,
        data = dt
    )

    dt[, sent_aligned := fitted(first_stage)]
    dt[, sent_orthogonal := residuals(first_stage)]

    cat("First-stage R2:", r2(first_stage, "r2"), "\n")
    cat("Correlation(aligned, orthogonal):",
        cor(dt$sent_aligned, dt$sent_orthogonal), "\n")

    return(dt)
}

estimate_orthogonal_models <- function(dt) {
    run <- function(f) {
        feols(as.formula(f), data = dt, cluster = ~cluster_id)
    }

    # Model 1: Key test - orthogonal sentiment + emotion
    m1 <- run(paste(
        "contribution ~ sent_orthogonal + emotion_valence",
        "| round + segment"
    ))

    # Model 2: Both decomposed components (sent_aligned may be collinear)
    m2 <- run(paste(
        "contribution ~ sent_aligned + sent_orthogonal",
        "| round + segment"
    ))

    return(list(m1 = m1, m2 = m2))
}

export_orthogonal_table <- function(models, filepath) {
    etable(
        models$m1, models$m2,
        file = filepath,
        tex = TRUE,
        fitstat = c("n", "r2"),
        dict = c(
            sent_orthogonal = "Sentiment (Orthogonal)",
            sent_aligned = "Sentiment (Aligned)",
            emotion_valence = "Emotion Valence"
        ),
        headers = c("Orthogonal + Emotion", "Decomposed Components"),
        title = "Orthogonal Decomposition: Sentiment Beyond Emotion"
    )
}

# =====
# Part B: Deception analysis
# =====
run_deception_analysis <- function(dt) {
    dt <- compute_emotion_sentiment_gap(dt)
    models <- estimate_deception_models(dt)
    export_deception_table(models, DECEPTION_TEX)
    cat("Deception table exported to:", DECEPTION_TEX, "\n")

    export_descriptive_table(dt, DESCRIPTIVE_TEX)
    cat("Descriptive table exported to:", DESCRIPTIVE_TEX, "\n")
}

compute_emotion_sentiment_gap <- function(dt) {
    # Min-max normalize both to [0,1] before computing gap
    # Valence ~0-100, sentiment ~-1 to 1: raw scales differ
    val_range <- max(dt$emotion_valence) - min(dt$emotion_valence)
    sent_range <- max(dt$sentiment_compound_mean) - min(dt$sentiment_compound_mean)
    if (val_range == 0) stop("Zero range in emotion_valence — cannot normalize")
    if (sent_range == 0) stop("Zero range in sentiment_compound_mean — cannot normalize")
    dt[, valence_norm := (emotion_valence - min(emotion_valence)) / val_range]
    dt[, sentiment_norm := (sentiment_compound_mean - min(sentiment_compound_mean)) / sent_range]

    # Positive gap = face happier than words
    dt[, emotion_sentiment_gap := valence_norm - sentiment_norm]

    cat("Gap range:", range(dt$emotion_sentiment_gap), "\n")
    cat("Gap mean:", mean(dt$emotion_sentiment_gap), "\n")

    return(dt)
}

prepare_promise_data <- function(dt) {
    dt_promise <- dt[made_promise == 1]
    dt_promise[, lied := as.integer(contribution < 20)]
    cat("Liar model N:", nrow(dt_promise),
        "  Lied:", sum(dt_promise$lied), "\n")
    return(dt_promise)
}

estimate_deception_models <- function(dt) {
    noncoop_formula <- paste(
        "noncooperative ~ emotion_sentiment_gap +",
        "emotion_valence + sentiment_compound_mean | round + segment"
    )
    m_noncoop <- run_logit_model(dt, noncoop_formula)
    m_liar <- run_liar_model(prepare_promise_data(dt))
    return(list(m_noncoop = m_noncoop, m_liar = m_liar))
}

run_logit_model <- function(dt, formula_str) {
    feglm(
        as.formula(formula_str),
        data = dt,
        family = binomial(link = "logit"),
        cluster = ~cluster_id
    )
}

run_pooled_liar_logit <- function(dt_promise) {
    cat("Falling back to pooled logit with cluster SEs\n")
    feglm(lied ~ emotion_sentiment_gap, data = dt_promise,
          family = binomial(link = "logit"), cluster = ~cluster_id)
}

run_liar_model <- function(dt_promise) {
    # Small sample: try FE logit, fall back to pooled if convergence fails
    tryCatch(
        feglm(lied ~ emotion_sentiment_gap | round + segment,
              data = dt_promise, family = binomial(link = "logit"),
              cluster = ~cluster_id),
        error = function(e) {
            convergence_patterns <- c("convergence", "singular", "separation",
                                       "not enough", "collinear")
            if (!any(sapply(convergence_patterns, grepl, e$message, ignore.case = TRUE))) {
                stop(e)
            }
            cat("FE logit convergence failed:", e$message, "\n")
            cat("Falling back to pooled logit (no FE)\n")
            run_pooled_liar_logit(dt_promise)
        }
    )
}

export_deception_table <- function(models, filepath) {
    etable(
        models$m_noncoop, models$m_liar,
        file = filepath,
        tex = TRUE,
        fitstat = c("n"),
        dict = c(
            emotion_sentiment_gap = "Emotion--Sentiment Gap",
            emotion_valence = "Emotion Valence",
            sentiment_compound_mean = "Sentiment (Compound)",
            noncooperative = "Noncooperative",
            lied = "Lied"
        ),
        title = "Strategic Deception: Emotion--Sentiment Gap"
    )
}

# =====
# Descriptive table
# =====
export_descriptive_table <- function(dt, filepath) {
    dt_promise <- dt[made_promise == 1]
    dt_promise[, lied := as.integer(contribution < 20)]

    desc_coop <- summarize_by_group(dt, "noncooperative",
                                     c("Cooperative", "Noncooperative"))
    desc_liar <- summarize_by_group(dt_promise, "lied",
                                     c("Honest", "Liar"))

    write_descriptive_tex(desc_coop, desc_liar, filepath)
}

summarize_by_group <- function(dt, group_var, labels) {
    result <- dt[, .(
        n = .N,
        mean_valence = mean(emotion_valence, na.rm = TRUE),
        mean_sentiment = mean(sentiment_compound_mean, na.rm = TRUE),
        mean_contribution = mean(contribution, na.rm = TRUE)
    ), by = group_var][order(get(group_var))]
    label_map <- setNames(labels, sort(unique(dt[[group_var]])))
    result[, group_label := label_map[as.character(get(group_var))]]
    return(result)
}

write_descriptive_tex <- function(desc_coop, desc_liar, filepath) {
    lines <- c(
        "\\begin{tabular}{lrrrr}",
        "  \\toprule",
        "  Group & N & Mean Valence & Mean Sentiment & Mean Contribution \\\\",
        "  \\midrule",
        "  \\emph{By cooperative state} \\\\",
        format_desc_row(desc_coop[1]),
        format_desc_row(desc_coop[2]),
        "  \\midrule",
        "  \\emph{By promise honesty} \\\\",
        format_desc_row(desc_liar[1]),
        format_desc_row(desc_liar[2]),
        "  \\bottomrule",
        "\\end{tabular}"
    )

    writeLines(lines, filepath)
}

format_desc_row <- function(row) {
    sprintf("  %s & %d & %.2f & %.3f & %.2f \\\\",
            row$group_label, row$n,
            row$mean_valence, row$mean_sentiment,
            row$mean_contribution)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
