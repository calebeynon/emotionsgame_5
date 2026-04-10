# Purpose: Panel regression of embedding projections on facial emotion measures
# Author: Claude Code
# Date: 2026-04-09
#
# Issue #54: Tests whether chat embedding projections (cooperative, promise,
# homogeneity, round-liar) predict facial emotion during contribution decisions.
#
# Models: emotion_Y ~ proj_X + word_count + sentiment_compound_mean | player_id + segment
# Clustering: session_code x segment x group
# Output: One LaTeX table per emotion DV (4 univariate + 1 combined column)

# nolint start
library(data.table)
library(fixest)

# FILE PATHS
INPUT_CSV <- "datastore/derived/merged_panel.csv"
OUTPUT_DIR <- "output/tables"

# PROJECTION COLUMNS (display name -> column name)
PROJECTION_COLS <- c(
    "Cooperative" = "proj_pr_dir_small",
    "Promise" = "proj_promise_pr_dir_small",
    "Homogeneity" = "proj_homog_pr_dir_small",
    "Round-liar" = "proj_rliar_pr_dir_small"
)

# EMOTION DEPENDENT VARIABLES (display name -> column name)
EMOTION_DVS <- c(
    "Valence" = "emotion_valence",
    "Joy" = "emotion_joy",
    "Anger" = "emotion_anger",
    "Contempt" = "emotion_contempt",
    "Surprise" = "emotion_surprise"
)

# CONTROL VARIABLES
CONTROLS <- c("word_count", "sentiment_compound_mean")

# =====
# Main function (FIRST - shows high-level flow)
# =====
main <- function() {
    dt <- load_and_prepare_data(INPUT_CSV)
    validate_data(dt)

    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

    for (i in seq_along(EMOTION_DVS)) {
        dv_label <- names(EMOTION_DVS)[i]
        dv_col <- EMOTION_DVS[i]
        run_emotion_analysis(dt, dv_col, dv_label)
    }

    export_combined_table(dt)
}

# =====
# Per-DV analysis pipeline
# =====
run_emotion_analysis <- function(dt, dv_col, dv_label) {
    univariate <- run_univariate_models(dt, dv_col)
    combined <- run_combined_model(dt, dv_col)

    cat(sprintf("\n=== %s Regressions ===\n", dv_label))
    print_model_summaries(univariate, combined, dv_label)
    compute_vif(dt, dv_col, dv_label)

    output_path <- file.path(OUTPUT_DIR, sprintf("issue_54_%s.tex", dv_col))
    export_emotion_table(univariate, combined, output_path, dv_label)
}

# =====
# Data loading and preparation
# =====
load_and_prepare_data <- function(filepath) {
    dt <- as.data.table(read.csv(filepath))
    dt <- dt[page_type == "Contribute"]
    dt[, player_id := paste(label, session_code, sep = "_")]
    dt[, cluster_id := paste(session_code, segment, group, sep = "_")]

    all_cols <- c(unname(PROJECTION_COLS), unname(EMOTION_DVS), CONTROLS)
    dt <- dt[complete.cases(dt[, ..all_cols])]

    cat(sprintf("Loaded %d player-rounds after filtering\n", nrow(dt)))
    return(dt)
}

# =====
# Data validation
# =====
validate_data <- function(dt) {
    required <- c(
        unname(PROJECTION_COLS), unname(EMOTION_DVS), CONTROLS,
        "player_id", "cluster_id", "segment"
    )
    missing <- setdiff(required, names(dt))
    if (length(missing) > 0) {
        stop("Missing required columns: ", paste(missing, collapse = ", "))
    }
    report_missing_values(dt, required)
}

report_missing_values <- function(dt, cols) {
    cat("\n=== Missing Values Report ===\n")
    for (col in cols) {
        n_na <- sum(is.na(dt[[col]]))
        if (n_na > 0) cat(sprintf("  %s: %d NA values\n", col, n_na))
    }
    cat("\n")
}

# =====
# Univariate regressions (one projection at a time)
# =====
run_univariate_models <- function(dt, dv_col) {
    models <- lapply(PROJECTION_COLS, function(proj_col) {
        run_single_regression(dt, dv_col, proj_col)
    })
    names(models) <- names(PROJECTION_COLS)
    return(models)
}

run_single_regression <- function(dt, dv_col, proj_col) {
    fml <- sprintf(
        "%s ~ %s + %s | player_id + segment",
        dv_col, proj_col, paste(CONTROLS, collapse = " + ")
    )
    feols(as.formula(fml), data = dt, cluster = ~cluster_id)
}

# =====
# Combined regression (all projections together)
# =====
run_combined_model <- function(dt, dv_col) {
    proj_terms <- paste(unname(PROJECTION_COLS), collapse = " + ")
    ctrl_terms <- paste(CONTROLS, collapse = " + ")
    fml <- sprintf("%s ~ %s + %s | player_id + segment", dv_col, proj_terms, ctrl_terms)
    feols(as.formula(fml), data = dt, cluster = ~cluster_id)
}

# =====
# VIF diagnostics for combined model
# =====
compute_vif <- function(dt, dv_col, dv_label) {
    cat(sprintf("\n--- VIF Diagnostics (%s combined) ---\n", dv_label))
    # VIF only for projection columns — controls (word_count, sentiment) always have low VIF
    proj_cols <- unname(PROJECTION_COLS)
    vifs <- sapply(proj_cols, function(col) {
        compute_single_vif(dt, col, proj_cols)
    })
    names(vifs) <- names(PROJECTION_COLS)
    print_vif_results(vifs)
}

compute_single_vif <- function(dt, target_col, all_cols) {
    other_cols <- setdiff(all_cols, target_col)
    fml <- sprintf("%s ~ %s", target_col, paste(other_cols, collapse = " + "))
    aux_fit <- lm(as.formula(fml), data = dt)
    r2 <- summary(aux_fit)$r.squared
    return(1 / (1 - r2))
}

print_vif_results <- function(vifs) {
    for (name in names(vifs)) {
        flag <- if (vifs[name] > 5) " *** HIGH ***" else ""
        cat(sprintf("  %s: %.2f%s\n", name, vifs[name], flag))
    }
    cat("\n")
}

# =====
# LaTeX output
# =====
export_emotion_table <- function(univariate, combined, filepath, dv_label) {
    model_list <- c(unname(univariate), list(combined))
    label_dict <- build_label_dict()
    col_headers <- c(names(PROJECTION_COLS), "Combined")

    do.call(etable, c(
        model_list,
        list(
            file = filepath,
            tex = TRUE,
            fitstat = c("n", "r2"),
            dict = label_dict,
            headers = col_headers,
            title = sprintf("%s ~ Embedding Projections", dv_label)
        )
    ))
    cat("Exported:", filepath, "\n")
}

build_label_dict <- function() {
    c(
        setNames(names(PROJECTION_COLS), PROJECTION_COLS),
        word_count = "Word Count",
        sentiment_compound_mean = "Sentiment (compound)"
    )
}

# =====
# Combined table (all emotions side-by-side)
# =====
export_combined_table <- function(dt) {
    models <- lapply(EMOTION_DVS, function(dv) run_combined_model(dt, dv))
    output_path <- file.path(OUTPUT_DIR, "issue_54_combined.tex")
    label_dict <- c(build_label_dict(), setNames(names(EMOTION_DVS), EMOTION_DVS))

    do.call(etable, c(
        unname(models),
        list(
            file = output_path,
            tex = TRUE,
            fitstat = c("n", "r2"),
            dict = label_dict,
            title = "Embedding Projections on Facial Emotion"
        )
    ))
    cat("Exported:", output_path, "\n")
}

# =====
# Console summary
# =====
print_model_summaries <- function(univariate, combined, dv_label) {
    for (name in names(univariate)) {
        cat(sprintf("\n--- %s (univariate) ---\n", name))
        print(summary(univariate[[name]]))
    }
    cat(sprintf("\n--- %s (combined) ---\n", dv_label))
    print(summary(combined))
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
