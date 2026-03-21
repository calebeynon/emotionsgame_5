# Purpose: Master runner for all Issue #39 emotion-sentiment analysis scripts
# Author: Claude Code
# Date: 2026-03-14

# =====
# Main function
# =====
main <- function() {
    message("=== Issue #39: Generating All Outputs ===\n")
    script_dir <- get_script_dir()
    failures <- run_all_scripts(script_dir)
    if (length(failures) > 0) {
        message(sprintf("\n=== %d script(s) FAILED: %s ===",
                        length(failures), paste(failures, collapse = ", ")))
        quit(status = 1)
    }
    message("\n=== All Issue #39 Outputs Generated ===")
    print_output_summary()
}

# =====
# Script runner
# =====
get_script_list <- function() {
    list(
        c("issue_39_plot_dotplots.R", "Dot Plots by Player Type"),
        c("issue_39_plot_negative_emotions.R", "Negative Emotion Dot Plots"),
        c("issue_39_regression_decomposition.R", "Decomposition Regression"),
        c("issue_39_gap_tests.R", "Gap Tests")
    )
}

run_all_scripts <- function(script_dir) {
    failures <- character(0)
    for (entry in get_script_list()) {
        if (!run_script(script_dir, entry[1], entry[2])) {
            failures <- c(failures, entry[1])
        }
    }
    return(failures)
}

# =====
# Helper functions
# =====
get_script_dir <- function() {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
        return(dirname(normalizePath(sub("--file=", "", file_arg))))
    }
    return(getwd())
}

run_script <- function(script_dir, script_name, description) {
    script_path <- file.path(script_dir, script_name)
    message(sprintf("[%s] Running %s...", description, script_name))
    tryCatch({
        source(script_path, local = new.env())
        message(sprintf("[%s] Complete.\n", description))
        return(TRUE)
    }, error = function(e) {
        message(sprintf("[%s] ERROR: %s\n", description, e$message))
        return(FALSE)
    })
}

print_output_summary <- function() {
    message("\nOutputs generated:")
    print_file_list("Plots: output/plots/", get_output_plots())
    print_file_list("Plots: _sandbox_data/", get_sandbox_plots())
    print_file_list("Tables: output/tables/", get_output_tables())
}

get_output_plots <- function() {
    c("emotion_sentiment_gap_by_cooperative_state.png",
      "emotion_sentiment_gap_by_liar_status.png",
      "emotion_sentiment_gap_by_sucker_status.png",
      "emotion_sentiment_gap_by_liar_x_state.png")
}

get_sandbox_plots <- function() {
    c("negative_emotion_by_cooperative_state.png",
      "negative_emotion_by_liar_status.png",
      "negative_emotion_by_sucker_status.png",
      "negative_emotion_by_liar_x_state.png")
}

get_output_tables <- function() {
    c("emotion_sentiment_orthogonal.tex",
      "emotion_sentiment_deception.tex",
      "emotion_sentiment_deception_descriptive.tex",
      "emotion_sentiment_gap_tests.tex")
}

print_file_list <- function(header, files) {
    message("  ", header)
    for (f in files) message("    - ", f)
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
