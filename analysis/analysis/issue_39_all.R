# Purpose: Master runner for all Issue #39 emotion-sentiment analysis scripts
# Author: Claude Code
# Date: 2026-03-14

# =====
# Main function
# =====
main <- function() {
    message("=== Issue #39: Generating All Outputs ===\n")

    script_dir <- get_script_dir()
    failures <- character(0)

    # Plot scripts
    if (!run_script(script_dir, "issue_39_plot_dotplots.R", "Dot Plots by Player Type")) {
        failures <- c(failures, "issue_39_plot_dotplots.R")
    }

    # Regression scripts
    if (!run_script(script_dir, "issue_39_regression_decomposition.R", "Decomposition Regression")) {
        failures <- c(failures, "issue_39_regression_decomposition.R")
    }
    if (!run_script(script_dir, "issue_39_gap_tests.R", "Gap Tests")) {
        failures <- c(failures, "issue_39_gap_tests.R")
    }

    if (length(failures) > 0) {
        message(sprintf("\n=== %d script(s) FAILED: %s ===",
                        length(failures), paste(failures, collapse = ", ")))
        quit(status = 1)
    }

    message("\n=== All Issue #39 Outputs Generated ===")
    print_output_summary()
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
    message("  Plots: output/plots/")
    message("    - emotion_sentiment_gap_by_cooperative_state.png")
    message("    - emotion_sentiment_gap_by_liar_status.png")
    message("    - emotion_sentiment_gap_by_sucker_status.png")
    message("    - emotion_sentiment_gap_by_liar_x_state.png")
    message("  Tables: output/tables/")
    message("    - emotion_sentiment_orthogonal.tex")
    message("    - emotion_sentiment_deception.tex")
    message("    - emotion_sentiment_deception_descriptive.tex")
    message("    - emotion_sentiment_gap_tests.tex")
}

# %%
if (interactive() || !exists("TESTING")) {
    main()
}
