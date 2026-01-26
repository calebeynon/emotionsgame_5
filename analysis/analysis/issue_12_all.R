#' Issue #12: Generate All Contribution Visualizations and Tables
#'
#' Master script that sources all individual Issue #12 scripts to generate
#' the complete set of plots and LaTeX tables for contribution analysis.
#'
#' Author: Claude
#' Date: 2026-01-25

# =====
# Main function
# =====

main <- function() {
    message("=== Issue #12: Generating All Outputs ===\n")

    script_dir <- get_script_dir()

    run_script(script_dir, "issue_12_cdf_plot.R", "CDF Plot")
    run_script(script_dir, "issue_12_mean_by_round.R", "Mean by Round Plot")
    run_script(script_dir, "issue_12_median_by_round.R", "Median by Round Plot")
    run_script(script_dir, "issue_12_mean_by_segment.R", "Mean by Segment Plot")
    run_script(script_dir, "issue_12_median_by_segment.R", "Median by Segment Plot")
    run_script(script_dir, "issue_12_table_contributions.R", "Contributions Table")
    run_script(script_dir, "issue_12_table_behavior.R", "Behavior Table")
    run_script(script_dir, "issue_12_table_contributions_aggregate.R", "Contributions Table (Aggregate)")
    run_script(script_dir, "issue_12_table_behavior_aggregate.R", "Behavior Table (Aggregate)")

    message("\n=== All Issue #12 Outputs Generated ===")
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
    }, error = function(e) {
        message(sprintf("[%s] ERROR: %s\n", description, e$message))
    })
}

print_output_summary <- function() {
    message("\nOutputs generated:")
    message("  Plots: analysis/output/plots/")
    message("    - contribution_cdf_by_treatment.png")
    message("    - mean_contribution_by_round.png")
    message("    - median_contribution_by_round.png")
    message("    - mean_contribution_by_segment.png")
    message("    - median_contribution_by_segment.png")
    message("  Tables: analysis/output/tables/")
    message("    - contributions_summary.tex")
    message("    - behavior_summary.tex")
    message("    - contributions_summary_aggregate.tex")
    message("    - behavior_summary_aggregate.tex")
}

# %%
if (interactive() || length(commandArgs(trailingOnly = TRUE)) == 0) {
    main()
}
