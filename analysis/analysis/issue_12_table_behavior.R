#' Purpose: Generate LaTeX table summarizing behavior classifications
#'          (promises, liars, suckers) by segment and session
#' Author: Claude
#' Date: 2026-01-25

# nolint start
library(data.table)
library(xtable)

# =====
# File paths
# =====
INPUT_CSV <- "../datastore/derived/behavior_classifications.csv"
OUTPUT_DIR <- "../output/tables"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "behavior_summary.tex")

# =====
# Main function (shows high-level flow)
# =====
main <- function() {
    dt <- load_behavior_data(INPUT_CSV)
    summary_dt <- aggregate_behavior_counts(dt)
    latex_table <- create_latex_table(summary_dt)
    ensure_output_dir(OUTPUT_DIR)
    save_latex_table(latex_table, OUTPUT_FILE)
    message("Behavior summary table saved to: ", OUTPUT_FILE)
}

# =====
# Data loading
# =====
load_behavior_data <- function(file_path) {
    # fread automatically converts True/False strings to logical
    dt <- fread(file_path)
    return(dt)
}

# =====
# Aggregation
# =====
aggregate_behavior_counts <- function(dt) {
    summary_dt <- dt[, .(
        N = .N,
        Promises = sum(made_promise),
        Liars_20 = sum(is_liar_20),
        Liars_5 = sum(is_liar_5),
        Suckers_20 = sum(is_sucker_20),
        Suckers_5 = sum(is_sucker_5)
    ), by = .(Session = session_code, Treatment = treatment, Segment = segment)]

    # Order by treatment, session, segment
    summary_dt <- summary_dt[order(Treatment, Session, Segment)]
    return(summary_dt)
}

# =====
# LaTeX table creation
# =====
create_latex_table <- function(summary_dt) {
    # Create xtable with booktabs style
    xt <- xtable(
        summary_dt,
        caption = "Behavior Classifications by Session and Segment",
        label = "tab:behavior_summary",
        align = c("l", "l", "c", "l", "r", "r", "r", "r", "r", "r")
    )
    return(xt)
}

# =====
# Output helpers
# =====
ensure_output_dir <- function(dir_path) {
    if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE)
    }
}

save_latex_table <- function(latex_table, file_path) {
    print(latex_table, file = file_path, include.rownames = FALSE,
          booktabs = TRUE, floating = FALSE, hline.after = NULL,
          add.to.row = list(pos = list(-1, 0, nrow(latex_table)),
                           command = c("\\toprule\n", "\\midrule\n", "\\bottomrule\n")),
          sanitize.colnames.function = format_column_names)
}

format_column_names <- function(x) {
    # Replace underscores with spaces for nicer column headers
    gsub("_", " ", x)
}

# =====
# Execute
# =====
if (sys.nframe() == 0) {
    main()
}
