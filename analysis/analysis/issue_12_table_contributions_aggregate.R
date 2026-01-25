#' Contributions Summary LaTeX Table (Aggregated Across Sessions)
#'
#' Generates a LaTeX table with contribution summary statistics
#' by treatment and segment, averaged across all sessions.
#'
#' Author: Claude
#' Date: 2026-01-25

library(data.table)
library(xtable)

# =====
# File paths
# =====
INPUT_FILE <- "../datastore/derived/contributions.csv"
OUTPUT_DIR <- "../output/tables"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "contributions_summary_aggregate.tex")

# =====
# Main function
# =====
main <- function() {
    dt <- load_contributions(INPUT_FILE)
    summary_dt <- compute_summary_stats(dt)
    ensure_output_dir(OUTPUT_DIR)
    write_latex_table(summary_dt, OUTPUT_FILE)
    message("LaTeX table written to: ", OUTPUT_FILE)
}

# =====
# Data loading
# =====
load_contributions <- function(file_path) {
    dt <- fread(file_path)
    dt[, segment := factor(segment, levels = paste0("supergame", 1:5))]
    return(dt)
}

# =====
# Summary statistics computation
# =====
compute_summary_stats <- function(dt) {
    summary_dt <- dt[, .(
        N = .N,
        Mean = round(mean(contribution, na.rm = TRUE), 2),
        SD = round(sd(contribution, na.rm = TRUE), 2),
        Min = min(contribution, na.rm = TRUE),
        Max = max(contribution, na.rm = TRUE),
        Median = median(contribution, na.rm = TRUE)
    ), by = .(treatment, segment)]

    setorder(summary_dt, treatment, segment)
    return(summary_dt)
}

# =====
# Output handling
# =====
ensure_output_dir <- function(dir_path) {
    if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE)
    }
}

write_latex_table <- function(summary_dt, output_file) {
    display_dt <- prepare_display_table(summary_dt)
    xtab <- create_xtable(display_dt)
    write_xtable_to_file(xtab, output_file, nrow(display_dt))
}

prepare_display_table <- function(summary_dt) {
    display_dt <- copy(summary_dt)
    setnames(display_dt, c("treatment", "segment"),
             c("Treatment", "Segment"))
    return(display_dt)
}

create_xtable <- function(display_dt) {
    xtable(display_dt, align = c("l", "c", "l", "r", "r", "r", "r", "r", "r"))
}

write_xtable_to_file <- function(xtab, output_file, nrows) {
    print(xtab, file = output_file, include.rownames = FALSE, booktabs = TRUE,
          floating = FALSE, hline.after = NULL,
          add.to.row = list(pos = list(-1, 0, nrows),
                           command = c("\\toprule\n", "\\midrule\n", "\\bottomrule\n")))
}

# =====
# Execute
# =====
if (sys.nframe() == 0) {
    main()
}
