# Issue #12: Add Contribution Tracking Visualizations

**Status:** Complete
**Branch:** issue_12_contribution_visuals
**Created:** 2026-01-25

## Objective

Create comprehensive contribution tracking visualizations and summary tables for the emotions game experiment, supporting comparative analysis across treatments.

## Scope

### Plots (5 total)
1. CDF of contributions by treatment
2. Mean contribution vs round (within-segment) by treatment
3. Median contribution vs round (within-segment) by treatment
4. Mean contribution vs segment by treatment
5. Median contribution vs segment by treatment

### LaTeX Tables (4 total)
1. Contributions summary by session/segment
2. Behavior classifications by session/segment
3. Contributions summary aggregated across sessions
4. Behavior classifications aggregated across sessions

## Implementation Details

### Design Choices
- Color scheme: #9E1B32 (Treatment 1) and #828A8F (Treatment 2)
- No plot titles (per CLAUDE.md guidelines)
- All functions ≤20 lines
- All scripts ≤300 lines
- Publication-quality themes and labels

### File Structure
- 10 R scripts in `analysis/analysis/`
- Master script: `issue_12_all.R` runs all individual scripts
- Output: `analysis/output/tables/` for LaTeX tables
- Output: `analysis/output/plots/` for generated plots (gitignored)

### Scripts Created

| Script | Output |
|--------|--------|
| `issue_12_cdf_plot.R` | `contribution_cdf_by_treatment.png` |
| `issue_12_mean_by_round.R` | `mean_contribution_by_round.png` |
| `issue_12_median_by_round.R` | `median_contribution_by_round.png` |
| `issue_12_mean_by_segment.R` | `mean_contribution_by_segment.png` |
| `issue_12_median_by_segment.R` | `median_contribution_by_segment.png` |
| `issue_12_table_contributions.R` | `contributions_summary.tex` |
| `issue_12_table_behavior.R` | `behavior_summary.tex` |
| `issue_12_table_contributions_aggregate.R` | `contributions_summary_aggregate.tex` |
| `issue_12_table_behavior_aggregate.R` | `behavior_summary_aggregate.tex` |
| `issue_12_all.R` | Runs all above scripts |

## Testing

- [x] All scripts run without errors
- [x] Plots generate correctly
- [x] LaTeX tables have valid syntax
- [x] Color scheme is consistent across all visualizations

## Dependencies

- R packages: data.table, ggplot2, xtable
- Input: `analysis/datastore/derived/contributions.csv`
- Input: `analysis/datastore/derived/behavior_classifications.csv`
