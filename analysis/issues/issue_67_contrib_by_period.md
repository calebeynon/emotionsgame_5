# Issue #67: Replace Figures 1 & 2 with mean contribution over all 22 rounds by treatment

## Summary
Figures 1 and 2 in `Paper.tex` (`median_contribution_by_round.png`, `mean_contribution_by_round.png`) previously showed median and mean contributions by round *within* each supergame. This hid the key pattern: contributions drop back down in round 1 of each new supergame, and the drop differs by treatment. Both figures are replaced with a single AER-style plot showing mean contribution + 95% CI across all 22 sequential rounds (supergames 1-5: 3+4+3+7+5), overlaid by treatment, with dashed vertical lines marking supergame boundaries.

## Methodology
1. Load `datastore/derived/contributions.csv` (panel of session × segment × round × participant → contribution). Fail loudly on missing file, missing required columns (`segment`, `round`, `treatment`, `contribution`), or any NA in `contribution`.
2. Map each `(segment, round)` pair to a cumulative sequential round 1-22 via `SEGMENT_ROUNDS = c(supergame1=3, supergame2=4, supergame3=3, supergame4=7, supergame5=5)`. Any segment value not in this map aborts with an error.
3. Aggregate by `(treatment, round)` to compute mean, `se = sd / sqrt(n)`, and 95% CI = mean ± 1.96·se. `n = 80` per `(treatment, round)` cell (4 sessions × 20 subjects per treatment).
4. Render with `ggplot2` in AER style: black filled circles (T1) and triangles (T2) dodged by 0.35 rounds, vertical error bars, no connecting lines, serif font, dashed vertical lines at round boundaries 3.5, 7.5, 10.5, 17.5 (derived from `cumsum(SEGMENT_ROUNDS)`).
5. Replace the two-minipage figure block in `Paper.tex` with a single figure; minimally rewrite the surrounding prose to reference all 22 rounds instead of within-segment rounds.

## Outputs
| Output | Script | Inputs |
|---|---|---|
| `analysis/output/plots/mean_contribution_by_period.png` | `analysis/analysis/issue_67_mean_by_period.R` | `datastore/derived/contributions.csv` |

## Paper Changes
- `analysis/paper/Paper.tex`:
  - Replaced the two-minipage figure (`median_contribution_by_round.png` + `mean_contribution_by_round.png`) with a single figure referencing `mean_contribution_by_period.png`. Added `\usepackage{adjustbox}`.
  - Rewrote the first paragraph of `\subsection{Summary Statistics}` to reference the single new figure and "22 sequential rounds".
  - Flipped `\input@path` and `\graphicspath` order to prefer `../output/tables/` and `../output/plots/` locally, so stale content in the Action-managed `paper/tables/` and `paper/plots/` dirs never shadows the canonical output. On Overleaf the fall-through still resolves via the sync Action.
- Side fix (discovered while recompiling): the existing `dynamic_regression.tex` overflowed the page because `\resizebox{\textwidth}{!}` scaled a narrow-but-tall tabular up in a `\doublespacing` document. Swapped for `adjustbox{max width=\textwidth}` (never scales up), wrapped the `\input` in `\begin{spacing}{1}`, and changed the float spec from `[H]` to `[t]`. Matching change made in `analysis/analysis/dynamic_regression.R` so future regenerations stay fixed.

## Notes
- Original `median_contribution_by_round.png` and `mean_contribution_by_round.png` remain on disk per the issue's explicit instruction ("only the `\includegraphics` references in `Paper.tex` change").
- `SEGMENT_BOUNDARIES` is derived from `SEGMENT_ROUNDS` (not hand-maintained) so there is no risk of the vertical lines drifting if segment lengths change.
- `dynamic_regression.R:66-68` has a separate pre-existing silent-failure concern (blanket NA→0 for chat columns) that was flagged in PR review but left out of scope.
