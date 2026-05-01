# Project Wiki

<!-- AUTO:stats -->
Total articles: 20
<!-- /AUTO:stats -->

## Articles by Type
<!-- AUTO:listing -->

### concept
- [Chat-Round Pairing Semantics (Critical)](concepts/chat-round-pairing.md) — Chat from round N is paired with round N+1's contribution because chat happens AFTER contribution in oTree
- [Project Glossary](concepts/glossary.md) — Definitions of terms used throughout the project: treatments, segments, liar, sucker, thresholds

### method
- [Behavior Classification: Promises, Liars, Suckers](methods/behavior-classification.md) — Pipeline that classifies chat as promises (LLM) and players as liars/suckers per round and per segment
- [Cooperative State Classification](methods/cooperative-state.md) — Two flavors of cooperative-state classification: group-level (everyone shares) and player-level (others' contributions only)
- [Dynamic Panel Regression (Arellano-Bond)](methods/dynamic-regression.md) — Two-step difference GMM of contribution dynamics. Baseline (4-col, §4.2) + extended (12-col, §4.7) tables. Aligned to Stata Table DP1.
- [Chat Embeddings & Hanaki External Validation](methods/embeddings-validation.md) — OpenAI text embeddings of chat, centroid projections by behavior class, and external validation on Hanaki & Ozkes (2023) data
- [Facial Valence Regressions for Liars and Suckers](methods/facial-valence-regressions.md) — Issue #52: AFFDEX facial valence regressed on liar and sucker flags, two face windows
- [Liar Diff-in-Means by Treatment and Gender (Issue #64)](methods/liar-diff-in-means.md) — Welch t-tests on participant-level 'ever lied' indicator — replaces earlier logit (issue #53)
- [Liar Flag: Cumulative vs Round-Specific](methods/liar-flag-comparison.md) — Comparison of cumulative (is_liar_20) and round-specific (lied_this_round_20) liar flag approaches for analysis
- [Lying Contagion Regression (Issue #72)](methods/lying-contagion.md) — Tests whether own lying responds to group lying in prior round/supergame, especially under feedback treatment
- [Merged Panel Construction](methods/merged-panel.md) — How the unified merged_panel.csv (10,683 rows × 34 cols) is built from oTree state, VADER sentiment, and AFFDEX emotion sources
- [Sentiment Analysis & Sentiment-Contribution Regressions](methods/sentiment-analysis.md) — VADER sentiment scoring of chat plus the regressions that link it to contributions and to facial emotion
- [Promise, Sucker, and Treatment Effects on Contributions](methods/sentiment-and-promise-regressions.md) — OLS panel regressions of contribution on promise/sucker flags and treatment, with two threshold specifications
- [Sucker DiD Event Study](methods/sucker-did.md) — Event-study DiD around the round a player gets suckered, with heterogeneous treatment effects by IF/AF

### paper
- [Main Paper: Facial Emotions vs Verbal Sentiments](papers/main-paper.md) — Structure, key tables, and key claims of analysis/paper/Paper.tex

### tool
- [Analysis Pipeline](tools/analysis-pipeline.md) — Data processing pipeline: experiment_data.py loader, derived classifications, R regressions, and paper output
- [Project Architecture](tools/architecture.md) — Top-level project structure: oTree experiment apps, analysis pipeline, and paper output
- [Datastore Files Reference](tools/datastore-files.md) — Cheat sheet for files in analysis/datastore/derived/ — what they contain, who produces them, who consumes them
- [experiment_data.py — Hierarchical Data Module](tools/experiment-data-module.md) — Core hierarchical data model: Experiment > Session > Segment > Round > Group > Player, with built-in VADER sentiment
- [Key Scripts Reference](tools/key-scripts.md) — Per-script index of analysis/derived/ and analysis/analysis/ — what each script does and what it produces
- [oTree Experiment Apps](tools/otree-apps.md) — 7 oTree apps implementing a public goods game with chat across 5 supergames

<!-- /AUTO:listing -->
