---
title: "Project Architecture"
type: tool
tags: [architecture, otree, analysis, structure]
summary: "Top-level project structure: oTree experiment apps, analysis pipeline, and paper output"
status: draft
last_verified: "2026-04-06"
---

## Summary

This project is a behavioral economics experiment built with oTree, studying how communication affects contribution behavior in a public goods game. The codebase has two major halves: the **oTree experiment** (game logic, pages, templates) and the **analysis module** (data processing, classification, regression, and paper output).

## Key Points

- **Framework**: oTree 5.11+ with Python 3.12+, managed via `uv`
- **Experiment**: 7 sequential apps for 16 participants in groups of 4
- **Analysis**: Python + R pipeline producing tables, plots, and a LaTeX paper synced to Overleaf
- **Data storage**: Symlinked Box folder (`analysis/datastore`) holds raw CSVs and derived outputs

## Directory Structure

```
emotionsgame_5/
├── settings.py                 # oTree session config (app sequence, rooms, currency)
├── pyproject.toml              # uv project config, dependencies, pytest settings
├── participant_labels.txt      # A-R labels (skipping I, O)
├── introduction/               # Instruction pages, practice round, quiz
├── chatpractice/               # Chat practice round (not in main sequence)
├── supergame1-5/               # 5 supergame apps (core game logic)
├── finalresults/               # Survey + payment page
├── _templates/global/          # Base page template
├── _static/                    # Static assets
├── src/emotionsgame/           # Python package (minimal placeholder)
├── analysis/                   # Analysis module (see analysis article)
│   ├── experiment_data.py      # Core data model (hierarchical classes)
│   ├── analysis/               # Analysis scripts (Python + R)
│   ├── derived/                # Data derivation scripts (embeddings, classification)
│   ├── annotations/            # Video annotation pipeline
│   ├── tests/                  # pytest test suite
│   ├── output/                 # Generated plots, tables, summary stats
│   ├── paper/                  # LaTeX paper (git subtree → Overleaf)
│   ├── datastore/              # Symlink to Box shared folder
│   └── issues/                 # Issue documentation
├── wiki/                       # Project knowledge base
└── .github/workflows/          # CI (Overleaf sync action)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Experiment | oTree 5.x (Python) |
| Package manager | uv |
| Analysis (data) | Python (pandas, numpy, scipy, nltk) |
| Analysis (stats) | R (fixest for panel regressions) |
| Embeddings | OpenAI API, Anthropic API |
| LLM classification | OpenAI (GPT-5-mini), Google Gemini |
| Visualization | matplotlib, seaborn, ggplot2 |
| Paper | LaTeX, synced to Overleaf via git subtree + GitHub Action |
| Testing | pytest |
| Data storage | Box (symlinked as `analysis/datastore`) |

## Configuration

- **`settings.py`**: Defines `SESSION_CONFIGS` with the 7-app sequence, 16 demo participants, room-based label assignment, 0.1 USD/point exchange rate, $7.50 participation fee
- **`pyproject.toml`**: Package metadata, all Python dependencies, pytest config (tests in `analysis/tests/`)
- **`.github/workflows/sync-overleaf.yml`**: Auto-syncs paper files to Overleaf on push to main

## Related

- [oTree Experiment Apps](wiki/tools/otree-apps.md)
- [Analysis Pipeline](wiki/tools/analysis-pipeline.md)
