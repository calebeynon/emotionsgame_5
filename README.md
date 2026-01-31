# emotionsgame_5

An oTree behavioral economics experiment implementing a public goods game with chat, studying how communication affects contribution behavior.

## Analysis Directory Structure

```
analysis/
├── datastore/         # Symlink to external data storage (Box)
├── derived/           # Scripts that compute derived data from raw inputs
├── analysis/          # Scripts that produce final outputs (plots, tables)
├── annotations/       # Video annotation pipeline for behavioral coding
├── output/            # Generated plots and LaTeX tables (gitignored)
├── paper/             # LaTeX paper (synced to Overleaf via GitHub Action)
├── issues/            # Issue documentation and specifications
├── tests/             # Test suite for experiment_data module
├── experiment_data.py # Core data loading and access module
└── archive/           # Historical scripts and plots
```

## Datastore

The `datastore/` symlink points to an external Box folder containing all experimental data. This keeps large data files out of version control while maintaining a consistent path structure. It contains `raw/` (oTree exports), `derived/` (processed data), and `annotations/` (video annotation data).

## Derived vs Analysis

### `derived/` — Data Processing

Scripts in `derived/` transform raw data into derived datasets. They:
- Read from `datastore/raw/`
- Write to `datastore/derived/`
- Perform computationally intensive operations (LLM API calls, classification)
- Are run infrequently (when raw data changes or methods update)

**Example workflow:**
```bash
uv run python analysis/derived/classify_promises.py   # Outputs: datastore/derived/promises.csv
uv run python analysis/derived/classify_behavior.py   # Outputs: datastore/derived/behavior_classifications.csv
```

### `analysis/` — Analysis and Visualization

Scripts in `analysis/` produce final outputs from derived data. They:
- Read from `datastore/derived/`
- Write to `analysis/output/` (plots and tables)
- Are run frequently during analysis iteration
- Outputs are synced to the paper via GitHub Action

**Example workflow:**
```bash
uv run python analysis/analysis/analysis_plots.py        # Outputs: output/plots/
uv run python analysis/analysis/multi_session_analysis.py
Rscript analysis/analysis/issue_12_all.R                  # Outputs: output/tables/, output/plots/
```

## Issues and Pull Requests

### Issue Workflow

Each analysis task is tracked as a GitHub issue with a corresponding markdown file in `analysis/issues/`. This provides:
- Clear specification of objectives and scope
- Documentation of implementation decisions
- Record of completed work

**Issue file naming:** `issue_N_short_description.md`

### Pull Request Workflow

1. Create a branch from `main` for each issue
2. Implement changes, committing frequently
3. Update the issue markdown with implementation notes
4. Open a PR referencing the issue number
5. Merge after review

### Contributing

- Check existing issues for planned work
- Open an issue before starting new analysis
- Follow existing patterns in `derived/` and `analysis/`
- Ensure outputs are generated correctly before PR

## Quick Start

```bash
# Install dependencies
uv sync

# Run the experiment locally
uv run otree devserver

# Run analysis
uv run python analysis/analysis/analysis_plots.py
```

## Repository

[github.com/calebeynon/emotionsgame_5](https://github.com/calebeynon/emotionsgame_5)
