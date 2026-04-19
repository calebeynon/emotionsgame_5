# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an oTree behavioral economics experiment implementing a **public goods game with chat**. The experiment studies how communication affects contribution behavior across multiple rounds with strategic regrouping between supergames.

- **16 participants**, 4 players per group
- **7 sequential apps**: introduction → supergame1-5 → finalresults
- **Game mechanics**: 25-point endowment, 0.4 multiplier, payoff = endowment - contribution + (group_total × multiplier)

## Key Commands

```bash
# Development
uv run otree devserver              # Run dev server
uv run otree test                   # Run all app tests
uv run otree test supergame1        # Test specific app
uv run otree resetdb                # Reset database

# Production
uv run otree prodserver
uv run otree zip

# Overleaf (analysis/paper synced via git subtree)
# Auto-syncs to Overleaf on push to main via GitHub Action (.github/workflows/sync-overleaf.yml)
# Action parses .tex files for \input and \includegraphics, copies referenced files from output/
git subtree pull --prefix=analysis/paper overleaf master --squash  # Pull from Overleaf
git subtree push --prefix=analysis/paper overleaf master           # Manual push to Overleaf

# Analysis (run from analysis/ directory)
uv run python analysis/analysis_plots.py
uv run python analysis/multi_session_analysis.py
uv run python derived/classify_promises.py
uv run python derived/classify_behavior.py
uv run python annotations/build_edited_data_csv.py
uv run python annotations/generate_annotations.py
```

### Paper Directory (`analysis/paper/`)
All `.tex` files in the paper directory **must** include the following path resolution preamble so they compile both locally and on Overleaf:

```latex
%%%%%%%%% Path resolution (Overleaf: tables/, plots/; Local: ../output/) %%%%%%%%%
\graphicspath{{plots/}{../output/plots/}}
\makeatletter
\def\input@path{{tables/}{../output/tables/}}
\makeatother
```

- Use **bare filenames** in `\input{}` and `\includegraphics{}` (no directory prefix)
- Locally, LaTeX resolves via `../output/tables/` and `../output/plots/`
- On Overleaf, the GitHub Action copies referenced files into `tables/` and `plots/`
- Do NOT manually commit files to `analysis/paper/tables/` or `analysis/paper/plots/` — the Action manages these

## Experimental Architecture

### Hierarchical Data Structure
```
Experiment (multi-session with treatments)
  └─ Session (16 participants, session_code)
      └─ Segment/Supergame (introduction, supergame1-5, finalresults)
          ├─ orphan_chats: Dict[str, List[ChatMessage]] (last round's chat, keyed by player label)
          └─ Round (numbered within segment)
              └─ Group (4 players, group_id_in_subsession)
                  └─ Player (label A-R, skips I/O to avoid confusion with 1/0)
                      └─ chat_messages: List[ChatMessage] (chat that INFLUENCED this contribution)
```

### Supergame Configuration
| Supergame | Rounds | Chat Timeout |
|-----------|--------|--------------|
| supergame1 | 3 | 120s first round, 30s after |
| supergame2 | 4 | 120s first round, 30s after |
| supergame3 | 3 | 120s first round, 30s after |
| supergame4 | 7 | 120s first round, 30s after |
| supergame5 | 5 | 120s first round, 30s after |

Each supergame uses different grouping matrices defined in `creating_session()` to study repeated game effects.

### oTree App Pattern (supergame1-5)
Each supergame app follows this structure:

**Models:**
- `C.ENDOWMENT = cu(25)`, `C.MULTIPLIER = 0.4`, `C.PLAYERS_PER_GROUP = 4`
- `Player.contribution` (0 to ENDOWMENT)
- `Group.total_contribution`, `Group.individual_share`

**Key Functions:**
- `creating_session()`: Sets group matrix via `subsession.set_group_matrix(grouping)`
- `set_payoffs()`: Calculates payoffs, stores in `participant.vars['payoff_list']` and `participant.vars['payoff_sum_X']`
- `set_payoffs_0()`: Resets payoff tracking between supergames

**Page Sequence:**
1. `StartPage` - Shows group members (round 1 only)
2. `ChatFirst` - 120s chat (round 1 only)
3. `Chat` - 30s chat with previous round feedback (rounds 2+)
4. `Contribute` - Main decision
5. `Results` - Round outcomes with cumulative table
6. `RegroupingMessage` - End-of-supergame transition

## Analysis Module

**Use `/experiment-data` skill when working with the `experiment_data.py` data object.**

### Data Loading
```python
from experiment_data import load_experiment_data

# Load multi-session experiment
file_pairs = [
    ('session1_data.csv', 'session1_chat.csv', 1),  # Treatment 1
    ('session2_data.csv', 'session2_chat.csv', 2),  # Treatment 2
]
experiment = load_experiment_data(file_pairs, name="My Experiment")

# Access nested data
player = experiment.get_session('abc123') \
    .get_supergame(1) \
    .get_round(1) \
    .get_player('A')
```

### Sentiment Analysis
Chat messages automatically include VADER sentiment analysis:
- Access via `player.get_chat_sentiment()`, `group.get_chat_sentiment()`, etc.
- Available at all hierarchy levels up to `experiment.get_overall_sentiment()`

### Annotation Pipeline
Two-stage process for behavioral video annotation:
1. `build_edited_data_csv.py`: PageTimes CSV + timesheet → normalized event data
2. `generate_annotations.py`: Event data → annotation markers with duration filtering

## Important Implementation Details

### Payoff Tracking
- Per-supergame payoffs stored in `participant.vars['payoff_list']`
- Cumulative sums stored as `participant.vars['payoff_sum_1']` through `payoff_sum_5`
- `set_payoffs_0()` resets the list between supergames

### Chat Implementation
- Uses oTree's built-in `{{ chat }}` with participant labels as nicknames
- Previous round feedback (others' contributions, own payoff) shown during chat phases
- Complex `vars_for_template()` logic handles round-specific display

**Chat-Round Pairing Semantics (Analysis Module):**
- In the oTree experiment, chat happens AFTER contribution in each round
- In `experiment_data.py`, `chat_messages` is paired with the contribution it INFLUENCED (next round's contribution)
- Round 1 of each supergame has empty `chat_messages` (no prior chat to influence it)
- Last round's chat influenced no subsequent contribution and is stored as `segment.orphan_chats`
- `orphan_chats` is a `Dict[str, List[ChatMessage]]` keyed by player label

### Participant Labels
- Labels A-R loaded from `participant_labels.txt`
- I and O are skipped to avoid confusion with 1 and 0
- Consistent across sessions via room-based assignment

## Environment Variables
- `GEMINI_API_KEY`: Required for promise classification in `analysis_up8.py`

## Symlinks
- `analysis/datastore` → `/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP` (shared Box folder with experiment data)
- When creating git worktrees, recreate this symlink: `ln -s /Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP <worktree>/analysis/datastore`

## Data Files
- **Input**: oTree data CSV, oTree chat CSV, PageTimes CSV, timesheet.xlsx (in `analysis/datastore/`)
- **Output**: Analysis results in `analysis/results/`, annotations in `analysis/datastore/annotations/`

## Issue Documentation
- All issue documentation (`.md` files) goes in `analysis/issues/`

## Wiki Maintenance (REQUIRED before pushing or opening a PR to main)

The project wiki at `wiki/` is the authoritative knowledge base for AI agents working on this repo. **Before any `git push` to main or `gh pr create` targeting main**, Claude must update the wiki to reflect the changes in the push/PR.

### What to do

1. **Diff the changes** about to be pushed/PR'd: `git diff main...HEAD` (or `git diff origin/main...HEAD`).
2. **Identify wiki articles affected** by those changes — check `wiki/_index.md` and skim relevant articles in `wiki/concepts/`, `wiki/methods/`, `wiki/tools/`, and `wiki/papers/`.
3. **Update, create, or delete** wiki articles so they remain accurate:
   - **Update** an existing article when its claims (results, file paths, sample sizes, script names, coefficients, etc.) are no longer correct.
   - **Create** a new article when the change introduces a new method, dataset, script, or major concept that an AI agent would need to know about.
   - **Delete** an article when the thing it documents has been removed from the codebase.
4. **Bump `last_verified`** in the frontmatter of any article you touch to today's date (use the current date from the environment).
5. **Regenerate `_index.md`** so article counts and listings stay in sync.
6. **Commit the wiki changes in the same push/PR** as the code changes — never as a follow-up.
7. **Run `/kb sync`** after the push/PR is merged to mirror to the GitHub Wiki.

### When to skip

Skip wiki updates only for:
- Pure formatting / typo fixes with no functional change.
- Changes to files the wiki does not document (e.g., `.claude/`, `.gitignore`, lockfile-only updates).
- Work-in-progress branches that won't merge to main.

### How to check wiki coverage

Run `/kb search <keyword>` to find articles related to the change. If a non-trivial change has no matching article, that's a signal to **create** one rather than skip the update.

### Article style for AI consumption

Wiki articles are read by AI agents, not just humans. Prefer:
- Concrete file paths, script names, output filenames, and sample sizes.
- Tables over prose where the content is enumerable.
- A "Related" section linking sibling articles via relative paths (`../tools/foo.md`, never `wiki/...`).
- Keep `summary:` in frontmatter to one line — it shows up in search results.


# Repo Owner Github: calebeynon