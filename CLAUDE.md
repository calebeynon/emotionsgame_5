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

# Analysis (run from analysis/ directory)
uv run python analysis/analysis_plots.py
uv run python analysis/multi_session_analysis.py
uv run python analysis/classify_promises.py
uv run python analysis/classify_behavior.py
uv run python annotations/build_edited_data_csv.py
uv run python annotations/generate_annotations.py
```

## Experimental Architecture

### Hierarchical Data Structure
```
Experiment (multi-session with treatments)
  └─ Session (16 participants, session_code)
      └─ Segment/Supergame (introduction, supergame1-5, finalresults)
          └─ Round (numbered within segment)
              └─ Group (4 players, group_id_in_subsession)
                  └─ Player (label A-R, skips I/O to avoid confusion with 1/0)
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

### Participant Labels
- Labels A-R loaded from `participant_labels.txt`
- I and O are skipped to avoid confusion with 1 and 0
- Consistent across sessions via room-based assignment

## Environment Variables
- `GEMINI_API_KEY`: Required for promise classification in `analysis_up8.py`

## Data Files
- **Input**: oTree data CSV, oTree chat CSV, PageTimes CSV, timesheet.xlsx
- **Output**: Analysis results in `analysis/results/`, annotations in `analysis/datastore/annotations/`
