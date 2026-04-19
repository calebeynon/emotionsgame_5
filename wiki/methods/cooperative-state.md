---
title: "Cooperative State Classification"
type: method
tags: [classification, cooperative-state, group-level, player-level, issue-31]
summary: "Two flavors of cooperative-state classification: group-level (everyone shares) and player-level (others' contributions only)"
status: active
last_verified: "2026-04-19"
---

## Summary

Issue #31 introduced state classification at two granularities. The **group-level** version flags all four players in a group-round identically based on the group mean. The **player-level** version flags each player based on the *other three* players' contributions, so two players in the same group can have different states.

## Group-Level (`classify_states.py`)

- A group-round is **cooperative** if group mean contribution ≥ 75% of endowment (≥ 18.75).
- All 4 players in that group-round inherit the same state.
- Output: `datastore/derived/state_classification.csv`.

## Player-Level (`classify_player_states.py`)

- Each player's state is determined by the **other 3** players' total contributions.
- Cooperative if others' total ≥ 60.
- Two players in the same group can have different states (e.g., a high-contributor surrounded by defectors will see a noncooperative world).
- Output: `datastore/derived/player_state_classification.csv`.

## Sample Sizes (10 sessions, 3,520 obs)

| Metric | Group-Level | Player-Level |
|---|---|---|
| Cooperative state | 2,528 (71.8%) | 2,309 (65.6%) |
| Noncooperative state | 992 (28.2%) | 1,211 (34.4%) |

The player-level approach reclassifies 219 observations from cooperative to noncooperative — these are cases where a high contributor's own contribution inflated the group mean past the threshold even though their environment was uncooperative.

## When to Use Which

- **Group-level**: Group-as-unit analyses (e.g., does cooperative-state group chat differ from noncooperative-state group chat?).
- **Player-level**: Individual decisions and reactions (e.g., does a player's own subsequent behavior depend on whether *others* cooperated, holding their own contribution constant?). This is what most regressions in the paper want.

## Tests

- `tests/test_classify_states.py` (26 unit + 12 export tests, group-level)
- `tests/test_classify_player_states.py` (20 unit + 12 export tests, player-level)
- `tests/test_classify_states_integration.py`

## Related

- [Behavior Classification](behavior-classification.md)
- [Merged Panel Construction](merged-panel.md)
