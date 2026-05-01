---
title: "Project Glossary"
type: concept
tags: [glossary, terminology, definitions]
summary: "Definitions of terms used throughout the project: treatments, segments, liar, sucker, thresholds"
status: active
last_verified: "2026-05-01"
---

## Summary

Definitions for terms used across the experiment, analysis code, and paper. AI agents should consult this when an issue, script, or commit message uses a term whose meaning isn't obvious from context.

## Experiment-Level Terms

| Term | Definition |
|---|---|
| **Segment / Supergame** | A block of repeated-game rounds with stable groups. The experiment has 5 segments: `supergame1` (3 rounds), `supergame2` (4), `supergame3` (3), `supergame4` (7), `supergame5` (5) — 22 rounds total. The codebase uses `segment` and `supergame` interchangeably; `Segment` is the preferred name in `experiment_data.py`. |
| **Round** | A single contribution decision within a segment, numbered 1 through (segment length). Each round = chat (rounds 2+) → contribute → results. |
| **IF (Individual Feedback)** | Encoded as `treatment == 1` in datastore CSVs. Players see **individual contributions** of each group member (and the aggregate) after each round. Note: earlier docs/code mislabeled this as "No Feedback / Treatment 1" — corrected in issue #74 to align with experimental instructions and Hanaki–Ozkes literature. |
| **AF (Aggregate Feedback)** | Encoded as `treatment == 2` in datastore CSVs. Players see only the **group total** contribution after each round, no individual breakdown. Earlier docs/code labeled this as "Feedback / Treatment 2"; corrected in issue #74. |
| **Endowment** | 25 ECU (experimental currency units) given to each player at the start of every round. |
| **MPCR / Multiplier** | 0.4 — each token in the group account returns 1.6 ECU split four ways. |
| **Random ending** | Each round has a 20% chance of ending the segment (continuation prob = 0.8), following Lugovskyy et al. (2010) Sequence 1. |
| **Label** | Letter A–R identifying a participant within a session. I and O are skipped to avoid 1/0 confusion. 16 labels per session. |

## Behavior-Classification Terms

| Term | Definition |
|---|---|
| **Promise** | A chat message classified by GPT-5-mini as a commitment to contribute a specific amount. |
| **Liar (cumulative, `is_liar_20`)** | A player who has broken at least one promise so far in the current segment (threshold: contribution < 20). Once flagged, persists for the rest of the segment. |
| **Liar (round-specific, `lied_this_round_20`)** | A player who broke a promise in **this specific round**. Doesn't persist. |
| **Sucker (`is_sucker_20`)** | A player who contributed the full 25 in a round when a groupmate broke a promise. |
| **Threshold 20** | Promise considered broken if contribution < 20. The "high" threshold — more players qualify as liars. |
| **Threshold 5** | Promise considered broken if contribution < 5. The "low" threshold — fewer qualify. |
| **Liar bucket** (issue #53) | Participant-level classification by total lie count across the experiment: `never` (0), `one_time` (1), `moderate` (2-3), `severe` (4+). |
| **Cooperative state** | Group-round (or player-round) flagged cooperative if mean contribution ≥ 75% of endowment (group level) or others' total ≥ 60 (player level). See [Cooperative State Classification](../methods/cooperative-state.md). |

## Threshold Naming Note

Earlier code used `_strict` and `_lenient` suffixes — these were swapped in issue #19 to `_20` and `_5` because "strict" was confusingly the *lenient-for-the-liar* threshold. Always use `_20` / `_5` in new code.

## Sample Sizes (Reference)

- **Sessions**: 10 (5 in IF, 5 in AF)
- **Participants**: 160 (16 per session)
- **Player-rounds**: 3,520 (160 × 22)
- **Player-rounds with chat**: 2,298 (rounds 2+ only)
- **Merged panel rows**: 10,683 (3 page types × 3,520 + 123 instruction rows)

## Related

- [Behavior Classification](../methods/behavior-classification.md)
- [Cooperative State Classification](../methods/cooperative-state.md)
- [Chat-Round Pairing Semantics](chat-round-pairing.md)
