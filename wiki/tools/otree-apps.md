---
title: "oTree Experiment Apps"
type: tool
tags: [otree, experiment, public-goods, chat]
summary: "7 oTree apps implementing a public goods game with chat across 5 supergames"
status: draft
last_verified: "2026-04-06"
---

## Summary

The experiment consists of 7 sequential oTree apps: `introduction` (instructions/quiz), `supergame1-5` (core public goods game with chat), and `finalresults` (survey + payment). Each supergame has different round counts and group assignments to study repeated game effects with strategic regrouping.

## Key Points

- **Game mechanics**: 25-point endowment, 0.4 MPCR, payoff = 25 - contribution + (group_total x 0.4)
- **Groups**: 4 players per group, 16 participants total, regrouped between supergames
- **Chat**: 120s first-round chat, 30s subsequent rounds, showing previous contributions
- **Labels**: Participants labeled A-R (skipping I/O to avoid 1/0 confusion)

## App Sequence

| App | Purpose | Rounds |
|-----|---------|--------|
| `introduction` | Instructions, practice, comprehension quiz | 1 |
| `supergame1` | Groups: [1,5,9,13], [2,6,10,14], [3,7,11,15], [4,8,12,16] | 3 |
| `supergame2` | Groups: different matrix | 4 |
| `supergame3` | Groups: different matrix | 3 |
| `supergame4` | Groups: different matrix | 7 |
| `supergame5` | Groups: [1,8,10,15], [2,7,9,16], [3,6,12,13], [4,5,11,14] | 5 |
| `finalresults` | Demographics survey + final payment | 1 |

## Supergame Page Flow

Each supergame round follows this page sequence:

1. **StartPage** — Shows group member labels (round 1 only)
2. **RoundWaitPage** — Synchronizes players
3. **ChatFirst** — 120s chat window (round 1 only)
4. **Chat** — 30s chat with previous round feedback: others' contributions, own payoff (rounds 2+)
5. **Contribute** — Player enters contribution (0-25)
6. **ResultsWaitPage** — Triggers `set_payoffs()` after all players arrive
7. **ResultsOnly** — Timed results display (25s)
8. **Results** — Full results with cumulative table (60s round 1, 40s after)
9. **RegroupingMessage** — End-of-supergame transition (last round only), triggers `set_payoffs_0()`

## Payoff System

- **Per-round**: `payoff = endowment - contribution + (group_total x multiplier)`
- **Tracking**: Payoffs appended to `participant.vars['payoff_list']`, cumulative sum stored as `participant.vars['payoff_sum_N']` for supergame N
- **Reset**: `set_payoffs_0()` clears `payoff_list` between supergames
- **Final**: One supergame randomly chosen; `final_payoff = selected_supergame_payoff + 75 points`; converted at $0.10/point + $7.50 participation fee

## Survey Fields (finalresults)

| Field | Question |
|-------|----------|
| q1 | Gender identity |
| q2 | Ethnicity |
| q3 | Age |
| q4 | Major |
| q5 | Number of siblings |
| q6 | Importance of religion (5-point scale) |

## Related

- [Project Architecture](wiki/tools/architecture.md)
- [Analysis Pipeline](wiki/tools/analysis-pipeline.md)
