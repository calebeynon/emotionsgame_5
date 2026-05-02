---
title: "experiment_data.py — Hierarchical Data Module"
type: tool
tags: [experiment_data, data-model, sentiment, vader, otree-loader]
summary: "Core hierarchical data model: Experiment > Session > Segment > Round > Group > Player, with built-in VADER sentiment"
status: active
last_verified: "2026-05-01"
---

## Summary

`analysis/experiment_data.py` defines the canonical Python data model for the experiment. Every analysis script that needs structured access to chat, contributions, or per-player data should load through this module. There's a dedicated `/experiment-data` skill for working with it.

## Hierarchy

```
Experiment (multi-session)
  └─ Session (16 participants, session_code, treatment)
      └─ Segment (introduction, supergame1-5, finalresults)
          ├─ orphan_chats: Dict[label, List[ChatMessage]]   # last-round chat
          └─ Round (numbered within segment)
              └─ Group (4 players, group_id_in_subsession)
                  └─ Player (label A-R)
                      └─ chat_messages: List[ChatMessage]   # chat that INFLUENCED this round
```

## Loading

```python
from experiment_data import load_experiment_data

file_pairs = [
    ('session1_data.csv', 'session1_chat.csv', 1),  # IF (treatment code 1)
    ('session2_data.csv', 'session2_chat.csv', 2),  # AF (treatment code 2)
]
experiment = load_experiment_data(file_pairs, name="My Experiment")

# Hierarchical access
player = (experiment
          .get_session('abc123')
          .get_segment('supergame1')
          .get_round(2)
          .get_player('A'))

print(player.contribution, player.payoff)
print(player.get_chat_sentiment())  # SentimentAnalysis object
```

## Built-in Sentiment

- VADER (`nltk.sentiment.SentimentIntensityAnalyzer`) is initialized at module import.
- `ChatMessage.sentiment_scores` is lazy-cached on first access.
- Aggregation methods exist at every level: `Player.get_chat_sentiment()`, `Group.get_chat_sentiment()`, `Round.get_chat_sentiment()`, `Segment.get_overall_sentiment()`, `Experiment.get_overall_sentiment()`.
- `SentimentAnalysis` dataclass: `positive`, `negative`, `neutral`, `compound`, `message_count`. Convenience: `dominant_sentiment` ('positive'/'negative'/'neutral') and `sentiment_intensity` ('strong'/'moderate'/'weak').

## Critical Semantic: Chat-Round Pairing

Chat in oTree round N happens AFTER contribution in round N, so it influences round N+1. `experiment_data.py` re-pairs chat accordingly:

- Round 1 of every segment: `chat_messages = []`.
- The chat from the **last round** of a segment has no subsequent contribution to pair with — it lives in `Segment.orphan_chats[label]`.

This re-pairing is the foundation of every sentiment / embedding regression in the project. See [Chat-Round Pairing Semantics](../concepts/chat-round-pairing.md).

## Module Layout

| Class | Purpose |
|---|---|
| `ChatMessage` | nickname, body, timestamp, datetime, lazy sentiment scores |
| `SentimentAnalysis` (dataclass) | aggregated VADER scores |
| `Player` | label, participant_id, contribution, payoff, chat_messages |
| `Group` | players (by label), total_contribution, individual_share, chat_messages |
| `Round` | groups, players (flattened), chat_messages |
| `Segment` | rounds, orphan_chats |
| `Session` | segments, treatment, session_code |
| `Experiment` | sessions, name |

## When NOT to use it

For pure-numeric panel work (regressions, plotting), use the **derived CSVs** in `datastore/derived/` instead — they're faster and already incorporate the corrected chat pairing. Use `experiment_data.py` when you need structured chat-message access (e.g., classifying messages, building embeddings, exploring sentiment by group).

## Related

- [Chat-Round Pairing Semantics](../concepts/chat-round-pairing.md)
- [Datastore Files Reference](datastore-files.md)
- [Analysis Pipeline](analysis-pipeline.md)
