# Hierarchical Experimental Data Structure

This module provides a hierarchical data structure for oTree experimental data with **multi-session loading capability** for experiment-level analysis.

## Quick Start

```python
from experiment_data import load_experiment_data

# Load multiple sessions into an experiment
file_pairs = [
    ('session1_data.csv', 'session1_chat.csv', 1),  # Treatment 1
    ('session2_data.csv', 'session2_chat.csv', 2),  # Treatment 2
    ('session3_data.csv', None, 1),  # No chat data, Treatment 1
]
experiment = load_experiment_data(file_pairs, name="My Experiment")

# Experiment-level analysis
overall_sentiment = experiment.get_overall_sentiment()
df = experiment.to_dataframe_contributions()

# Access individual sessions (all existing functionality preserved)
session = experiment.get_session(list(experiment.sessions.keys())[0])
contribution = session.get_supergame(1).get_round(1).get_player('A').contribution

# Access chat messages (note: round 1 has no chat - use round 2+ for chat that influenced contribution)
# Chat is paired with the round it INFLUENCED, not the round it occurred in
chat_messages = session.get_supergame(1).get_round(2).get_group(1).chat_messages
for msg in chat_messages:
    print(f"{msg.datetime.strftime('%H:%M:%S')} - {msg.nickname}: {msg.body}")
```

## Data Structure

The hierarchy now follows this pattern:
- **Experiment** → **Session** → **Segment** → **Round** → **Group** → **Player**

### Classes

- **`Experiment`**: Top-level container for multiple experimental sessions with aggregation methods
- **`Session`**: Container for entire experimental session (unchanged functionality)
- **`Segment`**: Contains data for one part of experiment (e.g., 'supergame1', 'introduction')
- **`Round`**: Contains data for one round within a segment
- **`Group`**: Contains data for one group within a round
- **`Player`**: Contains data for one player within a group/round
- **`ChatMessage`**: Individual chat message with nickname, body, timestamp, and datetime

## Access Patterns

### Basic Access

```python
# Load experiment data
file_pairs = [('session1.csv', 'chat1.csv', 1), ('session2.csv', 'chat2.csv', 2)]
experiment = load_experiment_data(file_pairs, name="My Experiment")

# Get experiment info
print(f"Experiment: {experiment.name}")
print(f"Sessions: {list(experiment.sessions.keys())}")

# Access individual session
session = experiment.get_session(list(experiment.sessions.keys())[0])
print(f"Session: {session.session_code}")
print(f"Participants: {list(session.participant_labels.values())}")
print(f"Segments: {list(session.segments.keys())}")
```

### Segment Access

```python
# Get specific segment
introduction = session.get_segment('introduction')
supergame1 = session.get_supergame(1)  # Shortcut for supergames
finalresults = session.get_segment('finalresults')
```

### Round Access

```python
# Get specific round within a segment
round1 = supergame1.get_round(1)
print(f"Round 1 has {len(round1.groups)} groups and {len(round1.players)} players")
```

### Player Access

```python
# Access player in specific round
player = session.get_supergame(1).get_round(1).get_player('A')
print(f"Player A contributed {player.contribution} in SG1 R1")

# Access player across all rounds in a segment
player_data = supergame1.get_player_across_rounds('A')
for round_num, player in player_data.items():
    print(f"Round {round_num}: {player.contribution}")

# Access player across entire session
all_data = session.get_player_across_session('A')
for segment_name, rounds_data in all_data.items():
    print(f"{segment_name}: {len(rounds_data)} rounds")
```

### Group Access

```python
# Access specific group
group = session.get_supergame(2).get_round(1).get_group(1)
print(f"Group total contribution: {group.total_contribution}")
print(f"Players in group: {list(group.players.keys())}")

# Access all groups in a round
round_obj = session.get_supergame(1).get_round(1)
for group_id, group in round_obj.groups.items():
    print(f"Group {group_id}: {[p.label for p in group.players.values()]}")
```

### Chat Message Access

**Important: Chat-Round Pairing Semantics**

Chat messages are paired with the round they **influenced**, not the round they occurred in.
This means:
- **Round 1**: Always has empty `chat_messages` (no prior chat to influence the first contribution)
- **Rounds 2+**: Contains chat from the previous round that influenced this contribution
- **Orphan chats**: Chat after the last round (which influenced no contribution) is stored at the segment level

```python
# Access chat messages at different levels

# Round 1 has no chat (no prior chat influenced this contribution)
round1_messages = session.get_supergame(1).get_round(1).chat_messages
print(f"Round 1 has {len(round1_messages)} messages")  # Always 0

# Round 2 contains chat from round 1 that influenced round 2's contribution
round2_messages = session.get_supergame(1).get_round(2).chat_messages
print(f"Round 2 has {len(round2_messages)} messages from prior chat")

# Chat messages from a specific group
group_messages = session.get_supergame(1).get_round(2).get_group(1).chat_messages
for msg in group_messages:
    print(f"{msg.datetime.strftime('%H:%M:%S')} - {msg.nickname}: {msg.body}")

# Chat messages from a specific player in a specific round
player = session.get_supergame(1).get_round(2).get_player('A')
player_messages = player.chat_messages
print(f"Player A's chat (that influenced round 2): {len(player_messages)} messages")

# Access orphan chats (chat after last round that influenced no contribution)
supergame1 = session.get_supergame(1)
orphan_chats = supergame1.orphan_chats  # List of ChatMessage objects
print(f"Orphan chats after last round: {len(orphan_chats)} messages")

# Chat messages across all rounds for a player
all_player_messages = []
for segment_name in ['supergame1', 'supergame2', 'supergame3']:
    segment = session.get_segment(segment_name)
    if segment:
        for round_obj in segment.rounds.values():
            player = round_obj.get_player('A')
            if player:
                all_player_messages.extend(player.chat_messages)
        # Include orphan chats from this segment
        all_player_messages.extend(segment.orphan_chats.get('A', []))

print(f"Player A sent {len(all_player_messages)} total messages")
```

## Data Attributes

### Player Attributes
- `participant_id`: Original participant ID
- `label`: Player label (A, B, C, etc.)
- `id_in_group`: Position within group (1-4)
- `contribution`: Contribution amount (for supergames)
- `payoff`: Round payoff
- `group_id`: Group membership
- `data`: Dictionary of additional player-specific data
- `chat_messages`: List of ChatMessage objects from this player in this round
- `get_chat_sentiment()`: Get sentiment analysis for this player's messages in this round

### Group Attributes
- `group_id`: Group identifier
- `players`: Dictionary of players in group (label → Player)
- `total_contribution`: Sum of all contributions in group
- `individual_share`: Amount each player receives from group pool
- `data`: Dictionary of additional group-specific data
- `chat_messages`: List of all ChatMessage objects for this group in this round
- `get_chat_sentiment()`: Get sentiment analysis for this group's messages
- `get_player_sentiments()`: Get sentiment analysis for each player in this group

### Round Attributes
- `round_number`: Round number within segment
- `groups`: Dictionary of groups (group_id → Group)
- `players`: Dictionary of all players (label → Player)
- `data`: Dictionary of additional round-specific data
- `chat_messages`: List of all ChatMessage objects for this round across all groups

### Segment Attributes
- `name`: Segment name ('supergame1', 'introduction', etc.)
- `rounds`: Dictionary of rounds (round_number → Round)
- `orphan_chats`: List of ChatMessage objects from chat after the last round (no contribution to influence)
- `data`: Dictionary of additional segment-specific data

### Experiment Attributes
- `name`: Experiment name
- `sessions`: Dictionary of sessions (session_code → Session)
- `metadata`: Experiment-level metadata
- `get_overall_sentiment()`: Aggregate sentiment across all sessions
- `get_session_sentiments()`: Per-session sentiment breakdown
- `to_dataframe_contributions()`: Convert all contribution data to DataFrame (includes treatment column)

### Session Attributes
- `session_code`: Unique session identifier
- `treatment`: Treatment condition (1 or 2)
- `segments`: Dictionary of segments (name → Segment)
- `participant_labels`: Mapping of participant_id → label
- `metadata`: Session-level metadata (fees, currency conversion, etc.)

### ChatMessage Attributes
- `nickname`: Player label (A, B, C, etc.) who sent the message
- `body`: The actual message text content
- `timestamp`: Unix timestamp when message was sent
- `datetime`: Python datetime object converted from timestamp
- `sentiment`: Overall compound sentiment score (-1 to 1)
- `positive_sentiment`: Positive sentiment component (0-1)
- `negative_sentiment`: Negative sentiment component (0-1)
- `neutral_sentiment`: Neutral sentiment component (0-1)
- `sentiment_scores`: Full VADER sentiment scores dictionary

### SentimentAnalysis Attributes
- `positive`: Average positive sentiment score (0-1)
- `negative`: Average negative sentiment score (0-1)
- `neutral`: Average neutral sentiment score (0-1)
- `compound`: Average overall sentiment score (-1 to 1)
- `message_count`: Number of messages analyzed
- `dominant_sentiment`: 'positive', 'negative', or 'neutral'
- `sentiment_intensity`: 'strong', 'moderate', or 'weak'

## Experiment-level Analysis

```python
# Load experiment data
file_pairs = [('session1.csv', 'chat1.csv', 1), ('session2.csv', 'chat2.csv', 2)]
experiment = load_experiment_data(file_pairs)

# Experiment-level aggregation
overall_sentiment = experiment.get_overall_sentiment()
session_sentiments = experiment.get_session_sentiments()

# Convert to DataFrame for analysis
df = experiment.to_dataframe_contributions()
print(df.groupby('session_code')['contribution'].mean())
print(df.groupby('treatment')['contribution'].mean())  # Compare by treatment

# Access individual sessions for detailed analysis
for session_code in experiment.list_session_codes():
    session = experiment.get_session(session_code)
    sg1 = session.get_supergame(1)

for label in sorted(session.participant_labels.values()):
    player_data = sg1.get_player_across_rounds(label)
    contributions = [p.contribution for p in player_data.values() 
                    if p.contribution is not None]
    if contributions:
        avg = sum(contributions) / len(contributions)
        print(f"Player {label}: {avg:.1f}")

# Analyze chat activity by player
for label in sorted(session.participant_labels.values()):
    total_messages = 0
    for round_obj in sg1.rounds.values():
        player = round_obj.get_player(label)
        if player:
            total_messages += len(player.chat_messages)
    print(f"Player {label} sent {total_messages} messages in Supergame 1")

# Analyze sentiment at different levels
overall_sentiment = session.get_overall_sentiment()
print(f"Overall session sentiment: {overall_sentiment}")

# Player sentiment across entire session
player_sentiments = session.get_all_player_sentiments_across_session()
for label, sentiment in player_sentiments.items():
    if sentiment:
        print(f"Player {label}: {sentiment}")

# Individual message sentiment
for msg in sg1.get_round(1).chat_messages[:3]:
    print(f"{msg.nickname}: '{msg.body}' (sentiment: {msg.sentiment:.3f})")
```

## Files

- `experiment_data.py`: Main module with classes and loading function
- `verify_data_integrity.py`: Verification script for validating data loading
- `analysis_plots.py`: Generate summary plots and statistics
- `chat_example.py`: Example script demonstrating chat data access patterns
- `sentiment_example.py`: Comprehensive demonstration of sentiment analysis capabilities
- `README.md`: This usage guide

## Requirements

- pandas
- Python 3.7+

## Notes

- All participant/session columns are filtered out during loading (as requested)
- Missing data is handled gracefully with None values
- Player labels (A-R) are preserved from the original data
- Survey data from finalresults segment is stored in player.data dictionary
- Chat messages are automatically loaded and integrated when chat CSV is provided
- Chat messages are sorted chronologically and accessible at player, group, and round levels
- Groups are correctly mapped between game data and chat data using group.id_in_subsession
- **Sentiment analysis** is automatically computed for all chat messages using NLTK's VADER sentiment analyzer
- Sentiment scores are available at all levels: individual messages, players, groups, rounds, segments, and session
- Each sentiment analysis includes positive, negative, neutral, and compound scores plus message count

## Chat-Round Pairing

**Chat messages are paired with the round they influenced, not the round they occurred in.**

This semantic change ensures proper causal analysis:

| Chat occurred... | Paired with... | Rationale |
|------------------|----------------|-----------|
| Before round 1 contribution | (none - no prior chat exists) | N/A |
| After round 1, before round 2 | Round 2 | This chat influenced round 2 contribution |
| After round N-1, before round N | Round N | This chat influenced round N contribution |
| After last round (round N) | Segment orphan_chats | No contribution to influence |

**Implications:**
- `round.chat_messages` for round 1 is always empty
- `round.chat_messages` for round N contains chat from round N-1
- `segment.orphan_chats` contains chat after the last round
