#!/usr/bin/env python3
"""
Example script demonstrating chat data access and analysis.

This script shows various ways to access chat messages from the loaded experimental data,
including filtering by player, group, round, and performing basic text analysis.
"""

from experiment_data import load_experiment_data
import re
from collections import Counter

def main():
    print("============================================================")
    print("CHAT DATA ACCESS EXAMPLES")
    print("============================================================")
    
    # Load experiment with chat data
    session = load_experiment_data(
        "/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/all_apps_wide_2025-09-11.csv",
        chat_csv_path="/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/ChatMessages-2025-09-11.csv"
    )
    
    print(f"Session: {session.session_code}")
    
    # Count total chat messages
    total_messages = 0
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                total_messages += len(round_obj.chat_messages)
    
    print(f"Total chat messages: {total_messages}")
    print()
    
    # Example 1: Access all messages from a specific player
    print("📍 EXAMPLE 1: All messages from Player A")
    print("=" * 50)
    messages_a = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    if msg.nickname == 'A':
                        messages_a.append(msg)
    
    print(f"Player A sent {len(messages_a)} messages:")
    for i, msg in enumerate(messages_a[:3]):  # Show first 3
        print(f"  {i+1}. '{msg.body}'")
    if len(messages_a) > 3:
        print(f"  ... and {len(messages_a) - 3} more")
    print()
    
    # Example 2: Messages from a specific round and group
    print("📍 EXAMPLE 2: Supergame 1, Round 2 group chat")
    print("=" * 50)
    supergame1 = session.get_segment('supergame1')
    round2 = supergame1.get_round(2)
    if round2:
        group1 = round2.get_group(1)  # First group
        
        if group1:
            print(f"Group 1 members: {sorted(group1.players.keys())}")
            group_messages = group1.chat_messages
            print(f"Group 1 exchanged {len(group_messages)} messages in round 2:")
            for msg in group_messages[:5]:  # Show first 5
                print(f"  {msg.nickname}: '{msg.body}'")
            if len(group_messages) > 5:
                print(f"  ... and {len(group_messages) - 5} more")
        else:
            print("Group 1 not found in round 2")
    else:
        print("Round 2 not found in supergame 1")
    print()
    
    # Example 3: Text analysis - word frequency
    print("📍 EXAMPLE 3: Most common words in all chat")
    print("=" * 50)
    
    # Collect all message text
    all_messages = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    all_messages.append(msg.body.lower())
    
    all_text = " ".join(all_messages)
    # Simple word extraction (remove punctuation, split)
    words = re.findall(r'\b[a-z]{3,}\b', all_text)  # Words of 3+ letters
    word_counts = Counter(words)
    
    print("Top 10 most common words:")
    for word, count in word_counts.most_common(10):
        print(f"  '{word}': {count}")
    print()
    
    # Example 4: Communication patterns by round
    print("📍 EXAMPLE 4: Communication volume by round")
    print("=" * 50)
    round_message_counts = {}
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                round_key = f"{segment.name}_r{round_obj.round_number}"
                round_message_counts[round_key] = len(round_obj.chat_messages)
    
    print("Messages per round:")
    for round_name, count in sorted(round_message_counts.items()):
        print(f"  {round_name}: {count} messages")
    print()
    
    # Example 5: Player participation analysis
    print("📍 EXAMPLE 5: Player chat participation")
    print("=" * 50)
    player_message_counts = {}
    
    # Count messages for each player
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    if msg.nickname not in player_message_counts:
                        player_message_counts[msg.nickname] = 0
                    player_message_counts[msg.nickname] += 1
    
    # Sort by message count
    sorted_players = sorted(player_message_counts.items(), key=lambda x: x[1], reverse=True)
    print("Players ranked by chat activity:")
    for label, count in sorted_players:
        print(f"  Player {label}: {count} messages")
    print()
    
    print("============================================================")
    print("CHAT DATA ACCESS EXAMPLES COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()