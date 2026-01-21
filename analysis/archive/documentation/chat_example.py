#!/usr/bin/env python3
"""
Example script demonstrating chat data access and analysis.

This script shows various ways to access chat messages from the loaded experimental data,
including filtering by player, group, round, and performing basic text analysis.

IMPORTANT: Chat-Round Pairing Semantics
Chat messages are paired with the round they INFLUENCED, not the round they occurred in.
- Round 1 always has empty chat_messages (no prior chat to influence first contribution)
- Round N contains chat from round N-1 that influenced round N contribution
- Orphan chats (after last round) are stored at segment.orphan_chats
"""

from experiment_data import load_experiment_data
import re
from collections import Counter


# =====
# Main function
# =====
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

    # Count total chat messages (excluding round 1 which has no chat)
    total_messages = count_total_messages(session)
    print(f"Total chat messages: {total_messages}")
    print()

    # Run all examples
    example_1_round_1_empty(session)
    example_2_chat_influences_contribution(session)
    example_3_orphan_chats(session)
    example_4_all_messages_from_player(session)
    example_5_word_frequency(session)
    example_6_communication_by_round(session)
    example_7_player_participation(session)

    print("============================================================")
    print("CHAT DATA ACCESS EXAMPLES COMPLETE!")
    print("============================================================")


# =====
# Helper functions
# =====
def count_total_messages(session):
    """Count total chat messages across all supergames."""
    total = 0
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                total += len(round_obj.chat_messages)
            # Include orphan chats if available
            if hasattr(segment, 'orphan_chats'):
                total += len(segment.orphan_chats)
    return total


def example_1_round_1_empty(session):
    """Demonstrate that round 1 has no chat messages."""
    print("EXAMPLE 1: Round 1 has empty chat_messages")
    print("=" * 50)
    supergame1 = session.get_segment('supergame1')
    round1 = supergame1.get_round(1)

    print(f"Round 1 chat_messages count: {len(round1.chat_messages)}")
    print("(Round 1 is always empty - no prior chat influenced this contribution)")
    print()


def example_2_chat_influences_contribution(session):
    """Demonstrate that round N contains chat from round N-1."""
    print("EXAMPLE 2: Chat that influenced round 2 contribution")
    print("=" * 50)
    supergame1 = session.get_segment('supergame1')
    round2 = supergame1.get_round(2)

    if round2:
        print(f"Round 2 chat_messages count: {len(round2.chat_messages)}")
        print("(These are messages from round 1 that influenced round 2 contributions)")

        group1 = round2.get_group(1)
        if group1:
            print(f"\nGroup 1 members: {sorted(group1.players.keys())}")
            print("Messages that influenced their round 2 contributions:")
            for msg in group1.chat_messages[:5]:
                print(f"  {msg.nickname}: '{msg.body}'")
            if len(group1.chat_messages) > 5:
                print(f"  ... and {len(group1.chat_messages) - 5} more")
    print()


def example_3_orphan_chats(session):
    """Demonstrate accessing orphan chats at segment level."""
    print("EXAMPLE 3: Orphan chats (after last round)")
    print("=" * 50)
    supergame1 = session.get_segment('supergame1')

    if hasattr(supergame1, 'orphan_chats'):
        orphan_count = len(supergame1.orphan_chats)
        print(f"Orphan chats after last round: {orphan_count}")
        print("(These messages occurred after the last contribution decision)")

        if orphan_count > 0:
            print("\nFirst few orphan messages:")
            for msg in supergame1.orphan_chats[:3]:
                print(f"  {msg.nickname}: '{msg.body}'")
    else:
        print("segment.orphan_chats not yet implemented")
    print()


def example_4_all_messages_from_player(session):
    """Collect all messages from a specific player."""
    print("EXAMPLE 4: All messages from Player A")
    print("=" * 50)
    messages_a = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            # Messages from rounds 2+ (round 1 is empty)
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    if msg.nickname == 'A':
                        messages_a.append(msg)
            # Include orphan chats
            if hasattr(segment, 'orphan_chats'):
                messages_a.extend(segment.orphan_chats.get('A', []))

    print(f"Player A sent {len(messages_a)} messages:")
    for i, msg in enumerate(messages_a[:3]):
        print(f"  {i+1}. '{msg.body}'")
    if len(messages_a) > 3:
        print(f"  ... and {len(messages_a) - 3} more")
    print()


def example_5_word_frequency(session):
    """Analyze word frequency in all chat messages."""
    print("EXAMPLE 5: Most common words in all chat")
    print("=" * 50)

    all_messages = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    all_messages.append(msg.body.lower())
            if hasattr(segment, 'orphan_chats'):
                for msg in segment.get_orphan_chats_flat():
                    all_messages.append(msg.body.lower())

    all_text = " ".join(all_messages)
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    word_counts = Counter(words)

    print("Top 10 most common words:")
    for word, count in word_counts.most_common(10):
        print(f"  '{word}': {count}")
    print()


def example_6_communication_by_round(session):
    """Show communication volume by round."""
    print("EXAMPLE 6: Communication volume by round")
    print("=" * 50)
    round_message_counts = {}

    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                round_key = f"{segment.name}_r{round_obj.round_number}"
                round_message_counts[round_key] = len(round_obj.chat_messages)
            # Track orphan chats separately
            if hasattr(segment, 'orphan_chats'):
                orphan_key = f"{segment.name}_orphan"
                round_message_counts[orphan_key] = len(segment.get_orphan_chats_flat())

    print("Messages per round (note: round 1 always 0):")
    for round_name, count in sorted(round_message_counts.items()):
        print(f"  {round_name}: {count} messages")
    print()


def example_7_player_participation(session):
    """Analyze player chat participation."""
    print("EXAMPLE 7: Player chat participation")
    print("=" * 50)
    player_message_counts = {}

    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    if msg.nickname not in player_message_counts:
                        player_message_counts[msg.nickname] = 0
                    player_message_counts[msg.nickname] += 1
            # Include orphan chats
            if hasattr(segment, 'orphan_chats'):
                for msg in segment.get_orphan_chats_flat():
                    if msg.nickname not in player_message_counts:
                        player_message_counts[msg.nickname] = 0
                    player_message_counts[msg.nickname] += 1

    sorted_players = sorted(
        player_message_counts.items(), key=lambda x: x[1], reverse=True
    )
    print("Players ranked by chat activity:")
    for label, count in sorted_players:
        print(f"  Player {label}: {count} messages")
    print()


# %%
if __name__ == "__main__":
    main()
