#!/usr/bin/env python3
"""
Comprehensive example showing sentiment analysis capabilities.

This script demonstrates how to access sentiment data at all levels:
- Individual message sentiment
- Player-level sentiment (per round and aggregated)
- Group-level sentiment (per round)
- Round-level sentiment 
- Segment/Supergame-level sentiment
- Session-level sentiment
"""

from experiment_data import load_experiment_data

def main():
    print("=" * 70)
    print("COMPREHENSIVE SENTIMENT ANALYSIS EXAMPLE")
    print("=" * 70)
    
    # Load data
    csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/all_apps_wide_2025-09-11.csv'
    chat_csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/ChatMessages-2025-09-11.csv'
    session = load_experiment_data(csv_path, chat_csv_path)
    
    print("📊 DATA OVERVIEW")
    print("-" * 50)
    print(f"Session: {session.session_code}")
    print(f"Total messages: {len(session.get_all_chat_messages())}")
    overall_sentiment = session.get_overall_sentiment()
    print(f"Overall sentiment: {overall_sentiment}")
    print()
    
    # 1. Session-level analysis
    print("🌍 SESSION-LEVEL SENTIMENT ANALYSIS")
    print("-" * 50)
    print("Sentiment by supergame:")
    sg_sentiments = session.get_supergame_sentiments()
    for sg_num, sentiment in sg_sentiments.items():
        if sentiment:
            print(f"  Supergame {sg_num}: {sentiment.compound:.3f} ({sentiment.dominant_sentiment})")
    print()
    
    # 2. Player-level analysis across entire session
    print("👥 PLAYER-LEVEL SENTIMENT ANALYSIS (Entire Session)")
    print("-" * 50)
    player_sentiments = session.get_all_player_sentiments_across_session()
    # Sort by compound sentiment
    sorted_by_sentiment = sorted([(label, sentiment) for label, sentiment in player_sentiments.items() 
                                if sentiment is not None], 
                               key=lambda x: x[1].compound, reverse=True)
    
    print("Most positive to most negative players:")
    for i, (label, sentiment) in enumerate(sorted_by_sentiment, 1):
        print(f"  {i:2d}. Player {label}: {sentiment.compound:+.3f} ({sentiment.message_count} msgs)")
    print()
    
    # 3. Supergame-specific analysis
    print("🎮 SUPERGAME-SPECIFIC ANALYSIS: Supergame 3 (highest sentiment)")
    print("-" * 50)
    sg3 = session.get_supergame(3)
    if sg3:
        print(f"Supergame 3 overall: {sg3.get_chat_sentiment()}")
        
        # Round-by-round sentiment
        print("\\nRound-by-round sentiment in Supergame 3:")
        round_sentiments = sg3.get_round_sentiments()
        for round_num, sentiment in sorted(round_sentiments.items()):
            if sentiment:
                print(f"  Round {round_num}: {sentiment.compound:+.3f} ({sentiment.message_count} msgs)")
        
        # Player sentiments in this supergame
        print("\\nPlayer sentiment in Supergame 3:")
        player_sg3_sentiments = sg3.get_all_player_sentiments_across_rounds()
        for label, sentiment in sorted(player_sg3_sentiments.items()):
            if sentiment:
                print(f"  Player {label}: {sentiment.compound:+.3f} ({sentiment.message_count} msgs)")
    print()
    
    # 4. Detailed round analysis
    print("🎯 DETAILED ROUND ANALYSIS: Supergame 1, Round 1")
    print("-" * 50)
    sg1 = session.get_supergame(1)
    if sg1:
        round1 = sg1.get_round(1)
        if round1:
            print(f"Round 1 overall: {round1.get_chat_sentiment()}")
            
            # Group-by-group analysis
            print("\\nGroup-by-group sentiment:")
            group_sentiments = round1.get_group_sentiments()
            for group_id, sentiment in sorted(group_sentiments.items()):
                if sentiment:
                    group = round1.get_group(group_id)
                    players = sorted(group.players.keys())
                    print(f"  Group {group_id} ({', '.join(players)}): {sentiment.compound:+.3f} ({sentiment.message_count} msgs)")
            
            # Individual player sentiment in this round
            print("\\nPlayer sentiment in Round 1:")
            player_round_sentiments = round1.get_player_sentiments()
            for label, sentiment in sorted(player_round_sentiments.items()):
                if sentiment:
                    print(f"  Player {label}: {sentiment.compound:+.3f} ({sentiment.message_count} msgs)")
    print()
    
    # 5. Individual message analysis
    print("💬 INDIVIDUAL MESSAGE SENTIMENT EXAMPLES")
    print("-" * 50)
    all_messages = session.get_all_chat_messages()
    
    # Find most positive and negative messages
    sorted_messages = sorted(all_messages, key=lambda m: m.sentiment)
    
    print("Most negative messages:")
    negative_msgs = [msg for msg in sorted_messages if msg.sentiment < -0.1][:3]
    for msg in negative_msgs:
        print(f"  {msg.sentiment:+.3f}: {msg.nickname} - '{msg.body}'")
    
    print("\\nMost positive messages:")
    positive_msgs = [msg for msg in sorted_messages if msg.sentiment > 0.1][-3:]
    for msg in positive_msgs:
        print(f"  {msg.sentiment:+.3f}: {msg.nickname} - '{msg.body}'")
    print()
    
    # 6. Sentiment trends across supergames
    print("📈 SENTIMENT TRENDS ACROSS SUPERGAMES")
    print("-" * 50)
    print("Average sentiment by supergame:")
    for sg_num in range(1, 6):
        sentiment = sg_sentiments.get(sg_num)
        if sentiment:
            print(f"  Supergame {sg_num}: {sentiment.compound:+.3f} " + 
                  f"(pos: {sentiment.positive:.2f}, neg: {sentiment.negative:.2f}, " +
                  f"neu: {sentiment.neutral:.2f})")
    print()
    
    # 7. Usage examples for researchers
    print("🔬 PRACTICAL USAGE EXAMPLES FOR RESEARCHERS")
    print("-" * 50)
    print("# Example 1: Get sentiment for a specific player in a specific supergame")
    player_a_sg2_sentiment = session.get_supergame(2).get_player_sentiment_across_rounds('A')
    if player_a_sg2_sentiment:
        print(f"Player A sentiment in Supergame 2: {player_a_sg2_sentiment}")
    
    print("\\n# Example 2: Compare group sentiment in a specific round")
    round1 = session.get_supergame(1).get_round(1)
    if round1:
        for group_id, group in round1.groups.items():
            sentiment = group.get_chat_sentiment()
            if sentiment:
                print(f"Group {group_id} sentiment: {sentiment.compound:+.3f}")
    
    print("\\n# Example 3: Individual message sentiment scores")
    first_msg = all_messages[0]
    print(f"Message: '{first_msg.body}'")
    print(f"  Compound: {first_msg.sentiment:.3f}")
    print(f"  Positive: {first_msg.positive_sentiment:.3f}")
    print(f"  Negative: {first_msg.negative_sentiment:.3f}")
    print(f"  Neutral: {first_msg.neutral_sentiment:.3f}")
    
    print("\\n" + "=" * 70)
    print("SENTIMENT ANALYSIS EXAMPLE COMPLETE!")
    print("All sentiment data is now integrated into the data structure.")
    print("You can access sentiment at any level: message → player → group → round → segment → session")
    print("=" * 70)

if __name__ == "__main__":
    main()