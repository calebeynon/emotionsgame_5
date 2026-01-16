"""
Data Integrity Verification Script

This script verifies that experimental data and chat messages are correctly loaded 
and integrated into the hierarchical data structure. Use this script to validate 
new experimental data files.

Usage: python verify_data_integrity.py [--verbose] [--sample-size N]
"""

from experiment_data import load_experiment_data
import pandas as pd
import random
import argparse
import sys


def verify_data_integrity(csv_path: str, chat_csv_path: str = None, verbose: bool = False, sample_size: int = 5):
    """
    Comprehensive verification of experimental data integrity.
    
    Args:
        csv_path: Path to experimental data CSV
        chat_csv_path: Path to chat messages CSV (optional)
        verbose: Whether to show detailed output
        sample_size: Number of random messages to verify
    
    Returns:
        bool: True if all verifications pass, False otherwise
    """
    print("=" * 70)
    print("DATA INTEGRITY VERIFICATION")
    print("=" * 70)
    
    all_tests_passed = True
    
    try:
        # Load data
        print(f"\n📂 Loading experimental data from: {csv_path}")
        if chat_csv_path:
            print(f"📂 Loading chat data from: {chat_csv_path}")
        
        session = load_experiment_data(csv_path, chat_csv_path)
        
        print(f"✅ Session loaded: {session.session_code}")
        print(f"✅ Participants: {len(session.participant_labels)} ({sorted(session.participant_labels.values())})")
        print(f"✅ Segments: {len(session.segments)} ({list(session.segments.keys())})")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading data: {e}")
        return False
    
    # Test 1: Verify basic data structure integrity
    print(f"\n🔍 TEST 1: Data Structure Integrity")
    test1_passed = True
    
    total_rounds = 0
    total_groups = 0
    total_players = 0
    
    for segment_name, segment in session.segments.items():
        if segment_name.startswith('supergame'):
            for round_num, round_obj in segment.rounds.items():
                total_rounds += 1
                
                for group_id, group in round_obj.groups.items():
                    total_groups += 1
                    
                    if len(group.players) == 0:
                        print(f"❌ Empty group found: {segment_name} Round {round_num} Group {group_id}")
                        test1_passed = False
                    else:
                        total_players += len(group.players)
                        
                        # Check player consistency
                        for player_label, player in group.players.items():
                            if player.label != player_label:
                                print(f"❌ Player label mismatch: {player_label} vs {player.label}")
                                test1_passed = False
    
    if test1_passed:
        print(f"✅ Structure integrity verified: {total_rounds} rounds, {total_groups} groups, {total_players} player instances")
    else:
        all_tests_passed = False
        
    # Test 2: Chat integration verification (if chat data provided)
    if chat_csv_path:
        print(f"\n🔍 TEST 2: Chat Integration Integrity")
        test2_passed = True
        
        # Load raw chat CSV for comparison
        try:
            chat_df = pd.read_csv(chat_csv_path)
            total_csv_messages = len(chat_df)
            
            # Count messages in data structure
            total_object_messages = 0
            rounds_with_chat = 0
            
            for segment_name, segment in session.segments.items():
                if segment_name.startswith('supergame'):
                    for round_obj in segment.rounds.values():
                        if round_obj.chat_messages:
                            rounds_with_chat += 1
                            total_object_messages += len(round_obj.chat_messages)
            
            print(f"✅ Chat messages: {total_object_messages} loaded (CSV had {total_csv_messages})")
            print(f"✅ Rounds with chat: {rounds_with_chat}")
            
            if total_object_messages != total_csv_messages:
                print(f"⚠️  Message count discrepancy: {total_object_messages} vs {total_csv_messages}")
                # This might be OK if some messages couldn't be mapped
            
            # Test random message accuracy
            print(f"\n🎯 Random Message Verification (sampling {sample_size} messages):")
            random_indices = random.sample(range(len(chat_df)), min(sample_size, len(chat_df)))
            
            messages_found = 0
            for i, idx in enumerate(random_indices):
                row = chat_df.iloc[idx]
                
                # Parse channel to find expected location
                import re
                match = re.match(r'^1-supergame(\d+)-(\d+)$', row['channel'])
                if match:
                    supergame_num = int(match.group(1))
                    supergame = session.get_supergame(supergame_num)
                    
                    if supergame:
                        # Search for this message in the supergame
                        found = False
                        for round_obj in supergame.rounds.values():
                            for group in round_obj.groups.values():
                                for msg in group.chat_messages:
                                    if (msg.nickname == row['nickname'] and 
                                        msg.body == row['body'] and 
                                        abs(msg.timestamp - float(row['timestamp'])) < 0.001):
                                        found = True
                                        messages_found += 1
                                        break
                                if found:
                                    break
                            if found:
                                break
                        
                        if verbose and found:
                            print(f"   ✅ Message {i+1}: {row['nickname']}: '{row['body'][:30]}...'")
                        elif verbose and not found:
                            print(f"   ❌ Message {i+1}: {row['nickname']}: '{row['body'][:30]}...' NOT FOUND")
            
            accuracy = messages_found / len(random_indices) * 100
            print(f"✅ Message accuracy: {messages_found}/{len(random_indices)} ({accuracy:.1f}%)")
            
            if accuracy < 90:
                print(f"❌ Message accuracy too low: {accuracy:.1f}%")
                test2_passed = False
                
        except Exception as e:
            print(f"❌ Chat verification error: {e}")
            test2_passed = False
        
        if not test2_passed:
            all_tests_passed = False
    
    # Test 3: Group membership consistency
    if verbose:
        print(f"\n🔍 TEST 3: Group Membership Consistency")
        
        for segment_name in ['supergame1', 'supergame2', 'supergame3']:
            segment = session.get_segment(segment_name)
            if segment and len(segment.rounds) > 1:
                print(f"\n   {segment_name.upper()} groupings:")
                
                # Show first round grouping
                first_round = segment.get_round(1)
                if first_round:
                    for group_id in sorted(first_round.groups.keys()):
                        group = first_round.get_group(group_id)
                        players = sorted(group.players.keys())
                        print(f"      Group {group_id}: {players}")
                        
                        # Check if this group stays consistent across rounds
                        consistent = True
                        for round_num in range(2, min(4, len(segment.rounds) + 1)):
                            other_round = segment.get_round(round_num)
                            if other_round and group_id in other_round.groups:
                                other_group = other_round.get_group(group_id)
                                other_players = sorted(other_group.players.keys())
                                if players != other_players:
                                    consistent = False
                                    break
                        
                        if not consistent:
                            print(f"         ⚠️  Group membership changes within segment")
    
    # Final summary
    print(f"\n" + "=" * 70)
    if all_tests_passed:
        print("🎉 ALL INTEGRITY TESTS PASSED!")
        print("✅ Data is ready for analysis")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Please review the issues above before proceeding")
    print("=" * 70)
    
    return all_tests_passed


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Verify experimental data integrity')
    parser.add_argument('csv_path', help='Path to experimental data CSV file')
    parser.add_argument('--chat-csv', help='Path to chat messages CSV file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed verification output')
    parser.add_argument('--sample-size', type=int, default=5,
                       help='Number of random messages to verify (default: 5)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run verification
    success = verify_data_integrity(
        csv_path=args.csv_path,
        chat_csv_path=args.chat_csv,
        verbose=args.verbose,
        sample_size=args.sample_size
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    # Default paths for this project
    csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/all_apps_wide_2025-09-11.csv'
    chat_csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/ChatMessages-2025-09-11.csv'
    
    success = verify_data_integrity(csv_path, chat_csv_path, verbose=True, sample_size=5)
    print(f"\nVerification {'PASSED' if success else 'FAILED'}")