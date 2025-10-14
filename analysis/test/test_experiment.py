#!/usr/bin/env python3
"""
Test script to demonstrate the new multi-session Experiment functionality.
"""

import experiment_data

def main():
    """Test the new Experiment functionality with a simple example."""
    
    # For demonstration, we'll use the same file multiple times
    # In practice, you'd have different session files
    csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/all_apps_wide_2025-09-11.csv'
    chat_csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/ChatMessages-2025-09-11.csv'
    
    # Test 1: Create an experiment with a single session
    print("=== Testing Experiment Class ===")
    file_pairs = [
        (csv_path, chat_csv_path, 1),  # Treatment 1
        # In practice, you'd add more file pairs here:
        # ('/path/to/session2.csv', '/path/to/session2_chat.csv', 2),  # Treatment 2
        # ('/path/to/session3.csv', None, 1),  # No chat data, Treatment 1
    ]
    
    try:
        experiment = experiment_data.load_experiment_data(file_pairs, name="Test Experiment")
        
        print(f"Created experiment: {experiment.name}")
        print(f"Sessions loaded: {len(experiment.sessions)}")
        print(f"Session codes: {experiment.list_session_codes()}")
        
        # Test aggregation functions
        print("\n=== Testing Aggregation Functions ===")
        
        # Overall sentiment across all sessions
        overall_sentiment = experiment.get_overall_sentiment()
        if overall_sentiment:
            print(f"Overall chat sentiment: {overall_sentiment}")
        else:
            print("No chat messages found across sessions")
        
        # Session-level sentiments
        session_sentiments = experiment.get_session_sentiments()
        print(f"Per-session sentiments: {len(session_sentiments)} sessions")
        
        # Convert to DataFrame
        df = experiment.to_dataframe_contributions()
        if df is not None:
            print(f"Contribution DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Sessions in DataFrame: {df['session_code'].unique()}")
            print(f"Treatments in DataFrame: {df['treatment'].unique()}")
            print(f"Segments in DataFrame: {df['segment'].unique()}")
        else:
            print("No contribution data found")
            
        print("\n=== Single Session Access (via Experiment) ===")
        
        # Access individual sessions
        for session_code in experiment.list_session_codes():
            session = experiment.get_session(session_code)
            if session:
                print(f"Session {session_code} (Treatment {session.treatment}): {len(session.segments)} segments, "
                      f"{len(session.participant_labels)} participants")
                
                # Access data through the session within the experiment
                supergame1 = session.get_supergame(1)
                if supergame1:
                    round1 = supergame1.get_round(1)
                    if round1:
                        players = list(round1.players.keys())
                        print(f"  Supergame 1, Round 1 players: {players}")
        
        print("\nTest completed successfully!")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("This is expected if you don't have the test data files.")
        print("Replace the file paths with your actual session data files.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()