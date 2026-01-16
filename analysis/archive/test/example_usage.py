#!/usr/bin/env python3
"""
Example showing how to use the new Experiment class to aggregate data across multiple sessions.

This demonstrates the enhanced experiment_data.py script that can now handle multiple sessions.
"""

import experiment_data
from typing import List, Tuple, Optional

def main():
    """Example of how to load and analyze multiple sessions."""
    
    # Example 1: Load multiple sessions into an Experiment
    print("=== Example 1: Loading Multiple Sessions ===")
    
    # Define your session files as (csv_path, chat_csv_path, treatment) tuples
    # chat_csv_path can be None if no chat data is available
    # treatment should be 1 or 2
    file_pairs: List[Tuple[str, Optional[str], int]] = [
        ('/path/to/session1.csv', '/path/to/session1_chat.csv', 1),
        ('/path/to/session2.csv', '/path/to/session2_chat.csv', 2),
        ('/path/to/session3.csv', None, 1),  # No chat data, treatment 1
    ]
    
    # Load the experiment (this would work with real file paths)
    # experiment = experiment_data.load_experiment_data(file_pairs, name="My Experiment")
    
    # For demonstration, we'll create a simple experiment manually
    experiment = experiment_data.Experiment("Demo Experiment")
    print(f"Created experiment: {experiment.name}")
    
    # Example 2: Accessing experiment-level data
    print("\n=== Example 2: Experiment-level Analysis ===")
    
    # Get overall sentiment across all sessions
    # overall_sentiment = experiment.get_overall_sentiment()
    # if overall_sentiment:
    #     print(f"Overall sentiment: {overall_sentiment.dominant_sentiment}")
    #     print(f"Compound score: {overall_sentiment.compound:.3f}")
    
    # Get sentiment for each session
    # session_sentiments = experiment.get_session_sentiments()
    # for session_code, sentiment in session_sentiments.items():
    #     if sentiment:
    #         print(f"Session {session_code}: {sentiment.dominant_sentiment}")
    
    # Example 3: Convert to DataFrame for analysis
    print("\n=== Example 3: Converting to DataFrame ===")
    
    # Convert contributions to a pandas DataFrame for easy analysis
    # df = experiment.to_dataframe_contributions()
    # if df is not None:
    #     print(f"Total contributions recorded: {len(df)}")
    #     print(f"Sessions: {df['session_code'].nunique()}")
    #     print(f"Unique players: {df['label'].nunique()}")
    #     
    #     # Group by session and segment to see contribution patterns
    #     grouped = df.groupby(['session_code', 'segment'])['contribution'].mean()
    #     print("Average contributions by session and segment:")
    #     print(grouped)
    
    # Example 4: Individual session access
    print("\n=== Example 4: Accessing Individual Sessions ===")
    
    # You can still access individual sessions within the experiment
    # for session_code in experiment.list_session_codes():
    #     session = experiment.get_session(session_code)
    #     print(f"Session {session_code}:")
    #     
    #     # Access specific data points
    #     sg1 = session.get_supergame(1)
    #     if sg1:
    #         r1 = sg1.get_round(1)
    #         if r1:
    #             player_a = r1.get_player('A')
    #             if player_a:
    #                 print(f"  Player A's Round 1 contribution: {player_a.contribution}")
    
    # Example 5: Working with real data
    print("\n=== Example 5: How to use with your actual data ===")
    
    print("""
    To use this with your actual session data:
    
    1. Prepare your file pairs:
       file_pairs = [
           ('session1_data.csv', 'session1_chat.csv', 1),  # Treatment 1
           ('session2_data.csv', 'session2_chat.csv', 2),  # Treatment 2
           ('session3_data.csv', None, 1),  # No chat data, Treatment 1
       ]
    
    2. Load the experiment:
       experiment = experiment_data.load_experiment_data(file_pairs)
    
    3. Analyze across sessions:
       # Get aggregated data
       df = experiment.to_dataframe_contributions()
       overall_sentiment = experiment.get_overall_sentiment()
       
       # Perform your analysis
       import pandas as pd
       print(df.groupby('session_code')['contribution'].mean())
       print(df.groupby('treatment')['contribution'].mean())  # Compare by treatment
    
    4. Access individual sessions as needed:
       for session_code in experiment.list_session_codes():
           session = experiment.get_session(session_code)
           # Work with individual session data
    """)


if __name__ == '__main__':
    main()