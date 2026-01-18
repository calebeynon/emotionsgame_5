"""
Multi-session analysis and visualization script for experimental data.

This script generates comprehensive plots for experiments with multiple sessions including:
- Segment-by-segment analysis (averages across sessions)
- Session-by-session analysis (individual session breakdowns)
- Contribution patterns, rankings, chat analysis, and sentiment analysis

Usage:
    python multi_session_analysis.py
    
The script loads experimental data using the Experiment object which can contain
multiple sessions, and produces both aggregate and individual session analyses.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from experiment_data import load_experiment_data, Experiment
import os
import re
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Any, Tuple
import argparse

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Global variables for consistent styling
COLORS = plt.cm.Set3(np.linspace(0, 1, 12))
SESSION_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C']
DEFAULT_OUTPUT_DIR = 'multi_session_plots'
SEGMENT_OUTPUT_DIR = 'segment_analysis'
SESSION_OUTPUT_DIR = 'session_analysis'


def setup_output_directories(base_dir: str = DEFAULT_OUTPUT_DIR):
    """Create output directories for plots if they don't exist."""
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, SEGMENT_OUTPUT_DIR),
        os.path.join(base_dir, SESSION_OUTPUT_DIR)
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created output directory: {dir_path}")


def extract_contributions_data(experiment: Experiment) -> pd.DataFrame:
    """Extract all contribution data from the experiment into a DataFrame."""
    records = []
    
    for session_code, session in experiment.sessions.items():
        for segment_name, segment in session.segments.items():
            if not segment_name.startswith('supergame'):
                continue
                
            supergame_num = int(segment_name.replace('supergame', ''))
            
            for round_num, round_obj in segment.rounds.items():
                for group_id, group in round_obj.groups.items():
                    for label, player in group.players.items():
                        if player.contribution is not None:
                            records.append({
                                'session_code': session_code,
                                'treatment': session.treatment,
                                'supergame': supergame_num,
                                'round': round_num,
                                'group_id': group_id,
                                'player_label': label,
                                'participant_id': player.participant_id,
                                'contribution': player.contribution,
                                'payoff': player.payoff,
                                'group_total': group.total_contribution,
                                'individual_share': group.individual_share
                            })
    
    return pd.DataFrame(records)


def extract_chat_data(
    experiment: Experiment, include_orphans: bool = False
) -> pd.DataFrame:
    """Extract all chat data from the experiment into a DataFrame.

    Chat messages are stored on the round they INFLUENCED, not the round they
    occurred in. For example, chat that occurred after round 1's contribution
    is stored on round 2 (the round it influenced). Round 1 has no chat because
    no prior chat influenced it.

    The 'round' column shows the influenced round (the round where the contribution
    decision was made after reading the chat). To get the round where chat occurred,
    use 'chat_occurred_in_round' column (which equals influenced_round - 1).

    Args:
        experiment: Experiment object containing session data
        include_orphans: If True, include orphan chats (last-round chats that
            influenced no subsequent contribution because the supergame ended).
            Orphan chats have round=None and chat_occurred_in_round set to the
            last round of the supergame.

    Returns:
        DataFrame with chat data. The 'round' column shows the influenced round.
    """
    records = []

    for session_code, session in experiment.sessions.items():
        for segment_name, segment in session.segments.items():
            if not segment_name.startswith('supergame'):
                continue

            supergame_num = int(segment_name.replace('supergame', ''))
            max_round = max(segment.rounds.keys()) if segment.rounds else 0

            # Extract regular chat messages (assigned to influenced round)
            for round_num, round_obj in segment.rounds.items():
                for msg in round_obj.chat_messages:
                    records.append({
                        'session_code': session_code,
                        'treatment': session.treatment,
                        'supergame': supergame_num,
                        'round': round_num,
                        'chat_occurred_in_round': round_num - 1,
                        'player_label': msg.nickname,
                        'message_body': msg.body,
                        'timestamp': msg.timestamp,
                        'sentiment': msg.sentiment,
                        'positive_sentiment': msg.positive_sentiment,
                        'negative_sentiment': msg.negative_sentiment,
                        'neutral_sentiment': msg.neutral_sentiment
                    })

            # Extract orphan chats if requested
            if include_orphans:
                for player_label, messages in segment.orphan_chats.items():
                    for msg in messages:
                        records.append({
                            'session_code': session_code,
                            'treatment': session.treatment,
                            'supergame': supergame_num,
                            'round': None,  # No influenced round
                            'chat_occurred_in_round': max_round,
                            'player_label': player_label,
                            'message_body': msg.body,
                            'timestamp': msg.timestamp,
                            'sentiment': msg.sentiment,
                            'positive_sentiment': msg.positive_sentiment,
                            'negative_sentiment': msg.negative_sentiment,
                            'neutral_sentiment': msg.neutral_sentiment
                        })

    return pd.DataFrame(records)


def extract_survey_data(experiment: Experiment) -> pd.DataFrame:
    """Extract survey data from finalresults segments."""
    records = []
    
    for session_code, session in experiment.sessions.items():
        final_results = session.get_segment('finalresults')
        if not final_results or not final_results.get_round(1):
            continue
            
        final_round = final_results.get_round(1)
        
        for player in final_round.players.values():
            if hasattr(player, 'data') and player.data:
                record = {
                    'session_code': session_code,
                    'treatment': session.treatment,
                    'player_label': player.label,
                    'participant_id': player.participant_id
                }
                
                # Add all survey questions
                for key, value in player.data.items():
                    if key.startswith('q') or key == 'final_payoff':
                        record[key] = value
                
                records.append(record)
    
    return pd.DataFrame(records)


def plot_contributions_aggregate_by_segment(contrib_df: pd.DataFrame, output_dir: str):
    """Plot average contributions by supergame, aggregated across all sessions."""
    print("Generating aggregate contribution plot by segment...")
    
    # Calculate means and standard errors across sessions
    segment_stats = contrib_df.groupby(['supergame', 'session_code'])['contribution'].mean().reset_index()
    final_stats = segment_stats.groupby('supergame')['contribution'].agg(['mean', 'std', 'count']).reset_index()
    final_stats['se'] = final_stats['std'] / np.sqrt(final_stats['count'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot with error bars (standard error across sessions)
    bars = ax1.bar(final_stats['supergame'], final_stats['mean'], 
                   yerr=final_stats['se'], capsize=5, 
                   color=COLORS[:len(final_stats)], alpha=0.7)
    
    ax1.set_title('Average Contributions by Supergame\n(Across All Sessions)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Supergame')
    ax1.set_ylabel('Average Contribution')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(final_stats['supergame'])
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, final_stats['mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(final_stats['se'])*0.1,
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Box plot showing distribution across sessions
    for sg_num in final_stats['supergame']:
        sg_data = segment_stats[segment_stats['supergame'] == sg_num]['contribution']
        ax2.scatter([sg_num] * len(sg_data), sg_data, alpha=0.6, s=60)
    
    ax2.boxplot([segment_stats[segment_stats['supergame'] == sg]['contribution'] 
                 for sg in final_stats['supergame']], 
                positions=final_stats['supergame'], patch_artist=True)
    
    ax2.set_title('Distribution of Session Means\nby Supergame', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Supergame')
    ax2.set_ylabel('Session Mean Contribution')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(final_stats['supergame'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/aggregate_contributions_by_segment.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/aggregate_contributions_by_segment.png")
    plt.close()


def plot_contributions_by_session(contrib_df: pd.DataFrame, output_dir: str):
    """Plot contributions for each session separately."""
    print("Generating contributions by individual session...")
    
    sessions = contrib_df['session_code'].unique()
    n_sessions = len(sessions)
    
    if n_sessions == 0:
        print("No session data found!")
        return
    
    # Create subplots - arrange in grid
    cols = min(2, n_sessions)
    rows = (n_sessions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_sessions == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, session_code in enumerate(sessions):
        ax = axes[i] if n_sessions > 1 else axes[0]
        
        session_data = contrib_df[contrib_df['session_code'] == session_code]
        treatment = session_data['treatment'].iloc[0]
        
        # Calculate means by supergame for this session
        session_means = session_data.groupby('supergame')['contribution'].mean()
        session_stds = session_data.groupby('supergame')['contribution'].std()
        
        bars = ax.bar(session_means.index, session_means.values, 
                     yerr=session_stds.values, capsize=5,
                     color=SESSION_COLORS[i % len(SESSION_COLORS)], alpha=0.7)
        
        ax.set_title(f'Session {session_code}\n(Treatment {treatment})', fontweight='bold')
        ax.set_xlabel('Supergame')
        ax.set_ylabel('Average Contribution')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 25)
        
        # Add value labels
        for bar, mean_val in zip(bars, session_means.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Remove empty subplots
    for j in range(n_sessions, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SESSION_OUTPUT_DIR}/contributions_by_session.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SESSION_OUTPUT_DIR}/contributions_by_session.png")
    plt.close()


def plot_contributions_by_round_aggregate(contrib_df: pd.DataFrame, output_dir: str):
    """Plot average contributions by round within each supergame, aggregated across sessions."""
    print("Generating round-by-round aggregate analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for sg_num in range(1, 6):
        ax = axes[sg_num - 1]
        sg_data = contrib_df[contrib_df['supergame'] == sg_num]
        
        if sg_data.empty:
            ax.set_title(f'Supergame {sg_num} - No Data', fontweight='bold')
            continue
        
        # Calculate session-level means first, then aggregate
        session_round_means = sg_data.groupby(['session_code', 'round'])['contribution'].mean().reset_index()
        round_stats = session_round_means.groupby('round')['contribution'].agg(['mean', 'std', 'count']).reset_index()
        round_stats['se'] = round_stats['std'] / np.sqrt(round_stats['count'])
        
        if not round_stats.empty:
            bars = ax.bar(round_stats['round'], round_stats['mean'], 
                         yerr=round_stats['se'], capsize=5, 
                         color=COLORS[sg_num-1], alpha=0.7)
            
            # Add value labels
            for bar, mean_val in zip(bars, round_stats['mean']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(round_stats['se'])*0.1,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Supergame {sg_num} - Contributions by Round\n(Across Sessions)', fontweight='bold')
        ax.set_ylabel('Average Contribution')
        ax.set_xlabel('Round')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 25)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/contributions_by_round_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/contributions_by_round_aggregate.png")
    plt.close()


def plot_individual_trends_aggregate(contrib_df: pd.DataFrame, output_dir: str):
    """Plot individual player trends aggregated across sessions."""
    print("Generating individual player trend analysis (aggregate)...")
    
    # Calculate individual player averages by supergame and session
    player_trends = contrib_df.groupby(['session_code', 'player_label', 'supergame'])['contribution'].mean().reset_index()
    
    # Then average across sessions for each player label
    label_stats = player_trends.groupby(['player_label', 'supergame'])['contribution'].agg(['mean', 'std', 'count']).reset_index()
    label_stats['se'] = label_stats['std'] / np.sqrt(label_stats['count'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all unique player labels
    player_labels = sorted(label_stats['player_label'].unique())
    
    for i, player_label in enumerate(player_labels):
        player_data = label_stats[label_stats['player_label'] == player_label]
        
        if not player_data.empty:
            ax.errorbar(player_data['supergame'], player_data['mean'], 
                       yerr=player_data['se'], marker='o', linewidth=2, 
                       alpha=0.7, label=f'Player {player_label}', capsize=3)
    
    ax.set_title('Individual Player Contribution Trends\n(Averaged Across Sessions)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Supergame')
    ax.set_ylabel('Average Contribution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f'SG{i}' for i in range(1, 6)])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/individual_trends_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/individual_trends_aggregate.png")
    plt.close()


def plot_individual_trends_by_session(contrib_df: pd.DataFrame, output_dir: str):
    """Plot individual player trends for each session separately."""
    print("Generating individual player trends by session...")
    
    sessions = contrib_df['session_code'].unique()
    
    for session_code in sessions:
        session_data = contrib_df[contrib_df['session_code'] == session_code]
        treatment = session_data['treatment'].iloc[0]
        
        # Calculate individual player averages by supergame for this session
        player_trends = session_data.groupby(['player_label', 'supergame'])['contribution'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        player_labels = sorted(player_trends['player_label'].unique())
        
        for i, player_label in enumerate(player_labels):
            player_data = player_trends[player_trends['player_label'] == player_label]
            
            if not player_data.empty:
                ax.plot(player_data['supergame'], player_data['contribution'], 
                       marker='o', linewidth=2, alpha=0.7, label=f'Player {player_label}')
        
        ax.set_title(f'Individual Player Trends - Session {session_code}\n(Treatment {treatment})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Supergame')
        ax.set_ylabel('Average Contribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([f'SG{i}' for i in range(1, 6)])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{SESSION_OUTPUT_DIR}/individual_trends_{session_code}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual trend plots for {len(sessions)} sessions")


def plot_payoff_rankings_aggregate(contrib_df: pd.DataFrame, output_dir: str):
    """Plot player rankings by total payoff, aggregated across sessions."""
    print("Generating aggregate payoff rankings...")
    
    # Calculate total payoffs by session and player
    session_payoffs = contrib_df.groupby(['session_code', 'player_label'])['payoff'].sum().reset_index()
    
    # Average payoffs across sessions for each player label
    avg_payoffs = session_payoffs.groupby('player_label')['payoff'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(avg_payoffs)))
    bars = ax.bar(range(len(avg_payoffs)), avg_payoffs.values, color=colors)
    
    ax.set_xticks(range(len(avg_payoffs)))
    ax.set_xticklabels(avg_payoffs.index)
    ax.set_title('Player Rankings by Average Total Payoff\n(Across Sessions)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Player')
    ax.set_ylabel('Average Total Payoff')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and rankings
    for i, (bar, payoff) in enumerate(zip(bars, avg_payoffs.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_payoffs)*0.01,
               f'{payoff:.0f}', ha='center', va='bottom', fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'#{i+1}', ha='center', va='center', fontsize=12, fontweight='bold',
               color='white' if i < len(avg_payoffs)//2 else 'black')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/payoff_rankings_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/payoff_rankings_aggregate.png")
    plt.close()


def plot_contribution_rankings_aggregate(contrib_df: pd.DataFrame, output_dir: str):
    """Plot player rankings by average contribution, aggregated across sessions."""
    print("Generating aggregate contribution rankings...")
    
    # Calculate average contributions by session and player
    session_contribs = contrib_df.groupby(['session_code', 'player_label'])['contribution'].mean().reset_index()
    
    # Average across sessions for each player label
    avg_contribs = session_contribs.groupby('player_label')['contribution'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(avg_contribs)))
    bars = ax.bar(range(len(avg_contribs)), avg_contribs.values, color=colors)
    
    ax.set_xticks(range(len(avg_contribs)))
    ax.set_xticklabels(avg_contribs.index)
    ax.set_title('Player Rankings by Average Contribution\n(Across Sessions)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Player')
    ax.set_ylabel('Average Contribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and rankings
    for i, (bar, contrib) in enumerate(zip(bars, avg_contribs.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_contribs)*0.01,
               f'{contrib:.1f}', ha='center', va='bottom', fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'#{i+1}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/contribution_rankings_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/contribution_rankings_aggregate.png")
    plt.close()


def analyze_survey_responses_aggregate(survey_df: pd.DataFrame, output_dir: str):
    """Analyze survey responses aggregated across sessions."""
    if survey_df.empty:
        print("No survey data found!")
        return
        
    print("Generating aggregate survey response analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Gender distribution (q1)
    if 'q1' in survey_df.columns:
        gender_counts = survey_df['q1'].value_counts()
        ax = axes[0]
        colors = ['lightblue', 'lightpink', 'lightgreen'][:len(gender_counts)]
        wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                         autopct='%1.1f%%', colors=colors)
        ax.set_title('Gender Distribution (Q1)\nAcross All Sessions', fontweight='bold')
    
    # Age distribution (q3)
    if 'q3' in survey_df.columns:
        ages = survey_df['q3'].dropna()
        ax = axes[1]
        ax.hist(ages, bins=8, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_title('Age Distribution (Q3)\nAcross All Sessions', fontweight='bold')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {ages.mean():.1f}')
        ax.legend()
    
    # Experience with experiments (q5)
    if 'q5' in survey_df.columns:
        exp_counts = survey_df['q5'].value_counts().sort_index()
        ax = axes[2]
        bars = ax.bar(exp_counts.index, exp_counts.values, color='lightcoral', alpha=0.7)
        ax.set_title('Previous Experiment Experience (Q5)\nAcross All Sessions', fontweight='bold')
        ax.set_xlabel('Number of Previous Experiments')
        ax.set_ylabel('Count')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    # Treatment distribution
    if 'treatment' in survey_df.columns:
        treatment_counts = survey_df['treatment'].value_counts()
        ax = axes[3]
        bars = ax.bar(['Treatment 1', 'Treatment 2'], treatment_counts.values, 
                     color=['orange', 'green'], alpha=0.7)
        ax.set_title('Treatment Distribution\nAcross All Sessions', fontweight='bold')
        ax.set_ylabel('Number of Participants')
        
        for bar, count in zip(bars, treatment_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(treatment_counts)*0.02,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Session distribution
    if 'session_code' in survey_df.columns:
        session_counts = survey_df['session_code'].value_counts()
        ax = axes[4]
        bars = ax.bar(range(len(session_counts)), session_counts.values, 
                     color=SESSION_COLORS[:len(session_counts)], alpha=0.7)
        ax.set_title('Participants by Session', fontweight='bold')
        ax.set_xticks(range(len(session_counts)))
        ax.set_xticklabels(session_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Number of Participants')
        
        for bar, count in zip(bars, session_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(session_counts)*0.02,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/survey_responses_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/survey_responses_aggregate.png")
    plt.close()


def generate_chat_word_frequency_aggregate(chat_df: pd.DataFrame, output_dir: str):
    """Generate word frequency analysis aggregated across all sessions."""
    if chat_df.empty:
        print("No chat data found for word frequency analysis!")
        return
        
    print("Generating aggregate chat word frequency analysis...")
    
    # Clean and process text
    all_text = []
    for _, row in chat_df.iterrows():
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', str(row['message_body']).lower())
        all_text.append(cleaned_text)
    
    if not all_text:
        return
    
    combined_text = ' '.join(all_text)
    words = combined_text.split()
    
    # Filter stop words and short words
    stop_words = {'the', 'and', 'but', 'for', 'are', 'was', 'were', 'been', 'have', 
                  'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 
                  'must', 'can', 'cant', 'wont', 'dont', 'didnt', 'isnt', 'arent', 
                  'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wouldnt', 'couldnt', 
                  'shouldnt', 'mightnt', 'mustnt', 'not', 'now', 'just', 'like', 
                  'get', 'got', 'want', 'think', 'know', 'see', 'said', 'say'}
    
    filtered_words = [word for word in words if len(word) >= 3 and word not in stop_words]
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(15)
    
    if not top_words:
        print("No words found after filtering!")
        return
    
    words_list, frequencies = zip(*top_words)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(range(len(words_list)), frequencies, 
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(words_list))), 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Words', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Frequent Words in Chat Messages\n(Across All Sessions)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(words_list)))
    ax.set_xticklabels(words_list, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(frequencies)*0.01,
               str(freq), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    total_messages = len(all_text)
    total_words = len(words)
    unique_words = len(set(filtered_words))
    
    stats_text = f"Total messages: {total_messages}\nTotal words: {total_words:,}\nUnique words: {unique_words:,}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/chat_word_frequency_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/chat_word_frequency_aggregate.png")
    plt.close()


def plot_chat_messages_per_round_aggregate(chat_df: pd.DataFrame, output_dir: str):
    """Plot number of chat messages per round, aggregated across sessions."""
    if chat_df.empty:
        print("No chat data found!")
        return
        
    print("Generating aggregate chat messages per round analysis...")
    
    # Count messages by session, supergame, and round
    msg_counts = chat_df.groupby(['session_code', 'supergame', 'round']).size().reset_index(name='msg_count')
    
    # Calculate statistics across sessions
    round_stats = msg_counts.groupby(['supergame', 'round'])['msg_count'].agg(['mean', 'std', 'count']).reset_index()
    round_stats['se'] = round_stats['std'] / np.sqrt(round_stats['count'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for sg_num in range(1, 6):
        ax = axes[sg_num - 1]
        sg_data = round_stats[round_stats['supergame'] == sg_num]
        
        if not sg_data.empty:
            bars = ax.bar(sg_data['round'], sg_data['mean'], 
                         yerr=sg_data['se'], capsize=5,
                         color=COLORS[sg_num-1], alpha=0.7)
            
            # Add value labels
            for bar, mean_val in zip(bars, sg_data['mean']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(sg_data['se'])*0.1,
                       f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Supergame {sg_num} - Chat Messages per Round\n(Across Sessions)', fontweight='bold')
        ax.set_ylabel('Average Number of Messages')
        ax.set_xlabel('Round')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Chat Activity by Round Aggregated Across Sessions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/chat_messages_per_round_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/chat_messages_per_round_aggregate.png")
    plt.close()


def plot_sentiment_analysis_aggregate(chat_df: pd.DataFrame, output_dir: str):
    """Generate sentiment analysis plots aggregated across sessions."""
    if chat_df.empty or 'sentiment' not in chat_df.columns:
        print("No sentiment data found!")
        return
        
    print("Generating aggregate sentiment analysis...")
    
    # Sentiment distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of sentiment scores
    sentiments = chat_df['sentiment'].dropna()
    if len(sentiments) == 0:
        print("No valid sentiment scores found!")
        return
    
    n_bins = 40
    counts, bins, patches = ax1.hist(sentiments, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars based on sentiment
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < -0.05:
            patch.set_facecolor('#FF6B6B')  # Red for negative
        elif bin_center > 0.05:
            patch.set_facecolor('#4ECDC4')  # Teal for positive  
        else:
            patch.set_facecolor('#95A5A6')  # Gray for neutral
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Sentiment Score', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution of Chat Message Sentiment\n(Across All Sessions)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_sentiment = np.mean(sentiments)
    positive_msgs = sum(1 for s in sentiments if s > 0.05)
    negative_msgs = sum(1 for s in sentiments if s < -0.05)
    neutral_msgs = len(sentiments) - positive_msgs - negative_msgs
    
    stats_text = f"""Total Messages: {len(sentiments):,}
Mean Sentiment: {mean_sentiment:+.3f}

Positive: {positive_msgs} ({positive_msgs/len(sentiments)*100:.1f}%)
Neutral: {neutral_msgs} ({neutral_msgs/len(sentiments)*100:.1f}%)
Negative: {negative_msgs} ({negative_msgs/len(sentiments)*100:.1f}%)"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10)
    
    # Sentiment by supergame
    sg_sentiment = chat_df.groupby('supergame')['sentiment'].agg(['mean', 'std', 'count']).reset_index()
    sg_sentiment['se'] = sg_sentiment['std'] / np.sqrt(sg_sentiment['count'])
    
    bars = ax2.bar(sg_sentiment['supergame'], sg_sentiment['mean'], 
                   yerr=sg_sentiment['se'], capsize=5,
                   color=COLORS[:len(sg_sentiment)], alpha=0.7)
    
    ax2.set_xlabel('Supergame', fontweight='bold')
    ax2.set_ylabel('Average Sentiment', fontweight='bold')
    ax2.set_title('Average Sentiment by Supergame\n(Across All Sessions)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, mean_val in zip(bars, sg_sentiment['mean']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (max(sg_sentiment['mean']) - min(sg_sentiment['mean'])) * 0.05,
                f'{mean_val:+.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/sentiment_analysis_aggregate.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/sentiment_analysis_aggregate.png")
    plt.close()


def plot_treatment_comparison_contributions(contrib_df: pd.DataFrame, output_dir: str):
    """Plot treatment comparison for contributions across supergames."""
    print("Generating treatment comparison analysis...")
    
    # Calculate treatment means by supergame and session
    treatment_session_means = contrib_df.groupby(['treatment', 'session_code', 'supergame'])['contribution'].mean().reset_index()
    
    # Then calculate statistics across sessions for each treatment
    treatment_stats = treatment_session_means.groupby(['treatment', 'supergame'])['contribution'].agg(['mean', 'std', 'count']).reset_index()
    treatment_stats['se'] = treatment_stats['std'] / np.sqrt(treatment_stats['count'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Side-by-side bar plot with error bars
    supergames = sorted(treatment_stats['supergame'].unique())
    x = np.arange(len(supergames))
    width = 0.35
    
    treatment1_data = treatment_stats[treatment_stats['treatment'] == 1]
    treatment2_data = treatment_stats[treatment_stats['treatment'] == 2]
    
    bars1 = ax1.bar(x - width/2, treatment1_data['mean'], width, 
                    yerr=treatment1_data['se'], capsize=5,
                    label='Treatment 1', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, treatment2_data['mean'], width,
                    yerr=treatment2_data['se'], capsize=5, 
                    label='Treatment 2', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Supergame')
    ax1.set_ylabel('Average Contribution')
    ax1.set_title('Treatment Comparison: Average Contributions by Supergame', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'SG{i}' for i in supergames])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars, data in [(bars1, treatment1_data), (bars2, treatment2_data)]:
        for bar, mean_val in zip(bars, data['mean']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Box plot showing distribution of session means by treatment
    t1_sessions = treatment_session_means[treatment_session_means['treatment'] == 1]
    t2_sessions = treatment_session_means[treatment_session_means['treatment'] == 2]
    
    for i, sg_num in enumerate(supergames):
        t1_sg_data = t1_sessions[t1_sessions['supergame'] == sg_num]['contribution']
        t2_sg_data = t2_sessions[t2_sessions['supergame'] == sg_num]['contribution']
        
        # Scatter points for individual sessions
        ax2.scatter([i - 0.15] * len(t1_sg_data), t1_sg_data, 
                   color='#FF6B6B', alpha=0.6, s=60, label='Treatment 1' if i == 0 else "")
        ax2.scatter([i + 0.15] * len(t2_sg_data), t2_sg_data, 
                   color='#4ECDC4', alpha=0.6, s=60, label='Treatment 2' if i == 0 else "")
    
    # Add box plots
    t1_all_data = [t1_sessions[t1_sessions['supergame'] == sg]['contribution'] for sg in supergames]
    t2_all_data = [t2_sessions[t2_sessions['supergame'] == sg]['contribution'] for sg in supergames]
    
    bp1 = ax2.boxplot(t1_all_data, positions=x - 0.2, widths=0.3, patch_artist=True, 
                     boxprops=dict(facecolor='#FF6B6B', alpha=0.3))
    bp2 = ax2.boxplot(t2_all_data, positions=x + 0.2, widths=0.3, patch_artist=True,
                     boxprops=dict(facecolor='#4ECDC4', alpha=0.3))
    
    ax2.set_xlabel('Supergame')
    ax2.set_ylabel('Session Mean Contribution')
    ax2.set_title('Treatment Comparison: Distribution of Session Means', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'SG{i}' for i in supergames])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/treatment_comparison_contributions.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/treatment_comparison_contributions.png")
    plt.close()


def plot_treatment_comparison_chat_sentiment(chat_df: pd.DataFrame, output_dir: str):
    """Plot treatment comparison for chat and sentiment analysis."""
    if chat_df.empty or 'sentiment' not in chat_df.columns:
        print("No sentiment data found for treatment comparison!")
        return
        
    print("Generating treatment comparison for chat and sentiment...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Messages per session by treatment
    msg_per_session = chat_df.groupby(['treatment', 'session_code']).size().reset_index(name='msg_count')
    t1_msg = msg_per_session[msg_per_session['treatment'] == 1]['msg_count']
    t2_msg = msg_per_session[msg_per_session['treatment'] == 2]['msg_count']
    
    ax1.bar(['Treatment 1', 'Treatment 2'], [t1_msg.mean(), t2_msg.mean()], 
           yerr=[t1_msg.std()/np.sqrt(len(t1_msg)), t2_msg.std()/np.sqrt(len(t2_msg))],
           capsize=5, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_ylabel('Average Messages per Session')
    ax1.set_title('Chat Volume by Treatment', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean_val, se_val) in enumerate([(t1_msg.mean(), t1_msg.std()/np.sqrt(len(t1_msg))), 
                                           (t2_msg.mean(), t2_msg.std()/np.sqrt(len(t2_msg)))]):
        ax1.text(i, mean_val + se_val + 10, f'{mean_val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Sentiment distribution by treatment
    t1_sentiment = chat_df[chat_df['treatment'] == 1]['sentiment'].dropna()
    t2_sentiment = chat_df[chat_df['treatment'] == 2]['sentiment'].dropna()
    
    ax2.hist([t1_sentiment, t2_sentiment], bins=30, alpha=0.7, 
            label=['Treatment 1', 'Treatment 2'], color=['#FF6B6B', '#4ECDC4'])
    ax2.set_xlabel('Sentiment Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sentiment Distribution by Treatment', fontweight='bold')
    ax2.legend()
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Average sentiment by supergame and treatment
    sentiment_by_sg = chat_df.groupby(['treatment', 'session_code', 'supergame'])['sentiment'].mean().reset_index()
    sentiment_stats = sentiment_by_sg.groupby(['treatment', 'supergame'])['sentiment'].agg(['mean', 'std', 'count']).reset_index()
    sentiment_stats['se'] = sentiment_stats['std'] / np.sqrt(sentiment_stats['count'])
    
    supergames = sorted(sentiment_stats['supergame'].unique())
    x = np.arange(len(supergames))
    width = 0.35
    
    t1_sent_data = sentiment_stats[sentiment_stats['treatment'] == 1]
    t2_sent_data = sentiment_stats[sentiment_stats['treatment'] == 2]
    
    ax3.bar(x - width/2, t1_sent_data['mean'], width, yerr=t1_sent_data['se'], 
           capsize=5, label='Treatment 1', color='#FF6B6B', alpha=0.8)
    ax3.bar(x + width/2, t2_sent_data['mean'], width, yerr=t2_sent_data['se'], 
           capsize=5, label='Treatment 2', color='#4ECDC4', alpha=0.8)
    
    ax3.set_xlabel('Supergame')
    ax3.set_ylabel('Average Sentiment')
    ax3.set_title('Average Sentiment by Supergame and Treatment', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'SG{i}' for i in supergames])
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Sentiment categories by treatment
    def categorize_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    chat_df['sentiment_category'] = chat_df['sentiment'].apply(categorize_sentiment)
    sentiment_cats = chat_df.groupby(['treatment', 'sentiment_category']).size().unstack(fill_value=0)
    sentiment_cats_pct = sentiment_cats.div(sentiment_cats.sum(axis=1), axis=0) * 100
    
    x_pos = [0, 1]
    width = 0.6
    categories = ['Negative', 'Neutral', 'Positive']
    colors = ['#FF6B6B', '#95A5A6', '#4ECDC4']
    
    bottom = np.zeros(2)
    for i, cat in enumerate(categories):
        if cat in sentiment_cats_pct.columns:
            values = [sentiment_cats_pct.loc[1, cat], sentiment_cats_pct.loc[2, cat]]
            ax4.bar(x_pos, values, width, bottom=bottom, label=cat, color=colors[i], alpha=0.8)
            
            # Add percentage labels
            for j, val in enumerate(values):
                ax4.text(x_pos[j], bottom[j] + val/2, f'{val:.1f}%', 
                        ha='center', va='center', fontweight='bold')
            
            bottom += values
    
    ax4.set_xlabel('Treatment')
    ax4.set_ylabel('Percentage of Messages')
    ax4.set_title('Sentiment Categories by Treatment', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['Treatment 1', 'Treatment 2'])
    ax4.legend()
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/treatment_comparison_chat_sentiment.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/treatment_comparison_chat_sentiment.png")
    plt.close()


def plot_treatment_comparison_player_behavior(contrib_df: pd.DataFrame, output_dir: str):
    """Plot treatment comparison for individual player behavior patterns."""
    print("Generating treatment comparison for player behavior...")
    
    # Calculate player-level statistics by treatment
    player_stats = contrib_df.groupby(['treatment', 'session_code', 'player_label']).agg({
        'contribution': ['mean', 'std'],
        'payoff': ['mean', 'sum']
    }).reset_index()
    
    # Flatten column names
    player_stats.columns = ['treatment', 'session_code', 'player_label', 
                           'mean_contribution', 'std_contribution', 'mean_payoff', 'total_payoff']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of player average contributions
    t1_contrib = player_stats[player_stats['treatment'] == 1]['mean_contribution']
    t2_contrib = player_stats[player_stats['treatment'] == 2]['mean_contribution']
    
    ax1.hist([t1_contrib, t2_contrib], bins=20, alpha=0.7, 
            label=['Treatment 1', 'Treatment 2'], color=['#FF6B6B', '#4ECDC4'])
    ax1.set_xlabel('Player Average Contribution')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Player Average Contributions\nby Treatment', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    ax1.axvline(t1_contrib.mean(), color='#FF6B6B', linestyle='--', linewidth=2, 
               label=f'T1 Mean: {t1_contrib.mean():.1f}')
    ax1.axvline(t2_contrib.mean(), color='#4ECDC4', linestyle='--', linewidth=2,
               label=f'T2 Mean: {t2_contrib.mean():.1f}')
    
    # 2. Distribution of player total payoffs
    t1_payoff = player_stats[player_stats['treatment'] == 1]['total_payoff']
    t2_payoff = player_stats[player_stats['treatment'] == 2]['total_payoff']
    
    ax2.hist([t1_payoff, t2_payoff], bins=20, alpha=0.7,
            label=['Treatment 1', 'Treatment 2'], color=['#FF6B6B', '#4ECDC4'])
    ax2.set_xlabel('Player Total Payoff')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Player Total Payoffs\nby Treatment', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Contribution variability by treatment
    t1_std = player_stats[player_stats['treatment'] == 1]['std_contribution'].dropna()
    t2_std = player_stats[player_stats['treatment'] == 2]['std_contribution'].dropna()
    
    ax3.boxplot([t1_std, t2_std], labels=['Treatment 1', 'Treatment 2'], 
               patch_artist=True, 
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax3.set_ylabel('Standard Deviation of Contributions')
    ax3.set_title('Player Contribution Variability\nby Treatment', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Player label performance comparison
    label_performance = contrib_df.groupby(['treatment', 'player_label'])['contribution'].mean().unstack(level=0)
    
    # Scatter plot comparing labels across treatments
    if 1 in label_performance.columns and 2 in label_performance.columns:
        ax4.scatter(label_performance[1], label_performance[2], 
                   s=80, alpha=0.7, color='purple')
        
        # Add diagonal line for reference (equal performance)
        min_val = min(label_performance[1].min(), label_performance[2].min())
        max_val = max(label_performance[1].max(), label_performance[2].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add labels for each point
        for label in label_performance.index:
            ax4.annotate(label, (label_performance.loc[label, 1], label_performance.loc[label, 2]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Average Contribution in Treatment 1')
        ax4.set_ylabel('Average Contribution in Treatment 2')
        ax4.set_title('Player Label Performance\nAcross Treatments', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{SEGMENT_OUTPUT_DIR}/treatment_comparison_player_behavior.png', 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/{SEGMENT_OUTPUT_DIR}/treatment_comparison_player_behavior.png")
    plt.close()


def generate_treatment_comparison_statistics(contrib_df: pd.DataFrame, chat_df: pd.DataFrame, 
                                           survey_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive treatment comparison statistics."""
    print("Generating treatment comparison statistics...")
    
    stats = {}
    
    # Overall treatment comparison
    for treatment in [1, 2]:
        treat_data = contrib_df[contrib_df['treatment'] == treatment]
        treat_chat = chat_df[chat_df['treatment'] == treatment] if not chat_df.empty else pd.DataFrame()
        treat_survey = survey_df[survey_df['treatment'] == treatment] if not survey_df.empty else pd.DataFrame()
        
        stats[f'Treatment {treatment}'] = {
            'Sessions': treat_data['session_code'].nunique(),
            'Total observations': len(treat_data),
            'Mean contribution': treat_data['contribution'].mean(),
            'Std contribution': treat_data['contribution'].std(),
            'Mean payoff': treat_data['payoff'].mean() if treat_data['payoff'].notna().any() else 0,
            'Std payoff': treat_data['payoff'].std() if treat_data['payoff'].notna().any() else 0,
            'Chat messages': len(treat_chat) if not treat_chat.empty else 0,
            'Avg sentiment': treat_chat['sentiment'].mean() if not treat_chat.empty and 'sentiment' in treat_chat.columns else 'N/A',
            'Survey responses': len(treat_survey) if not treat_survey.empty else 0
        }
    
    # Statistical tests
    from scipy import stats as scipy_stats
    
    t1_contrib = contrib_df[contrib_df['treatment'] == 1]['contribution']
    t2_contrib = contrib_df[contrib_df['treatment'] == 2]['contribution']
    
    # T-test for contributions
    t_stat, p_val = scipy_stats.ttest_ind(t1_contrib, t2_contrib)
    
    stats['Statistical Tests'] = {
        'Contribution t-test statistic': t_stat,
        'Contribution t-test p-value': p_val,
        'Contribution difference significant (p<0.05)': 'Yes' if p_val < 0.05 else 'No',
        'Effect size (Cohens d)': (t1_contrib.mean() - t2_contrib.mean()) / np.sqrt(((t1_contrib.std()**2 + t2_contrib.std()**2) / 2))
    }
    
    # Session-level comparison
    session_means_t1 = contrib_df[contrib_df['treatment'] == 1].groupby('session_code')['contribution'].mean()
    session_means_t2 = contrib_df[contrib_df['treatment'] == 2].groupby('session_code')['contribution'].mean()
    
    stats['Session-Level Comparison'] = {
        'Treatment 1 session means': f"{session_means_t1.mean():.2f} ± {session_means_t1.std():.2f}",
        'Treatment 2 session means': f"{session_means_t2.mean():.2f} ± {session_means_t2.std():.2f}",
        'Between-session variability T1': session_means_t1.std(),
        'Between-session variability T2': session_means_t2.std()
    }
    
    # Save to file
    with open(f'{output_dir}/treatment_comparison_statistics.txt', 'w') as f:
        f.write("TREATMENT COMPARISON STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        for section, data in stats.items():
            f.write(f"{section}:\n")
            for key, value in data.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"Saved: {output_dir}/treatment_comparison_statistics.txt")


def generate_summary_statistics(contrib_df: pd.DataFrame, chat_df: pd.DataFrame, 
                              survey_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive summary statistics."""
    print("Generating summary statistics...")
    
    stats = {}
    
    # Overall statistics
    stats['Overall Statistics'] = {
        'Total sessions': contrib_df['session_code'].nunique(),
        'Total observations': len(contrib_df),
        'Total participants': contrib_df['participant_id'].nunique(),
        'Mean contribution': contrib_df['contribution'].mean(),
        'Std contribution': contrib_df['contribution'].std(),
        'Mean payoff': contrib_df['payoff'].mean() if contrib_df['payoff'].notna().any() else 0,
        'Std payoff': contrib_df['payoff'].std() if contrib_df['payoff'].notna().any() else 0
    }
    
    # By treatment
    if 'treatment' in contrib_df.columns:
        for treatment in sorted(contrib_df['treatment'].unique()):
            treat_data = contrib_df[contrib_df['treatment'] == treatment]
            stats[f'Treatment {treatment}'] = {
                'Sessions': treat_data['session_code'].nunique(),
                'Observations': len(treat_data),
                'Participants': treat_data['participant_id'].nunique(),
                'Mean contribution': treat_data['contribution'].mean(),
                'Std contribution': treat_data['contribution'].std(),
                'Mean payoff': treat_data['payoff'].mean() if treat_data['payoff'].notna().any() else 0,
                'Std payoff': treat_data['payoff'].std() if treat_data['payoff'].notna().any() else 0
            }
    
    # By supergame
    for sg_num in sorted(contrib_df['supergame'].unique()):
        sg_data = contrib_df[contrib_df['supergame'] == sg_num]
        stats[f'Supergame {sg_num}'] = {
            'Observations': len(sg_data),
            'Sessions': sg_data['session_code'].nunique(),
            'Mean contribution': sg_data['contribution'].mean(),
            'Std contribution': sg_data['contribution'].std(),
            'Mean payoff': sg_data['payoff'].mean() if sg_data['payoff'].notna().any() else 0,
            'Std payoff': sg_data['payoff'].std() if sg_data['payoff'].notna().any() else 0
        }
    
    # Chat statistics
    if not chat_df.empty:
        stats['Chat Statistics'] = {
            'Total messages': len(chat_df),
            'Sessions with chat': chat_df['session_code'].nunique(),
            'Messages per session': len(chat_df) / chat_df['session_code'].nunique(),
            'Average sentiment': chat_df['sentiment'].mean() if 'sentiment' in chat_df.columns else 'N/A',
            'Positive messages (%)': (chat_df['sentiment'] > 0.05).sum() / len(chat_df) * 100 if 'sentiment' in chat_df.columns else 'N/A',
            'Negative messages (%)': (chat_df['sentiment'] < -0.05).sum() / len(chat_df) * 100 if 'sentiment' in chat_df.columns else 'N/A'
        }
    
    # Survey statistics
    if not survey_df.empty:
        stats['Survey Statistics'] = {
            'Total responses': len(survey_df),
            'Sessions with surveys': survey_df['session_code'].nunique(),
            'Response rate': f"{len(survey_df)}/{contrib_df['participant_id'].nunique()} participants",
            'Average age': survey_df['q3'].mean() if 'q3' in survey_df.columns else 'N/A'
        }
    
    # Save to file
    with open(f'{output_dir}/summary_statistics.txt', 'w') as f:
        f.write("MULTI-SESSION EXPERIMENTAL DATA SUMMARY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        for section, data in stats.items():
            f.write(f"{section}:\n")
            for key, value in data.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.3f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"Saved: {output_dir}/summary_statistics.txt")


def main():
    """Main function to run all multi-session analyses."""
    print("=" * 80)
    print("MULTI-SESSION EXPERIMENTAL DATA ANALYSIS")
    print("=" * 80)
    
    # Setup directories
    setup_output_directories()
    
    # Load experiment data using the existing data loading functionality
    print("Loading experiment data using experiment_data.main()...")
    
    # Import and use the main function from experiment_data module
    from experiment_data import main as load_experiment_main
    experiment = load_experiment_main()
    
    print(f"\nLoaded experiment: {experiment.name}")
    print(f"Sessions: {list(experiment.sessions.keys())}")
    
    # Extract data into DataFrames
    print("\nExtracting data...")
    contrib_df = extract_contributions_data(experiment)
    chat_df = extract_chat_data(experiment)
    survey_df = extract_survey_data(experiment)
    
    print(f"Contribution records: {len(contrib_df)}")
    print(f"Chat records: {len(chat_df)}")
    print(f"Survey records: {len(survey_df)}")
    
    if contrib_df.empty:
        print("No contribution data found!")
        return
    
    # Generate aggregate analysis (across sessions)
    print("\n" + "=" * 40)
    print("SEGMENT-BY-SEGMENT ANALYSIS (ACROSS SESSIONS)")
    print("=" * 40)
    
    plot_contributions_aggregate_by_segment(contrib_df, DEFAULT_OUTPUT_DIR)
    plot_contributions_by_round_aggregate(contrib_df, DEFAULT_OUTPUT_DIR)
    plot_individual_trends_aggregate(contrib_df, DEFAULT_OUTPUT_DIR)
    plot_payoff_rankings_aggregate(contrib_df, DEFAULT_OUTPUT_DIR)
    plot_contribution_rankings_aggregate(contrib_df, DEFAULT_OUTPUT_DIR)
    
    if not survey_df.empty:
        analyze_survey_responses_aggregate(survey_df, DEFAULT_OUTPUT_DIR)
    
    if not chat_df.empty:
        print("\n" + "=" * 30)
        print("CHAT & SENTIMENT ANALYSIS (ACROSS SESSIONS)")
        print("=" * 30)
        generate_chat_word_frequency_aggregate(chat_df, DEFAULT_OUTPUT_DIR)
        plot_chat_messages_per_round_aggregate(chat_df, DEFAULT_OUTPUT_DIR)
        plot_sentiment_analysis_aggregate(chat_df, DEFAULT_OUTPUT_DIR)
    
    # Generate treatment comparison analysis
    print("\n" + "=" * 40)
    print("TREATMENT COMPARISON ANALYSIS")
    print("=" * 40)
    
    plot_treatment_comparison_contributions(contrib_df, DEFAULT_OUTPUT_DIR)
    plot_treatment_comparison_player_behavior(contrib_df, DEFAULT_OUTPUT_DIR)
    
    if not chat_df.empty:
        plot_treatment_comparison_chat_sentiment(chat_df, DEFAULT_OUTPUT_DIR)
    
    generate_treatment_comparison_statistics(contrib_df, chat_df, survey_df, DEFAULT_OUTPUT_DIR)
    
    # Generate individual session analysis
    print("\n" + "=" * 40)
    print("SESSION-BY-SESSION ANALYSIS")
    print("=" * 40)
    
    plot_contributions_by_session(contrib_df, DEFAULT_OUTPUT_DIR)
    plot_individual_trends_by_session(contrib_df, DEFAULT_OUTPUT_DIR)
    
    # Generate summary statistics
    print("\n" + "=" * 30)
    print("SUMMARY STATISTICS")
    print("=" * 30)
    generate_summary_statistics(contrib_df, chat_df, survey_df, DEFAULT_OUTPUT_DIR)
    
    print()
    print("=" * 80)
    print("MULTI-SESSION ANALYSIS COMPLETE!")
    print(f"All plots and statistics saved to: {DEFAULT_OUTPUT_DIR}/")
    print(f"  - Segment analysis (cross-session): {DEFAULT_OUTPUT_DIR}/{SEGMENT_OUTPUT_DIR}/")
    print(f"  - Session analysis (individual): {DEFAULT_OUTPUT_DIR}/{SESSION_OUTPUT_DIR}/")
    print(f"  - Treatment comparison included in segment analysis")
    print(f"  - Statistical summaries: {DEFAULT_OUTPUT_DIR}/*.txt")
    print("=" * 80)


if __name__ == '__main__':
    main()