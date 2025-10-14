"""
Analysis and visualization script for experimental data.

This script generates comprehensive plots including:
- Average contributions (aggregate and by rounds/periods)
- Survey response summaries
- Player payoff rankings
- Player contribution rankings
- Chat data analysis:
  * Word frequency distribution of most common chat words
  * Number of chat messages per round
  * Correlation between chat activity and group contributions
- Sentiment analysis:
  * Distribution of sentiment scores across all messages
  * Correlation between sentiment in round t-1 and contribution in round t
  * Box plots of sentiment scores grouped by supergame
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from experiment_data import load_experiment_data
import os
import re
from scipy.stats import pearsonr

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Global variables for consistent styling
COLORS = plt.cm.Set3(np.linspace(0, 1, 12))
OUTPUT_DIR = 'plots'


def setup_output_directory():
    """Create output directory for plots if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")


def plot_average_contributions_aggregate(session):
    """Plot overall average contributions across all supergames."""
    print("Generating aggregate contribution plot...")
    
    # Collect all contributions by supergame
    supergame_contributions = {}
    
    for sg_num in range(1, 6):  # supergames 1-5
        sg = session.get_supergame(sg_num)
        if sg:
            all_contributions = []
            for round_num in sg.rounds.keys():
                round_obj = sg.get_round(round_num)
                for player in round_obj.players.values():
                    if player.contribution is not None:
                        all_contributions.append(player.contribution)
            
            if all_contributions:
                supergame_contributions[f'Supergame {sg_num}'] = all_contributions
    
    # Create box plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    data_for_boxplot = list(supergame_contributions.values())
    labels = list(supergame_contributions.keys())
    
    box_plot = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], COLORS):
        patch.set_facecolor(color)
    
    ax1.set_title('Distribution of Contributions by Supergame', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Contribution Amount')
    ax1.grid(True, alpha=0.3)
    
    # Mean contributions bar plot
    means = [np.mean(contributions) for contributions in data_for_boxplot]
    stds = [np.std(contributions) for contributions in data_for_boxplot]
    
    bars = ax2.bar(labels, means, yerr=stds, capsize=5, color=COLORS[:len(labels)], alpha=0.7)
    ax2.set_title('Average Contributions by Supergame', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Contribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(stds)*0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/aggregate_contributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/aggregate_contributions.png")
    plt.close()


def plot_contributions_by_round(session):
    """Plot average contributions by round within each supergame."""
    print("Generating contributions by round plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for sg_num in range(1, 6):
        ax = axes[sg_num - 1]
        sg = session.get_supergame(sg_num)
        
        if sg:
            round_means = []
            round_stds = []
            round_labels = []
            
            for round_num in sorted(sg.rounds.keys()):
                round_obj = sg.get_round(round_num)
                contributions = [p.contribution for p in round_obj.players.values() 
                               if p.contribution is not None]
                
                if contributions:
                    round_means.append(np.mean(contributions))
                    round_stds.append(np.std(contributions))
                    round_labels.append(f'R{round_num}')
            
            if round_means:
                bars = ax.bar(round_labels, round_means, yerr=round_stds, 
                             capsize=5, color=COLORS[sg_num-1], alpha=0.7)
                
                # Add value labels
                for bar, mean in zip(bars, round_means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(round_stds)*0.1,
                           f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Supergame {sg_num} - Contributions by Round', fontweight='bold')
        ax.set_ylabel('Average Contribution')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 25)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/contributions_by_round.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/contributions_by_round.png")
    plt.close()


def plot_individual_contribution_trends(session):
    """Plot individual player contribution trends across supergames."""
    print("Generating individual contribution trends...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all player labels
    players = sorted(session.participant_labels.values())
    
    for i, player_label in enumerate(players):
        supergame_avgs = []
        supergame_labels = []
        
        for sg_num in range(1, 6):
            sg = session.get_supergame(sg_num)
            if sg:
                player_data = sg.get_player_across_rounds(player_label)
                contributions = [p.contribution for p in player_data.values() 
                               if p.contribution is not None]
                
                if contributions:
                    supergame_avgs.append(np.mean(contributions))
                    supergame_labels.append(sg_num)
        
        if supergame_avgs:
            ax.plot(supergame_labels, supergame_avgs, marker='o', linewidth=2, 
                   alpha=0.7, label=f'Player {player_label}')
    
    ax.set_title('Individual Player Contribution Trends Across Supergames', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Supergame')
    ax.set_ylabel('Average Contribution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f'SG{i}' for i in range(1, 6)])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/individual_trends.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/individual_trends.png")
    plt.close()


def analyze_survey_responses(session):
    """Analyze and plot survey responses from finalresults."""
    print("Generating survey response analysis...")
    
    final_results = session.get_segment('finalresults')
    if not final_results or not final_results.get_round(1):
        print("No survey data found!")
        return
    
    final_round = final_results.get_round(1)
    
    # Collect survey data
    survey_data = []
    for player in final_round.players.values():
        if hasattr(player, 'data') and player.data:
            player_survey = {'player': player.label}
            for key, value in player.data.items():
                if key.startswith('q'):
                    player_survey[key] = value
            player_survey['final_payoff'] = player.data.get('final_payoff', 0)
            survey_data.append(player_survey)
    
    if not survey_data:
        print("No survey data available!")
        return
    
    survey_df = pd.DataFrame(survey_data)
    
    # Create subplots for different survey questions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot gender distribution (q1)
    if 'q1' in survey_df.columns:
        gender_counts = survey_df['q1'].value_counts()
        ax = axes[0]
        colors = ['lightblue', 'lightpink', 'lightgreen'][:len(gender_counts)]
        wedges, texts, autotexts = ax.pie(gender_counts.values, labels=gender_counts.index, 
                                         autopct='%1.1f%%', colors=colors)
        ax.set_title('Gender Distribution (Q1)', fontweight='bold')
    
    # Plot ethnicity distribution (q2)
    if 'q2' in survey_df.columns:
        ethnicity_counts = survey_df['q2'].value_counts()
        ax = axes[1]
        ax.bar(range(len(ethnicity_counts)), ethnicity_counts.values, 
               color=COLORS[:len(ethnicity_counts)])
        ax.set_xticks(range(len(ethnicity_counts)))
        ax.set_xticklabels(ethnicity_counts.index, rotation=45, ha='right')
        ax.set_title('Ethnicity Distribution (Q2)', fontweight='bold')
        ax.set_ylabel('Count')
    
    # Plot age distribution (q3)
    if 'q3' in survey_df.columns:
        ages = survey_df['q3'].dropna()
        ax = axes[2]
        ax.hist(ages, bins=8, color='skyblue', alpha=0.7, edgecolor='black')
        ax.set_title('Age Distribution (Q3)', fontweight='bold')
        ax.set_xlabel('Age')
        ax.set_ylabel('Count')
        ax.axvline(ages.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {ages.mean():.1f}')
        ax.legend()
    
    # Plot major distribution (q4) - top categories only
    if 'q4' in survey_df.columns:
        major_counts = survey_df['q4'].value_counts().head(8)  # Top 8 majors
        ax = axes[3]
        bars = ax.bar(range(len(major_counts)), major_counts.values, 
                     color=COLORS[:len(major_counts)])
        ax.set_xticks(range(len(major_counts)))
        ax.set_xticklabels([maj[:15] + '...' if len(maj) > 15 else maj 
                           for maj in major_counts.index], rotation=45, ha='right')
        ax.set_title('Academic Major Distribution (Q4)', fontweight='bold')
        ax.set_ylabel('Count')
    
    # Plot experience with experiments (q5)
    if 'q5' in survey_df.columns:
        exp_counts = survey_df['q5'].value_counts().sort_index()
        ax = axes[4]
        bars = ax.bar(exp_counts.index, exp_counts.values, color='lightcoral', alpha=0.7)
        ax.set_title('Previous Experiment Experience (Q5)', fontweight='bold')
        ax.set_xlabel('Number of Previous Experiments')
        ax.set_ylabel('Count')
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
    
    # Plot importance ratings (q6)
    if 'q6' in survey_df.columns:
        importance_counts = survey_df['q6'].value_counts()
        ax = axes[5]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(importance_counts)]
        bars = ax.bar(range(len(importance_counts)), importance_counts.values, 
                     color=colors, alpha=0.7)
        ax.set_xticks(range(len(importance_counts)))
        ax.set_xticklabels(importance_counts.index, rotation=45, ha='right')
        ax.set_title('Importance of Money (Q6)', fontweight='bold')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/survey_responses.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/survey_responses.png")
    plt.close()


def plot_payoff_rankings(session):
    """Plot player rankings by total payoff."""
    print("Generating payoff rankings...")
    
    # Calculate total payoffs for each player
    player_payoffs = {}
    
    for player_id, label in session.participant_labels.items():
        total_payoff = 0
        
        # Sum payoffs across all supergames
        for sg_num in range(1, 6):
            sg = session.get_supergame(sg_num)
            if sg:
                player_data = sg.get_player_across_rounds(label)
                for player in player_data.values():
                    if player.payoff is not None:
                        total_payoff += player.payoff
        
        # Add final payoff if available
        final_results = session.get_segment('finalresults')
        if final_results and final_results.get_round(1):
            final_player = final_results.get_round(1).get_player(label)
            if final_player and hasattr(final_player, 'data'):
                final_payoff = final_player.data.get('final_payoff', 0)
                if final_payoff:
                    total_payoff = float(final_payoff)  # Use the calculated final payoff instead
        
        player_payoffs[label] = total_payoff
    
    # Sort by payoff
    sorted_payoffs = sorted(player_payoffs.items(), key=lambda x: x[1], reverse=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    players, payoffs = zip(*sorted_payoffs)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(players)))
    
    bars = ax.bar(range(len(players)), payoffs, color=colors)
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels(players)
    ax.set_title('Player Rankings by Total Payoff', fontsize=14, fontweight='bold')
    ax.set_xlabel('Player')
    ax.set_ylabel('Total Payoff')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, payoff) in enumerate(zip(bars, payoffs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(payoffs)*0.01,
               f'{payoff:.0f}', ha='center', va='bottom', fontweight='bold')
        # Add rank number
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'#{i+1}', ha='center', va='center', fontsize=12, fontweight='bold',
               color='white' if i < len(players)//2 else 'black')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/payoff_rankings.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/payoff_rankings.png")
    plt.close()


def plot_contribution_rankings(session):
    """Plot player rankings by average contribution."""
    print("Generating contribution rankings...")
    
    # Calculate average contributions for each player
    player_avg_contributions = {}
    
    for player_id, label in session.participant_labels.items():
        all_contributions = []
        
        # Collect contributions across all supergames
        for sg_num in range(1, 6):
            sg = session.get_supergame(sg_num)
            if sg:
                player_data = sg.get_player_across_rounds(label)
                for player in player_data.values():
                    if player.contribution is not None:
                        all_contributions.append(player.contribution)
        
        if all_contributions:
            player_avg_contributions[label] = np.mean(all_contributions)
        else:
            player_avg_contributions[label] = 0
    
    # Sort by average contribution
    sorted_contributions = sorted(player_avg_contributions.items(), 
                                 key=lambda x: x[1], reverse=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    players, avg_contribs = zip(*sorted_contributions)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(players)))
    
    bars = ax.bar(range(len(players)), avg_contribs, color=colors)
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels(players)
    ax.set_title('Player Rankings by Average Contribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Player')
    ax.set_ylabel('Average Contribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, avg_contrib) in enumerate(zip(bars, avg_contribs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(avg_contribs)*0.01,
               f'{avg_contrib:.1f}', ha='center', va='bottom', fontweight='bold')
        # Add rank number
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
               f'#{i+1}', ha='center', va='center', fontsize=12, fontweight='bold',
               color='white')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/contribution_rankings.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/contribution_rankings.png")
    plt.close()


def generate_summary_statistics(session):
    """Generate and save summary statistics."""
    print("Generating summary statistics...")
    
    stats = {}
    
    # Overall statistics
    all_contributions = []
    all_payoffs = []
    
    for sg_num in range(1, 6):
        sg = session.get_supergame(sg_num)
        if sg:
            for round_num in sg.rounds.keys():
                round_obj = sg.get_round(round_num)
                for player in round_obj.players.values():
                    if player.contribution is not None:
                        all_contributions.append(player.contribution)
                    if player.payoff is not None:
                        all_payoffs.append(player.payoff)
    
    stats['Overall'] = {
        'Total observations': len(all_contributions),
        'Mean contribution': np.mean(all_contributions),
        'Std contribution': np.std(all_contributions),
        'Mean payoff': np.mean(all_payoffs),
        'Std payoff': np.std(all_payoffs)
    }
    
    # By supergame statistics
    for sg_num in range(1, 6):
        sg = session.get_supergame(sg_num)
        if sg:
            sg_contributions = []
            sg_payoffs = []
            
            for round_num in sg.rounds.keys():
                round_obj = sg.get_round(round_num)
                for player in round_obj.players.values():
                    if player.contribution is not None:
                        sg_contributions.append(player.contribution)
                    if player.payoff is not None:
                        sg_payoffs.append(player.payoff)
            
            if sg_contributions:
                stats[f'Supergame {sg_num}'] = {
                    'Observations': len(sg_contributions),
                    'Rounds': len(sg.rounds),
                    'Mean contribution': np.mean(sg_contributions),
                    'Std contribution': np.std(sg_contributions),
                    'Mean payoff': np.mean(sg_payoffs),
                    'Std payoff': np.std(sg_payoffs)
                }
    
    # Save to file
    with open(f'{OUTPUT_DIR}/summary_statistics.txt', 'w') as f:
        f.write("EXPERIMENTAL DATA SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        for section, data in stats.items():
            f.write(f"{section}:\n")
            for key, value in data.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"Saved: {OUTPUT_DIR}/summary_statistics.txt")


def generate_chat_word_frequency(session):
    """Generate word frequency bar chart from all chat messages."""
    print("Generating chat word frequency analysis...")
    
    # Collect all chat text
    all_text = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    # Clean text - remove punctuation and normalize
                    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', msg.body.lower())
                    all_text.append(cleaned_text)
    
    if not all_text:
        print("No chat messages found for word frequency analysis!")
        return
    
    # Combine all text and extract words
    combined_text = ' '.join(all_text)
    words = combined_text.split()
    
    # Filter out common stop words and short words
    stop_words = {'the', 'and', 'but', 'for', 'are', 'was', 'were', 'been', 'have', 
                  'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 
                  'must', 'can', 'cant', 'wont', 'dont', 'didnt', 'isnt', 'arent', 
                  'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wouldnt', 'couldnt', 
                  'shouldnt', 'mightnt', 'mustnt', 'not', 'now', 'just', 'like', 
                  'get', 'got', 'want', 'think', 'know', 'see', 'said', 'say', 
                  'going', 'go', 'went', 'come', 'came', 'way', 'well', 'back', 
                  'time', 'make', 'made', 'take', 'took', 'look', 'looked'}
    
    # Filter words (remove stop words and words shorter than 3 characters)
    filtered_words = [word for word in words if len(word) >= 3 and word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get top 15 most common words
    top_words = word_counts.most_common(15)
    
    if not top_words:
        print("No words found after filtering!")
        return
    
    # Prepare data for plotting
    words_list, frequencies = zip(*top_words)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(range(len(words_list)), frequencies, 
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(words_list))), 
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Words', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Frequent Words in Chat Messages', 
                fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(range(len(words_list)))
    ax.set_xticklabels(words_list, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(frequencies)*0.01,
               str(freq), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add statistics text
    total_messages = len(all_text)
    total_words = len(words)
    unique_words = len(set(filtered_words))
    
    stats_text = f"Total messages: {total_messages}\nTotal words: {total_words:,}\nUnique words (filtered): {unique_words:,}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/chat_word_frequency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/chat_word_frequency.png")
    plt.close()


def plot_chat_messages_per_round(session):
    """Plot number of chat messages per round across all supergames."""
    print("Generating chat messages per round plot...")
    
    # Collect data for each supergame
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for sg_num in range(1, 6):
        ax = axes[sg_num - 1]
        sg = session.get_supergame(sg_num)
        
        if sg:
            round_labels = []
            message_counts = []
            
            for round_num in sorted(sg.rounds.keys()):
                round_obj = sg.get_round(round_num)
                round_labels.append(f'R{round_num}')
                message_counts.append(len(round_obj.chat_messages))
            
            if round_labels:
                bars = ax.bar(round_labels, message_counts, 
                             color=COLORS[sg_num-1], alpha=0.7)
                
                # Add value labels on bars
                for bar, count in zip(bars, message_counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Supergame {sg_num} - Chat Messages per Round', fontweight='bold')
        ax.set_ylabel('Number of Messages')
        ax.set_xlabel('Round')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from 0
        ax.set_ylim(0, max(message_counts) * 1.1 if message_counts else 1)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Chat Activity by Round Across Supergames', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/chat_messages_per_round.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/chat_messages_per_round.png")
    plt.close()


def plot_chat_contribution_correlation(session):
    """Plot correlation between number of chat messages per round and group total contributions."""
    print("Generating chat-contribution correlation analysis...")
    
    # Collect data points (chat messages vs group contributions)
    chat_counts = []
    group_contributions = []
    round_labels = []
    colors_list = []
    
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'purple'}
    
    for sg_num in range(1, 6):
        sg = session.get_supergame(sg_num)
        if sg:
            for round_num in sorted(sg.rounds.keys()):
                round_obj = sg.get_round(round_num)
                
                # For each group in this round
                for group_id, group in round_obj.groups.items():
                    if group.total_contribution is not None:
                        # Count chat messages for this group
                        group_chat_count = len(group.chat_messages)
                        
                        chat_counts.append(group_chat_count)
                        group_contributions.append(group.total_contribution)
                        round_labels.append(f'SG{sg_num}R{round_num}G{group_id}')
                        colors_list.append(color_map[sg_num])
    
    if len(chat_counts) < 2:
        print("Insufficient data for correlation analysis!")
        return
    
    # Calculate correlation
    correlation, p_value = pearsonr(chat_counts, group_contributions)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    for sg_num in range(1, 6):
        sg_color = color_map[sg_num]
        sg_chat = [chat_counts[i] for i, c in enumerate(colors_list) if c == sg_color]
        sg_contrib = [group_contributions[i] for i, c in enumerate(colors_list) if c == sg_color]
        
        if sg_chat:
            ax1.scatter(sg_chat, sg_contrib, c=sg_color, alpha=0.7, 
                       label=f'Supergame {sg_num}', s=60)
    
    # Add trend line
    if len(chat_counts) > 1:
        z = np.polyfit(chat_counts, group_contributions, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(chat_counts), max(chat_counts), 100)
        ax1.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Number of Chat Messages per Group per Round')
    ax1.set_ylabel('Group Total Contribution')
    ax1.set_title(f'Chat Messages vs Group Contributions\n(r = {correlation:.3f}, p = {p_value:.3f})', 
                 fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of chat counts
    ax2.hist(chat_counts, bins=max(chat_counts)+1 if max(chat_counts) < 20 else 20, 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Number of Chat Messages')
    ax2.set_ylabel('Frequency (Group-Rounds)')
    ax2.set_title('Distribution of Chat Activity\nper Group per Round', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"""Correlation Statistics:
Pearson r = {correlation:.3f}
p-value = {p_value:.3f}
Sample size = {len(chat_counts)} group-rounds

Mean messages per group-round: {np.mean(chat_counts):.1f}
Mean group contribution: {np.mean(group_contributions):.1f}"""
    
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/chat_contribution_correlation.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/chat_contribution_correlation.png")
    print(f"Correlation: r = {correlation:.3f}, p = {p_value:.3f}")
    plt.close()


def plot_sentiment_distribution(session):
    """Plot distribution of chat sentiment scores from -1 to 1."""
    print("Generating sentiment distribution plot...")
    
    # Collect all individual message sentiment scores
    all_sentiments = []
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                for msg in round_obj.chat_messages:
                    all_sentiments.append(msg.sentiment)
    
    if not all_sentiments:
        print("No sentiment data found!")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram
    n_bins = 40
    counts, bins, patches = ax.hist(all_sentiments, bins=n_bins, 
                                   alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars based on sentiment (red for negative, gray for neutral, green for positive)
    for i, (count, patch) in enumerate(zip(counts, patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < -0.05:
            patch.set_facecolor('#FF6B6B')  # Red for negative
        elif bin_center > 0.05:
            patch.set_facecolor('#4ECDC4')  # Teal for positive  
        else:
            patch.set_facecolor('#95A5A6')  # Gray for neutral
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Neutral (0.0)')
    
    # Customize plot
    ax.set_xlabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (Number of Messages)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Chat Message Sentiment Scores', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sentiment region labels
    ax.text(-0.7, max(counts) * 0.8, 'Negative\nSentiment', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF6B6B', alpha=0.3))
    ax.text(0.7, max(counts) * 0.8, 'Positive\nSentiment', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#4ECDC4', alpha=0.3))
    
    # Add statistics
    mean_sentiment = np.mean(all_sentiments)
    std_sentiment = np.std(all_sentiments)
    positive_msgs = sum(1 for s in all_sentiments if s > 0.05)
    negative_msgs = sum(1 for s in all_sentiments if s < -0.05)
    neutral_msgs = len(all_sentiments) - positive_msgs - negative_msgs
    
    stats_text = f"""Total Messages: {len(all_sentiments):,}
Mean Sentiment: {mean_sentiment:+.3f}
Std Deviation: {std_sentiment:.3f}

Positive: {positive_msgs} ({positive_msgs/len(all_sentiments)*100:.1f}%)
Neutral: {neutral_msgs} ({neutral_msgs/len(all_sentiments)*100:.1f}%)
Negative: {negative_msgs} ({negative_msgs/len(all_sentiments)*100:.1f}%)"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
           fontsize=10)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/sentiment_distribution.png")
    plt.close()


def plot_sentiment_contribution_correlation(session):
    """Plot correlation between player sentiment in round t-1 and contribution in round t."""
    print("Generating sentiment-contribution correlation plot...")
    
    # Collect data points: sentiment in round t-1 vs contribution in round t
    sentiment_t_minus_1 = []
    contribution_t = []
    supergame_labels = []
    player_labels = []
    
    color_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'purple'}
    
    for sg_num in range(1, 6):
        sg = session.get_supergame(sg_num)
        if sg and len(sg.rounds) > 1:  # Need at least 2 rounds for t-1 and t
            # For each player, look at their sentiment in round t-1 and contribution in round t
            for player_label in session.participant_labels.values():
                rounds = sorted(sg.rounds.keys())
                
                for i in range(1, len(rounds)):  # Start from second round
                    round_t_minus_1 = sg.get_round(rounds[i-1])
                    round_t = sg.get_round(rounds[i])
                    
                    if (round_t_minus_1 and round_t and 
                        player_label in round_t_minus_1.players and 
                        player_label in round_t.players):
                        
                        player_prev = round_t_minus_1.players[player_label]
                        player_curr = round_t.players[player_label]
                        
                        # Get sentiment from previous round
                        prev_sentiment = player_prev.get_chat_sentiment()
                        
                        if (prev_sentiment and prev_sentiment.message_count > 0 and 
                            player_curr.contribution is not None):
                            
                            sentiment_t_minus_1.append(prev_sentiment.compound)
                            contribution_t.append(player_curr.contribution)
                            supergame_labels.append(sg_num)
                            player_labels.append(player_label)
    
    if len(sentiment_t_minus_1) < 2:
        print("Insufficient data for sentiment-contribution correlation!")
        return
    
    # Calculate correlation
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(sentiment_t_minus_1, contribution_t)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot colored by supergame
    for sg_num in range(1, 6):
        sg_color = color_map[sg_num]
        sg_sentiment = [sentiment_t_minus_1[i] for i, sg in enumerate(supergame_labels) if sg == sg_num]
        sg_contrib = [contribution_t[i] for i, sg in enumerate(supergame_labels) if sg == sg_num]
        
        if sg_sentiment:
            ax1.scatter(sg_sentiment, sg_contrib, c=sg_color, alpha=0.7, 
                       label=f'Supergame {sg_num}', s=60)
    
    # Add trend line
    if len(sentiment_t_minus_1) > 1:
        z = np.polyfit(sentiment_t_minus_1, contribution_t, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(sentiment_t_minus_1), max(sentiment_t_minus_1), 100)
        ax1.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Player Sentiment in Round t-1')
    ax1.set_ylabel('Player Contribution in Round t')
    ax1.set_title(f'Sentiment → Contribution Correlation\n(r = {correlation:.3f}, p = {p_value:.3f})', 
                 fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of sentiment scores used
    ax2.hist(sentiment_t_minus_1, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Sentiment Score (t-1)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Sentiment Scores\nUsed in Correlation', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"""Correlation Analysis:
Pearson r = {correlation:.3f}
p-value = {p_value:.3f}
Sample size = {len(sentiment_t_minus_1)} observations

Mean sentiment (t-1): {np.mean(sentiment_t_minus_1):.3f}
Mean contribution (t): {np.mean(contribution_t):.1f}

Interpretation:
{'Strong' if abs(correlation) >= 0.5 else 'Moderate' if abs(correlation) >= 0.3 else 'Weak'} {'positive' if correlation > 0 else 'negative'} correlation
{'Significant' if p_value < 0.05 else 'Not significant'} at α = 0.05"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sentiment_contribution_correlation.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/sentiment_contribution_correlation.png")
    print(f"Sentiment → Contribution correlation: r = {correlation:.3f}, p = {p_value:.3f}")
    plt.close()


def plot_sentiment_by_segment(session):
    """Plot box and whisker plot of sentiment scores grouped by segment/supergame."""
    print("Generating sentiment by segment box plot...")
    
    # Collect sentiment data by supergame
    supergame_sentiments = {}
    
    for sg_num in range(1, 6):
        sg = session.get_supergame(sg_num)
        if sg:
            # Get all individual message sentiments for this supergame
            sg_sentiments = []
            for round_obj in sg.rounds.values():
                for msg in round_obj.chat_messages:
                    sg_sentiments.append(msg.sentiment)
            
            if sg_sentiments:
                supergame_sentiments[f'Supergame {sg_num}'] = sg_sentiments
    
    if not supergame_sentiments:
        print("No sentiment data found for segment analysis!")
        return
    
    # Prepare data for box plot
    data_for_boxplot = list(supergame_sentiments.values())
    labels = list(supergame_sentiments.keys())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Box plot
    box_plot = ax1.boxplot(data_for_boxplot, tick_labels=labels, patch_artist=True,
                          showmeans=True, meanline=True)
    
    # Color the boxes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize box plot
    ax1.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Sentiment Scores by Supergame', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add statistical annotations
    means = [np.mean(sentiments) for sentiments in data_for_boxplot]
    stds = [np.std(sentiments) for sentiments in data_for_boxplot]
    
    for i, (mean_val, std_val, label) in enumerate(zip(means, stds, labels)):
        ax1.text(i+1, max([max(s) for s in data_for_boxplot]) * 1.02, 
                f'μ = {mean_val:+.3f}\nσ = {std_val:.3f}', 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Bar plot showing summary statistics
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Supergame', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax2.set_title('Average Sentiment Score by Supergame (with Standard Deviation)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'SG{i+1}' for i in range(len(labels))])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add value labels on bars
    for bar, mean_val, n_msgs in zip(bars, means, [len(s) for s in data_for_boxplot]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
                height + (max(means) + max(stds)) * 0.02,
                f'{mean_val:+.3f}\n(n={n_msgs})', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add overall statistics text
    all_sentiments = [s for sentiments in data_for_boxplot for s in sentiments]
    overall_stats = f"""Overall Statistics:
Total Messages: {len(all_sentiments):,}
Grand Mean: {np.mean(all_sentiments):+.3f}
Grand Std: {np.std(all_sentiments):.3f}

Supergame Comparison:
Highest: {labels[np.argmax(means)]} ({max(means):+.3f})
Lowest: {labels[np.argmin(means)]} ({min(means):+.3f})
Range: {max(means) - min(means):.3f}"""
    
    ax2.text(0.98, 0.02, overall_stats, transform=ax2.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sentiment_by_segment.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/sentiment_by_segment.png")
    plt.close()


def main():
    """Main function to run all analyses."""
    print("=" * 60)
    print("EXPERIMENTAL DATA ANALYSIS")
    print("=" * 60)
    
    # Setup
    setup_output_directory()
    
    # Load data with chat integration
    csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/all_apps_wide_2025-09-11.csv'
    chat_csv_path = '/Users/caleb/Library/CloudStorage/Box-Box/SharedFolder_LPCP/pilot/session data/ChatMessages-2025-09-11.csv'
    print(f"Loading data from: {csv_path}")
    print(f"Loading chat data from: {chat_csv_path}")
    session = load_experiment_data(csv_path, chat_csv_path)
    print(f"Loaded session: {session.session_code}")
    print(f"Participants: {len(session.participant_labels)}")
    print(f"Segments: {list(session.segments.keys())}")
    
    # Count chat messages
    total_chat_messages = 0
    for segment in session.segments.values():
        if segment.name.startswith('supergame'):
            for round_obj in segment.rounds.values():
                total_chat_messages += len(round_obj.chat_messages)
    print(f"Total chat messages loaded: {total_chat_messages}")
    print()
    
    # Generate all plots
    plot_average_contributions_aggregate(session)
    plot_contributions_by_round(session)
    plot_individual_contribution_trends(session)
    analyze_survey_responses(session)
    plot_payoff_rankings(session)
    plot_contribution_rankings(session)
    generate_summary_statistics(session)
    
    # Generate chat analysis plots
    if total_chat_messages > 0:
        print("\n" + "=" * 30)
        print("CHAT DATA ANALYSIS")
        print("=" * 30)
        generate_chat_word_frequency(session)
        plot_chat_messages_per_round(session)
        plot_chat_contribution_correlation(session)
        
        print("\n" + "=" * 30)
        print("SENTIMENT ANALYSIS")
        print("=" * 30)
        plot_sentiment_distribution(session)
        plot_sentiment_contribution_correlation(session)
        plot_sentiment_by_segment(session)
    else:
        print("\n⚠️  No chat messages found - skipping chat and sentiment analysis")
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"All plots and statistics saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()