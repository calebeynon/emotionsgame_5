"""
Promise classification script for experimental chat messages.

Iterates through experiment data, classifies each message using dual LLM
classifiers (OpenAI + Anthropic), and outputs aggregated player-round level results.

Author: Claude Code
Date: 2026-01-16
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd


def log(msg: str):
    """Print with immediate flush for real-time logging."""
    print(msg, flush=True)

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_data import load_experiment_data, Experiment
from dual_classifier import classify_batch_parallel, calculate_agreement_rate
from llm_clients import get_openai_cost_estimate, get_anthropic_cost_estimate

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
OUTPUT_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'


# =====
# Main function
# =====
def main():
    """Main execution flow for promise classification."""
    experiment = load_experiment()
    messages_data = collect_all_messages(experiment)

    total_messages = sum(len(m['messages']) for m in messages_data)
    log(f"\nCollected {len(messages_data)} player-rounds with {total_messages} messages")

    if total_messages == 0:
        log("No messages found to classify.")
        return

    print_cost_estimate(total_messages, messages_data)

    results = classify_all_messages(messages_data)
    save_results(pd.DataFrame.from_records(results))
    print_summary(results)


# =====
# Data loading
# =====
def load_experiment() -> Experiment:
    """Load experiment data from raw session files."""
    return load_experiment_data(build_file_pairs(), name="Promise Classification")


def build_file_pairs() -> list:
    """Build list of (data_csv, chat_csv, treatment) tuples from raw directory."""
    file_pairs = []
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = 1 if '_t1_' in data_file.name else (2 if '_t2_' in data_file.name else 0)
        chat_file = data_file.with_name(data_file.name.replace("_data", "_chat"))
        chat_path = str(chat_file) if chat_file.exists() else None
        file_pairs.append((str(data_file), chat_path, treatment))
    return file_pairs


def extract_treatment(filename: str) -> int:
    """Extract treatment number from filename like '01_t1_data.csv'."""
    if '_t1_' in filename:
        return 1
    return 2 if '_t2_' in filename else 0


# =====
# Message collection
# =====
def collect_all_messages(experiment: Experiment) -> list:
    """Collect all messages organized by player-round from experiment data."""
    messages_data = []
    for session_code, session in experiment.sessions.items():
        for segment_name, segment in session.segments.items():
            if not segment_name.startswith('supergame'):
                continue
            for round_num, round_obj in segment.rounds.items():
                for group_id, group in round_obj.groups.items():
                    all_msgs = sorted(group.chat_messages, key=lambda m: m.timestamp)
                    for label, player in group.players.items():
                        player_msgs = [m.body for m in all_msgs if m.nickname == label]
                        if player_msgs:
                            messages_data.append({
                                'session_code': session_code,
                                'treatment': session.treatment,
                                'segment': segment_name,
                                'round': round_num,
                                'group': group_id,
                                'label': label,
                                'participant_id': player.participant_id,
                                'contribution': player.contribution,
                                'payoff': player.payoff,
                                'messages': player_msgs,
                                'all_group_msgs': all_msgs,
                            })
    return messages_data


# =====
# Cost estimation
# =====
def print_cost_estimate(total_messages: int, messages_data: list):
    """Print cost estimate for the classification run."""
    avg_context = calculate_avg_context(messages_data)
    openai_cost = get_openai_cost_estimate(total_messages, avg_context)
    anthropic_cost = get_anthropic_cost_estimate(total_messages, avg_context)
    total_cost = openai_cost['estimated_cost_usd'] + anthropic_cost['estimated_cost_usd']

    log("\n" + "=" * 50)
    log("COST ESTIMATE")
    log("=" * 50)
    log(f"Messages to classify: {total_messages}")
    log(f"Average context length: {avg_context:.1f} messages")
    log(f"\nOpenAI estimated cost: ${openai_cost['estimated_cost_usd']:.4f}")
    log(f"Anthropic estimated cost: ${anthropic_cost['estimated_cost_usd']:.4f}")
    log(f"Total estimated cost: ${total_cost:.4f}")
    log("=" * 50 + "\n")


def calculate_avg_context(messages_data: list) -> float:
    """Calculate average context length across all messages."""
    total_context, total_messages = 0, 0
    for player_data in messages_data:
        all_msgs = player_data['all_group_msgs']
        for i, msg in enumerate(all_msgs):
            if msg.nickname == player_data['label']:
                total_context += i
                total_messages += 1
    return total_context / max(total_messages, 1)


# =====
# Classification
# =====
def classify_all_messages(messages_data: list, max_workers: int = 20) -> list:
    """Classify all messages in parallel and aggregate to player-round level."""
    flat_messages, index_map = flatten_messages(messages_data)
    log(f"Classifying {len(flat_messages)} messages with {max_workers} workers...")

    classifications = classify_batch_parallel(flat_messages, max_workers=max_workers)

    return aggregate_results(messages_data, classifications, index_map)


def flatten_messages(messages_data: list) -> tuple:
    """Flatten all messages into a single list for batch processing."""
    flat_messages = []
    index_map = []

    for player_idx, player_data in enumerate(messages_data):
        all_group_msgs = player_data['all_group_msgs']
        label = player_data['label']
        for msg_body in player_data['messages']:
            context = build_context(all_group_msgs, msg_body, label)
            flat_messages.append({'message': msg_body, 'context': context})
            index_map.append(player_idx)

    return flat_messages, index_map


def aggregate_results(messages_data: list, classifications: list, index_map: list) -> list:
    """Aggregate flat classification results back to player-round level."""
    player_results = {i: {'cls': [], 'openai': [], 'anthropic': [], 'disputed': 0}
                      for i in range(len(messages_data))}

    for cls_idx, result in enumerate(classifications):
        player_idx = index_map[cls_idx]
        player_results[player_idx]['cls'].append(result['consensus'])
        player_results[player_idx]['openai'].append(result['openai'])
        player_results[player_idx]['anthropic'].append(result['anthropic'])
        if result['disputed']:
            player_results[player_idx]['disputed'] += 1

    return [build_result_record(messages_data[i], player_results[i]['cls'],
                                player_results[i]['openai'], player_results[i]['anthropic'],
                                player_results[i]['disputed'])
            for i in range(len(messages_data))]


def build_context(all_group_msgs: list, target_msg: str, target_label: str) -> list:
    """Build context list of prior messages for classification."""
    context = []
    for msg in all_group_msgs:
        if msg.body == target_msg and msg.nickname == target_label:
            break
        context.append({'sender': msg.nickname, 'body': msg.body})
    return context


def build_result_record(player_data, classifications, openai_cls, anthropic_cls, disputed) -> dict:
    """Build final result record for a player-round."""
    promise_count = sum(1 for c in classifications if c == 1)
    message_count = len(classifications)
    return {
        'session_code': player_data['session_code'],
        'treatment': player_data['treatment'],
        'segment': player_data['segment'],
        'round': player_data['round'],
        'group': player_data['group'],
        'label': player_data['label'],
        'participant_id': player_data['participant_id'],
        'contribution': player_data['contribution'],
        'payoff': player_data['payoff'],
        'message_count': message_count,
        'promise_count': promise_count,
        'promise_percentage': promise_count / max(message_count, 1) * 100,
        'disputed_count': disputed,
        'messages': json.dumps(player_data['messages']),
        'classifications': json.dumps(classifications),
        'openai_classifications': json.dumps(openai_cls),
        'anthropic_classifications': json.dumps(anthropic_cls),
    }


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save results DataFrame to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log(f"\nResults saved to: {OUTPUT_FILE}")


def print_summary(results: list):
    """Print final summary statistics."""
    all_cls, total_disputed = [], 0
    for r in results:
        all_cls.extend(json.loads(r['classifications']))
        total_disputed += r['disputed_count']

    total = len(all_cls)
    agreement_rate = (total - total_disputed) / max(total, 1) * 100

    log("\n" + "=" * 50)
    log("CLASSIFICATION SUMMARY")
    log("=" * 50)
    log(f"Total messages classified: {total}")
    log(f"Agreement rate: {agreement_rate:.1f}%")
    log(f"Disputed classifications: {total_disputed}")
    log("=" * 50)


# %%
if __name__ == "__main__":
    main()
