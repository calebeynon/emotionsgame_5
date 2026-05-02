"""
Interactive verification tool for LLM promise classifications.

Presents each chat message with full group conversation context and current
LLM-assigned label. Reviewer confirms or flips each label (0 = not promise,
1 = promise). Progress saves automatically for pause/resume.

Usage:
    uv run python analysis/derived/verify_promises.py           # Start/resume review
    uv run python analysis/derived/verify_promises.py --export  # Export without reviewing

Author: Claude Code
Date: 2026-04-10
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_data import load_experiment_data

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
INPUT_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'
PROGRESS_FILE = DATA_DIR / 'derived' / 'promise_verification_progress.json'
OUTPUT_FILE = DATA_DIR / 'derived' / 'promise_classifications_verified.csv'


# =====
# Main function
# =====
def main():
    """Main execution flow."""
    args = parse_args()
    df = pd.read_csv(INPUT_FILE)
    review_items = build_review_items(df)
    progress = load_progress()
    remaining = print_status(review_items, progress)

    if args.export or remaining == 0:
        export_results(df, progress)
        return

    context_lookup = load_context()
    progress = run_review(review_items, progress, context_lookup)
    export_results(df, progress)


def print_status(review_items: list, progress: dict) -> int:
    """Print current review status and return remaining count."""
    reviewed = sum(1 for r in review_items if r['key'] in progress)
    remaining = len(review_items) - reviewed
    print(f"\nLoaded {len(review_items)} messages ({reviewed} reviewed, {remaining} remaining)")
    if remaining == 0:
        print("All messages already reviewed!")
    return remaining


def load_context() -> dict:
    """Load experiment data and build conversation context lookup."""
    print("Loading experiment data for conversation context...")
    return build_context_lookup(load_experiment())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify promise classifications")
    parser.add_argument('--export', action='store_true',
                        help="Export current progress to CSV without reviewing")
    return parser.parse_args()


# =====
# Data loading
# =====
def load_experiment():
    """Load experiment data for conversation context."""
    file_pairs = []
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = 1 if '_t1_' in data_file.name else (2 if '_t2_' in data_file.name else 0)
        chat_file = data_file.with_name(data_file.name.replace("_data", "_chat"))
        chat_path = str(chat_file) if chat_file.exists() else None
        file_pairs.append((str(data_file), chat_path, treatment))
    return load_experiment_data(file_pairs, name="Verification")


def build_context_lookup(experiment) -> dict:
    """Build lookup: (session, segment, round, group) -> sorted chat messages."""
    lookup = {}
    for session_code, session in experiment.sessions.items():
        for seg_name, segment in session.segments.items():
            if not seg_name.startswith('supergame'):
                continue
            for round_num, round_obj in segment.rounds.items():
                for group_id, group in round_obj.groups.items():
                    key = (session_code, seg_name, round_num, group_id)
                    lookup[key] = sorted(group.chat_messages, key=lambda m: m.timestamp)
    return lookup


# =====
# Review item construction
# =====
def build_review_items(df: pd.DataFrame) -> list:
    """Flatten CSV rows into individual message review items."""
    items = []
    for row_idx, row in df.iterrows():
        messages = json.loads(row['messages'])
        classifications = json.loads(row['classifications'])
        for msg_idx, (msg, cls) in enumerate(zip(messages, classifications)):
            items.append(build_item(row, row_idx, msg_idx, msg, cls))
    return items


def build_item(row, row_idx: int, msg_idx: int, msg: str, cls: int) -> dict:
    """Build a single review item from a CSV row and message index."""
    return {
        'key': make_key(row, msg_idx),
        'row_idx': row_idx,
        'msg_idx': msg_idx,
        'message': msg,
        'original_classification': cls,
        'session_code': row['session_code'],
        'segment': row['segment'],
        'round': int(row['round']),
        'group': int(row['group']),
        'label': row['label'],
    }


def make_key(row, msg_idx: int) -> str:
    """Create unique key for a message within a player-round."""
    return f"{row['session_code']}|{row['segment']}|{row['round']}|{row['group']}|{row['label']}|{msg_idx}"


# =====
# Progress management
# =====
def load_progress() -> dict:
    """Load review progress from JSON file."""
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress: dict):
    """Save review progress to JSON file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# =====
# Interactive review
# =====
def run_review(items: list, progress: dict, context_lookup: dict) -> dict:
    """Run the interactive review loop."""
    unreviewed = [i for i in items if i['key'] not in progress]
    total = len(items)
    done_count = total - len(unreviewed)
    print(f"\nStarting review. {len(unreviewed)} messages remaining.")
    print("Controls: [Enter]=Confirm  [f]=Flip  [s]=Skip  [q]=Save & Quit\n")

    for item in unreviewed:
        done_count += 1
        display_message(item, done_count, total, context_lookup)
        done_count, should_quit = process_action(item, progress, done_count, total)
        if should_quit:
            return progress

    return finish_review(progress, total, total, complete=True)


def process_action(item: dict, progress: dict, done: int, total: int) -> tuple:
    """Handle a single review action. Returns (done_count, should_quit)."""
    action = get_user_action(item['original_classification'])
    if action == 'quit':
        finish_review(progress, done - 1, total, complete=False)
        return done, True
    if action == 'skip':
        return done - 1, False
    record_review(progress, item, action)
    if done % 50 == 0:
        save_progress(progress)
    return done, False


def record_review(progress: dict, item: dict, verified_cls: int):
    """Record a single review decision."""
    progress[item['key']] = {
        'verified': verified_cls,
        'original': item['original_classification'],
        'changed': verified_cls != item['original_classification'],
    }


def finish_review(progress: dict, done: int, total: int, *, complete: bool) -> dict:
    """Save progress and print completion message."""
    save_progress(progress)
    if complete:
        print(f"\nReview complete! All {total} messages reviewed.")
    else:
        print(f"\nProgress saved. {done} of {total} reviewed.")
    return progress


def display_message(item: dict, current: int, total: int, context_lookup: dict):
    """Display a message with its group conversation context."""
    print_header(item, current, total)
    ctx_key = (item['session_code'], item['segment'], item['round'], item['group'])
    group_msgs = context_lookup.get(ctx_key, [])
    print_conversation(group_msgs, item)
    print_label(item['original_classification'])


def print_header(item: dict, current: int, total: int):
    """Print the message header with location info."""
    print("=" * 70)
    print(f"  Message {current}/{total}  |  {item['session_code']}  |  "
          f"{item['segment']}, Round {item['round']}, Group {item['group']}")
    print("=" * 70)


def print_conversation(group_msgs: list, item: dict):
    """Print the group conversation with the target message highlighted."""
    target_idx = find_target_index(group_msgs, item['label'], item['msg_idx'])
    print("\nGroup conversation:")
    if not group_msgs:
        print(f"  \033[1;33m► {item['label']}: {item['message']}\033[0m  ◄")
        return
    for i, msg in enumerate(group_msgs):
        if i == target_idx:
            print(f"  \033[1;33m► {msg.nickname}: {msg.body}\033[0m  ◄")
        else:
            print(f"    {msg.nickname}: {msg.body}")


def print_label(cls: int):
    """Print the current classification label."""
    label = "\033[1;32mPROMISE (1)\033[0m" if cls == 1 else "NOT PROMISE (0)"
    print(f"\nCurrent label: {label}")


def find_target_index(group_msgs: list, label: str, msg_idx: int) -> int | None:
    """Find the global index of the player's N-th message in the group conversation."""
    player_count = 0
    for i, msg in enumerate(group_msgs):
        if msg.nickname == label:
            if player_count == msg_idx:
                return i
            player_count += 1
    return None


def get_user_action(current_cls: int):
    """Get user action: returns verified classification (int), 'skip', or 'quit'."""
    flip_label = "PROMISE (1)" if current_cls == 0 else "NOT PROMISE (0)"
    try:
        action = input(f"[Enter] Confirm  [f] Flip→{flip_label}  [s] Skip  [q] Quit > ")
    except (EOFError, KeyboardInterrupt):
        return 'quit'

    action = action.strip().lower()
    if action == 'q':
        return 'quit'
    if action == 's':
        return 'skip'
    if action == 'f':
        return 1 - current_cls
    return current_cls


# =====
# Export results
# =====
def export_results(df: pd.DataFrame, progress: dict):
    """Export verified results to CSV with review status columns."""
    if not progress:
        print("\nNo reviews to export.")
        return

    row_results = [compute_row_verification(row, progress) for _, row in df.iterrows()]

    df_out = df.copy()
    df_out['verified_classifications'] = [r[0] for r in row_results]
    df_out['review_status'] = [r[1] for r in row_results]
    df_out['num_changes'] = [r[2] for r in row_results]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print_summary(progress)
    print(f"\nVerified results saved to: {OUTPUT_FILE}")


def compute_row_verification(row, progress: dict) -> tuple:
    """Compute verified classifications and status for a single row."""
    messages = json.loads(row['messages'])
    verified = json.loads(row['classifications'])
    changes, reviewed = 0, 0

    for msg_idx in range(len(messages)):
        key = make_key(row, msg_idx)
        if key in progress:
            verified[msg_idx] = progress[key]['verified']
            changes += int(progress[key]['changed'])
            reviewed += 1

    status = 'verified' if reviewed == len(messages) else ('partial' if reviewed > 0 else 'unreviewed')
    return json.dumps(verified), status, changes


def print_summary(progress: dict):
    """Print agreement statistics."""
    total = len(progress)
    changed = sum(1 for p in progress.values() if p['changed'])
    confirmed = total - changed
    flipped_to_1 = sum(1 for p in progress.values() if p['changed'] and p['verified'] == 1)
    flipped_to_0 = sum(1 for p in progress.values() if p['changed'] and p['verified'] == 0)

    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Messages reviewed:  {total}")
    print(f"Confirmed (agree):  {confirmed} ({confirmed / max(total, 1) * 100:.1f}%)")
    print(f"Changed (disagree): {changed} ({changed / max(total, 1) * 100:.1f}%)")
    print(f"  0→1 (missed promises): {flipped_to_1}")
    print(f"  1→0 (false promises):  {flipped_to_0}")
    print("=" * 50)


# %%
if __name__ == "__main__":
    main()
