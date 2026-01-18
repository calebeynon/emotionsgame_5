"""
Behavior classification: detect liars/suckers in public goods game.
Author: Claude Code | Date: 2026-01-17
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_data import load_experiment_data, Experiment

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
RAW_DIR = DATA_DIR / 'raw'
PROMISE_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'
OUTPUT_FILE = DATA_DIR / 'derived' / 'behavior_classifications.csv'

# THRESHOLDS
STRICT_THRESHOLD = 20
LENIENT_THRESHOLD = 5


# =====
# Main function
# =====
def main():
    """Main execution flow for behavior classification."""
    experiment = load_experiment()
    promise_df = load_promise_data()
    promise_lookup = build_promise_lookup(promise_df)

    records = build_all_records(experiment, promise_lookup)
    df = pd.DataFrame.from_records(records)

    save_results(df)
    print_summary(df)


# =====
# Data loading
# =====
def load_experiment() -> Experiment:
    """Load experiment data from raw session files."""
    return load_experiment_data(build_file_pairs(), name="Behavior Classification")


def build_file_pairs() -> list:
    """Build list of (data_csv, chat_csv, treatment) tuples from raw directory."""
    file_pairs = []
    for data_file in sorted(RAW_DIR.glob("*_data.csv")):
        treatment = extract_treatment(data_file.name)
        chat_file = data_file.with_name(data_file.name.replace("_data", "_chat"))
        chat_path = str(chat_file) if chat_file.exists() else None
        file_pairs.append((str(data_file), chat_path, treatment))
    return file_pairs


def extract_treatment(filename: str) -> int:
    """Extract treatment number from filename like '01_t1_data.csv'."""
    if '_t1_' in filename:
        return 1
    return 2 if '_t2_' in filename else 0


def load_promise_data() -> pd.DataFrame:
    """Load promise classifications CSV."""
    if not PROMISE_FILE.exists():
        print(f"Warning: Promise file not found at {PROMISE_FILE}")
        return pd.DataFrame()
    return pd.read_csv(PROMISE_FILE)


# =====
# Promise lookup
# =====
def build_promise_lookup(df: pd.DataFrame) -> dict:
    """Build lookup dict: (session, segment, round, label) -> made_promise bool."""
    lookup = {}
    for _, row in df.iterrows():
        key = (row['session_code'], row['segment'], row['round'], row['label'])
        lookup[key] = row['promise_count'] > 0
    return lookup


def player_made_promise(lookup: dict, session: str, segment: str, rnd: int, label: str) -> bool:
    """Check if player made a promise in given round."""
    return lookup.get((session, segment, rnd, label), False)


# =====
# Record building
# =====
def build_all_records(experiment: Experiment, promise_lookup: dict) -> list:
    """Build classification records for all player-rounds."""
    records = []
    for session_code, session in experiment.sessions.items():
        for segment_name, segment in session.segments.items():
            if not segment_name.startswith('supergame'):
                continue
            records.extend(build_segment_records(
                session_code, session.treatment, segment_name, segment, promise_lookup
            ))
    return records


def build_segment_records(session_code, treatment, segment_name, segment, promise_lookup) -> list:
    """Build records for all player-rounds in a segment."""
    records = []
    for round_num in sorted(segment.rounds.keys()):
        round_obj = segment.get_round(round_num)
        for group_id, group in round_obj.groups.items():
            for label, player in group.players.items():
                record = build_player_record(
                    session_code, treatment, segment_name, round_num,
                    group_id, label, player, segment, promise_lookup
                )
                records.append(record)
    return records


def build_player_record(session_code, treatment, segment_name, round_num,
                        group_id, label, player, segment, promise_lookup) -> dict:
    """Build single player-round record with liar/sucker flags."""
    base_record = _build_base_record(
        session_code, treatment, segment_name, round_num, group_id, label, player
    )
    flag_record = _build_flag_record(
        session_code, segment_name, round_num, label, segment, promise_lookup
    )
    return {**base_record, **flag_record}


def _build_base_record(session_code, treatment, segment_name, round_num,
                       group_id, label, player) -> dict:
    """Build base player-round record without behavioral flags."""
    return {
        'session_code': session_code, 'treatment': treatment,
        'segment': segment_name, 'round': round_num,
        'group': group_id, 'label': label,
        'participant_id': player.participant_id,
        'contribution': player.contribution, 'payoff': player.payoff,
    }


def _build_flag_record(session_code, segment_name, round_num, label,
                       segment, promise_lookup) -> dict:
    """Build behavioral flag record (made_promise, liar, sucker flags)."""
    made_promise = player_made_promise(promise_lookup, session_code, segment_name, round_num, label)
    liar_strict, liar_lenient = compute_liar_flags(
        session_code, segment_name, round_num, label, segment, promise_lookup
    )
    sucker_strict, sucker_lenient = compute_sucker_flags(
        session_code, segment_name, round_num, label, segment, promise_lookup
    )
    return {
        'made_promise': made_promise,
        'is_liar_strict': liar_strict, 'is_liar_lenient': liar_lenient,
        'is_sucker_strict': sucker_strict, 'is_sucker_lenient': sucker_lenient,
    }


# =====
# Liar detection
# =====
def compute_liar_flags(session_code, segment_name, round_num, label, segment, promise_lookup):
    """Compute is_liar flags by checking prior rounds for broken promises."""
    if round_num == 1:
        return False, False

    is_liar_strict, is_liar_lenient = False, False
    for prior_round in range(1, round_num):
        strict, lenient = _check_liar_in_prior_round(
            session_code, segment_name, prior_round, label, segment, promise_lookup
        )
        is_liar_strict = is_liar_strict or strict
        is_liar_lenient = is_liar_lenient or lenient
        if is_liar_strict and is_liar_lenient:
            break
    return is_liar_strict, is_liar_lenient


def _check_liar_in_prior_round(session_code, segment_name, prior_round, label, segment, promise_lookup):
    """Check if player broke a promise in a specific prior round."""
    prior_round_obj = segment.get_round(prior_round)
    if not prior_round_obj:
        return False, False

    prior_player = prior_round_obj.get_player(label)
    if not prior_player:
        return False, False

    made_promise = player_made_promise(promise_lookup, session_code, segment_name, prior_round, label)
    contribution = prior_player.contribution or 0
    strict = made_promise and contribution < STRICT_THRESHOLD
    lenient = made_promise and contribution < LENIENT_THRESHOLD
    return strict, lenient


# =====
# Sucker detection
# =====
def compute_sucker_flags(session_code, segment_name, round_num, label, segment, promise_lookup):
    """Compute is_sucker flags by checking if player was suckered in prior rounds."""
    if round_num == 1:
        return False, False

    is_sucker_strict, is_sucker_lenient = False, False
    for prior_round in range(1, round_num):
        strict, lenient = check_sucker_in_round(
            session_code, segment_name, prior_round, label, segment, promise_lookup
        )
        is_sucker_strict = is_sucker_strict or strict
        is_sucker_lenient = is_sucker_lenient or lenient
        if is_sucker_strict and is_sucker_lenient:
            break
    return is_sucker_strict, is_sucker_lenient


def check_sucker_in_round(session_code, segment_name, round_num, label, segment, promise_lookup):
    """Check if player was suckered in a specific round."""
    round_obj = segment.get_round(round_num)
    if not round_obj:
        return False, False

    player = round_obj.get_player(label)
    if not player or player.contribution != 25:
        return False, False

    group = round_obj.get_group(player.group_id)
    if not group:
        return False, False

    return check_group_for_broken_promises(
        session_code, segment_name, round_num, label, group, promise_lookup
    )


def check_group_for_broken_promises(session_code, segment_name, round_num, label, group, promise_lookup):
    """Check if any group member broke a promise in this round."""
    is_sucker_strict = False
    is_sucker_lenient = False

    for member_label, member in group.players.items():
        if member_label == label:
            continue

        made_promise = player_made_promise(promise_lookup, session_code, segment_name, round_num, member_label)
        contribution = member.contribution or 0

        if made_promise and contribution < STRICT_THRESHOLD:
            is_sucker_strict = True
        if made_promise and contribution < LENIENT_THRESHOLD:
            is_sucker_lenient = True

    return is_sucker_strict, is_sucker_lenient


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save results DataFrame to CSV."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Total records: {len(df)}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 50)
    print("BEHAVIOR CLASSIFICATION SUMMARY")
    print("=" * 50)
    _print_details(df)
    print("=" * 50)


def _print_details(df: pd.DataFrame):
    """Print promise and flag summary details."""
    count, total, pct = df['made_promise'].sum(), len(df), df['made_promise'].mean() * 100
    print(f"\nPromise makers: {count} / {total} ({pct:.1f}%)")
    for flag, thresh in [('is_liar_strict', STRICT_THRESHOLD), ('is_liar_lenient', LENIENT_THRESHOLD),
                         ('is_sucker_strict', STRICT_THRESHOLD), ('is_sucker_lenient', LENIENT_THRESHOLD)]:
        series = df[flag]
        print(f"\n{flag}: {series.sum()} ({series.mean()*100:.1f}%)")


# %%
if __name__ == "__main__":
    main()
