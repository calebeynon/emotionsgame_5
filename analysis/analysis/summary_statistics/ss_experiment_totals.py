"""
Purpose: Generate experiment-level summary tables: overall totals (participants,
         sessions, groups, messages, rounds) and mean page timing by treatment.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
from pathlib import Path

import pandas as pd

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    PLAYERS_PER_SESSION,
    SESSIONS_DIR,
    SESSION_CODE_TO_TREATMENT,
    SUPERGAME_ROUNDS,
    load_chat_raw,
    load_contributions,
    write_tex_table,
)

# PAGE TIMES MAPPING: contributions_session_code -> (relative path, filter_code)
_PAGETIMES_MAP = {
    # Leading space in path is intentional — matches the actual directory name on disk
    'sa7mprty': (' pilot session 1/session data/PageTimes-2025-09-11.csv', 'sa7mprty'),
    'irrzlgk2': ('session 3 oct 2/s3 data/PageTimes-2025-10-02.csv', 'z8dowljr'),
    '6uv359rf': ('session 4 oct 3/PageTimes-2025-10-03.csv', '6uv359rf'),
    'umbzdj98': ('session 5 oct 8/PageTimes-2025-10-08.csv', 'umbzdj98'),
    'j3ki5tli': ('session 6 oct 9/PageTimes-2025-10-09.csv', 'j3ki5tli'),
    'r5dj4yfl': ('session 7 oct 13/PageTimes-2025-10-13.csv', 'r5dj4yfl'),
    'sylq2syi': ('session 8 oct 14/PageTimes-2025-10-14.csv', 'sylq2syi'),
    'iiu3xixz': ('session 9 oct 21/PageTimes-2025-10-21.csv', 'iiu3xixz'),
    '6ucza025': ('session 10 oct 22/PageTimes-2025-10-22.csv', '6ucza025'),
    '6sdkxl2q': ('session 11 nov 17/PageTimes-2025-11-17.csv', '6sdkxl2q'),
}

# Pages to report timing for (excluding wait pages and init)
_TIMED_PAGES = ['Contribute', 'Results', 'StartPage', 'RegroupingMessage']

# =====
# Main function
# =====

def main():
    """Generate experiment totals and timing tables."""

    contribs = load_contributions()
    chat = load_chat_raw()

    totals = compute_experiment_totals(contribs, chat)
    write_tex_table(totals, 'experiment_totals.tex', 'lr')

    pagetimes = load_all_pagetimes()
    durations = compute_page_durations(pagetimes)
    timing = compute_timing_stats(durations)
    write_tex_table(timing, 'experiment_timing.tex', 'l' + 'r' * (len(timing.columns) - 1))


# =====
# Experiment totals
# =====

def compute_experiment_totals(contribs, chat):
    """Build a summary table of high-level experiment counts."""
    n_sessions = contribs['session_code'].nunique()
    n_per_treatment = _count_per_treatment(contribs)
    n_groups = contribs.groupby(
        ['session_code', 'segment', 'round', 'group'],
    ).ngroups
    rows = [
        ['Sessions', n_sessions],
        ['Participants', n_sessions * PLAYERS_PER_SESSION],
        ['Participants (T1)', n_per_treatment[1]],
        ['Participants (T2)', n_per_treatment[2]],
        ['Player-rounds', len(contribs)],
        ['Groups (total)', n_groups],
        ['Chat messages', len(chat)],
        ['Supergame rounds', sum(SUPERGAME_ROUNDS.values())],
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Value'])


def _count_per_treatment(contribs):
    """Count unique participants per treatment."""
    result = {}
    for treatment in [1, 2]:
        t_sessions = contribs[contribs['treatment'] == treatment]['session_code'].nunique()
        result[treatment] = t_sessions * PLAYERS_PER_SESSION
    return result


# =====
# PageTimes loading
# =====

def load_all_pagetimes():
    """Load and concatenate PageTimes CSVs for all sessions."""
    frames = []
    for session_code, (rel_path, filter_code) in _PAGETIMES_MAP.items():
        df = _load_single_pagetimes(rel_path, filter_code)
        treatment = SESSION_CODE_TO_TREATMENT[session_code]
        df['treatment'] = treatment
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_single_pagetimes(rel_path, filter_code):
    """Load one PageTimes CSV, filtering to the correct session_code."""
    path = SESSIONS_DIR / rel_path
    df = pd.read_csv(path)
    return df[df['session_code'] == filter_code].copy()


# =====
# Duration computation
# =====

def compute_page_durations(pagetimes):
    """Compute time-on-page by differencing consecutive epoch times per participant."""
    pagetimes = pagetimes.sort_values(
        ['session_code', 'participant_code', 'page_index'],
    )
    pagetimes['prev_epoch'] = pagetimes.groupby(
        ['session_code', 'participant_code'],
    )['epoch_time_completed'].shift(1)
    pagetimes['duration'] = pagetimes['epoch_time_completed'] - pagetimes['prev_epoch']

    # Drop first page per participant (no prior timestamp), wait pages, and init
    mask = (
        pagetimes['prev_epoch'].notna()
        & (pagetimes['is_wait_page'] == 0)
        & (pagetimes['page_name'] != 'InitializeParticipant')
    )
    return pagetimes[mask].copy()


def compute_timing_stats(durations):
    """Mean time per page type by treatment, restricted to timed pages."""
    df = durations[durations['page_name'].isin(_TIMED_PAGES)]
    grouped = df.groupby(['treatment', 'page_name'])['duration'].mean()
    table = grouped.unstack(fill_value=0).round(1)
    table.index = [f'Treatment {t}' for t in table.index]
    table.index.name = 'Treatment'
    # Reorder columns to match _TIMED_PAGES order
    cols = [c for c in _TIMED_PAGES if c in table.columns]
    return table[cols].reset_index()


# %%
if __name__ == "__main__":
    main()
