"""
Purpose: Generate chat summary statistics tables for the public goods experiment.
         Outputs volume, length, participation, word frequency, and orphan chat
         tables by treatment and supergame.
Author: Caleb Eynon
Date: 2026-03-02
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import (
    GROUPS_PER_SESSION,
    PLAYERS_PER_SESSION,
    SUPERGAME_ROUNDS,
    load_chat_raw,
    write_tex_table,
)
_STOPWORDS = set(stopwords.words('english'))
_TOP_N_WORDS = 20


# =====
# Main function
# =====

def main():
    """Generate all chat summary statistics tables."""
    chat = _prepare_chat(load_chat_raw())
    _write_all_tables(chat)


def _write_all_tables(chat):
    """Write all chat statistics .tex tables to the output directory."""
    write_tex_table(compute_volume_stats(chat), 'chat_volume.tex', 'clrr')
    write_tex_table(compute_length_stats(chat), 'chat_length.tex', 'clrr')
    write_tex_table(compute_participation(chat), 'chat_participation.tex', 'clr')
    write_tex_table(compute_word_frequency(chat), 'chat_word_frequency.tex', 'lrlrlr')
    write_tex_table(compute_orphan_volume(chat), 'chat_orphan_volume.tex', 'clrr')


# =====
# Data preparation
# =====

def _prepare_chat(chat):
    """Parse channel column to extract supergame, page index, and text stats."""
    parts = chat['channel'].str.split('-', expand=True)
    chat['supergame'] = parts[1]
    chat['ch_page'] = parts[2].astype(int)
    chat['sg_num'] = chat['supergame'].str.extract(r'(\d+)$')[0].astype(int)
    chat['word_count'] = chat['body'].astype(str).str.split().str.len()
    chat['char_count'] = chat['body'].astype(str).str.len()
    return chat


# =====
# Volume statistics
# =====

def compute_volume_stats(chat):
    """Total messages and mean per player-round by treatment x supergame."""
    rows = []
    for (trt, sg), grp in chat.groupby(['treatment', 'supergame']):
        total = len(grp)
        n_sessions = grp['session_code'].nunique()
        n_rounds = SUPERGAME_ROUNDS[grp['sg_num'].iloc[0]]
        player_rounds = PLAYERS_PER_SESSION * n_rounds * n_sessions
        mean_per_pr = round(total / player_rounds, 2)
        rows.append([trt, sg, total, mean_per_pr])
    cols = ['Treatment', 'Supergame', 'Total Msgs', 'Mean/Player-Round']
    return pd.DataFrame(rows, columns=cols)


# =====
# Length statistics
# =====

def compute_length_stats(chat):
    """Mean message length in chars and words by treatment x supergame."""
    grouped = chat.groupby(['treatment', 'supergame'])
    stats = grouped.agg(
        mean_chars=('char_count', 'mean'),
        mean_words=('word_count', 'mean'),
    ).round(1).reset_index()
    stats.columns = ['Treatment', 'Supergame', 'Mean Chars', 'Mean Words']
    return stats


# =====
# Participation statistics
# =====

def compute_participation(chat):
    """Pct of player-rounds with at least one message by treatment x supergame."""
    rows = []
    for (trt, sg), grp in chat.groupby(['treatment', 'supergame']):
        n_sessions = grp['session_code'].nunique()
        n_rounds = SUPERGAME_ROUNDS[grp['sg_num'].iloc[0]]
        total_pr = PLAYERS_PER_SESSION * n_rounds * n_sessions
        active = grp.groupby(['session_code', 'nickname', 'ch_page']).ngroups
        pct = round(100 * active / total_pr, 1)
        rows.append([trt, sg, pct])
    return pd.DataFrame(rows, columns=['Treatment', 'Supergame', 'Pct Active'])


# =====
# Word frequency statistics
# =====

def compute_word_frequency(chat):
    """Top N words excluding stopwords, overall and by treatment."""
    overall = _top_words(chat['body'])
    t1 = _top_words(chat[chat['treatment'] == 1]['body'])
    t2 = _top_words(chat[chat['treatment'] == 2]['body'])
    rows = []
    for i in range(_TOP_N_WORDS):
        row = _word_row(overall, i) + _word_row(t1, i) + _word_row(t2, i)
        rows.append(row)
    cols = ['Overall', 'Count', 'T1 Word', 'T1 Count', 'T2 Word', 'T2 Count']
    return pd.DataFrame(rows, columns=cols)


def _word_row(word_list, index):
    """Extract (word, count) at index, or empty strings if out of range."""
    if index < len(word_list):
        return [word_list[index][0], word_list[index][1]]
    return ['', '']


def _top_words(body_series):
    """Count non-stopword tokens and return top N as (word, count) list."""
    counter = Counter()
    for text in body_series.dropna().astype(str):
        tokens = re.findall(r'[a-z]+', text.lower())
        counter.update(t for t in tokens if t not in _STOPWORDS and len(t) > 1)
    return counter.most_common(_TOP_N_WORDS)


# =====
# Orphan chat volume
# =====

def compute_orphan_volume(chat):
    """Count orphan (last round) chat messages by treatment x supergame."""
    tagged = _tag_orphan_messages(chat)
    rows = []
    for (trt, sg), grp in tagged.groupby(['treatment', 'supergame']):
        n_orphan = grp['is_orphan'].sum()
        total = len(grp)
        pct = round(100 * n_orphan / total, 1) if total else 0
        rows.append([trt, sg, n_orphan, pct])
    cols = ['Treatment', 'Supergame', 'Orphan Msgs', 'Pct of Total']
    return pd.DataFrame(rows, columns=cols)


def _tag_orphan_messages(chat):
    """Tag messages from the last round per session x supergame (vectorized)."""
    chat = chat.copy()
    bases = chat.groupby(['session_code', 'supergame'])['ch_page'].transform('min')
    n_rounds = chat['sg_num'].map(SUPERGAME_ROUNDS)
    orphan_start = bases + (n_rounds - 1) * GROUPS_PER_SESSION
    chat['is_orphan'] = chat['ch_page'] >= orphan_start
    return chat


# %%
if __name__ == "__main__":
    main()
