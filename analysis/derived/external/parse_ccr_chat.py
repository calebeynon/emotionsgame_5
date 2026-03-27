"""
Low-level parsers for CCR raw chat files.

Handles two formats:
  - .txt files (Original + Science sites): "nickname: message" per line
  - z-Tree .xls files (NUS + NTU sites): tab-separated contracts tables

Author: Claude Code
Date: 2026-03-26
"""

import re
from pathlib import Path

import pandas as pd

# System messages to remove from .txt files
SYSTEM_PATTERNS = [
    "has just joined the discussion",
    "has just left the discussion",
]

# Verified session mapping for Original site (word counts match chat_lines.dta)
ORIGINAL_SESSION_MAP = {
    "Ingroup_2008-02-08_1": 1,
    "Ingroup_2008-02-22_1": 2,
    "Ingroup_2008-03-11_1": 3,
    "Outgroup_2008-02-14_1": 4,
    "Outgroup_2008-03-07_1": 5,
    "Outgroup_2008-03-07_2": 6,
}


# =====
# Public API
# =====
def parse_txt_sites(original_dir, science_dir):
    """Parse all .txt chat files from Original and Science sites."""
    original = _parse_original_site(original_dir)
    science = _parse_science_site(science_dir)
    return pd.concat([original, science], ignore_index=True)


def parse_ztree_sites(nus_dir, ntu_dir):
    """Parse z-Tree .xls files from NUS (sessions 21-30) and NTU (31-58)."""
    nus = _parse_ztree_site(nus_dir, start_session=21)
    ntu = _parse_ztree_site(ntu_dir, start_session=31)
    return pd.concat([nus, ntu], ignore_index=True)


# =====
# Original site (.txt)
# =====
def _parse_original_site(chats_dir):
    """Parse Original site .txt files, map to sessions 1-6."""
    all_rows = []
    for filepath in sorted(chats_dir.glob("*.txt")):
        color, date_key = _parse_original_filename(filepath.name)
        session = ORIGINAL_SESSION_MAP[date_key]
        red = 1 if color == "Red" else 0
        for msg in _parse_txt_file(filepath):
            all_rows.append(_msg_row(session, red, msg))
    return pd.DataFrame(all_rows)


def _parse_original_filename(filename):
    """Extract color and date key from Original site filename."""
    pattern = (
        r"(Green|Red) \((Ingroup|Outgroup) (\d{4}-\d{2}-\d{2})"
        r"(?:\s+Session\s+(\d+))?\)"
    )
    m = re.match(pattern, filename)
    if not m:
        raise ValueError(f"Cannot parse Original filename: {filename}")
    color, treatment, date, session_within = m.groups()
    date_key = f"{treatment}_{date}_{session_within or '1'}"
    return color, date_key


# =====
# Science site (.txt)
# =====
def _parse_science_site(chats_dir):
    """Parse Science site .txt files, map to sessions 7-20."""
    all_rows = []
    for filepath in sorted(chats_dir.glob("*.txt")):
        session, red = _parse_science_filename(filepath.name)
        for msg in _parse_txt_file(filepath):
            all_rows.append(_msg_row(session, red, msg))
    return pd.DataFrame(all_rows)


def _parse_science_filename(filename):
    """Extract session number and color from Science site filename.

    Ingroup Session N -> session 6+N; Outgroup Session N -> session 13+N.
    """
    m = re.match(
        r"(Ingroup|Outgroup)\s+Session\s*(\d+)-(Green|Red)\.txt",
        filename,
    )
    if not m:
        raise ValueError(f"Cannot parse Science filename: {filename}")
    treatment, session_num, color = m.groups()
    offset = 6 if treatment == "Ingroup" else 13
    return offset + int(session_num), 1 if color == "Red" else 0


# =====
# Shared .txt parsing
# =====
def _parse_txt_file(filepath):
    """Parse a .txt chat file, returning list of {nickname, message} dicts."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    messages = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if _is_header_line(line) or not line or ":" not in line:
            continue
        nickname, message = line.split(":", 1)
        message = message.strip()
        if _is_system_message(message) or not message:
            continue
        messages.append({"nickname": nickname.strip(), "message": message})
    return messages


def _is_header_line(line):
    """Check if line is a word/line count header."""
    lower = line.lower()
    return lower.startswith("word count:") or lower.startswith("lines:")


def _is_system_message(message):
    """Check if message is a system join/leave notification."""
    return any(pat in message for pat in SYSTEM_PATTERNS)


# =====
# z-Tree parsing (NUS + NTU)
# =====
def _parse_ztree_site(directory, start_session):
    """Parse z-Tree .xls files in sorted order, mapping to session numbers.

    senderColor 1 = Green (red=0), senderColor 2 = Red (red=1).
    """
    files = sorted(directory.glob("*.xls"))
    all_rows = []
    for i, filepath in enumerate(files):
        session = start_session + i
        for msg in _parse_ztree_file(filepath):
            all_rows.append({
                "session": session,
                "red": msg["red"],
                "nickname": f"subject_{msg['subject']}",
                "message": msg["message"],
            })
    return pd.DataFrame(all_rows)


def _parse_ztree_file(filepath):
    """Parse a z-Tree .xls file. Takes last contracts section (skips practice)."""
    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()
    sections = _extract_contracts_sections(lines)
    if not sections:
        raise ValueError(f"No contracts sections found in {filepath}")
    return [m for row in sections[-1] if (m := _parse_ztree_row(row))]


def _extract_contracts_sections(lines):
    """Split z-Tree lines into contracts data sections."""
    sections = []
    current = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 5 or parts[2] != "contracts":
            continue
        if parts[3] == "Period":
            if current:
                sections.append(current)
            current = []
        else:
            current.append(parts)
    if current:
        sections.append(current)
    return sections


def _parse_ztree_row(parts):
    """Parse a single z-Tree contracts row into a message dict."""
    if len(parts) < 7:
        return None
    try:
        sender_color = int(parts[4])
        subject = int(parts[5])
    except ValueError:
        return None
    chat_text = parts[6].strip('"')
    if not chat_text:
        return None
    return {"red": 1 if sender_color == 2 else 0, "subject": subject, "message": chat_text}


# =====
# Helpers
# =====
def _msg_row(session, red, msg):
    """Build a message row dict from session/red and a parsed message."""
    return {
        "session": session,
        "red": red,
        "nickname": msg["nickname"],
        "message": msg["message"],
    }
