"""
Structural + pinning tests for dynamic_regression_extended.tex.

Verifies the extended table exposes 12 data columns and that the Chat/Facial
coefficients added by +Chat and +Chat+Facial specs match their pinned values.
Coefficient-parity against Stata Table DP1 lives in test_dynamic_regression_parity.py.

Author: Claude Code
Date: 2026-04-20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dynreg_tex import check_coef  # noqa: E402

# Chat/Facial coefficient pins for extended cols 1,2,4,5,7,8,10,11 (i.e. +Chat
# and +Chat+Facial specs for each baseline). Values frozen from current R
# output so a regression changes these visibly.
EXTENDED_CHAT_FACIAL_PINS = {
    # T1 mean +Chat (col 1), +Chat+Facial (col 2)
    ("Word Count", 1):             0.079,
    ("Word Count", 2):             0.155,
    ("Sentiment (compound)", 1):   1.244,
    ("Sentiment (compound)", 2):   0.881,
    ("Emotion Valence", 2):       -0.043,
    # T2 mean +Chat (col 4), +Chat+Facial (col 5)
    ("Word Count", 4):             0.014,
    ("Word Count", 5):             0.031,
    ("Sentiment (compound)", 4):   1.662,
    ("Sentiment (compound)", 5):   0.009,
    ("Emotion Valence", 5):        0.004,
    # T1 min/med/max +Chat (col 7), +Chat+Facial (col 8)
    ("Word Count", 7):             0.090,
    ("Word Count", 8):             0.151,
    ("Sentiment (compound)", 7):   1.090,
    ("Sentiment (compound)", 8):   0.401,
    ("Emotion Valence", 8):       -0.044,
    # T2 min/med/max +Chat (col 10), +Chat+Facial (col 11)
    ("Word Count", 10):            0.010,
    ("Word Count", 11):            0.022,
    ("Sentiment (compound)", 10):  1.111,
    ("Sentiment (compound)", 11): -0.686,
    ("Emotion Valence", 11):       0.005,
}


def test_extended_has_twelve_data_columns(extended_coefs):
    """Extended .tex must expose 12 model columns (4 baselines x 3 specs)."""
    for label, row in extended_coefs.items():
        assert len(row) == 12, (
            f"Row '{label}' in extended has {len(row)} columns, expected 12"
        )


def test_extended_chat_facial_coefficients_pinned(extended_coefs):
    """Chat/Facial coefficients across extended cols 1,2,4,5,7,8,10,11 match pins."""
    failures = []
    for (label, col_idx), expected in EXTENDED_CHAT_FACIAL_PINS.items():
        if label not in extended_coefs:
            failures.append(f"{label}: missing from extended .tex")
            continue
        actual = extended_coefs[label][col_idx]
        msg = check_coef(label, expected, actual, col_idx, f"extended[{col_idx}]")
        if msg:
            failures.append(msg)
    assert not failures, (
        "Extended chat/facial pin mismatches:\n  " + "\n  ".join(failures)
    )
