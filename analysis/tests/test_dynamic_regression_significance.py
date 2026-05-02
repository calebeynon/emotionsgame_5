"""
Structural + pinning tests for dynamic_regression_extended.tex.

Verifies the extended table exposes 6 data columns and that the Chat/Facial
coefficients added by +Chat and +Chat+Facial specs match their pinned values.
Coefficient-parity against Stata Table DP1 lives in test_dynamic_regression_parity.py.

Issue #74 dropped the min/med/max columns; the extended table now has 6 cols
(2 baselines × 3 specs) instead of the prior 12.

Layout (0-indexed):
    0: IF Base       3: AF Base
    1: IF +Chat      4: AF +Chat
    2: IF +Chat+Fac  5: AF +Chat+Fac

Author: Claude Code
Date: 2026-05-01
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _dynreg_tex import check_coef  # noqa: E402

# Chat/Facial coefficient pins for the +Chat and +Chat+Facial specs of each
# baseline. Frozen from current R output so a regression changes these visibly.
EXTENDED_CHAT_FACIAL_PINS = {
    # IF +Chat (col 1), +Chat+Facial (col 2)
    ("Word Count", 1):             0.079,
    ("Word Count", 2):             0.155,
    ("Sentiment (compound)", 1):   1.244,
    ("Sentiment (compound)", 2):   0.881,
    ("Emotion Valence", 2):       -0.043,
    # AF +Chat (col 4), +Chat+Facial (col 5)
    ("Word Count", 4):             0.014,
    ("Word Count", 5):             0.031,
    ("Sentiment (compound)", 4):   1.662,
    ("Sentiment (compound)", 5):   0.009,
    ("Emotion Valence", 5):        0.004,
}


def test_extended_has_six_data_columns(extended_coefs):
    """Extended .tex must expose 6 model columns (2 baselines x 3 specs)."""
    for label, row in extended_coefs.items():
        assert len(row) == 6, (
            f"Row '{label}' in extended has {len(row)} columns, expected 6"
        )


def test_extended_chat_facial_coefficients_pinned(extended_coefs):
    """Chat/Facial coefficients across extended cols 1,2,4,5 match pins."""
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
