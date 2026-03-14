"""
Session mapping utilities for linking iMotions session numbers to oTree session codes.

Provides dictionaries mapping session numbers to codes and treatments, plus parsers
for participant IDs and annotation strings from iMotions export data.

Author: Claude Code
Date: 2026-03-11
"""

import logging
import re

logger = logging.getLogger(__name__)

# SESSION MAPPINGS
SESSION_NUM_TO_CODE = {
    1: "sa7mprty",
    3: "irrzlgk2",
    4: "6uv359rf",
    5: "umbzdj98",
    6: "j3ki5tli",
    7: "r5dj4yfl",
    8: "sylq2syi",
    9: "iiu3xixz",
    10: "6ucza025",
    11: "6sdkxl2q",
}

SESSION_NUM_TO_TREATMENT = {
    1: 1,
    3: 2,
    4: 2,
    5: 1,
    6: 2,
    7: 1,
    8: 2,
    9: 1,
    10: 2,
    11: 1,
}

# Regex for standard annotations like s1r1Contribute, s3r2Results, s5r4ResultsOnly
_STANDARD_PATTERN = re.compile(r'^s(\d+)r(\d+)(Contribute|Results|ResultsOnly)$')

# Regex for irregular annotations like S4result1, S4Result2, S4Results4
_IRREGULAR_PATTERN = re.compile(r'^S(\d+)[Rr]esults?(\d+)$')


# =====
# Main function (demo / validation)
# =====
def main():
    """Print session mappings and run parsing examples for validation."""
    _print_session_mappings()
    _print_id_examples()
    _print_annotation_examples()


def _print_session_mappings():
    """Print all session number to code/treatment mappings."""
    print("Session mappings:")
    for num, code in sorted(SESSION_NUM_TO_CODE.items()):
        print(f"  Session {num}: {code} (treatment {SESSION_NUM_TO_TREATMENT[num]})")


def _print_id_examples():
    """Print participant ID parsing examples."""
    print("\nParsing examples:")
    for id_str, session in [("A3", 3), ("F3", 1), ("B10", 10), ("Q1", 1)]:
        label = parse_participant_id(id_str, session)
        print(f"  {id_str} (session {session}) -> {label}")


def _print_annotation_examples():
    """Print annotation parsing examples."""
    print("\nAnnotation examples:")
    for ann in ["s1r1Contribute", "s4r3Results", "S4result1",
                "S4Result2", "S4Results4", "all_instructions"]:
        print(f"  {ann} -> {parse_annotation(ann)}")


# =====
# Participant ID parsing
# =====
def parse_participant_id(id_str: str, session_num: int) -> str:
    """Extract player label from iMotions participant ID.

    IDs are <letter><session_number> (e.g., A3, B10).
    Special case: F3 in session 1 is a known misentry.
    """
    label = id_str[0]
    if session_num == 1 and id_str == "F3":
        logger.warning("Known misentry: 'F3' in session 1 corrected to 'F'")
        return label
    if id_str[1:] != str(session_num):
        logger.warning("Unexpected ID '%s' in session %d", id_str, session_num)
    return label


# =====
# Annotation parsing
# =====
def parse_annotation(annotation_str: str) -> tuple:
    """Parse iMotions annotation string into (segment, round, page_type).

    For 'all_instructions', segment and round are None.
    """
    if annotation_str == "all_instructions":
        return (None, None, "all_instructions")
    standard = _STANDARD_PATTERN.match(annotation_str)
    if standard:
        seg_num, round_num, page_type = standard.groups()
        return (f"supergame{seg_num}", int(round_num), page_type)
    irregular = _IRREGULAR_PATTERN.match(annotation_str)
    if irregular:
        seg_num, round_num = irregular.groups()
        return (f"supergame{seg_num}", int(round_num), "Results")
    logger.warning("Unrecognized annotation format: '%s'", annotation_str)
    return (None, None, annotation_str)


# %%
if __name__ == "__main__":
    main()
