"""
Tests for round structure in experiment data.

Verifies that rounds per supergame are correct and sequential.
Expected rounds: supergame1=3, supergame2=4, supergame3=3, supergame4=7, supergame5=5 (22 total)

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
from experiment_data import Session

# EXPECTED ROUND COUNTS
EXPECTED_ROUNDS_PER_SUPERGAME = {
    1: 3,
    2: 4,
    3: 3,
    4: 7,
    5: 5,
}
EXPECTED_TOTAL_ROUNDS = 22


# =====
# Helper functions
# =====
def get_round_count_for_supergame(session: Session, supergame_num: int) -> int:
    """Get the number of rounds in a supergame."""
    supergame = session.get_supergame(supergame_num)
    if supergame is None:
        return 0
    return len(supergame.rounds)


def get_round_numbers_for_supergame(session: Session, supergame_num: int) -> list:
    """Get sorted list of round numbers in a supergame."""
    supergame = session.get_supergame(supergame_num)
    if supergame is None:
        return []
    return sorted(supergame.rounds.keys())


def verify_round_structure(session: Session) -> None:
    """Verify round structure for a session - used by both t1 and t2 tests."""
    # Check each supergame has expected round count
    for sg_num, expected_count in EXPECTED_ROUNDS_PER_SUPERGAME.items():
        actual_count = get_round_count_for_supergame(session, sg_num)
        assert actual_count == expected_count, (
            f"Supergame {sg_num}: expected {expected_count} rounds, got {actual_count}"
        )

    # Check round numbers are sequential (1, 2, 3, ...)
    for sg_num in range(1, 6):
        round_numbers = get_round_numbers_for_supergame(session, sg_num)
        expected_numbers = list(range(1, len(round_numbers) + 1))
        assert round_numbers == expected_numbers, (
            f"Supergame {sg_num}: expected sequential rounds {expected_numbers}, "
            f"got {round_numbers}"
        )


# =====
# T1 Session Tests
# =====
def test_supergame1_has_3_rounds(loaded_t1_session: Session):
    """Verify supergame1 has exactly 3 rounds."""
    actual = get_round_count_for_supergame(loaded_t1_session, 1)
    assert actual == 3, f"Expected 3 rounds, got {actual}"


def test_supergame2_has_4_rounds(loaded_t1_session: Session):
    """Verify supergame2 has exactly 4 rounds."""
    actual = get_round_count_for_supergame(loaded_t1_session, 2)
    assert actual == 4, f"Expected 4 rounds, got {actual}"


def test_supergame3_has_3_rounds(loaded_t1_session: Session):
    """Verify supergame3 has exactly 3 rounds."""
    actual = get_round_count_for_supergame(loaded_t1_session, 3)
    assert actual == 3, f"Expected 3 rounds, got {actual}"


def test_supergame4_has_7_rounds(loaded_t1_session: Session):
    """Verify supergame4 has exactly 7 rounds."""
    actual = get_round_count_for_supergame(loaded_t1_session, 4)
    assert actual == 7, f"Expected 7 rounds, got {actual}"


def test_supergame5_has_5_rounds(loaded_t1_session: Session):
    """Verify supergame5 has exactly 5 rounds."""
    actual = get_round_count_for_supergame(loaded_t1_session, 5)
    assert actual == 5, f"Expected 5 rounds, got {actual}"


def test_round_numbers_sequential(loaded_t1_session: Session):
    """Verify round numbers are 1 through N with no gaps for each supergame."""
    for sg_num in range(1, 6):
        round_numbers = get_round_numbers_for_supergame(loaded_t1_session, sg_num)
        expected_count = EXPECTED_ROUNDS_PER_SUPERGAME[sg_num]
        expected_numbers = list(range(1, expected_count + 1))
        assert round_numbers == expected_numbers, (
            f"Supergame {sg_num}: expected {expected_numbers}, got {round_numbers}"
        )


def test_total_supergame_rounds(loaded_t1_session: Session):
    """Verify total of 22 rounds across all supergames."""
    total = sum(
        get_round_count_for_supergame(loaded_t1_session, sg_num)
        for sg_num in range(1, 6)
    )
    assert total == EXPECTED_TOTAL_ROUNDS, (
        f"Expected {EXPECTED_TOTAL_ROUNDS} total rounds, got {total}"
    )


# =====
# T2 Session Tests
# =====
def test_round_structure_t2(loaded_t2_session: Session):
    """Verify same round structure for t2 session."""
    verify_round_structure(loaded_t2_session)
