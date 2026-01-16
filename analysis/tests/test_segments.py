"""
Tests for segment structure in Session objects.

Verifies that sessions contain the expected segments:
- introduction
- supergame1 through supergame5
- finalresults

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
from experiment_data import Session


# =====
# Expected segment names
# =====
EXPECTED_SEGMENTS = [
    'introduction',
    'supergame1',
    'supergame2',
    'supergame3',
    'supergame4',
    'supergame5',
    'finalresults',
]

SUPERGAME_NAMES = [
    'supergame1',
    'supergame2',
    'supergame3',
    'supergame4',
    'supergame5',
]


# =====
# T1 Session tests
# =====
def test_all_segments_present(loaded_t1_session: Session):
    """Session has introduction, supergame1-5, and finalresults."""
    segment_names = set(loaded_t1_session.segments.keys())

    for expected in EXPECTED_SEGMENTS:
        assert expected in segment_names, f"Missing segment: {expected}"


def test_segment_count(loaded_t1_session: Session):
    """Session has exactly 7 segments total."""
    assert len(loaded_t1_session.segments) == 7


def test_supergame_segment_names(loaded_t1_session: Session):
    """Supergame names are exactly 'supergame1' through 'supergame5'."""
    segment_names = loaded_t1_session.segments.keys()
    supergames = [name for name in segment_names if name.startswith('supergame')]

    assert sorted(supergames) == SUPERGAME_NAMES


def test_get_supergame_returns_correct_segment(loaded_t1_session: Session):
    """session.get_supergame(N) returns segment named 'supergameN'."""
    for n in range(1, 6):
        segment = loaded_t1_session.get_supergame(n)
        assert segment is not None, f"get_supergame({n}) returned None"
        assert segment.name == f'supergame{n}'


def test_introduction_segment_exists(loaded_t1_session: Session):
    """Session has 'introduction' segment accessible via get_segment."""
    segment = loaded_t1_session.get_segment('introduction')
    assert segment is not None
    assert segment.name == 'introduction'


def test_finalresults_segment_exists(loaded_t1_session: Session):
    """Session has 'finalresults' segment accessible via get_segment."""
    segment = loaded_t1_session.get_segment('finalresults')
    assert segment is not None
    assert segment.name == 'finalresults'


# =====
# T2 Session tests
# =====
def test_segments_t2(loaded_t2_session: Session):
    """T2 session has same segment structure as T1."""
    # Verify segment count
    assert len(loaded_t2_session.segments) == 7

    # Verify all expected segments present
    segment_names = set(loaded_t2_session.segments.keys())
    for expected in EXPECTED_SEGMENTS:
        assert expected in segment_names, f"T2 missing segment: {expected}"

    # Verify supergame names
    supergames = [name for name in segment_names if name.startswith('supergame')]
    assert sorted(supergames) == SUPERGAME_NAMES

    # Verify get_supergame works correctly
    for n in range(1, 6):
        segment = loaded_t2_session.get_supergame(n)
        assert segment is not None, f"T2 get_supergame({n}) returned None"
        assert segment.name == f'supergame{n}'

    # Verify introduction and finalresults
    intro = loaded_t2_session.get_segment('introduction')
    assert intro is not None and intro.name == 'introduction'

    final = loaded_t2_session.get_segment('finalresults')
    assert final is not None and final.name == 'finalresults'
