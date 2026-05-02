"""
Pytest fixtures for experiment_data module tests.

Author: Test Infrastructure
Date: 2026-01-15
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Tests directory for importing sibling helper modules (_dynreg_tex, _helpers).
sys.path.insert(0, str(Path(__file__).parent))

from experiment_data import (
    Experiment, Group, Player, Round, Segment, Session, load_experiment_data,
)
from _dynreg_tex import (
    BASELINE_TEX, EXTENDED_TEX,
    ensure_tex_outputs_current,
    parse_tex_coefficients, parse_tex_with_details, parse_tex_gof_rows,
)

# FILE PATHS
RAW_DATA_DIR = Path(__file__).parent.parent / "datastore" / "raw"


# =====
# Shared synthetic-data builders (used across test modules)
# =====
def make_player(label, contribution, pid=1):
    """Create a Player with given label and contribution."""
    p = Player(participant_id=pid, label=label, id_in_group=1)
    p.contribution = contribution
    return p


def make_group(group_id, players_data):
    """Create a Group from list of (label, contribution, pid) tuples."""
    g = Group(group_id)
    for label, contribution, pid in players_data:
        g.add_player(make_player(label, contribution, pid))
    return g


def _make_round(num, groups):
    """Create a Round from a list of groups."""
    r = Round(num)
    for g in groups:
        r.add_group(g)
    return r


def make_experiment_1sg(rounds_data):
    """Build single-session, single-supergame experiment from rounds data.

    Args:
        rounds_data: list of lists of (label, contribution, pid) per round.
    """
    seg = Segment("supergame1")
    for i, players in enumerate(rounds_data, 1):
        seg.add_round(_make_round(i, [make_group(1, players)]))
    sess = Session("s1", 1)
    sess.add_segment(seg)
    for rnd in seg.rounds.values():
        for label, player in rnd.players.items():
            sess.participant_labels[player.participant_id] = label
    exp = Experiment(name="Test")
    exp.add_session(sess)
    return exp


# =====
# Path fixtures
# =====
@pytest.fixture
def raw_data_dir() -> Path:
    """Return Path to raw data directory."""
    return RAW_DATA_DIR


@pytest.fixture
def t1_session_paths(raw_data_dir: Path) -> tuple:
    """Return tuple (data_csv_path, chat_csv_path) for session 01_t1."""
    data_path = raw_data_dir / "01_t1_data.csv"
    chat_path = raw_data_dir / "01_t1_chat.csv"

    if not data_path.exists():
        pytest.skip(f"T1 data file not found: {data_path}")

    return (data_path, chat_path)


@pytest.fixture
def t2_session_paths(raw_data_dir: Path) -> tuple:
    """Return tuple (data_csv_path, chat_csv_path) for session 03_t2."""
    data_path = raw_data_dir / "03_t2_data.csv"
    chat_path = raw_data_dir / "03_t2_chat.csv"

    if not data_path.exists():
        pytest.skip(f"T2 data file not found: {data_path}")

    return (data_path, chat_path)


# =====
# Raw DataFrame fixtures
# =====
@pytest.fixture
def t1_raw_df(t1_session_paths: tuple) -> pd.DataFrame:
    """Return pandas DataFrame of raw t1 game data."""
    data_path, _ = t1_session_paths
    return pd.read_csv(data_path)


@pytest.fixture
def t2_raw_df(t2_session_paths: tuple) -> pd.DataFrame:
    """Return pandas DataFrame of raw t2 game data."""
    data_path, _ = t2_session_paths
    return pd.read_csv(data_path)


@pytest.fixture
def t1_chat_df(t1_session_paths: tuple) -> pd.DataFrame:
    """Return pandas DataFrame of raw t1 chat data."""
    _, chat_path = t1_session_paths

    if not chat_path.exists():
        pytest.skip(f"T1 chat file not found: {chat_path}")

    return pd.read_csv(chat_path)


@pytest.fixture
def t2_chat_df(t2_session_paths: tuple) -> pd.DataFrame:
    """Return pandas DataFrame of raw t2 chat data."""
    _, chat_path = t2_session_paths

    if not chat_path.exists():
        pytest.skip(f"T2 chat file not found: {chat_path}")

    return pd.read_csv(chat_path)


# =====
# Session object fixtures
# =====
@pytest.fixture
def loaded_t1_session(t1_session_paths: tuple) -> Session:
    """Load and return Session object for t1 using experiment_data module."""
    data_path, chat_path = t1_session_paths

    # Chat path may not exist, handle gracefully
    chat_str = str(chat_path) if chat_path.exists() else None

    file_pairs = [(str(data_path), chat_str, 1)]
    experiment = load_experiment_data(file_pairs, name="T1 Test")

    # Return first (and only) session
    session_codes = experiment.list_session_codes()
    return experiment.get_session(session_codes[0])


@pytest.fixture
def loaded_t2_session(t2_session_paths: tuple) -> Session:
    """Load and return Session object for t2 using experiment_data module."""
    data_path, chat_path = t2_session_paths

    # Chat path may not exist, handle gracefully
    chat_str = str(chat_path) if chat_path.exists() else None

    file_pairs = [(str(data_path), chat_str, 2)]
    experiment = load_experiment_data(file_pairs, name="T2 Test")

    # Return first (and only) session
    session_codes = experiment.list_session_codes()
    return experiment.get_session(session_codes[0])


# =====
# Experiment fixture
# =====
@pytest.fixture
def sample_experiment(t1_session_paths: tuple, t2_session_paths: tuple) -> Experiment:
    """Return Experiment with both t1 and t2 sessions loaded."""
    t1_data, t1_chat = t1_session_paths
    t2_data, t2_chat = t2_session_paths

    # Build file pairs, handling missing chat files
    file_pairs = []

    t1_chat_str = str(t1_chat) if t1_chat.exists() else None
    file_pairs.append((str(t1_data), t1_chat_str, 1))

    t2_chat_str = str(t2_chat) if t2_chat.exists() else None
    file_pairs.append((str(t2_data), t2_chat_str, 2))

    return load_experiment_data(file_pairs, name="Sample Experiment")


# =====
# Chat pairing verification helpers
# =====
@pytest.fixture
def chat_pairing_helper():
    """Provide helper functions for verifying chat-round pairing semantics.

    Chat-round pairing rules:
    - Round 1 of each supergame has empty chat_messages
    - Round N (N>1) has chat from round N-1 (chat influences next decision)
    - Last round's chat is in segment.orphan_chats (no subsequent round)
    - segment.get_all_chat_messages(include_orphans=True) returns all messages
    """
    class ChatPairingHelper:
        """Helper class for chat pairing verification."""

        @staticmethod
        def get_all_session_chats(session: Session, include_orphans: bool = True):
            """Get all chat messages from a session's supergames.

            Args:
                session: Session object to extract chats from
                include_orphans: Include orphan chats from last rounds

            Returns:
                List of all ChatMessage objects
            """
            all_messages = []
            for segment in session.segments.values():
                if segment.name.startswith('supergame'):
                    all_messages.extend(
                        segment.get_all_chat_messages(include_orphans=include_orphans)
                    )
            return all_messages

        @staticmethod
        def verify_round1_has_no_chat(session: Session) -> list:
            """Verify round 1 of each supergame has empty chat_messages.

            Returns:
                List of violations (empty if all pass)
            """
            violations = []
            for sg_num in range(1, 6):
                supergame = session.get_supergame(sg_num)
                if supergame is None:
                    continue
                round1 = supergame.get_round(1)
                if round1 and len(round1.chat_messages) > 0:
                    violations.append({
                        'supergame': sg_num,
                        'round': 1,
                        'chat_count': len(round1.chat_messages)
                    })
            return violations

        @staticmethod
        def get_orphan_chat_counts(session: Session) -> dict:
            """Get orphan chat counts per supergame.

            Returns:
                Dict mapping supergame number to orphan chat count
            """
            counts = {}
            for sg_num in range(1, 6):
                supergame = session.get_supergame(sg_num)
                if supergame is None:
                    continue
                counts[sg_num] = len(supergame.get_orphan_chats_flat())
            return counts

        @staticmethod
        def verify_chat_message_totals(session: Session, raw_chat_df) -> dict:
            """Compare loaded chat totals with raw CSV.

            Must include orphans to match raw CSV totals.

            Returns:
                Dict with 'loaded', 'raw', and 'match' keys
            """
            loaded_count = len(
                ChatPairingHelper.get_all_session_chats(session, include_orphans=True)
            )
            raw_count = len(raw_chat_df)
            return {
                'loaded': loaded_count,
                'raw': raw_count,
                'match': loaded_count == raw_count
            }

    return ChatPairingHelper()


# =====
# Dynamic regression tex-parsing fixtures
# =====
@pytest.fixture(scope="session")
def baseline_coefs():
    """Parse dynamic_regression_baseline.tex -> {label: [coef per column]}."""
    ensure_tex_outputs_current()
    if not BASELINE_TEX.exists():
        raise FileNotFoundError(
            f"Missing baseline table: {BASELINE_TEX}. "
            f"Run: cd analysis && Rscript analysis/dynamic_regression.R"
        )
    return parse_tex_coefficients(BASELINE_TEX, num_data_cols=2)


@pytest.fixture(scope="session")
def extended_coefs():
    """Parse dynamic_regression_extended.tex -> {label: [coef per column]}."""
    ensure_tex_outputs_current()
    if not EXTENDED_TEX.exists():
        raise FileNotFoundError(
            f"Missing extended table: {EXTENDED_TEX}. "
            f"Run: cd analysis && Rscript analysis/dynamic_regression.R"
        )
    return parse_tex_coefficients(EXTENDED_TEX, num_data_cols=6)


@pytest.fixture(scope="session")
def baseline_details():
    """Parse baseline .tex with SE+stars: {label: [(coef, se, stars), ...]}."""
    ensure_tex_outputs_current()
    if not BASELINE_TEX.exists():
        raise FileNotFoundError(
            f"Missing baseline table: {BASELINE_TEX}. "
            f"Run: cd analysis && Rscript analysis/dynamic_regression.R"
        )
    return parse_tex_with_details(BASELINE_TEX, num_data_cols=2)


@pytest.fixture(scope="session")
def baseline_gof():
    """Parse the GoF rows (Observations, AR/Sargan/Wald p-values) from baseline.tex."""
    ensure_tex_outputs_current()
    if not BASELINE_TEX.exists():
        raise FileNotFoundError(
            f"Missing baseline table: {BASELINE_TEX}. "
            f"Run: cd analysis && Rscript analysis/dynamic_regression.R"
        )
    return parse_tex_gof_rows(BASELINE_TEX, num_data_cols=2)
