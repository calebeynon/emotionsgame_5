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

from experiment_data import load_experiment_data, Session, Experiment

# FILE PATHS
RAW_DATA_DIR = Path(__file__).parent.parent / "datastore" / "raw"


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
