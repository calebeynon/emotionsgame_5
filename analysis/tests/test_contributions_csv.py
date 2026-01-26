"""
Test schema and structure validation for contributions.csv.

Verifies that contributions.csv has the expected structure, columns,
data types, and value ranges for downstream analysis.

Author: Claude Code
Date: 2026-01-24
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# FILE PATHS
CONTRIBUTIONS_CSV = Path(__file__).parent.parent / 'datastore' / 'derived' / 'contributions.csv'

# CONSTANTS
EXPECTED_ROW_COUNT = 3520  # 10 sessions x 16 players x 22 rounds
ENDOWMENT = 25
EXPECTED_COLUMNS = [
    'Unnamed: 0',  # Index artifact from pandas to_csv
    'session_code',
    'treatment',
    'segment',
    'round',
    'group',
    'label',
    'participant_id',
    'contribution',
    'payoff',
    'role',  # Currently all null - documented in test_role_column_notes
]
REQUIRED_COLUMNS = [
    'session_code',
    'treatment',
    'segment',
    'round',
    'group',
    'label',
    'participant_id',
    'contribution',
    'payoff',
]
EXPECTED_SEGMENTS = ['supergame1', 'supergame2', 'supergame3', 'supergame4', 'supergame5']
ROUNDS_PER_SUPERGAME = {1: 3, 2: 4, 3: 3, 4: 7, 5: 5}  # Total: 22
NUM_SESSIONS = 10
NUM_PLAYERS_PER_SESSION = 16
NUM_GROUPS_PER_ROUND = 4
# Valid player labels (skips I and O to avoid confusion with 1 and 0)
VALID_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R']
EXPECTED_SESSION_CODES = [
    '6sdkxl2q', '6ucza025', '6uv359rf', 'iiu3xixz', 'irrzlgk2',
    'j3ki5tli', 'r5dj4yfl', 'sa7mprty', 'sylq2syi', 'umbzdj98'
]
EXPECTED_TREATMENT_MAPPING = {
    '6sdkxl2q': 1, 'iiu3xixz': 1, 'r5dj4yfl': 1, 'sa7mprty': 1, 'umbzdj98': 1,  # T1
    '6ucza025': 2, '6uv359rf': 2, 'irrzlgk2': 2, 'j3ki5tli': 2, 'sylq2syi': 2,  # T2
}


# =====
# Fixtures
# =====
@pytest.fixture
def contributions_df() -> pd.DataFrame:
    """Load contributions.csv as a DataFrame."""
    if not CONTRIBUTIONS_CSV.exists():
        pytest.skip(f"contributions.csv not found: {CONTRIBUTIONS_CSV}")
    return pd.read_csv(CONTRIBUTIONS_CSV)


# =====
# Schema tests
# =====
class TestContributionsCsvSchema:
    """Test suite for contributions.csv schema and structure validation."""

    def test_contributions_csv_exists(self):
        """Verify contributions.csv file exists at expected path."""
        assert CONTRIBUTIONS_CSV.exists(), (
            f"contributions.csv not found at {CONTRIBUTIONS_CSV}"
        )

    def test_contributions_csv_has_expected_columns(self, contributions_df):
        """Verify all expected columns are present."""
        actual_columns = contributions_df.columns.tolist()
        for col in EXPECTED_COLUMNS:
            assert col in actual_columns, f"Missing expected column: {col}"

    def test_contributions_csv_row_count(self, contributions_df):
        """Verify exactly 3520 rows (10 sessions x 16 players x 22 rounds)."""
        assert len(contributions_df) == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} rows, found {len(contributions_df)}"
        )

    def test_contributions_csv_data_types(self, contributions_df):
        """Verify columns have correct data types."""
        expected_dtypes = {
            'Unnamed: 0': 'int64',
            'session_code': 'object',
            'treatment': 'int64',
            'segment': 'object',
            'round': 'int64',
            'group': 'int64',
            'label': 'object',
            'participant_id': 'int64',
            'contribution': 'float64',
            'payoff': 'float64',
            'role': 'float64',  # All NaN, stored as float
        }
        for col, expected_dtype in expected_dtypes.items():
            assert str(contributions_df[col].dtype) == expected_dtype, (
                f"Column '{col}' has dtype {contributions_df[col].dtype}, "
                f"expected {expected_dtype}"
            )

    def test_contributions_csv_value_ranges(self, contributions_df):
        """Verify contribution in [0, 25] and treatment in {1, 2}."""
        # Contribution range
        assert contributions_df['contribution'].min() >= 0, (
            f"Contribution below 0: {contributions_df['contribution'].min()}"
        )
        assert contributions_df['contribution'].max() <= ENDOWMENT, (
            f"Contribution above {ENDOWMENT}: {contributions_df['contribution'].max()}"
        )

        # Treatment values
        unique_treatments = set(contributions_df['treatment'].unique())
        assert unique_treatments == {1, 2}, (
            f"Expected treatments {{1, 2}}, found {unique_treatments}"
        )

        # Segment values
        unique_segments = set(contributions_df['segment'].unique())
        assert unique_segments == set(EXPECTED_SEGMENTS), (
            f"Expected segments {set(EXPECTED_SEGMENTS)}, found {unique_segments}"
        )

    def test_contributions_csv_no_missing_required_values(self, contributions_df):
        """Verify no NaN values in required columns."""
        for col in REQUIRED_COLUMNS:
            missing_count = contributions_df[col].isnull().sum()
            assert missing_count == 0, (
                f"Column '{col}' has {missing_count} missing values"
            )

    def test_contributions_csv_unique_identifiers(self, contributions_df):
        """Verify each (session_code, segment, round, label) is unique."""
        id_columns = ['session_code', 'segment', 'round', 'label']
        duplicates = contributions_df.duplicated(subset=id_columns, keep=False)
        duplicate_count = duplicates.sum()
        assert duplicate_count == 0, (
            f"Found {duplicate_count // 2} duplicate identifier combinations. "
            f"First duplicate: {contributions_df[duplicates].head(1)[id_columns].to_dict()}"
        )

    def test_role_column_notes(self, contributions_df):
        """Document that role column is all null (expected behavior).

        The 'role' column exists in the CSV but contains no data.
        This is expected because the public goods game does not use roles.
        The column is retained for compatibility with other oTree exports.
        """
        assert contributions_df['role'].isnull().all(), (
            "Expected role column to be all null, but found some values"
        )
        # Explicit note for documentation
        role_null_count = contributions_df['role'].isnull().sum()
        assert role_null_count == EXPECTED_ROW_COUNT, (
            f"Expected {EXPECTED_ROW_COUNT} null values in role, found {role_null_count}"
        )

    def test_index_column_notes(self, contributions_df):
        """Document Unnamed: 0 column artifact from pandas to_csv.

        The 'Unnamed: 0' column is an index artifact created when saving
        a DataFrame with to_csv() without index=False. It contains
        sequential integers from 0 to n-1.
        """
        # Verify it's a sequential index
        expected_index = list(range(EXPECTED_ROW_COUNT))
        actual_index = contributions_df['Unnamed: 0'].tolist()
        assert actual_index == expected_index, (
            "Unnamed: 0 column should contain sequential integers 0 to n-1"
        )


# =====
# Coverage tests
# =====
class TestContributionsCsvCoverage:
    """Test suite for verifying complete data coverage in contributions.csv."""

    def test_contributions_csv_has_all_sessions(self, contributions_df):
        """Verify all 10 unique session codes are present."""
        actual_sessions = set(contributions_df['session_code'].unique())
        expected_sessions = set(EXPECTED_SESSION_CODES)

        assert actual_sessions == expected_sessions, (
            f"Session mismatch.\n"
            f"  Missing: {expected_sessions - actual_sessions}\n"
            f"  Extra: {actual_sessions - expected_sessions}"
        )
        assert contributions_df['session_code'].nunique() == NUM_SESSIONS, (
            f"Expected {NUM_SESSIONS} sessions, found "
            f"{contributions_df['session_code'].nunique()}"
        )

    def test_contributions_csv_session_treatment_mapping(self, contributions_df):
        """Verify exact session-treatment mapping (5 T1, 5 T2 with correct codes)."""
        # Verify specific session-treatment mapping for every session
        session_treatments = contributions_df.groupby('session_code')['treatment'].first()
        mismatches = []
        for session_code, expected_treatment in EXPECTED_TREATMENT_MAPPING.items():
            actual_treatment = session_treatments.get(session_code)
            if actual_treatment != expected_treatment:
                mismatches.append(
                    f"  {session_code}: expected T{expected_treatment}, "
                    f"found T{actual_treatment}"
                )

        assert len(mismatches) == 0, (
            f"Session-treatment mapping errors:\n" + "\n".join(mismatches)
        )

        # Verify treatment counts as a sanity check
        treatment_counts = contributions_df.groupby('treatment')['session_code'].nunique()
        assert treatment_counts.get(1, 0) == 5, (
            f"Expected 5 treatment=1 sessions, found {treatment_counts.get(1, 0)}"
        )
        assert treatment_counts.get(2, 0) == 5, (
            f"Expected 5 treatment=2 sessions, found {treatment_counts.get(2, 0)}"
        )

    def test_contributions_csv_rounds_per_supergame(self, contributions_df):
        """Verify correct round counts: SG1=3, SG2=4, SG3=3, SG4=7, SG5=5."""
        for sg_num, expected_rounds in ROUNDS_PER_SUPERGAME.items():
            segment_name = f'supergame{sg_num}'
            segment_df = contributions_df[contributions_df['segment'] == segment_name]
            actual_rounds = segment_df['round'].nunique()
            assert actual_rounds == expected_rounds, (
                f"{segment_name}: expected {expected_rounds} rounds, found {actual_rounds}"
            )

            # Verify round numbers are 1 to expected_rounds
            expected_round_nums = set(range(1, expected_rounds + 1))
            actual_round_nums = set(segment_df['round'].unique())
            assert actual_round_nums == expected_round_nums, (
                f"{segment_name}: expected rounds {expected_round_nums}, "
                f"found {actual_round_nums}"
            )

    def test_contributions_csv_players_per_session(self, contributions_df):
        """Verify 16 unique player labels (A-R, skipping I/O) in each session."""
        for session_code in EXPECTED_SESSION_CODES:
            session_df = contributions_df[contributions_df['session_code'] == session_code]
            actual_labels = set(session_df['label'].unique())
            expected_labels = set(VALID_LABELS)

            assert actual_labels == expected_labels, (
                f"Session {session_code} label mismatch.\n"
                f"  Missing: {expected_labels - actual_labels}\n"
                f"  Extra: {actual_labels - expected_labels}"
            )
            assert session_df['label'].nunique() == NUM_PLAYERS_PER_SESSION, (
                f"Session {session_code}: expected {NUM_PLAYERS_PER_SESSION} players, "
                f"found {session_df['label'].nunique()}"
            )

    def test_contributions_csv_groups_per_round(self, contributions_df):
        """Verify 4 groups per round in each session/supergame."""
        errors = []
        for session_code in EXPECTED_SESSION_CODES:
            for segment_name in EXPECTED_SEGMENTS:
                sg_num = int(segment_name.replace('supergame', ''))
                for round_num in range(1, ROUNDS_PER_SUPERGAME[sg_num] + 1):
                    round_df = contributions_df[
                        (contributions_df['session_code'] == session_code) &
                        (contributions_df['segment'] == segment_name) &
                        (contributions_df['round'] == round_num)
                    ]
                    num_groups = round_df['group'].nunique()
                    if num_groups != NUM_GROUPS_PER_ROUND:
                        errors.append(
                            f"{session_code}/{segment_name}/R{round_num}: "
                            f"{num_groups} groups (expected {NUM_GROUPS_PER_ROUND})"
                        )

        assert len(errors) == 0, (
            f"Found {len(errors)} rounds with incorrect group counts:\n"
            + "\n".join(errors[:10])  # Show first 10 errors
        )


# =====
# Raw data matching tests
# =====
RAW_DATA_DIR = Path(__file__).parent.parent / 'datastore' / 'raw'

# Session code to raw file mapping (extracted from raw CSVs)
SESSION_TO_RAW_FILE = {
    'sa7mprty': '01_t1_data.csv',
    'irrzlgk2': '03_t2_data.csv',
    '6uv359rf': '04_t2_data.csv',
    'umbzdj98': '05_t1_data.csv',
    'j3ki5tli': '06_t2_data.csv',
    'r5dj4yfl': '07_t1_data.csv',
    'sylq2syi': '08_t2_data.csv',
    'iiu3xixz': '09_t1_data.csv',
    '6ucza025': '10_t2_data.csv',
    '6sdkxl2q': '11_t1_data.csv',
}


def build_raw_session_map() -> dict:
    """Build mapping of session_code to raw data file path."""
    return {
        session_code: RAW_DATA_DIR / filename
        for session_code, filename in SESSION_TO_RAW_FILE.items()
    }


def get_raw_value(raw_df: pd.DataFrame, segment: str, round_num: int,
                  label: str, field: str) -> float:
    """Extract a player field value from raw CSV data.

    Args:
        raw_df: Raw data DataFrame (one row per participant)
        segment: Segment name (e.g., 'supergame1')
        round_num: Round number within segment
        label: Player label (e.g., 'A')
        field: Field to extract ('contribution' or 'payoff')

    Returns:
        The value for the specified player/round/field
    """
    # Build column name: supergame{N}.{R}.player.{field}
    column_name = f"{segment}.{round_num}.player.{field}"

    # Find row by participant label
    player_row = raw_df[raw_df['participant.label'] == label]
    if player_row.empty:
        raise ValueError(f"No player found with label '{label}'")

    return player_row[column_name].values[0]


class TestContributionsCsvRawMatch:
    """Test suite for verifying contributions.csv matches raw oTree data files."""

    @pytest.fixture
    def raw_session_map(self) -> dict:
        """Get mapping of session_code to raw file path."""
        session_map = build_raw_session_map()
        # Verify all raw files exist
        missing = [s for s, p in session_map.items() if not p.exists()]
        if missing:
            pytest.skip(f"Raw data files not found for sessions: {missing}")
        return session_map

    @pytest.fixture
    def raw_dataframes(self, raw_session_map) -> dict:
        """Load all raw data files into DataFrames."""
        return {
            session_code: pd.read_csv(path)
            for session_code, path in raw_session_map.items()
        }

    def test_contributions_csv_session_codes_match_raw_files(
        self, contributions_df, raw_session_map
    ):
        """Verify each session_code in contributions.csv has a raw data file."""
        csv_sessions = set(contributions_df['session_code'].unique())
        raw_sessions = set(raw_session_map.keys())

        missing_in_raw = csv_sessions - raw_sessions
        extra_in_raw = raw_sessions - csv_sessions

        assert len(missing_in_raw) == 0, (
            f"Sessions in contributions.csv but not in raw files: {missing_in_raw}"
        )
        assert len(extra_in_raw) == 0, (
            f"Sessions in raw files but not in contributions.csv: {extra_in_raw}"
        )

    def test_contributions_csv_matches_raw_contributions(
        self, contributions_df, raw_dataframes
    ):
        """Verify every contribution value matches the raw CSV data."""
        mismatches = []

        for _, row in contributions_df.iterrows():
            session_code = row['session_code']
            segment = row['segment']
            round_num = row['round']
            label = row['label']
            csv_contribution = row['contribution']

            raw_df = raw_dataframes[session_code]
            raw_contribution = get_raw_value(
                raw_df, segment, round_num, label, 'contribution'
            )

            if csv_contribution != pytest.approx(raw_contribution):
                mismatches.append({
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                    'csv_value': csv_contribution,
                    'raw_value': raw_contribution,
                })

            # Stop early if too many mismatches
            if len(mismatches) >= 20:
                break

        if mismatches:
            # Format first 10 mismatches for error message
            mismatch_lines = [
                f"  {m['session']}/{m['segment']}/R{m['round']}/{m['label']}: "
                f"csv={m['csv_value']} vs raw={m['raw_value']}"
                for m in mismatches[:10]
            ]
            more_msg = ""
            if len(mismatches) > 10:
                more_msg = f"\n  ... and {len(mismatches) - 10} more"

            assert False, (
                f"Found {len(mismatches)} contribution mismatches:\n"
                + "\n".join(mismatch_lines) + more_msg
            )

    def test_contributions_csv_matches_raw_payoffs(
        self, contributions_df, raw_dataframes
    ):
        """Verify every payoff value matches the raw CSV data."""
        mismatches = []

        for _, row in contributions_df.iterrows():
            session_code = row['session_code']
            segment = row['segment']
            round_num = row['round']
            label = row['label']
            csv_payoff = row['payoff']

            raw_df = raw_dataframes[session_code]
            raw_payoff = get_raw_value(
                raw_df, segment, round_num, label, 'payoff'
            )

            if csv_payoff != pytest.approx(raw_payoff):
                mismatches.append({
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                    'csv_value': csv_payoff,
                    'raw_value': raw_payoff,
                })

            # Stop early if too many mismatches
            if len(mismatches) >= 20:
                break

        if mismatches:
            # Format first 10 mismatches for error message
            mismatch_lines = [
                f"  {m['session']}/{m['segment']}/R{m['round']}/{m['label']}: "
                f"csv={m['csv_value']} vs raw={m['raw_value']}"
                for m in mismatches[:10]
            ]
            more_msg = ""
            if len(mismatches) > 10:
                more_msg = f"\n  ... and {len(mismatches) - 10} more"

            assert False, (
                f"Found {len(mismatches)} payoff mismatches:\n"
                + "\n".join(mismatch_lines) + more_msg
            )


# =====
# Experiment object matching tests
# =====
# Import experiment_data module for object-based matching
from experiment_data import load_experiment_data

# File mapping for all 10 sessions (number prefix -> (data_file, treatment))
ALL_SESSION_FILES = {
    '01': ('01_t1_data.csv', '01_t1_chat.csv', 1),
    '03': ('03_t2_data.csv', '03_t2_chat.csv', 2),
    '04': ('04_t2_data.csv', '04_t2_chat.csv', 2),
    '05': ('05_t1_data.csv', '05_t1_chat.csv', 1),
    '06': ('06_t2_data.csv', '06_t2_chat.csv', 2),
    '07': ('07_t1_data.csv', '07_t1_chat.csv', 1),
    '08': ('08_t2_data.csv', '08_t2_chat.csv', 2),
    '09': ('09_t1_data.csv', '09_t1_chat.csv', 1),
    '10': ('10_t2_data.csv', '10_t2_chat.csv', 2),
    '11': ('11_t1_data.csv', '11_t1_chat.csv', 1),
}


def segment_to_supergame_num(segment_name: str) -> int:
    """Convert segment name (e.g., 'supergame1') to supergame number (1)."""
    return int(segment_name.replace('supergame', ''))


def load_all_sessions_experiment():
    """Load all 10 sessions into a single Experiment object."""
    file_pairs = []
    for prefix, (data_file, chat_file, treatment) in ALL_SESSION_FILES.items():
        data_path = RAW_DATA_DIR / data_file
        chat_path = RAW_DATA_DIR / chat_file
        if data_path.exists():
            chat_str = str(chat_path) if chat_path.exists() else None
            file_pairs.append((str(data_path), chat_str, treatment))
    return load_experiment_data(file_pairs, name="All Sessions Test")


class TestContributionsCsvExperimentMatch:
    """Test suite for verifying contributions.csv matches experiment_data objects."""

    def test_contributions_csv_matches_sample_experiment(
        self, contributions_df, sample_experiment
    ):
        """Verify contributions.csv matches sample_experiment fixture (2 sessions)."""
        mismatches = []
        sample_session_codes = sample_experiment.list_session_codes()

        # Filter CSV to only sample experiment sessions
        sample_df = contributions_df[
            contributions_df['session_code'].isin(sample_session_codes)
        ]

        for _, row in sample_df.iterrows():
            session_code = row['session_code']
            segment = row['segment']
            round_num = int(row['round'])
            label = row['label']
            csv_contribution = row['contribution']
            csv_payoff = row['payoff']

            # Navigate experiment hierarchy
            session = sample_experiment.get_session(session_code)
            if session is None:
                mismatches.append({
                    'type': 'session_not_found',
                    'session': session_code,
                })
                continue

            sg_num = segment_to_supergame_num(segment)
            supergame = session.get_supergame(sg_num)
            if supergame is None:
                mismatches.append({
                    'type': 'supergame_not_found',
                    'session': session_code,
                    'segment': segment,
                })
                continue

            round_obj = supergame.get_round(round_num)
            if round_obj is None:
                mismatches.append({
                    'type': 'round_not_found',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                })
                continue

            player = round_obj.get_player(label)
            if player is None:
                mismatches.append({
                    'type': 'player_not_found',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                })
                continue

            # Compare contribution
            if csv_contribution != pytest.approx(player.contribution):
                mismatches.append({
                    'type': 'contribution_mismatch',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                    'csv_value': csv_contribution,
                    'obj_value': player.contribution,
                })

            # Compare payoff
            if csv_payoff != pytest.approx(player.payoff):
                mismatches.append({
                    'type': 'payoff_mismatch',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                    'csv_value': csv_payoff,
                    'obj_value': player.payoff,
                })

            # Stop early if too many mismatches
            if len(mismatches) >= 20:
                break

        if mismatches:
            mismatch_lines = self._format_mismatches(mismatches[:10])
            more_msg = f"\n  ... and {len(mismatches) - 10} more" if len(mismatches) > 10 else ""
            assert False, (
                f"Found {len(mismatches)} mismatches between CSV and experiment object:\n"
                + "\n".join(mismatch_lines) + more_msg
            )

    @pytest.mark.slow
    def test_contributions_csv_matches_all_sessions_experiment(self, contributions_df):
        """Verify all 3,520 rows in contributions.csv match experiment_data objects.

        Loads all 10 sessions and verifies every contribution and payoff value
        in the CSV matches the corresponding value in the experiment object.
        """
        # Skip if raw data files are missing
        missing_files = [
            f for f, (df, _, _) in ALL_SESSION_FILES.items()
            if not (RAW_DATA_DIR / df).exists()
        ]
        if missing_files:
            pytest.skip(f"Raw data files missing for sessions: {missing_files}")

        # Load all sessions
        experiment = load_all_sessions_experiment()

        mismatches = []
        for _, row in contributions_df.iterrows():
            session_code = row['session_code']
            segment = row['segment']
            round_num = int(row['round'])
            label = row['label']
            csv_contribution = row['contribution']
            csv_payoff = row['payoff']

            # Navigate experiment hierarchy
            session = experiment.get_session(session_code)
            if session is None:
                mismatches.append({
                    'type': 'session_not_found',
                    'session': session_code,
                })
                continue

            sg_num = segment_to_supergame_num(segment)
            supergame = session.get_supergame(sg_num)
            if supergame is None:
                mismatches.append({
                    'type': 'supergame_not_found',
                    'session': session_code,
                    'segment': segment,
                })
                continue

            round_obj = supergame.get_round(round_num)
            if round_obj is None:
                mismatches.append({
                    'type': 'round_not_found',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                })
                continue

            player = round_obj.get_player(label)
            if player is None:
                mismatches.append({
                    'type': 'player_not_found',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                })
                continue

            # Compare contribution
            if csv_contribution != pytest.approx(player.contribution):
                mismatches.append({
                    'type': 'contribution_mismatch',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                    'csv_value': csv_contribution,
                    'obj_value': player.contribution,
                })

            # Compare payoff
            if csv_payoff != pytest.approx(player.payoff):
                mismatches.append({
                    'type': 'payoff_mismatch',
                    'session': session_code,
                    'segment': segment,
                    'round': round_num,
                    'label': label,
                    'csv_value': csv_payoff,
                    'obj_value': player.payoff,
                })

            # Stop early if too many mismatches
            if len(mismatches) >= 20:
                break

        if mismatches:
            mismatch_lines = self._format_mismatches(mismatches[:10])
            more_msg = f"\n  ... and {len(mismatches) - 10} more" if len(mismatches) > 10 else ""
            assert False, (
                f"Found {len(mismatches)} mismatches between CSV and experiment object:\n"
                + "\n".join(mismatch_lines) + more_msg
            )

    def _format_mismatches(self, mismatches: list) -> list:
        """Format mismatch records into readable error lines."""
        lines = []
        for m in mismatches:
            if m['type'] == 'session_not_found':
                lines.append(f"  Session not found: {m['session']}")
            elif m['type'] == 'supergame_not_found':
                lines.append(f"  Supergame not found: {m['session']}/{m['segment']}")
            elif m['type'] == 'round_not_found':
                lines.append(
                    f"  Round not found: {m['session']}/{m['segment']}/R{m['round']}"
                )
            elif m['type'] == 'player_not_found':
                lines.append(
                    f"  Player not found: {m['session']}/{m['segment']}/R{m['round']}/{m['label']}"
                )
            elif m['type'] in ('contribution_mismatch', 'payoff_mismatch'):
                field = 'contribution' if m['type'] == 'contribution_mismatch' else 'payoff'
                lines.append(
                    f"  {field.capitalize()} mismatch: {m['session']}/{m['segment']}/R{m['round']}/{m['label']}: "
                    f"csv={m['csv_value']} vs obj={m['obj_value']}"
                )
        return lines


# =====
# Integration tests
# =====
class TestContributionsCsvIntegration:
    """Test suite for verifying contributions.csv round-trip consistency.

    These tests verify that contributions.csv can be regenerated from the
    experiment_data object and matches the existing CSV file.
    """

    @pytest.mark.slow
    def test_contributions_csv_roundtrip_consistency(self, contributions_df):
        """Verify contributions.csv matches DataFrame generated from experiment_data.

        This test loads all 10 sessions into an Experiment object, generates
        a DataFrame using to_dataframe_contributions(), and compares it with
        the existing contributions.csv file to ensure round-trip consistency.
        """
        # Skip if raw data files are missing
        missing_files = [
            f for f, (df, _, _) in ALL_SESSION_FILES.items()
            if not (RAW_DATA_DIR / df).exists()
        ]
        if missing_files:
            pytest.skip(f"Raw data files missing for sessions: {missing_files}")

        # Load all sessions and generate DataFrame
        experiment = load_all_sessions_experiment()
        generated_df = experiment.to_dataframe_contributions()

        assert generated_df is not None, "to_dataframe_contributions() returned None"

        # Normalize both DataFrames for comparison
        csv_df = self._normalize_dataframe(contributions_df)
        gen_df = self._normalize_dataframe(generated_df)

        # Compare row counts
        assert len(gen_df) == len(csv_df), (
            f"Row count mismatch: generated={len(gen_df)}, csv={len(csv_df)}"
        )

        # Compare column sets (excluding index artifact)
        gen_cols = set(gen_df.columns)
        csv_cols = set(csv_df.columns) - {'Unnamed: 0'}

        missing_cols = csv_cols - gen_cols
        extra_cols = gen_cols - csv_cols

        assert len(missing_cols) == 0, (
            f"Generated DataFrame missing columns: {missing_cols}"
        )
        # Note: extra columns in generated are acceptable (future extensions)

        # Merge on primary key to compare values
        key_cols = ['session_code', 'segment', 'round', 'label']
        merged = gen_df.merge(
            csv_df[key_cols + ['contribution', 'payoff']],
            on=key_cols,
            suffixes=('_gen', '_csv'),
            how='outer',
            indicator=True
        )

        # Check for unmatched rows
        left_only = merged[merged['_merge'] == 'left_only']
        right_only = merged[merged['_merge'] == 'right_only']

        if len(left_only) > 0:
            sample = left_only[key_cols].head(5).to_dict('records')
            assert False, (
                f"Found {len(left_only)} rows in generated but not in CSV:\n"
                f"Sample: {sample}"
            )

        if len(right_only) > 0:
            sample = right_only[key_cols].head(5).to_dict('records')
            assert False, (
                f"Found {len(right_only)} rows in CSV but not in generated:\n"
                f"Sample: {sample}"
            )

        # Compare contribution values
        contribution_mismatches = self._find_value_mismatches(
            merged, 'contribution_gen', 'contribution_csv', key_cols
        )
        if contribution_mismatches:
            assert False, (
                f"Found {len(contribution_mismatches)} contribution mismatches:\n"
                + "\n".join(contribution_mismatches[:10])
            )

        # Compare payoff values
        payoff_mismatches = self._find_value_mismatches(
            merged, 'payoff_gen', 'payoff_csv', key_cols
        )
        if payoff_mismatches:
            assert False, (
                f"Found {len(payoff_mismatches)} payoff mismatches:\n"
                + "\n".join(payoff_mismatches[:10])
            )

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame for consistent comparison.

        Sorts by primary key and resets index for deterministic comparison.
        """
        key_cols = ['session_code', 'segment', 'round', 'label']
        return df.sort_values(key_cols).reset_index(drop=True)

    def _find_value_mismatches(
        self, merged: pd.DataFrame, col_gen: str, col_csv: str, key_cols: list
    ) -> list:
        """Find rows where generated and CSV values differ."""
        mismatches = []
        both = merged[merged['_merge'] == 'both']

        for _, row in both.iterrows():
            gen_val = row[col_gen]
            csv_val = row[col_csv]

            # Handle NaN comparison
            if pd.isna(gen_val) and pd.isna(csv_val):
                continue

            if pd.isna(gen_val) or pd.isna(csv_val):
                key_str = "/".join(str(row[k]) for k in key_cols)
                mismatches.append(
                    f"  {key_str}: gen={gen_val} vs csv={csv_val}"
                )
                continue

            if gen_val != pytest.approx(csv_val):
                key_str = "/".join(str(row[k]) for k in key_cols)
                mismatches.append(
                    f"  {key_str}: gen={gen_val} vs csv={csv_val}"
                )

            # Stop early if too many mismatches
            if len(mismatches) >= 20:
                break

        return mismatches


# =====
# Run tests directly
# =====
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
