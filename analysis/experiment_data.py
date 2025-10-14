"""
Hierarchical data structure for oTree experimental data.

This module provides classes to store and access experimental data in a structured way.
Usage: session.get_segment('supergame1').get_round(1).get_player('A').contribution
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import re
from datetime import datetime
from dataclasses import dataclass
from statistics import mean
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results for chat messages."""
    positive: float  # Positive sentiment score (0-1)
    negative: float  # Negative sentiment score (0-1)
    neutral: float   # Neutral sentiment score (0-1)
    compound: float  # Overall sentiment (-1 to 1, negative to positive)
    message_count: int  # Number of messages analyzed
    
    @property
    def dominant_sentiment(self) -> str:
        """Return the dominant sentiment category."""
        if self.compound >= 0.05:
            return 'positive'
        elif self.compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    @property
    def sentiment_intensity(self) -> str:
        """Return sentiment intensity category."""
        abs_compound = abs(self.compound)
        if abs_compound >= 0.6:
            return 'strong'
        elif abs_compound >= 0.2:
            return 'moderate'
        else:
            return 'weak'
    
    def __str__(self):
        return f"Sentiment(compound={self.compound:.3f}, {self.dominant_sentiment}, {self.sentiment_intensity}, n={self.message_count})"


class ChatMessage:
    """Individual chat message."""
    
    def __init__(self, nickname: str, body: str, timestamp: float):
        self.nickname = nickname  # Player label (A, B, C, etc.)
        self.body = body  # Message content
        self.timestamp = timestamp  # Unix timestamp
        self.datetime = datetime.fromtimestamp(timestamp)  # Converted datetime
        self._sentiment_scores = None  # Cache for sentiment analysis
    
    @property
    def sentiment_scores(self) -> Dict[str, float]:
        """Get VADER sentiment scores for this message."""
        if self._sentiment_scores is None:
            self._sentiment_scores = sia.polarity_scores(self.body)
        return self._sentiment_scores
    
    @property
    def sentiment(self) -> float:
        """Get compound sentiment score (-1 to 1)."""
        return self.sentiment_scores['compound']
    
    @property
    def positive_sentiment(self) -> float:
        """Get positive sentiment component (0-1)."""
        return self.sentiment_scores['pos']
    
    @property
    def negative_sentiment(self) -> float:
        """Get negative sentiment component (0-1)."""
        return self.sentiment_scores['neg']
    
    @property
    def neutral_sentiment(self) -> float:
        """Get neutral sentiment component (0-1)."""
        return self.sentiment_scores['neu']
    
    def __str__(self):
        return f"{self.nickname}: {self.body}"
    
    def __repr__(self):
        return self.__str__()


def analyze_messages_sentiment(messages: List[ChatMessage]) -> Optional[SentimentAnalysis]:
    """Analyze sentiment for a list of chat messages."""
    if not messages:
        return None
    
    # Collect all sentiment scores
    positive_scores = []
    negative_scores = []
    neutral_scores = []
    compound_scores = []
    
    for msg in messages:
        scores = msg.sentiment_scores
        positive_scores.append(scores['pos'])
        negative_scores.append(scores['neg'])
        neutral_scores.append(scores['neu'])
        compound_scores.append(scores['compound'])
    
    # Calculate average sentiment scores
    return SentimentAnalysis(
        positive=mean(positive_scores),
        negative=mean(negative_scores),
        neutral=mean(neutral_scores),
        compound=mean(compound_scores),
        message_count=len(messages)
    )


class Player:
    """Individual player data for a specific round."""
    
    def __init__(self, participant_id: int, label: str, id_in_group: int):
        self.participant_id = participant_id
        self.label = label  # A, B, C, etc.
        self.id_in_group = id_in_group
        self.role = None
        self.payoff = None
        self.contribution = None
        self.group_id = None
        # Store any additional player-specific data
        self.data = {}
        # Chat messages from this player in this round
        self.chat_messages = []
    
    def get_chat_sentiment(self) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis for this player's chat messages in this round."""
        return analyze_messages_sentiment(self.chat_messages)
    
    def __str__(self):
        return f"Player {self.label} (ID: {self.participant_id})"
    
    def __repr__(self):
        return self.__str__()


class Group:
    """Group data for a specific round."""
    
    def __init__(self, group_id: int):
        self.group_id = group_id
        self.players: Dict[str, Player] = {}  # label -> Player
        self.total_contribution = None
        self.individual_share = None
        self.data = {}
        # All chat messages for this group in this round
        self.chat_messages = []
    
    def add_player(self, player: Player):
        """Add a player to this group."""
        self.players[player.label] = player
        player.group_id = self.group_id
    
    def get_player(self, label: str) -> Optional[Player]:
        """Get player by label (A, B, C, etc.)."""
        return self.players.get(label)
    
    def get_player_by_id(self, participant_id: int) -> Optional[Player]:
        """Get player by participant ID."""
        for player in self.players.values():
            if player.participant_id == participant_id:
                return player
        return None
    
    def get_chat_sentiment(self) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis for this group's chat messages in this round."""
        return analyze_messages_sentiment(self.chat_messages)
    
    def get_player_sentiments(self) -> Dict[str, Optional[SentimentAnalysis]]:
        """Get sentiment analysis for each player in this group."""
        sentiments = {}
        for label, player in self.players.items():
            sentiments[label] = player.get_chat_sentiment()
        return sentiments
    
    def __str__(self):
        return f"Group {self.group_id} ({len(self.players)} players)"


class Round:
    """Data for a single round within a segment."""
    
    def __init__(self, round_number: int):
        self.round_number = round_number
        self.groups: Dict[int, Group] = {}  # group_id -> Group
        self.players: Dict[str, Player] = {}  # label -> Player (for easy access)
        self.data = {}
        # All chat messages for this round across all groups
        self.chat_messages = []
    
    def add_group(self, group: Group):
        """Add a group to this round."""
        self.groups[group.group_id] = group
        # Also add players to round-level access
        for label, player in group.players.items():
            self.players[label] = player
    
    def get_group(self, group_id: int) -> Optional[Group]:
        """Get group by ID."""
        return self.groups.get(group_id)
    
    def get_player(self, label: str) -> Optional[Player]:
        """Get player by label."""
        return self.players.get(label)
    
    def get_player_by_id(self, participant_id: int) -> Optional[Player]:
        """Get player by participant ID."""
        for player in self.players.values():
            if player.participant_id == participant_id:
                return player
        return None
    
    def get_all_chat_messages(self) -> List[ChatMessage]:
        """Get all chat messages for this round across all groups."""
        return self.chat_messages
    
    def get_chat_sentiment(self) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis for all chat messages in this round."""
        return analyze_messages_sentiment(self.chat_messages)
    
    def get_group_sentiments(self) -> Dict[int, Optional[SentimentAnalysis]]:
        """Get sentiment analysis for each group in this round."""
        sentiments = {}
        for group_id, group in self.groups.items():
            sentiments[group_id] = group.get_chat_sentiment()
        return sentiments
    
    def get_player_sentiments(self) -> Dict[str, Optional[SentimentAnalysis]]:
        """Get sentiment analysis for each player in this round."""
        sentiments = {}
        for label, player in self.players.items():
            sentiments[label] = player.get_chat_sentiment()
        return sentiments
    
    def __str__(self):
        return f"Round {self.round_number} ({len(self.groups)} groups, {len(self.players)} players)"


class Segment:
    """A segment of the experiment (introduction, supergame1-5, finalresults)."""
    
    def __init__(self, name: str):
        self.name = name
        self.rounds: Dict[int, Round] = {}  # round_number -> Round
        self.data = {}
    
    def add_round(self, round_obj: Round):
        """Add a round to this segment."""
        self.rounds[round_obj.round_number] = round_obj
    
    def get_round(self, round_number: int) -> Optional[Round]:
        """Get round by number."""
        return self.rounds.get(round_number)
    
    def get_player_across_rounds(self, label: str) -> Dict[int, Player]:
        """Get a player's data across all rounds in this segment."""
        player_data = {}
        for round_num, round_obj in self.rounds.items():
            if label in round_obj.players:
                player_data[round_num] = round_obj.players[label]
        return player_data
    
    def get_all_chat_messages(self) -> List[ChatMessage]:
        """Get all chat messages for this segment across all rounds."""
        all_messages = []
        for round_obj in self.rounds.values():
            all_messages.extend(round_obj.chat_messages)
        return all_messages
    
    def get_chat_sentiment(self) -> Optional[SentimentAnalysis]:
        """Get sentiment analysis for all chat messages in this segment."""
        all_messages = self.get_all_chat_messages()
        return analyze_messages_sentiment(all_messages)
    
    def get_round_sentiments(self) -> Dict[int, Optional[SentimentAnalysis]]:
        """Get sentiment analysis for each round in this segment."""
        sentiments = {}
        for round_num, round_obj in self.rounds.items():
            sentiments[round_num] = round_obj.get_chat_sentiment()
        return sentiments
    
    def get_player_sentiment_across_rounds(self, label: str) -> Optional[SentimentAnalysis]:
        """Get aggregated sentiment analysis for a player across all rounds in this segment."""
        player_messages = []
        for round_obj in self.rounds.values():
            if label in round_obj.players:
                player = round_obj.players[label]
                player_messages.extend(player.chat_messages)
        return analyze_messages_sentiment(player_messages)
    
    def get_all_player_sentiments_across_rounds(self) -> Dict[str, Optional[SentimentAnalysis]]:
        """Get aggregated sentiment analysis for all players across all rounds in this segment."""
        # Get all unique player labels
        all_labels = set()
        for round_obj in self.rounds.values():
            all_labels.update(round_obj.players.keys())
        
        sentiments = {}
        for label in all_labels:
            sentiments[label] = self.get_player_sentiment_across_rounds(label)
        return sentiments
    
    def __str__(self):
        return f"Segment '{self.name}' ({len(self.rounds)} rounds)"


class Session:
    """Complete session data containing all segments."""
    
    def __init__(self, session_code: str, treatment: Optional[int] = None):
        self.session_code = session_code
        self.treatment = treatment  # Treatment condition (1 or 2)
        self.segments: Dict[str, Segment] = {}
        # Store participant mapping: participant_id -> label
        self.participant_labels: Dict[int, str] = {}
        # Store session-level metadata
        self.metadata = {}
    
    def add_segment(self, segment: Segment):
        """Add a segment to this session."""
        self.segments[segment.name] = segment
    
    def get_segment(self, name: str) -> Optional[Segment]:
        """Get segment by name (e.g., 'supergame1', 'introduction')."""
        return self.segments.get(name)
    
    def get_supergame(self, number: int) -> Optional[Segment]:
        """Get supergame by number (1-5)."""
        return self.get_segment(f'supergame{number}')
    
    def get_player_across_session(self, label: str) -> Dict[str, Dict[int, Player]]:
        """Get a player's data across all segments and rounds."""
        player_data = {}
        for segment_name, segment in self.segments.items():
            player_data[segment_name] = segment.get_player_across_rounds(label)
        return player_data
    
    def get_participant_data(self, participant_id: int) -> Dict[str, Dict[int, Player]]:
        """Get all data for a specific participant by ID."""
        if participant_id not in self.participant_labels:
            return {}
        
        label = self.participant_labels[participant_id]
        return self.get_player_across_session(label)
    
    def get_all_chat_messages(self) -> List[ChatMessage]:
        """Get all chat messages for this session across all segments."""
        all_messages = []
        for segment in self.segments.values():
            if segment.name.startswith('supergame'):
                all_messages.extend(segment.get_all_chat_messages())
        return all_messages
    
    def get_overall_sentiment(self) -> Optional[SentimentAnalysis]:
        """Get overall sentiment analysis for all chat messages in the session."""
        all_messages = self.get_all_chat_messages()
        return analyze_messages_sentiment(all_messages)
    
    def get_segment_sentiments(self) -> Dict[str, Optional[SentimentAnalysis]]:
        """Get sentiment analysis for each segment."""
        sentiments = {}
        for segment_name, segment in self.segments.items():
            if segment_name.startswith('supergame'):
                sentiments[segment_name] = segment.get_chat_sentiment()
        return sentiments
    
    def get_player_sentiment_across_session(self, label: str) -> Optional[SentimentAnalysis]:
        """Get aggregated sentiment analysis for a player across the entire session."""
        player_messages = []
        for segment in self.segments.values():
            if segment.name.startswith('supergame'):
                player_messages.extend([msg for round_obj in segment.rounds.values() 
                                      for msg in round_obj.chat_messages 
                                      if msg.nickname == label])
        return analyze_messages_sentiment(player_messages)
    
    def get_all_player_sentiments_across_session(self) -> Dict[str, Optional[SentimentAnalysis]]:
        """Get aggregated sentiment analysis for all players across the entire session."""
        sentiments = {}
        for label in self.participant_labels.values():
            sentiments[label] = self.get_player_sentiment_across_session(label)
        return sentiments
    
    def get_supergame_sentiments(self) -> Dict[int, Optional[SentimentAnalysis]]:
        """Get sentiment analysis for each supergame (1-5)."""
        sentiments = {}
        for sg_num in range(1, 6):
            segment = self.get_supergame(sg_num)
            if segment:
                sentiments[sg_num] = segment.get_chat_sentiment()
        return sentiments
    
    def __str__(self):
        return f"Session {self.session_code} ({len(self.segments)} segments)"


class Experiment:
    """Experiment-level container that holds multiple sessions and provides aggregation."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or 'experiment'
        self.sessions: Dict[str, Session] = {}  # session_code -> Session
        self.metadata: Dict[str, Any] = {}
    
    def add_session(self, session: Session):
        """Add a Session to the experiment."""
        self.sessions[session.session_code] = session
    
    def get_session(self, session_code: str) -> Optional[Session]:
        return self.sessions.get(session_code)
    
    def list_session_codes(self) -> List[str]:
        return list(self.sessions.keys())
    
    def get_all_chat_messages(self) -> List[ChatMessage]:
        messages: List[ChatMessage] = []
        for sess in self.sessions.values():
            messages.extend(sess.get_all_chat_messages())
        return messages
    
    def get_overall_sentiment(self) -> Optional[SentimentAnalysis]:
        return analyze_messages_sentiment(self.get_all_chat_messages())
    
    def get_session_sentiments(self) -> Dict[str, Optional[SentimentAnalysis]]:
        return {code: sess.get_overall_sentiment() for code, sess in self.sessions.items()}
    
    def get_all_player_sentiments_across_experiment(self) -> Dict[str, Optional[SentimentAnalysis]]:
        """Aggregate player sentiments across the entire experiment.
        Keys are namespaced as '<session_code>:<label>' to avoid collisions.
        """
        sentiments: Dict[str, Optional[SentimentAnalysis]] = {}
        for code, sess in self.sessions.items():
            for label in sess.participant_labels.values():
                key = f"{code}:{label}"
                sentiments[key] = sess.get_player_sentiment_across_session(label)
        return sentiments
    
    def to_dataframe_contributions(self) -> Optional[pd.DataFrame]:
        """Flatten contributions across all sessions into a DataFrame.
        Columns: session_code, treatment, segment, round, group, label, participant_id, contribution, payoff, role
        Returns None if no data found.
        """
        records: List[Dict[str, Any]] = []
        for code, sess in self.sessions.items():
            for segment_name, segment in sess.segments.items():
                if not segment_name.startswith('supergame'):
                    continue
                for round_num, rnd in segment.rounds.items():
                    for group_id, grp in rnd.groups.items():
                        for label, player in grp.players.items():
                            records.append({
                                'session_code': code,
                                'treatment': sess.treatment,
                                'segment': segment_name,
                                'round': round_num,
                                'group': group_id,
                                'label': label,
                                'participant_id': player.participant_id,
                                'contribution': player.contribution,
                                'payoff': player.payoff,
                                'role': player.role
                            })
        if not records:
            return None
        return pd.DataFrame.from_records(records)


def load_chat_data(chat_csv_path: str) -> Dict[int, Dict[int, Dict[str, List[ChatMessage]]]]:
    """
    Load chat data and organize it by supergame and chatgroup.
    
    We'll map chatgroups to actual rounds and groups later when we have 
    the session data loaded.
    
    Args:
        chat_csv_path: Path to the chat CSV file
        
    Returns:
        Dictionary structure: {supergame: {chatgroup: {nickname: [ChatMessage, ...]}}}
    """
    print(f"Loading chat data from: {chat_csv_path}")
    
    # Read chat CSV
    chat_df = pd.read_csv(chat_csv_path)
    print(f"Loaded {len(chat_df)} chat messages")
    
    # Parse channel information to extract supergame and chatgroup
    channel_pattern = re.compile(r'^1\-supergame(\d+)\-(.+)$')
    
    chat_data = {}  # supergame -> chatgroup -> nickname -> [ChatMessage]
    
    for _, row in chat_df.iterrows():
        channel = row['channel']
        match = channel_pattern.match(channel)
        
        if match:
            supergame = int(match.group(1))
            chatgroup = int(match.group(2))
            nickname = row['nickname']
            
            # Create chat message
            message = ChatMessage(
                nickname=nickname,
                body=row['body'],
                timestamp=float(row['timestamp'])
            )
            
            # Initialize data structure if needed
            if supergame not in chat_data:
                chat_data[supergame] = {}
            
            if chatgroup not in chat_data[supergame]:
                chat_data[supergame][chatgroup] = {}
                
            if nickname not in chat_data[supergame][chatgroup]:
                chat_data[supergame][chatgroup][nickname] = []
            
            chat_data[supergame][chatgroup][nickname].append(message)
    
    return chat_data


def assign_chat_messages(session: Session, chat_data: Dict[int, Dict[int, Dict[str, List[ChatMessage]]]]):
    """
    Assign chat messages to the appropriate groups and players.
    
    The chatgroup numbering system:
    - Each round has 4 consecutive chatgroup numbers (for 4 groups)
    - Round 1: chatgroups N, N+1, N+2, N+3
    - Round 2: chatgroups N+4, N+5, N+6, N+7
    - Round 3: chatgroups N+8, N+9, N+10, N+11
    - And so on...
    
    Args:
        session: Session object with loaded experimental data
        chat_data: Chat data organized by supergame and chatgroup
    """
    print("Assigning chat messages to groups and players...")
    
    for supergame_num in chat_data.keys():
        supergame = session.get_supergame(supergame_num)
        if not supergame:
            continue
            
        # Get chat data for this supergame
        sg_chat_data = chat_data[supergame_num]
        chatgroups = sorted(sg_chat_data.keys())
        
        if not chatgroups:
            continue
            
        # Find the starting chatgroup number for this supergame
        min_chatgroup = min(chatgroups)
        
        # Map chatgroups to rounds and groups
        for round_num in sorted(supergame.rounds.keys()):
            round_obj = supergame.get_round(round_num)
            if not round_obj:
                continue
            
            # Calculate the chatgroup range for this round
            # Round 1: min_chatgroup to min_chatgroup + 3
            # Round 2: min_chatgroup + 4 to min_chatgroup + 7
            # Round 3: min_chatgroup + 8 to min_chatgroup + 11
            round_start_chatgroup = min_chatgroup + (round_num - 1) * 4
            round_chatgroups = list(range(round_start_chatgroup, round_start_chatgroup + 4))
            
            # Map each chatgroup to a game group
            for i, chatgroup_id in enumerate(round_chatgroups):
                if chatgroup_id in sg_chat_data:
                    game_group_id = i + 1  # Game groups are numbered 1-4
                    game_group = round_obj.get_group(game_group_id)
                    
                    if game_group:
                        # Get all chat messages for this chatgroup
                        for nickname, messages in sg_chat_data[chatgroup_id].items():
                            # Add messages to group
                            game_group.chat_messages.extend(messages)
                            
                            # Add messages to individual player if they exist in this group
                            if nickname in game_group.players:
                                player = game_group.players[nickname]
                                player.chat_messages.extend(messages)
                            
                            # Add messages to round
                            round_obj.chat_messages.extend(messages)
    
    # Sort chat messages by timestamp
    for segment in session.segments.values():
        for round_obj in segment.rounds.values():
            round_obj.chat_messages.sort(key=lambda x: x.timestamp)
            for group in round_obj.groups.values():
                group.chat_messages.sort(key=lambda x: x.timestamp)
                for player in group.players.values():
                    player.chat_messages.sort(key=lambda x: x.timestamp)
    
    print(f"Chat assignment complete!")


def _load_single_session_data(csv_path: str, chat_csv_path: Optional[str] = None, treatment: Optional[int] = None) -> Session:
    """
    Internal function to load experimental data from CSV file into hierarchical structure.
    This is now a private function used by load_experiment_data.
    
    Args:
        csv_path: Path to the CSV file containing experimental data
        chat_csv_path: Optional path to chat CSV file
        treatment: Treatment condition (1 or 2)
        
    Returns:
        Session object containing all experimental data
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Load chat data if provided
    chat_data = {}
    if chat_csv_path:
        chat_data = load_chat_data(chat_csv_path)
    
    # Get session info from first row
    session_code = df['session.code'].iloc[0]
    session = Session(session_code, treatment)
    
    # Store session metadata
    session.metadata['is_demo'] = df['session.is_demo'].iloc[0]
    session.metadata['participation_fee'] = df['session.config.participation_fee'].iloc[0]
    session.metadata['real_world_currency_per_point'] = df['session.config.real_world_currency_per_point'].iloc[0]
    session.metadata['room'] = df['session.config.room'].iloc[0]
    
    # Create participant label mapping
    for _, row in df.iterrows():
        participant_id = row['participant.id_in_session']
        label = row['participant.label']
        session.participant_labels[participant_id] = label
    
    # Get all column names to parse segments
    columns = df.columns.tolist()
    
    # Parse segments from column names
    segment_pattern = r'^(introduction|supergame\d+|finalresults)\.(\d+)\.(player|group|subsession)\.(.+)$'
    
    segments_found = set()
    for col in columns:
        match = re.match(segment_pattern, col)
        if match:
            segment_name = match.group(1)
            segments_found.add(segment_name)
    
    # Process each segment
    for segment_name in sorted(segments_found):
        segment = Segment(segment_name)
        
        # Find all rounds for this segment
        rounds_found = set()
        segment_cols = [col for col in columns if col.startswith(f'{segment_name}.')]
        
        for col in segment_cols:
            match = re.match(segment_pattern, col)
            if match:
                round_num = int(match.group(2))
                rounds_found.add(round_num)
        
        # Process each round
        for round_num in sorted(rounds_found):
            round_obj = Round(round_num)
            
            # Get all player data for this round
            players_in_round = {}
            groups_in_round = {}
            
            for _, row in df.iterrows():
                participant_id = row['participant.id_in_session']
                label = row['participant.label']
                
                # Get player data
                player_cols = [col for col in columns if col.startswith(f'{segment_name}.{round_num}.player.')]
                if player_cols and not pd.isna(row.get(f'{segment_name}.{round_num}.player.id_in_group')):
                    
                    player = Player(participant_id, label, int(row[f'{segment_name}.{round_num}.player.id_in_group']))
                    
                    # Add player attributes
                    for col in player_cols:
                        attr_name = col.split('.')[-1]
                        value = row[col]
                        if pd.notna(value):
                            if attr_name == 'payoff':
                                player.payoff = float(value)
                            elif attr_name == 'contribution':
                                player.contribution = float(value) if value != '' else None
                            elif attr_name == 'role':
                                player.role = value
                            else:
                                player.data[attr_name] = value
                    
                    players_in_round[label] = player
                    
                    # Get group data - use group.id_in_subsession as the actual group identifier
                    group_id_col = f'{segment_name}.{round_num}.group.id_in_subsession'
                    if group_id_col in df.columns and not pd.isna(row[group_id_col]):
                        group_id = int(row[group_id_col])
                        if group_id not in groups_in_round:
                            groups_in_round[group_id] = Group(group_id)
                        
                        groups_in_round[group_id].add_player(player)
            
            # Add group-level data
            for group_id, group in groups_in_round.items():
                group_cols = [col for col in columns if col.startswith(f'{segment_name}.{round_num}.group.')]
                if group_cols:
                    # Use first player in group to get group data
                    first_player_row = None
                    for _, row in df.iterrows():
                        if (not pd.isna(row.get(f'{segment_name}.{round_num}.player.id_in_group')) and
                            int(row[f'{segment_name}.{round_num}.player.id_in_group']) == group_id):
                            first_player_row = row
                            break
                    
                    if first_player_row is not None:
                        for col in group_cols:
                            attr_name = col.split('.')[-1]
                            value = first_player_row[col]
                            if pd.notna(value):
                                if attr_name == 'total_contribution':
                                    group.total_contribution = float(value)
                                elif attr_name == 'individual_share':
                                    group.individual_share = float(value)
                                else:
                                    group.data[attr_name] = value
                
                round_obj.add_group(group)
            
            segment.add_round(round_obj)
        
        session.add_segment(segment)
    
    # Assign chat messages to groups and players
    if chat_data:
        assign_chat_messages(session, chat_data)
    
    return session

def load_experiment_data(file_pairs: List[Tuple[str, Optional[str], int]], name: Optional[str] = None) -> Experiment:
    """Load experimental data from multiple sessions into an Experiment object.
    
    Args:
        file_pairs: List of (all_data_csv_path, chat_data_csv_path, treatment) tuples. 
                   chat_data_csv_path can be None if no chat data is available.
                   treatment should be 1 or 2 to indicate treatment condition.
        name: Optional experiment name.
        
    Returns:
        Experiment object containing all sessions' data
    """
    experiment = Experiment(name=name)
    
    for csv_path, chat_csv_path, treatment in file_pairs:
        print(f"Loading session data from: {csv_path} (Treatment {treatment})")
        session = _load_single_session_data(csv_path, chat_csv_path, treatment)
        experiment.add_session(session)
        print(f"Added session: {session.session_code} (Treatment {treatment})")
    
    print(f"Experiment '{experiment.name}' loaded with {len(experiment.sessions)} sessions")
    return experiment


def main():
    """Example usage of the experiment data structure with multi-session loading."""
    
    DATA = os.environ.get('lpcp_data')
    
    d = [
        (file, int(m.group(1)))
        for file in os.listdir(DATA)
        if (m := re.search(r'_t(\d+)', file)) and 'data' in file
    ]
    # Build file_pairs of (csv_path, chat_csv_path, treatment)
    file_pairs: List[Tuple[str, Optional[str], int]] = []
    for i,file in enumerate(d):
        nums = re.findall(r'\d+', file[0])
        first_two_numbers = nums[:2] if len(nums) >= 2 else nums
        chat_file = f"{first_two_numbers[0]}_t{file[1]}_chat.csv"
        chat_path = os.path.join(DATA, chat_file) if os.path.exists(os.path.join(DATA, chat_file)) else None
        data_path = os.path.join(DATA, file[0])
        file_pairs.append((data_path, chat_path, file[1]))
    
    
    experiment = load_experiment_data(file_pairs, name="Demo Experiment")
    
    print(f"\nLoaded experiment: {experiment.name}")
    print(f"Sessions: {list(experiment.sessions.keys())}")
    
    # Example access patterns
    print("\n=== Example Access Patterns ===")
    
    # Access individual sessions within the experiment
    for session_code in experiment.list_session_codes():
        session = experiment.get_session(session_code)
        if session:
            print(f"\nSession {session_code} (Treatment {session.treatment}):")
            print(f"  Participants: {list(session.participant_labels.values())}")
            print(f"  Segments: {list(session.segments.keys())}")
            
            # Access a specific player's contribution in supergame 1, round 1
            supergame1 = session.get_supergame(1)
            if supergame1:
                round1 = supergame1.get_round(1)
                if round1:
                    player_a = round1.get_player('A')
                    if player_a:
                        print(f"  Player A's contribution in Supergame 1, Round 1: {player_a.contribution}")
    
    # Experiment-level aggregation
    print("\n=== Experiment-level Aggregation ===")
    
    overall_sent = experiment.get_overall_sentiment()
    if overall_sent:
        print(f"Overall chat sentiment across sessions: {overall_sent}")
    else:
        print("No chat messages found across sessions")
    
    df = experiment.to_dataframe_contributions()
    if df is not None:
        print(f"Contribution DataFrame shape: {df.shape}")
        print(f"Sessions in data: {df['session_code'].unique()}")
        print(f"Treatments in data: {df['treatment'].unique()}")
        print(f"Segments in data: {df['segment'].unique()}")
    else:
        print("No contribution data found")


if __name__ == '__main__':
    main()