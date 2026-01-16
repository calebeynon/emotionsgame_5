import experiment_data
import os     
import re
from typing import List, Optional, Tuple

DATA = os.environ.get('lpcp_data')

def load_experiment(): 
    d = [
        (file, int(m.group(1)))
        for file in os.listdir(DATA)
        if (m := re.search(r'_t(\d+)', file)) and 'data' in file
    ]
        # Build file_pairs of (csv_path, chat_csv_path, treatment)
    file_pairs: List[Tuple[str, Optional[str], int]] = []
    for i,file in enumerate(d):
        # Extract the part before '_t' to get the session identifier
        # This handles both "11_t1_data.csv" -> "11" and "1_11_t1_data.csv" -> "1_11"
        filename = file[0]
        before_t = filename.split('_t')[0]
        # Use the entire part before '_t' as the session identifier
        # This preserves multi-digit numbers and compound identifiers
        session_id = before_t
        chat_file = f"{session_id}_t{file[1]}_chat.csv"
        chat_path = os.path.join(DATA, chat_file) if os.path.exists(os.path.join(DATA, chat_file)) else None
        data_path = os.path.join(DATA, file[0])
        file_pairs.append((data_path, chat_path, file[1]))
        
        
    experiment = experiment_data.load_experiment_data(file_pairs, name="Experiment")
    return experiment

df = experiment.to_dataframe_contributions()

