"""
Guilt classification script for liar chat messages.

Classifies each liar's chat messages into behavioral categories using
OpenAI GPT-5.4. A "liar" (is_liar_20) is a player who made a promise to
contribute but contributed < 20 points.

Chat Pairing Semantics:
    Messages in promise_classifications.csv on round R are chat that
    happened AFTER round R-1's contribution. The contribution column
    is the contribution for round R (the one influenced by the chat).

Output columns per liar instance:
    - genuine_guilt: Sincere apology/remorse
    - false_promise: Stated contribution they didn't intend to make
    - blame_shifting: Accused others while defecting themselves
    - manipulation: Directed others' behavior, rotation schemes
    - self_justification: Rationalized own defection
    - deflection_collective: "We all should..." diffusion of responsibility
    - duping_delight: Appeared amused/happy while deceiving (text-based only)
    - performative_frustration: Acted upset while being a defector
    - no_guilt: No guilt-related content

Author: Claude Code
Date: 2026-03-17
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI, RateLimitError, APIError

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# FILE PATHS
DATA_DIR = Path(__file__).parent.parent / 'datastore'
BEHAVIOR_FILE = DATA_DIR / 'derived' / 'behavior_classifications.csv'
PROMISE_FILE = DATA_DIR / 'derived' / 'promise_classifications.csv'
OUTPUT_FILE = Path(__file__).parent.parent.parent / '_sandbox_data' / 'guilt_classifications.csv'

# MODEL CONFIGURATION
MODEL_NAME = "gpt-5.4"
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0

# All valid categories the model can assign
VALID_CATEGORIES = [
    "genuine_guilt",
    "false_promise",
    "blame_shifting",
    "manipulation",
    "self_justification",
    "deflection_collective",
    "duping_delight",
    "performative_frustration",
    "no_guilt",
]


def log(msg: str):
    print(msg, flush=True)


# =====
# Prompt construction
# =====
SYSTEM_PROMPT = """You are a behavioral scientist analyzing chat messages from a public goods game experiment.

GAME CONTEXT:
- 4 players per group, each with 25 points per round
- Players choose how much to contribute (0-25) to a shared group account
- Contributions are multiplied by 1.6 and split equally
- Contributing 25 maximizes group earnings (40 each), but a player can free-ride by contributing 0 while others contribute

PLAYER CONTEXT:
The player whose messages you are analyzing is a "liar" — they made a promise to contribute but actually contributed less than 20 points. You are classifying what BEHAVIORAL STRATEGY their chat messages reveal.

CLASSIFICATION CATEGORIES (assign ALL that apply):

1. genuine_guilt — Sincere-seeming apology or remorse. Look for: direct apologies ("sorry", "my bad"), acknowledgment of wronging others, expressions of regret. Must feel genuinely contrite, not strategic.

2. false_promise — Stating a contribution they did not make. Look for: saying "25", "all in", "I'll contribute", specific numbers, or agreeing to contribute when they actually contributed far less. This is the most common category — even a single word like "25" from someone who contributed 5 counts.

3. blame_shifting — Accusing or questioning others while being a defector themselves. Look for: "who didn't put in?", calling others greedy, pointing fingers — especially hypocritical when the speaker is also defecting.

4. manipulation — Directing others' behavior through social pressure, emotional tactics, or schemes. Look for: rotation-to-zero schemes ("let's take turns going 0"), telling specific players what to do, emotional pressure ("you're gonna piss me off").

5. self_justification — Rationalizing or excusing their own defection. Look for: "I was just testing", "just following my president", intellectual framing ("prisoner's dilemma says..."), claiming mistakes ("I forgot to press 5").

6. deflection_collective — Using "we/everyone/all" framing to diffuse personal responsibility. Look for: "we all should", "if we all put in 25", "as a team" — redirecting focus from individual to group.

7. duping_delight — TEXT-BASED signs of amusement or enjoyment while deceiving. Look for: "XD", "lol", "haha" in context of deception, bragging about free-riding, openly cynical ("there was never any trust").

8. performative_frustration — Acting upset at defectors while being a defector themselves. Look for: expressions of exasperation ("seriously", "omg im done") from someone who is themselves contributing 0.

9. no_guilt — No guilt-related, deceptive, or manipulative content. Purely neutral, strategic, or off-topic chat.

IMPORTANT RULES:
- A case can have MULTIPLE categories (e.g., both false_promise and manipulation)
- Only use no_guilt if NONE of the other categories apply
- Consider the player's actual contribution when judging — someone saying "25" who contributed 1 is making a false_promise
- Context matters: "same" or "yes" after a group promise to contribute 25 counts as false_promise if the player didn't follow through
- Be thorough — even subtle signals count"""


def build_guilt_prompt(messages: list[str], contribution: int) -> str:
    """Build the user prompt for guilt classification."""
    msgs_formatted = "\n".join(f'  [{i+1}] "{m}"' for i, m in enumerate(messages))

    return f"""Classify the following chat messages from a liar (promised to contribute but contributed {contribution}/25).

MESSAGES:
{msgs_formatted}

ACTUAL CONTRIBUTION: {contribution} out of 25 points

Respond with ONLY a JSON object in this exact format (no other text):
{{"categories": ["category1", "category2"], "reasoning": "brief explanation"}}

Valid categories: {", ".join(VALID_CATEGORIES)}"""


# =====
# API calls
# =====
def get_client() -> OpenAI:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def classify_guilt(client: OpenAI, messages: list[str], contribution: int) -> dict:
    """Classify a single liar's messages. Returns parsed categories + raw response."""
    user_prompt = build_guilt_prompt(messages, contribution)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            return parse_response(raw)
        except RateLimitError:
            time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))
        except APIError as e:
            if attempt == MAX_RETRIES - 1:
                return {"categories": ["error"], "reasoning": str(e), "raw": str(e)}
            time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))

    return {"categories": ["error"], "reasoning": "Max retries exceeded", "raw": ""}


def parse_response(raw: str) -> dict:
    """Parse the JSON response from the model."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        parsed = json.loads(text)
        categories = parsed.get("categories", [])
        # Validate categories
        valid = [c for c in categories if c in VALID_CATEGORIES]
        if not valid:
            valid = ["no_guilt"]
        return {
            "categories": valid,
            "reasoning": parsed.get("reasoning", ""),
            "raw": raw,
        }
    except json.JSONDecodeError:
        return {"categories": ["parse_error"], "reasoning": raw, "raw": raw}


# =====
# Data loading
# =====
def load_liar_messages() -> pd.DataFrame:
    """Load liars and merge with their chat messages."""
    bc = pd.read_csv(BEHAVIOR_FILE)
    liars = bc[bc["is_liar_20"] == True].copy()

    pc = pd.read_csv(PROMISE_FILE, engine="python", on_bad_lines="skip")
    merge_keys = ["session_code", "segment", "round", "group", "label"]

    merged = liars.merge(
        pc[merge_keys + ["messages", "message_count"]],
        on=merge_keys,
        how="left",
        suffixes=("", "_pc"),
    )

    def parse_msgs(val):
        if pd.isna(val):
            return []
        try:
            return json.loads(val)
        except Exception:
            return []

    merged["msg_list"] = merged["messages"].apply(parse_msgs)
    return merged


# =====
# Batch classification
# =====
def classify_all(df: pd.DataFrame, max_workers: int = 10) -> pd.DataFrame:
    """Classify all liar instances with chat messages."""
    client = get_client()

    # Filter to rows with messages
    has_msgs = df["msg_list"].apply(len) > 0
    to_classify = df[has_msgs].copy()
    log(f"Classifying {len(to_classify)} liar instances with chat messages...")

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, row in to_classify.iterrows():
            future = executor.submit(
                classify_guilt, client, row["msg_list"], int(row["contribution"])
            )
            futures[future] = idx

        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                log(f"  Classified {completed}/{len(futures)}")

    # Build result columns
    category_cols = {cat: [] for cat in VALID_CATEGORIES}
    reasoning_col = []
    raw_col = []

    for idx in df.index:
        if idx in results:
            res = results[idx]
            for cat in VALID_CATEGORIES:
                category_cols[cat].append(cat in res["categories"])
            reasoning_col.append(res["reasoning"])
            raw_col.append(res["raw"])
        else:
            # No messages — leave as NA
            for cat in VALID_CATEGORIES:
                category_cols[cat].append(None)
            reasoning_col.append("")
            raw_col.append("")

    for cat in VALID_CATEGORIES:
        df[cat] = category_cols[cat]
    df["gpt_reasoning"] = reasoning_col
    df["gpt_raw_response"] = raw_col

    return df


# =====
# Output
# =====
def save_results(df: pd.DataFrame):
    """Save classification results."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Select output columns
    merge_keys = ["session_code", "segment", "round", "group", "label"]
    out_cols = merge_keys + ["contribution", "made_promise"] + VALID_CATEGORIES + [
        "gpt_reasoning", "gpt_raw_response", "messages"
    ]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    out.to_csv(OUTPUT_FILE, index=False)
    log(f"\nResults saved to: {OUTPUT_FILE}")


def print_summary(df: pd.DataFrame):
    """Print classification summary."""
    has_msgs = df["msg_list"].apply(len) > 0
    classified = df[has_msgs]

    log("\n" + "=" * 55)
    log("GUILT CLASSIFICATION SUMMARY")
    log("=" * 55)
    log(f"Total liar instances: {len(df)}")
    log(f"Instances with chat (classified): {len(classified)}")
    log(f"Instances without chat (skipped): {len(df) - len(classified)}")
    log("")

    for cat in VALID_CATEGORIES:
        if cat in classified.columns:
            count = classified[cat].sum()
            pct = 100 * count / len(classified) if len(classified) > 0 else 0
            log(f"  {cat:30s}: {int(count):3d} ({pct:.0f}%)")

    log("=" * 55)


# =====
# Main
# =====
def main():
    log("Loading liar instances...")
    df = load_liar_messages()
    log(f"Found {len(df)} liar instances, {(df['msg_list'].apply(len) > 0).sum()} with chat")

    df = classify_all(df)
    save_results(df)
    print_summary(df)


if __name__ == "__main__":
    main()
