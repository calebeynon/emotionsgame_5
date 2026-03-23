"""
Liar communication strategy classifier for chat messages.

Classifies each liar's chat messages into communication strategy categories
using OpenAI GPT-5.4. A "liar" (lied_this_round_20) is a player who made a
promise AND contributed < 20 points in THAT SPECIFIC round (not cumulatively
flagged).

These categories capture the heterogeneous ways liars communicate after
breaking promises — strategic, emotional, and rhetorical patterns — rather
than measuring "guilt" per se.

Chat Pairing Semantics:
    Messages in promise_classifications.csv on round R are chat that
    happened AFTER round R-1's contribution. The contribution column
    is the contribution for round R (the one influenced by the chat).

Output columns per liar instance (column names preserved for compatibility):
    - genuine_guilt: Sincere apology/remorse
    - false_promise: Stated contribution they didn't intend to make
    - blame_shifting: Accused others while defecting themselves
    - manipulation: Directed others' behavior, rotation schemes
    - self_justification: Rationalized own defection
    - deflection_collective: "We all should..." diffusion of responsibility
    - duping_delight: Appeared amused/happy while deceiving (text-based only)
    - performative_frustration: Acted upset while being a defector
    - no_guilt: No strategy-related content detected

Author: Claude Code
Date: 2026-03-17
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI, RateLimitError, APIError

from classify_guilt_prompt import SYSTEM_PROMPT

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


# =====
# Logging
# =====
def log(msg: str):
    print(msg, flush=True)


# =====
# Prompt construction
# =====
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
        result = _attempt_classify(client, user_prompt, attempt)
        if result is not None:
            return result
    log("ERROR: Max retries exceeded for classification request")
    return {"categories": ["error"], "reasoning": "Max retries exceeded", "raw": ""}


def _call_api(client: OpenAI, user_prompt: str) -> str:
    """Call the model and return the raw response text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=300,
    )
    content = response.choices[0].message.content
    if content is None:
        raise APIError(
            "Content filtered: model returned None content",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            body=None,
        )
    return content.strip()


def _attempt_classify(client: OpenAI, user_prompt: str, attempt: int) -> dict | None:
    """Single classification attempt. Returns result dict or None to retry."""
    try:
        return parse_response(_call_api(client, user_prompt))
    except RateLimitError:
        time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))
        return None
    except APIError as e:
        if attempt == MAX_RETRIES - 1:
            return {"categories": ["error"], "reasoning": str(e), "raw": str(e)}
        time.sleep(BASE_DELAY_SECONDS * (2 ** attempt))
        return None


def _strip_markdown(raw: str) -> str:
    """Strip markdown code fences from model response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return text


def _validate_categories(categories: list[str]) -> list[str]:
    """Validate and resolve contradictions in category list."""
    valid = [c for c in categories if c in VALID_CATEGORIES]
    if not valid:
        return ["no_guilt"]
    if len(valid) > 1 and "no_guilt" in valid:
        return [c for c in valid if c != "no_guilt"]
    return valid


def parse_response(raw: str) -> dict:
    """Parse the JSON response from the model."""
    text = _strip_markdown(raw)
    try:
        parsed = json.loads(text)
        valid = _validate_categories(parsed.get("categories", []))
        return {"categories": valid, "reasoning": parsed.get("reasoning", ""), "raw": raw}
    except json.JSONDecodeError:
        log(f"ERROR: JSON decode failed for response: {raw[:100]}")
        return {"categories": ["parse_error"], "reasoning": raw, "raw": raw}


# =====
# Data loading
# =====
def _parse_msgs(val) -> list:
    """Parse a JSON-encoded message list, returning [] on missing/invalid."""
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return []


def load_liar_messages() -> pd.DataFrame:
    """Load liars and merge with their chat messages."""
    bc = pd.read_csv(BEHAVIOR_FILE)
    liars = bc[bc["lied_this_round_20"] == True].copy()
    pc = pd.read_csv(PROMISE_FILE, engine="python", on_bad_lines="warn")
    merge_keys = ["session_code", "segment", "round", "group", "label"]
    merged = liars.merge(
        pc[merge_keys + ["messages", "message_count"]],
        on=merge_keys,
        how="left",
        suffixes=("", "_pc"),
    )
    merged["msg_list"] = merged["messages"].apply(_parse_msgs)
    return merged


# =====
# Batch classification
# =====
def _run_parallel(to_classify: pd.DataFrame, client, max_workers: int) -> dict:
    """Submit classification tasks in parallel and return {idx: result}."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(classify_guilt, client, row["msg_list"], int(row["contribution"])): idx
            for idx, row in to_classify.iterrows()
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                log(f"ERROR: Classification failed for index {idx}: {exc}")
                results[idx] = {"categories": ["error"], "reasoning": str(exc), "raw": ""}
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                log(f"  Classified {completed}/{len(futures)}")
    return results


def _build_result_columns(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Append classification columns to df from results dict."""
    category_cols = {cat: [] for cat in VALID_CATEGORIES}
    reasoning_col, raw_col = [], []
    for idx in df.index:
        res = results.get(idx)
        for cat in VALID_CATEGORIES:
            category_cols[cat].append(cat in res["categories"] if res else None)
        reasoning_col.append(res["reasoning"] if res else "")
        raw_col.append(res["raw"] if res else "")
    for cat in VALID_CATEGORIES:
        df[cat] = category_cols[cat]
    df["gpt_reasoning"] = reasoning_col
    df["gpt_raw_response"] = raw_col
    return df


def classify_all(df: pd.DataFrame, max_workers: int = 10) -> pd.DataFrame:
    """Classify all liar instances with chat messages."""
    client = get_client()
    has_msgs = df["msg_list"].apply(len) > 0
    to_classify = df[has_msgs].copy()
    log(f"Classifying {len(to_classify)} liar instances with chat messages...")
    results = _run_parallel(to_classify, client, max_workers)
    return _build_result_columns(df, results)


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
    log("LIAR COMMUNICATION STRATEGY SUMMARY")
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


if __name__ == "__main__":
    main()
