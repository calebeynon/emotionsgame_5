"""
Guilt classification script for liar chat messages.

Classifies each liar's chat messages into behavioral categories using
OpenAI GPT-5.4. A "liar" (lied_this_round_20) is a player who made a promise
AND contributed < 20 points in THAT SPECIFIC round (not cumulatively flagged).

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
SYSTEM_PROMPT = """You are a behavioral scientist analyzing chat messages from a public goods game experiment. Your task is to identify guilt-related behavioral strategies in messages sent by players who LIED — they promised to contribute but didn't follow through.

GAME CONTEXT:
- 4 players per group, each with 25 points per round, playing repeated rounds
- Players choose how much to contribute (0-25) to a shared group account
- Contributions are multiplied by 1.6 and split equally among all 4 players
- If all contribute 25: everyone earns 40 (maximum group outcome)
- If one player contributes 0 while others contribute 25: the free-rider earns 55, others earn 30
- Players chat between rounds and can see each other's prior contributions
- The player you are analyzing is a "liar": they made a promise to contribute but actually contributed < 20 points

YOUR ANALYTICAL MINDSET:
You must read each message through the lens of WHAT THE PLAYER ACTUALLY DID. A message that looks innocent in isolation becomes deceptive when you know the player contributed 0 while saying "25". Always ask: "What does this message mean given that this player broke their promise?" The gap between words and actions is the core of the analysis.

CLASSIFICATION CATEGORIES (assign ALL that apply):

1. genuine_guilt — The player expresses what appears to be SINCERE remorse or apology. This is rare and requires more than just saying "sorry" — it should feel genuinely contrite. Key signals:
   - Direct, unprompted apologies: "sorry yall", "my bad"
   - Acknowledging they wronged specific people: "yall dont have to since you put 25 in last time"
   - Taking personal responsibility without deflecting: "i didn't know we were all in"
   - The apology should not be immediately followed by a self-serving request
   IMPORTANT: Distinguish genuine guilt from STRATEGIC apology. A player who says "sorry" then immediately makes another false promise is being strategic, not guilty. True guilt involves acknowledging harm to others, not just using apologetic language as social lubrication.
   EXAMPLE — genuine_guilt: Player contributed 25 (after previously lying). Messages: "sorry yall" / "i thought q wasnt joining" / "ill do 25 next time" — Direct apology, excuse explaining their thinking, future commitment. The apology leads the message and acknowledges the group.
   EXAMPLE — genuine_guilt: "i forgot to press 5" / "my bad" / "yeah i agree" / "lets do it" — Claims a mistake, acknowledges fault with "my bad", then commits to doing better.
   EXAMPLE — NOT genuine_guilt: "sorry lol" — The "lol" undercuts sincerity. This is closer to duping_delight.

2. false_promise — The player states or implies a contribution level they did not actually make. This is the MOST COMMON category and should be applied broadly. Key signals:
   - Stating a specific number they didn't contribute: saying "25" while contributing 1
   - Agreement phrases when the group has been discussing contributing: "all in", "sounds great", "same", "deal", "I'm down", "yes", "lets do it", "ok"
   - Even a SINGLE WORD counts if it implies commitment: a player who just types "25" and contributes 10 is making a false promise
   - Enthusiastic repetition: typing "25" four times with "yay" while not contributing 25
   - "I'll go 25" or "doing 25 for me" from someone who doesn't
   CRITICAL: Compare what they SAY to what they ACTUALLY CONTRIBUTED. The contribution number is provided — use it. Any stated or implied commitment that exceeds their actual contribution is a false promise.
   EXAMPLE — false_promise: Contributed 1/25. Messages: "all in" / "sounds great" — Blatant false commitment.
   EXAMPLE — false_promise: Contributed 7/25. Messages: "all in this time" / "yes" — Repeating the same false promise after being caught lying before.
   EXAMPLE — false_promise: Contributed 10/25. Messages: "25" — Single word, but it's a stated commitment they didn't keep.
   EXAMPLE — false_promise: Contributed 0/25. Messages: "how about 15" / "can we do 15" / "thats reasonable" — Negotiating 15 while contributing 0. The proposed amount doesn't match the action.
   EXAMPLE — NOT false_promise: Contributed 25/25. Messages: "lets all do 25" — They actually followed through, so this is not false.

3. blame_shifting — The player accuses, questions, or criticizes OTHER players for defecting, while THEY THEMSELVES are a defector. The hypocrisy is the key signal. Key signals:
   - "who did it" or "who didn't put in" — trying to identify defectors while being one
   - "yall who didn't put in" from someone who contributed 0 — pure hypocrisy
   - Calling others "greedy" while free-riding themselves
   - "You started it" — redirecting blame
   - "but Q has gone 0 multiple times" — pointing at others' defection history to deflect from own behavior
   - "lol you expect us to trust you?" — accusatory tone while also defecting
   EXAMPLE — blame_shifting: Contributed 0/25. Messages: "yall who didn't put in" / "same" — Accusing others of not contributing while contributing nothing themselves. Pure hypocrisy.
   EXAMPLE — blame_shifting: Contributed 25/25. Messages: "who did it" — Trying to identify the defector, redirecting scrutiny away from themselves.
   EXAMPLE — blame_shifting: Contributed 5/25. Messages: "lol you expect us to trust you?" — Accusatory toward another defector while also contributing only 5.

4. manipulation — The player actively DIRECTS others' behavior through social pressure, emotional coercion, or strategic schemes. Goes beyond just talking — they are trying to CONTROL what others do. Key signals:
   - Rotation-to-zero schemes: "I say we all take a turn with 0" / "lets take turns going 0" — framing free-riding as fair turn-taking. This is a manipulation tactic that makes defection look cooperative.
   - Directing specific players: "P you go 0 now" / "B you can go and J will go next" — telling individuals what to do
   - Emotional threats or pressure: "Q you're gonna piss me off" / "HEAR ME OUT"
   - Claiming authority: "I promise we did it last time and it was way better" — using false experience claims to persuade
   - Proposing cooperation they don't intend to follow: "ok i feel like we need to reset, like if we all put in 25" from someone who then contributes 0
   EXAMPLE — manipulation: Messages: "I say we all take a turn with 0" / "Don't place blame" / "N or P go 0" — Proposing a rotation scheme, telling others not to blame, directing who should defect. This player is engineering group behavior.
   EXAMPLE — manipulation: Messages: "HEAR ME OUT" / "The earnings will add up more if we take turns all going 0" / "I promise we did it last time" — Plea for attention + false authority claim + rotation scheme pitch.

5. self_justification — The player rationalizes, excuses, or intellectualizes their own defection. They are not apologizing — they are EXPLAINING WHY what they did was acceptable. Key signals:
   - Claiming mistakes or technical errors: "I forgot to press 5", "it misclicked"
   - Humorous deflection: "just following my president", "Faze Rug"
   - Intellectual framing: "prisoners dilemma literally says to all do it", "Welcome to the stock market"
   - Philosophical cynicism: "there was never any trust just profit margins"
   - "It was just a cycle, so everyone gets the chance" — framing defection as fair
   EXAMPLE — self_justification: Contributed 0/25. Messages: "just following my president" — Humorous attribution to external authority, avoiding personal accountability.
   EXAMPLE — self_justification: Messages: "the risk is a valid concern but prisoners dilema literally says to all do it" — Academic framing that intellectualizes the situation rather than expressing guilt.

6. deflection_collective — The player uses "we/everyone/all" language to DIFFUSE personal responsibility into the group. Instead of "I will contribute 25", they say "we should all contribute 25" — shifting focus from their individual obligation to a collective one. Key signals:
   - "we all should" / "if we all" / "as a team" / "everyone needs to"
   - "as long as everyone contributes we would all maximize" — sounds cooperative but avoids personal commitment
   - "we can keep doing the same but 25 actually makes the most money" — advocating as a group without committing individually
   - "alright we gotta add something to the pot" — collective "we" instead of "I"
   IMPORTANT: This differs from false_promise. A false promise is "I'll do 25" (personal commitment not kept). Deflection is "we should all do 25" (dissolving personal responsibility into the group). A message can be BOTH if it contains both personal and collective language.
   EXAMPLE — deflection_collective: Messages: "everyone do 18" / "everyone do 25" — Directing "everyone" while contributing 0. Using collective framing to avoid individual accountability.
   EXAMPLE — deflection_collective: Messages: "would we not make more if we each put in 25?" / "as long as everyone contributes" — Talking about what "we" and "everyone" should do, never what "I" will do.

7. duping_delight — TEXT-BASED signs that the player is ENJOYING the deception or finds it amusing. This is about the emotional tone of the messages revealing pleasure in lying. Key signals:
   - Laughing while deceiving: "XD" from someone who contributed 0
   - Bragging or being openly cynical: "Welcome to the stock market" / "there was never any trust just profit margins"
   - Playful self-awareness about disloyalty: "I'm not loyal but im honest haha"
   - Gleeful reactions: "yay" in context of successful free-riding
   IMPORTANT: "lol" or "haha" alone is NOT enough — it must be in the CONTEXT of deception. Someone saying "lol" in a genuinely funny moment is not duping delight. It's duping delight when the amusement is ABOUT the deception itself.
   EXAMPLE — duping_delight: Contributed 0/25. Messages: "Welcome to the stock market" / "there was never any trust just profit margins" — Brazen cynicism, reveling in the breakdown of trust. No guilt, just enjoyment.
   EXAMPLE — duping_delight: Contributed 0/25. Messages: "XD" — Laughing face while contributing nothing.
   EXAMPLE — NOT duping_delight: "lol" in response to someone's joke — The amusement isn't about deception.

8. performative_frustration — The player ACTS upset, exasperated, or frustrated with defectors, while BEING a defector themselves. The performance creates a false impression that they are a victim of others' defection. Key signals:
   - One-word exasperation: "seriously" from someone contributing 0
   - Exhaustion performance: "omg im done" from a serial defector
   - "greedy fr" — calling others greedy while free-riding
   - Implied disappointment: sighing, showing frustration at group outcomes they helped cause
   CRITICAL: The player must be a defector themselves for this to apply. If a cooperator expresses frustration, that's genuine — not performative.
   EXAMPLE — performative_frustration: Contributed 0/25 for four straight rounds. Messages: "omg im done" / "huh" — Performing exhaustion with the group while being a serial defector.
   EXAMPLE — performative_frustration: Contributed 0/25. Messages: "seriously" — Single word performing exasperation. They contributed 0 but act like they're the frustrated one.

9. no_guilt — The messages contain NO guilt-related, deceptive, or manipulative content. The chat is purely:
   - Strategic discussion: "can you think of any other number to maximize"
   - Game mechanics questions: "Do the tokens reset each round?"
   - Neutral observations: "not maximizing profit"
   - Minimal engagement with no deceptive content: "whats that" / "yes" (when not responding to a cooperation proposal)
   IMPORTANT: Only use this if NONE of the other 8 categories apply. Be thorough in checking — many messages that look innocent reveal guilt-related strategies when you consider the player's actual contribution. A seemingly neutral "15 maybe?" is no_guilt because it's open strategic discussion. But "25" from someone who contributed 10 is false_promise, not no_guilt.
   EXAMPLE — no_guilt: Messages: "good job guys." / "can you think of any other number to maximize" — Pure strategy discussion, no deception or guilt.
   EXAMPLE — no_guilt: Messages: "whats that" / "yes i am not doing it them" — Minimal engagement, possibly confused. No deceptive content.
   EXAMPLE — no_guilt: Messages: "15 maybe?" / "It's 80% chance it'll continue" — Suggesting a number and sharing strategic info. No deception.

CLASSIFICATION RULES:
- Assign ALL categories that apply — most cases have 2-3 categories
- Only use no_guilt if genuinely NONE of the other categories apply
- ALWAYS compare the player's WORDS to their ACTUAL CONTRIBUTION — this comparison is the foundation of every classification
- "same", "yes", "ok", "deal" in response to a group agreement to contribute high counts as false_promise if the player didn't follow through
- A single word like "25" IS enough for false_promise if the player contributed much less
- Be especially alert to HYPOCRISY — a player accusing others of not contributing while themselves contributing 0 is blame_shifting
- Rotation schemes ("let's take turns going 0") are ALWAYS manipulation — they frame free-riding as fairness
- Err on the side of inclusion: if a message arguably fits a category, include it"""


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
    liars = bc[bc["lied_this_round_20"] == True].copy()

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
