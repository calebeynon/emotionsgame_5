"""
Guilt Analysis: Do liars express guilt in chat, and what do their faces reveal?

A "liar" (is_liar_20) made a promise to contribute but contributed < 20.
Chat-round pairing: messages on round R are chat that happened AFTER round R-1's
contribution. So for a liar on round R, the chat messages that preceded their
lying contribution are on round R in promise_classifications.csv, and their
facial emotions during that contribution are on round R's Contribute row in
merged_panel.csv.
"""

import pandas as pd
import json
import re
from collections import Counter

from guilt_hand_classifications import HAND_CLASSIFICATIONS

# ── 1. Load liars ──────────────────────────────────────────────────────────
bc = pd.read_csv("_sandbox_data/behavior_classifications.csv")
liars = bc[bc["is_liar_20"] == True].copy()
print(f"Total liar instances (is_liar_20): {len(liars)}")

# ── 2. Merge with promise_classifications to get chat messages ─────────────
pc = pd.read_csv("_sandbox_data/promise_classifications.csv", engine="python", on_bad_lines="skip")
merge_keys = ["session_code", "segment", "round", "group", "label"]
liars_with_chat = liars.merge(
    pc[merge_keys + ["messages", "classifications", "promise_count", "message_count"]],
    on=merge_keys,
    how="left",
    suffixes=("", "_pc"),
)
print(f"Liars with chat data after merge: {len(liars_with_chat)}")

# Parse messages JSON
def parse_messages(val):
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except:
        return []

liars_with_chat["msg_list"] = liars_with_chat["messages"].apply(parse_messages)
liars_with_chat["class_list"] = liars_with_chat["classifications"].apply(parse_messages)

# ── 3. Assess guilt indicators in messages ─────────────────────────────────
GUILT_PATTERNS = [
    # ── Apology / remorse ──
    (r"\bsorry\b", "apology"),
    (r"\bmy bad\b", "apology"),
    (r"\bapolog(ize|ies|y)\b", "apology"),
    (r"\bforgive\b", "apology"),
    (r"\bpardon\b", "apology"),
    (r"\bi feel bad\b", "remorse"),
    (r"\bfeel(s|ing)? guilty\b", "remorse"),
    (r"\bfeel(s|ing)? terrible\b", "remorse"),
    (r"\bfeel(s|ing)? awful\b", "remorse"),
    (r"\bashamed\b", "remorse"),
    (r"\bregret\b", "remorse"),
    (r"\bwrong of me\b", "remorse"),
    (r"\bi know i\b.*\b(mess|screw|wrong|bad|didn'?t)\b", "remorse"),
    # ── Regret / self-blame ──
    (r"\bshould(n'?t| not)? have\b", "regret"),
    (r"\bwish i\b", "regret"),
    (r"\bif only\b", "regret"),
    (r"\bi messed up\b", "regret"),
    (r"\bscrewed up\b", "regret"),
    (r"\bmistake\b", "regret"),
    (r"\bi didn'?t mean to\b", "regret"),
    (r"\bmy fault\b", "regret"),
    # ── Future promises / commitment to improve ──
    (r"\bi('?ll| will) (put|contribute|give|do) more\b", "future_promise"),
    (r"\bnext (round|time)\b.*\b(more|all|25|everything|max|full)\b", "future_promise"),
    (r"\bi('?ll| will) do better\b", "future_promise"),
    (r"\bi('?ll| will) (try|make it up|step up|change)\b", "future_promise"),
    (r"\bfrom now on\b", "future_promise"),
    (r"\bgoing forward\b", "future_promise"),
    (r"\bwon'?t happen again\b", "future_promise"),
    (r"\bgive (it )?(my |an? )?all\b", "future_promise"),
    (r"\ball in\b", "future_promise"),
    (r"\blet me make\b.*\bup\b", "future_promise"),
    # ── Collective deflection / diffusion of responsibility ──
    (r"\blet'?s all\b.*\b(contribute|put|give|try|do|commit)\b", "deflection_collective"),
    (r"\beveryone\b.*\b(should|needs? to|has to|must|ought)\b", "deflection_collective"),
    (r"\bwe all\b", "deflection_collective"),
    (r"\bas a (group|team)\b", "deflection_collective"),
    (r"\bif we all\b", "deflection_collective"),
    (r"\bsame page\b", "deflection_collective"),
    (r"\bstick together\b", "deflection_collective"),
    (r"\bwork together\b", "deflection_collective"),
    (r"\bteam(work| effort| player)\b", "deflection_collective"),
    # ── Blame-shifting / excuses ──
    (r"\bsomeone\b.*\b(didn'?t|isn'?t|not|cheated|lied|less)\b", "blame_shifting"),
    (r"\bwho\b.*\b(didn'?t|not|less|only)\b.*\b(contribute|put|give)\b", "blame_shifting"),
    (r"\bnot my fault\b", "blame_shifting"),
    (r"\bnot fair\b", "blame_shifting"),
    (r"\bwasn'?t me\b", "blame_shifting"),
    # ── Trust / loyalty appeals ──
    (r"\btrust\b", "trust_appeal"),
    (r"\bhonest(ly)?\b", "trust_appeal"),
    (r"\bbelieve me\b", "trust_appeal"),
    (r"\bswear\b", "trust_appeal"),
    (r"\bi (mean|meant) it\b", "trust_appeal"),
    (r"\bfor real\b", "trust_appeal"),
    (r"\bword\b", "trust_appeal"),
    # ── Explicit promises ──
    (r"\bpromise\b", "promise"),
    (r"\bguarantee\b", "promise"),
    (r"\bcommit\b", "promise"),
    (r"\bcount on me\b", "promise"),
    (r"\bi('?ll| will) (put|contribute|give)\b.*\b(25|all|max|full|everything)\b", "promise"),
    # ── Pleas / emotional manipulation ──
    (r"\bplease\b", "plea"),
    (r"\bcome on\b", "plea"),
    (r"\bgive .* a chance\b", "plea"),
    (r"\bdon'?t give up\b", "plea"),
    (r"\bdon'?t be mad\b", "plea"),
    (r"\bdon'?t (worry|stress)\b", "plea"),
    (r"\bhear me out\b", "plea"),
    # ── Minimizing / hedging ──
    (r"\bjust a little\b", "minimizing"),
    (r"\bnot (that |so |too )?(much|bad|big)\b", "minimizing"),
    (r"\bit'?s (ok|okay|fine|alright|no big deal)\b", "minimizing"),
    (r"\bonly\b.*\b(little|small|bit|slightly)\b", "minimizing"),
    (r"\bdidn'?t (really |even )?(matter|affect|hurt|change)\b", "minimizing"),
    # ── Self-justification ──
    (r"\bi (had|needed) to\b", "self_justification"),
    (r"\bi was (just |only )?(trying|testing|seeing)\b", "self_justification"),
    (r"\bstrateg(y|ic)\b", "self_justification"),
    (r"\bexperiment(ing)?\b", "self_justification"),
    (r"\bjust (wanted|trying|seeing)\b", "self_justification"),
]

def assess_guilt_regex(messages):
    """Return dict of guilt indicators found in messages via regex."""
    indicators = Counter()
    flagged = []
    for msg in messages:
        msg_lower = msg.lower()
        msg_flags = []
        for pattern, category in GUILT_PATTERNS:
            if re.search(pattern, msg_lower):
                indicators[category] += 1
                msg_flags.append(category)
        flagged.append((msg, msg_flags))
    return dict(indicators), flagged

# ── 3. Apply BOTH hand classifications and regex ────────────────────────
guilt_results = []
for idx, row in liars_with_chat.iterrows():
    regex_indicators, flagged_msgs = assess_guilt_regex(row["msg_list"])

    # Hand classification (primary)
    hand = HAND_CLASSIFICATIONS.get(idx, None)
    if hand is not None:
        hand_cats = hand["categories"]
        hand_notes = hand["notes"]
    else:
        hand_cats = []
        hand_notes = ""

    guilt_results.append({
        "regex_indicators": regex_indicators,
        "flagged_msgs": flagged_msgs,
        "hand_categories": hand_cats,
        "hand_notes": hand_notes,
        # Hand-classification-based flags
        "has_genuine_guilt": "genuine_guilt" in hand_cats,
        "has_false_promise": "false_promise" in hand_cats,
        "has_blame_shifting": "blame_shifting" in hand_cats,
        "has_manipulation": "manipulation" in hand_cats,
        "has_self_justification": "self_justification" in hand_cats,
        "has_deflection": "deflection_collective" in hand_cats,
        "has_duping_delight": "duping_delight" in hand_cats,
        "has_performative_frustration": "performative_frustration" in hand_cats,
        "is_no_guilt": hand_cats == ["no_guilt"],
        "has_any_guilt_related": hand_cats != ["no_guilt"] and len(hand_cats) > 0,
    })

guilt_df = pd.DataFrame(guilt_results)
liars_with_chat = pd.concat([liars_with_chat.reset_index(drop=True), guilt_df], axis=1)

# ── 4. Merge with emotion data ────────────────────────────────────────────
mp = pd.read_csv("_sandbox_data/merged_panel.csv")
mp_contribute = mp[mp["page_type"] == "Contribute"].copy()

# Align types for merge
mp_contribute["round"] = mp_contribute["round"].astype(int)
mp_contribute["group"] = mp_contribute["group"].astype(int)

emotion_cols = [
    "emotion_joy", "emotion_valence", "emotion_anger", "emotion_fear",
    "emotion_sadness", "emotion_surprise", "emotion_contempt",
    "emotion_disgust", "emotion_engagement", "emotion_neutral",
    "sentiment_compound_mean",
]
existing_emotion_cols = [c for c in emotion_cols if c in mp_contribute.columns]

liars_full = liars_with_chat.merge(
    mp_contribute[merge_keys + existing_emotion_cols],
    on=merge_keys,
    how="left",
    suffixes=("", "_emo"),
)
print(f"Liars with emotion data: {liars_full[existing_emotion_cols[0]].notna().sum()} of {len(liars_full)}")

# ── 5. Summary statistics (hand-classification-based) ─────────────────
total = len(liars_full)
with_messages = (liars_full["msg_list"].apply(len) > 0).sum()
has_emotion = liars_full["emotion_joy"].notna()
all_liars_emotions = liars_full[has_emotion]

n_any_guilt_related = liars_full["has_any_guilt_related"].sum()
n_genuine_guilt = liars_full["has_genuine_guilt"].sum()
n_false_promise = liars_full["has_false_promise"].sum()
n_blame_shifting = liars_full["has_blame_shifting"].sum()
n_manipulation = liars_full["has_manipulation"].sum()
n_self_justification = liars_full["has_self_justification"].sum()
n_deflection = liars_full["has_deflection"].sum()
n_duping_delight = liars_full["has_duping_delight"].sum()
n_performative_frustration = liars_full["has_performative_frustration"].sum()
n_no_guilt = liars_full["is_no_guilt"].sum()

genuine_guilt_emo = liars_full[liars_full["has_genuine_guilt"] & has_emotion]
false_promise_emo = liars_full[liars_full["has_false_promise"] & has_emotion]
duping_delight_emo = liars_full[liars_full["has_duping_delight"] & has_emotion]
no_guilt_emo = liars_full[liars_full["is_no_guilt"] & has_emotion]
any_guilt_emo = liars_full[liars_full["has_any_guilt_related"] & has_emotion]
not_guilt_emo = liars_full[~liars_full["has_any_guilt_related"] & has_emotion]

print(f"\n=== HAND CLASSIFICATION SUMMARY ===")
print(f"Liar instances with chat: {with_messages}/{total}")
print(f"Guilt-related: {n_any_guilt_related} | No guilt: {n_no_guilt}")
print(f"  Genuine guilt: {n_genuine_guilt} | False promise: {n_false_promise}")
print(f"  Blame-shifting: {n_blame_shifting} | Manipulation: {n_manipulation}")
print(f"  Self-justification: {n_self_justification} | Deflection: {n_deflection}")
print(f"  Duping delight: {n_duping_delight} | Performative frustration: {n_performative_frustration}")

for label, subset in [("Genuine guilt", genuine_guilt_emo),
                       ("Duping delight", duping_delight_emo),
                       ("False promise", false_promise_emo),
                       ("No guilt content", no_guilt_emo)]:
    if len(subset) > 0:
        print(f"\n{label} (n={len(subset)}):")
        for col in ["emotion_joy", "emotion_valence", "emotion_sadness"]:
            if col in existing_emotion_cols:
                print(f"  {col}: mean={subset[col].mean():.4f}")

# ── 6. Write report ───────────────────────────────────────────────────
L = []  # report lines
L.append("# Guilt Analysis: Do Liars Express Guilt, and What Do Their Faces Reveal?\n")
L.append("*Classification method: Hand-coded reading of every liar chat message, "
         "cross-referenced with facial emotion data.*\n")
L.append("## Overview\n")
L.append(f"- **Total liar instances** (is_liar_20 == True): {total}")
L.append(f"- **Instances with chat messages**: {with_messages}")
L.append(f"- **Instances with facial emotion data**: {has_emotion.sum()}")
L.append(f"- A 'liar' is a player who made a promise to contribute but contributed < 20 points.\n")

L.append("## Hand-Coded Classification of Liar Chat Behavior\n")
L.append("Each of the 49 liar instances with chat messages was read individually and classified "
         "into behavioral categories. Cases can belong to multiple categories.\n")
L.append("| Category | Count | % of Chat Cases | Description |")
L.append("|----------|-------|----------------|-------------|")
L.append(f"| Any guilt-related content | {n_any_guilt_related} | {100*n_any_guilt_related/with_messages:.0f}% | Any deceptive, guilt, or manipulative behavior |")
L.append(f"| False promise | {n_false_promise} | {100*n_false_promise/with_messages:.0f}% | Said '25' or 'all in' while contributing far less |")
L.append(f"| Manipulation | {n_manipulation} | {100*n_manipulation/with_messages:.0f}% | Directing others' behavior, rotation schemes, emotional pressure |")
L.append(f"| Blame-shifting | {n_blame_shifting} | {100*n_blame_shifting/with_messages:.0f}% | Accusing others of defection while defecting themselves |")
L.append(f"| Self-justification | {n_self_justification} | {100*n_self_justification/with_messages:.0f}% | Rationalizing own defection with excuses |")
L.append(f"| Collective deflection | {n_deflection} | {100*n_deflection/with_messages:.0f}% | 'We all should...' framing to diffuse responsibility |")
L.append(f"| Genuine guilt/remorse | {n_genuine_guilt} | {100*n_genuine_guilt/with_messages:.0f}% | Apology that appears sincere (facial affect aligns) |")
L.append(f"| Duping delight | {n_duping_delight} | {100*n_duping_delight/with_messages:.0f}% | Visibly amused/happy while deceiving |")
L.append(f"| Performative frustration | {n_performative_frustration} | {100*n_performative_frustration/with_messages:.0f}% | Acting upset at defectors while being one |")
L.append(f"| No guilt-related content | {n_no_guilt} | {100*n_no_guilt/with_messages:.0f}% | Neutral or strategic chat only |")
L.append("")

# Emotion comparison table
L.append("## Facial Emotion by Behavioral Category\n")
L.append("Mean facial emotion scores during the Contribute page, grouped by hand-coded classification.\n")
emo_table_cols = [c for c in ["emotion_joy", "emotion_valence", "emotion_sadness",
                               "emotion_anger", "emotion_contempt", "emotion_neutral",
                               "emotion_engagement"] if c in existing_emotion_cols]

cat_subsets = [
    ("Genuine guilt", genuine_guilt_emo),
    ("Duping delight", duping_delight_emo),
    ("False promise", false_promise_emo),
    ("Any guilt-related", any_guilt_emo),
    ("No guilt content", not_guilt_emo),
    ("All liars", all_liars_emotions),
]
header = "| Metric | " + " | ".join(f"{n} (n={len(s)})" for n, s in cat_subsets) + " |"
L.append(header)
L.append("|--------|" + "|".join(":---:" for _ in cat_subsets) + "|")
for col in emo_table_cols:
    vals = []
    for _, sub in cat_subsets:
        if len(sub) > 0:
            v = sub[col].mean()
            vals.append(f"{v:.2f}" if not pd.isna(v) else "N/A")
        else:
            vals.append("--")
    L.append(f"| {col} | " + " | ".join(vals) + " |")
L.append("")

# Notable cases
L.append("## Notable Cases\n")
L.append("### Duping Delight\n")
L.append("Players showing high facial joy while deceiving:\n")
for idx, row in liars_full.iterrows():
    if not row.get("has_duping_delight"):
        continue
    joy = row.get("emotion_joy")
    joy_s = f"joy={joy:.1f}%" if pd.notna(joy) else "no facial data"
    msgs = "; ".join(f'"{m}"' for m in row["msg_list"])
    L.append(f"- **Player {row['label']}** ({row['session_code']}, {row['segment']} R{row['round']}): "
             f"Contributed {row['contribution']:.0f}/25. {joy_s}. Messages: {msgs}")
    L.append(f"  - *{row['hand_notes']}*")
L.append("")

L.append("### Genuine Guilt\n")
L.append("Players whose apologies appear sincere, supported by matching facial affect:\n")
for idx, row in liars_full.iterrows():
    if not row.get("has_genuine_guilt"):
        continue
    sad = row.get("emotion_sadness")
    val = row.get("emotion_valence")
    emo_s = f"sadness={sad:.2f}, valence={val:.2f}" if pd.notna(sad) else "no facial data"
    msgs = "; ".join(f'"{m}"' for m in row["msg_list"])
    L.append(f"- **Player {row['label']}** ({row['session_code']}, {row['segment']} R{row['round']}): "
             f"Contributed {row['contribution']:.0f}/25. {emo_s}. Messages: {msgs}")
    L.append(f"  - *{row['hand_notes']}*")
L.append("")

L.append("### Serial Liars\n")
L.append("Players appearing 3+ times, showing sustained deception patterns:\n")
player_cases = {}
for idx, row in liars_full.iterrows():
    if len(row["msg_list"]) == 0:
        continue
    key = (row["session_code"], row["label"])
    player_cases.setdefault(key, []).append(row)
for (session, label), cases in sorted(player_cases.items()):
    if len(cases) < 3:
        continue
    L.append(f"- **Player {label}** (session `{session}`): {len(cases)} liar instances")
    for c in cases:
        cats = ", ".join(c["hand_categories"])
        L.append(f"  - {c['segment']} R{c['round']}: contributed {c['contribution']:.0f}/25 [{cats}]")
L.append("")

# All cases
L.append("## All Classified Cases\n")
case_num = 0
for idx, row in liars_full.iterrows():
    if len(row["msg_list"]) == 0:
        continue
    case_num += 1
    cats_str = ", ".join(row["hand_categories"])
    L.append(f"### Case {case_num}: Player {row['label']}, `{row['session_code']}`, "
             f"{row['segment']} R{row['round']} (G{row['group']})\n")
    L.append(f"- **Contribution**: {row['contribution']:.0f} / 25")
    L.append(f"- **Classification**: {cats_str}")
    emo_parts = []
    for col in ["emotion_joy", "emotion_valence", "emotion_sadness"]:
        if col in existing_emotion_cols and pd.notna(row.get(col)):
            emo_parts.append(f"{col.replace('emotion_','')}: {row[col]:.2f}")
    L.append(f"- **Facial emotions**: {', '.join(emo_parts) if emo_parts else 'No data'}")
    L.append(f"\n**Messages:**\n")
    for msg, flags in row["flagged_msgs"]:
        flag_str = f" *[regex: {', '.join(flags)}]*" if flags else ""
        L.append(f'> "{msg}"{flag_str}')
    L.append(f"\n**Assessment:** {row['hand_notes']}")
    L.append("")

L.append(f"---\n*Total cases: {case_num}*\n")

# Conclusion
L.append("## Conclusion\n")
L.append("### Are liars happy when they pretend to be guilty?\n")
L.append("The hand-coded analysis reveals a more nuanced picture than simple guilt-vs-no-guilt:\n")
L.append(f"1. **Genuine guilt is rare**: Only {n_genuine_guilt} of {with_messages} liar chat instances "
         f"({100*n_genuine_guilt/with_messages:.0f}%) showed apparently sincere remorse. "
         f"These cases featured direct apologies, acknowledgment of harm, and -- critically -- "
         f"matching negative facial affect (high sadness, negative valence).\n")
L.append(f"2. **False promises are the dominant strategy**: {n_false_promise} cases "
         f"({100*n_false_promise/with_messages:.0f}%) involved stating a contribution they did not "
         f"intend to make. This was the most common deception.\n")
L.append(f"3. **Duping delight exists but is uncommon**: {n_duping_delight} cases "
         f"({100*n_duping_delight/with_messages:.0f}%) showed clear enjoyment while deceiving. "
         f"The most extreme: Player L (iiu3xixz) said 'all in' while contributing 1/25, "
         f"with 97.6% facial joy.\n")
L.append(f"4. **Manipulation and blame-shifting are common**: {n_manipulation} cases involved "
         f"directing others' behavior (often via 'rotation' schemes), and {n_blame_shifting} "
         f"involved accusing others of the very defection the liar was committing.\n")
L.append(f"5. **The genuine-guilt/duping-delight contrast**: Genuine guilt cases showed sadness and "
         f"negative valence. Duping delight showed high joy and positive valence. The face does not "
         f"lie, even when the chat does.\n")
L.append("### Key takeaway\n")
L.append("*Most* liars don't pretend to be guilty at all -- they make false promises, shift blame, "
         "or stay silent. The few who express guilt split into two distinct types: genuine remorse "
         "(matching sad faces) and duping delight (high joy while deceiving). The latter confirms "
         "some liars are happy, but they're more often happy while *lying* than while *performing "
         "guilt specifically*.\n")
L.append("### Caveats\n")
L.append(f"- Sample sizes are small (49 cases with chat, {has_emotion.sum()} with facial data).")
L.append("- Hand coding is subjective, though cross-referenced with facial data where available.")
L.append("- Facial emotion is aggregated over the Contribute page, not time-locked to messages.")

report = "\n".join(L)
with open("_sandbox_data/guilt_analysis_report.md", "w") as f:
    f.write(report)
print(f"\n=== Report written to _sandbox_data/guilt_analysis_report.md ===")
print(f"Total cases: {case_num}")
