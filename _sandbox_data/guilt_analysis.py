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
    (r"\bsorry\b", "apology"),
    (r"\bmy bad\b", "apology"),
    (r"\bapologize\b", "apology"),
    (r"\bi feel bad\b", "remorse"),
    (r"\bfeel guilty\b", "remorse"),
    (r"\bshould have\b", "regret"),
    (r"\bshouldn'?t have\b", "regret"),
    (r"\bi('?ll| will) (put|contribute|give|do) more\b", "future_promise"),
    (r"\bnext (round|time)\b.*\b(more|all|25|everything)\b", "future_promise"),
    (r"\bi('?ll| will) do better\b", "future_promise"),
    (r"\blet'?s all\b.*\b(contribute|put|give)\b", "deflection_collective"),
    (r"\beveryone\b.*\b(should|needs? to|has to)\b", "deflection_collective"),
    (r"\bwe all\b", "deflection_collective"),
    (r"\btrust\b", "trust_appeal"),
    (r"\bpromise\b", "promise"),
    (r"\bplease\b", "plea"),
    (r"\bcome on\b", "plea"),
]

def assess_guilt(messages):
    """Return dict of guilt indicators found in messages."""
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

guilt_results = []
for idx, row in liars_with_chat.iterrows():
    indicators, flagged_msgs = assess_guilt(row["msg_list"])
    guilt_results.append({
        "indicators": indicators,
        "flagged_msgs": flagged_msgs,
        "has_guilt": bool(indicators),
        "has_apology": indicators.get("apology", 0) > 0 or indicators.get("remorse", 0) > 0,
        "has_future_promise": indicators.get("future_promise", 0) > 0 or indicators.get("promise", 0) > 0,
        "has_deflection": indicators.get("deflection_collective", 0) > 0,
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

# ── 5. Summary statistics ─────────────────────────────────────────────────
total = len(liars_full)
with_messages = (liars_full["msg_list"].apply(len) > 0).sum()
with_any_guilt = liars_full["has_guilt"].sum()
with_apology = liars_full["has_apology"].sum()
with_future_promise = liars_full["has_future_promise"].sum()
with_deflection = liars_full["has_deflection"].sum()

has_emotion = liars_full["emotion_joy"].notna()
guilty_liars_emotions = liars_full[liars_full["has_guilt"] & has_emotion]
non_guilty_liars_emotions = liars_full[~liars_full["has_guilt"] & has_emotion]
all_liars_emotions = liars_full[has_emotion]

print(f"\n=== GUILT INDICATOR SUMMARY ===")
print(f"Liar instances with chat messages: {with_messages}/{total}")
print(f"Instances with ANY guilt indicator: {with_any_guilt}/{total}")
print(f"  - Apology/remorse: {with_apology}")
print(f"  - Future promises: {with_future_promise}")
print(f"  - Collective deflection: {with_deflection}")

if len(guilty_liars_emotions) > 0 and len(non_guilty_liars_emotions) > 0:
    print(f"\n=== EMOTION COMPARISON ===")
    print(f"Guilty-expressing liars (n={len(guilty_liars_emotions)}):")
    for col in ["emotion_joy", "emotion_valence", "emotion_sadness"]:
        if col in existing_emotion_cols:
            print(f"  {col}: mean={guilty_liars_emotions[col].mean():.4f}")
    print(f"Non-guilty-expressing liars (n={len(non_guilty_liars_emotions)}):")
    for col in ["emotion_joy", "emotion_valence", "emotion_sadness"]:
        if col in existing_emotion_cols:
            print(f"  {col}: mean={non_guilty_liars_emotions[col].mean():.4f}")

# ── 6. Write report ───────────────────────────────────────────────────────
lines = []
lines.append("# Guilt Analysis: Do Liars Express Guilt, and What Do Their Faces Reveal?\n")
lines.append("## Overview\n")
lines.append(f"- **Total liar instances** (is_liar_20 == True): {total}")
lines.append(f"- **Instances with chat messages**: {with_messages}")
lines.append(f"- **Instances with facial emotion data**: {has_emotion.sum()}")
lines.append(f"- A 'liar' is a player who made a promise to contribute but contributed < 20 points.\n")

lines.append("## Guilt Indicators in Chat Messages\n")
lines.append(f"| Indicator | Count | % of Liars |")
lines.append(f"|-----------|-------|------------|")
lines.append(f"| Any guilt indicator | {with_any_guilt} | {100*with_any_guilt/total:.1f}% |")
lines.append(f"| Apology/remorse | {with_apology} | {100*with_apology/total:.1f}% |")
lines.append(f"| Future promises (\"I'll do better\") | {with_future_promise} | {100*with_future_promise/total:.1f}% |")
lines.append(f"| Collective deflection (\"we all should\") | {with_deflection} | {100*with_deflection/total:.1f}% |")
lines.append("")

# Emotion comparison table
lines.append("## Facial Emotion Comparison\n")
if len(guilty_liars_emotions) > 0 and len(non_guilty_liars_emotions) > 0:
    lines.append(f"Comparing facial emotions during the Contribute page for liars who expressed guilt-like language vs. those who did not.\n")
    lines.append(f"| Metric | Guilt-Expressing (n={len(guilty_liars_emotions)}) | Non-Guilt (n={len(non_guilty_liars_emotions)}) | All Liars (n={len(all_liars_emotions)}) |")
    lines.append(f"|--------|:---:|:---:|:---:|")
    for col in ["emotion_joy", "emotion_valence", "emotion_sadness", "emotion_anger", "emotion_contempt", "emotion_neutral", "emotion_engagement", "sentiment_compound_mean"]:
        if col in existing_emotion_cols:
            g_mean = guilty_liars_emotions[col].mean()
            ng_mean = non_guilty_liars_emotions[col].mean()
            all_mean = all_liars_emotions[col].mean()
            g_val = f"{g_mean:.4f}" if not pd.isna(g_mean) else "N/A"
            ng_val = f"{ng_mean:.4f}" if not pd.isna(ng_mean) else "N/A"
            all_val = f"{all_mean:.4f}" if not pd.isna(all_mean) else "N/A"
            lines.append(f"| {col} | {g_val} | {ng_val} | {all_val} |")
    lines.append("")
elif len(all_liars_emotions) > 0:
    lines.append("Not enough data to split by guilt expression. Overall liar emotion means:\n")
    for col in existing_emotion_cols:
        val = all_liars_emotions[col].mean()
        lines.append(f"- **{col}**: {val:.4f}" if not pd.isna(val) else f"- **{col}**: N/A")
    lines.append("")
else:
    lines.append("No facial emotion data available for liar instances.\n")

# Detailed case studies
lines.append("## Detailed Liar Cases\n")
lines.append("Below are all liar instances where the player sent chat messages, showing their messages, guilt indicators, and facial emotion scores.\n")

case_num = 0
for idx, row in liars_full.iterrows():
    if len(row["msg_list"]) == 0:
        continue
    case_num += 1
    lines.append(f"### Case {case_num}: Session `{row['session_code']}`, {row['segment']}, Round {row['round']}, Player {row['label']} (Group {row['group']})\n")
    lines.append(f"- **Contribution**: {row['contribution']:.0f} / 25")
    lines.append(f"- **Promise made**: Yes (is_liar_20)")

    # Emotions
    emo_parts = []
    for col in ["emotion_joy", "emotion_valence", "emotion_sadness", "emotion_contempt", "emotion_neutral"]:
        if col in existing_emotion_cols and pd.notna(row.get(col)):
            emo_parts.append(f"{col.replace('emotion_','')}: {row[col]:.4f}")
    if emo_parts:
        lines.append(f"- **Facial emotions** (Contribute page): {', '.join(emo_parts)}")
    else:
        lines.append(f"- **Facial emotions**: No data available")

    if row.get("sentiment_compound_mean") and pd.notna(row.get("sentiment_compound_mean")):
        lines.append(f"- **Chat sentiment** (VADER compound mean): {row['sentiment_compound_mean']:.4f}")

    # Messages
    lines.append(f"\n**Messages** ({len(row['msg_list'])} total):\n")
    for msg, flags in row["flagged_msgs"]:
        flag_str = f" **[{', '.join(flags)}]**" if flags else ""
        lines.append(f"> \"{msg}\"{flag_str}")

    # Guilt assessment
    if row["has_guilt"]:
        ind_strs = [f"{k} ({v})" for k, v in row["indicators"].items()]
        lines.append(f"\nGuilt indicators detected: {', '.join(ind_strs)}")
    else:
        lines.append(f"\nNo explicit guilt indicators detected.")
    lines.append("")

lines.append(f"---\n*Total cases with messages shown: {case_num}*\n")

# Conclusion
lines.append("## Conclusion: Are Liars Happy When They Pretend to Be Guilty?\n")

# Build conclusion from data
if len(guilty_liars_emotions) > 0 and len(non_guilty_liars_emotions) > 0:
    joy_guilt = guilty_liars_emotions["emotion_joy"].mean()
    joy_no_guilt = non_guilty_liars_emotions["emotion_joy"].mean()
    val_guilt = guilty_liars_emotions["emotion_valence"].mean()
    val_no_guilt = non_guilty_liars_emotions["emotion_valence"].mean()

    joy_diff = joy_guilt - joy_no_guilt
    val_diff = val_guilt - val_no_guilt

    lines.append(f"### Key Findings\n")
    lines.append(f"1. **Guilt expression is {'common' if with_any_guilt/total > 0.3 else 'uncommon' if with_any_guilt/total > 0.1 else 'rare'}**: "
                 f"Only {with_any_guilt} of {total} liar instances ({100*with_any_guilt/total:.1f}%) contained any guilt-related language "
                 f"(apologies, remorse, future promises, or collective deflection).\n")

    lines.append(f"2. **Dominant strategy -- {'deflection and future promises' if with_deflection + with_future_promise > with_apology else 'direct apology' if with_apology > 0 else 'silence'}**: "
                 f"Rather than expressing genuine guilt, liars more commonly "
                 f"{'used collective deflection (\"we all should...\") or made promises for future rounds' if with_deflection + with_future_promise > with_apology else 'stayed silent or used neutral language'}.\n")

    if not pd.isna(joy_guilt) and not pd.isna(joy_no_guilt):
        if joy_diff > 0.005:
            joy_finding = (f"Liars who used guilt-related language showed **higher** facial joy "
                          f"({joy_guilt:.4f}) compared to non-guilt-expressing liars ({joy_no_guilt:.4f}), "
                          f"suggesting their guilt expressions may not reflect genuine remorse.")
        elif joy_diff < -0.005:
            joy_finding = (f"Liars who used guilt-related language showed **lower** facial joy "
                          f"({joy_guilt:.4f}) compared to non-guilt-expressing liars ({joy_no_guilt:.4f}), "
                          f"which could indicate some genuine discomfort.")
        else:
            joy_finding = (f"Facial joy was similar between guilt-expressing ({joy_guilt:.4f}) and "
                          f"non-guilt-expressing liars ({joy_no_guilt:.4f}), making it hard to distinguish "
                          f"genuine from performed guilt.")
        lines.append(f"3. **Facial emotions**: {joy_finding}\n")

    if not pd.isna(val_guilt) and not pd.isna(val_no_guilt):
        if val_diff > 0.005:
            val_finding = (f"Emotional valence was **more positive** for guilt-expressing liars "
                          f"({val_guilt:.4f} vs {val_no_guilt:.4f}), consistent with the hypothesis "
                          f"that liars feel satisfaction (duping delight) even while performing guilt.")
        elif val_diff < -0.005:
            val_finding = (f"Emotional valence was **more negative** for guilt-expressing liars "
                          f"({val_guilt:.4f} vs {val_no_guilt:.4f}), suggesting some liars may "
                          f"experience genuine negative affect when breaking promises.")
        else:
            val_finding = (f"Emotional valence was similar ({val_guilt:.4f} vs {val_no_guilt:.4f}).")
        lines.append(f"4. **Valence**: {val_finding}\n")

    lines.append("### Interpretation\n")
    if joy_diff > 0.005 and not pd.isna(joy_diff):
        lines.append("The evidence tentatively supports the 'duping delight' hypothesis: liars who deploy "
                     "guilt-laden language in chat display **more positive facial affect**, not less. "
                     "This pattern is consistent with strategic guilt performance -- players who verbally "
                     "express remorse appear to be emotionally unbothered, or even pleased, during the "
                     "contribution decision. The guilt expression functions as a social tool for maintaining "
                     "group cooperation norms while personally free-riding.\n")
    elif joy_diff < -0.005 and not pd.isna(joy_diff):
        lines.append("The evidence does not clearly support 'duping delight.' Liars who express guilt "
                     "show somewhat lower joy, which may indicate genuine discomfort with norm violation. "
                     "However, sample sizes are small and the differences may not be statistically significant. "
                     "The guilt expression could reflect real internal conflict rather than pure strategic manipulation.\n")
    else:
        lines.append("The facial emotion evidence is inconclusive. Guilt-expressing and non-guilt-expressing "
                     "liars show similar emotional profiles, making it difficult to determine whether guilt "
                     "expressions are strategic performances or genuine reactions. Larger samples or more "
                     "granular temporal analysis (e.g., emotion during chat vs. during contribution) may be needed.\n")
else:
    lines.append("Insufficient data to draw conclusions about the relationship between guilt expression "
                 "and facial emotions. More liar instances with both chat and facial emotion data are needed.\n")

lines.append(f"### Caveats\n")
lines.append(f"- Sample size is limited ({total} liar instances, {has_emotion.sum()} with facial data).")
lines.append(f"- Guilt detection uses keyword matching, which may miss subtle or implicit guilt expressions.")
lines.append(f"- Facial emotion data is aggregated over the entire Contribute page, not synchronized to specific moments.")
lines.append(f"- No statistical significance testing was performed; differences should be treated as descriptive.")

report_text = "\n".join(lines)
with open("_sandbox_data/guilt_analysis_report.md", "w") as f:
    f.write(report_text)

print(f"\n=== Report written to _sandbox_data/guilt_analysis_report.md ===")
print(f"Total cases with messages: {case_num}")
