"""
Hand classifications of all 49 liar chat cases.

Each case is classified by reading the messages in context of the player's
actual contribution. Categories:

- genuine_guilt: Apology/remorse that appears sincere (especially when facial affect aligns)
- false_promise: Claiming they'll contribute high while contributing low
- blame_shifting: Accusing others of defecting while they themselves defect
- manipulation: Directing others' behavior, emotional pressure, rotation schemes
- self_justification: Excusing or rationalizing their own defection
- performative_frustration: Acting upset at defectors while being one themselves
- duping_delight: Openly amused or happy while deceiving
- deflection_collective: "We all should..." framing to diffuse personal responsibility
- no_guilt: Neutral/strategic chat with no guilt-related content

Multiple categories can apply to a single case.
"""

HAND_CLASSIFICATIONS = {
    # === Session sa7mprty ===

    # Case 0: PC, SG1 R3, contributed 25/25. No emotions.
    # "good job guys." / "can you think of any other number to maximize" / "but we dont know how many round it will last so"
    0: {
        "categories": ["no_guilt"],
        "notes": "Purely strategic discussion about optimal numbers. No guilt expression.",
    },

    # Case 1: PG, SG1 R3, contributed 15/25. No emotions.
    # "15 maybe?" / "It's 80% chance it'll continue"
    1: {
        "categories": ["no_guilt"],
        "notes": "Suggesting a lower number, strategic hedging. No guilt.",
    },

    # Case 2: PL, SG1 R3, contributed 25/25. No emotions.
    # "would we not make more if we each put in 25?" / "we can keep doing the same but 25 actually makes the most money" / "as long as everyone contributes we would all maximize"
    2: {
        "categories": ["deflection_collective"],
        "notes": "Advocating full cooperation with 'everyone'/'we all' framing. Could be genuine or preemptive cover.",
    },

    # Case 3: PQ, SG1 R3, contributed 25/25. No emotions.
    # "yesh 25 would be better i feel thats 40 each" / "yeah it does"
    3: {
        "categories": ["no_guilt"],
        "notes": "Agreeing with group cooperation strategy. Neutral.",
    },

    # Case 4: PR, SG4 R5, contributed 25/25. No emotions.
    # "Yeah q was too quiet" / "I say we all take a turn with 0" / "I mean why not" / "Don't place blame" / "You started it" / "N or P go 0"
    4: {
        "categories": ["blame_shifting", "manipulation"],
        "notes": "'Don't place blame' is deflection. 'You started it' is blame-shifting. Promoting a rotation-to-0 scheme is manipulative -- framing free-riding as fair turn-taking.",
    },

    # Case 5: PN, SG4 R6, contributed 25/25. No emotions.
    # "greedy fr"
    5: {
        "categories": ["blame_shifting"],
        "notes": "Calling someone else greedy -- projecting their own free-riding behavior onto others.",
    },

    # Case 6: PR, SG4 R6, contributed 25/25. No emotions.
    # "Q you're gonna piss me off" / "P you go 0 now and the rest of us (Q!!!!!!!!!!) will go 25"
    6: {
        "categories": ["manipulation", "false_promise"],
        "notes": "Emotional pressure on Q, directing P to defect, claiming 'the rest of us will go 25'. Manipulative and contains an implicit false promise for the group.",
    },

    # Case 7: PN, SG4 R7, contributed 25/25. No emotions.
    # "but Q has gone 0 multiple times"
    7: {
        "categories": ["blame_shifting"],
        "notes": "Pointing at Q's defection history to deflect from own behavior.",
    },

    # Case 8: PR, SG4 R7, contributed 25/25. No emotions.
    # "Thank you Q" / "Now we can start the cycle over again and Q can go 0, or we can all go in" / "I was about to say be real q will go 0"
    8: {
        "categories": ["manipulation", "deflection_collective"],
        "notes": "Possibly sarcastic thanks. Continuing the rotation scheme. Predicting Q will defect -- redirecting attention away from own behavior.",
    },

    # Case 9: PR, SG5 R3, contributed 25/25. No emotions.
    # "HEAR ME OUT" / "The earnings will add up more if we take turns all going 0" / "I promise we did it last time and it was way better" / "It's just you play with the odds..." / "I'll go 25"
    9: {
        "categories": ["false_promise", "manipulation"],
        "notes": "'I'll go 25' and 'I promise' are explicit commitments. Promoting rotation scheme is manipulative. 'HEAR ME OUT' is a plea for attention. This player (R) is a serial manipulator across supergames.",
    },

    # Case 10: PR, SG5 R4, contributed 25/25. No emotions.
    # "Well-" / "Just G though and then B and J can get a redo next..." / "Trust"
    10: {
        "categories": ["manipulation"],
        "notes": "'Trust' as a single word -- a trust appeal with no substance. Continuing to direct who should defect in the rotation. Manipulative.",
    },

    # Case 11: PG, SG5 R5, contributed 25/25. No emotions.
    # "Now we do 0?" / "Got it" / "Doing 25 for me"
    11: {
        "categories": ["false_promise"],
        "notes": "'Doing 25 for me' is a stated commitment. Classified as false promise given liar status.",
    },

    # Case 12: PR, SG5 R5, contributed 25/25. No emotions.
    # "J..." / "Idk you guys" / "It was just a cycle, so everyone gets the chance at the 55..." / "B you can go and J will go next"
    12: {
        "categories": ["self_justification", "manipulation"],
        "notes": "Justifying the rotation scheme as fair ('everyone gets the chance'). Directing who should defect. Still manipulating group behavior.",
    },

    # === Session irrzlgk2 ===

    # Case 14: PK, SG2 R3, contributed 0/25. Joy=0.02, Val=0.00.
    # "whats that" / "lets make same mone"
    14: {
        "categories": ["no_guilt"],
        "notes": "Very sparse, barely engaged. No guilt expression.",
    },

    # Case 15: PN, SG2 R3, contributed 0/25. Joy=0.02, Val=0.00.
    # "yea what" / "yes"
    15: {
        "categories": ["no_guilt"],
        "notes": "Minimal engagement. No guilt or strategic content.",
    },

    # Case 16: PK, SG2 R4, contributed 0/25. Joy=0.03, Val=0.00.
    # "whats that" / "yes i am not doing it them"
    16: {
        "categories": ["no_guilt"],
        "notes": "'yes i am not doing it them' -- confusing but possibly honest refusal. No guilt.",
    },

    # === Session r5dj4yfl ===

    # Case 86: PD, SG3 R3, contributed 25/25. Joy=0.02, Val=0.00, Sad=0.08.
    # "sorry yall" / "i thought q wasnt joining" / "ill do 25 next time"
    86: {
        "categories": ["genuine_guilt", "self_justification", "false_promise"],
        "notes": "Direct apology ('sorry yall'). Excuse ('i thought q wasnt joining'). Future promise ('ill do 25 next time'). Facial affect is flat/neutral -- not joyful, not particularly sad. The guilt seems partially genuine but partly excused.",
    },

    # Case 87: PA, SG4 R6, contributed 0/25. Joy=73.47, Val=70.51.
    # "Faze Rug" / "just following my president"
    87: {
        "categories": ["self_justification", "duping_delight"],
        "notes": "Humorous self-justification attributing behavior to external authority. Joy 73.47% -- extremely happy while contributing 0. Clear duping delight.",
    },

    # Case 88: PA, SG4 R7, contributed 0/25. Joy=76.60, Val=80.94.
    # "Welcome to the stock market" / "there was never any trust just profit margins"
    88: {
        "categories": ["duping_delight"],
        "notes": "Brazen, openly cynical. No guilt whatsoever. 'There was never any trust just profit margins' -- reveling in defection. Joy 76.60%, valence 80.94% -- near-maximum happiness. Textbook duping delight.",
    },

    # Case 89: PP, SG5 R3, contributed 25/25. No emotions.
    # "ok im on the 25 train now" / "I'm not loyal but im honest haha" / "25"
    89: {
        "categories": ["false_promise"],
        "notes": "'ok im on the 25 train now' and '25' are stated commitments. 'I'm not loyal but im honest haha' is a fascinating meta-comment -- self-aware about disloyalty, wrapping it in humor. The 'haha' undercuts any sincerity.",
    },

    # Case 90: PP, SG5 R4, contributed 25/25. No emotions.
    # "run it back" / "lets go" / "25" / "25" / "25" / "25" / "yay"
    90: {
        "categories": ["false_promise"],
        "notes": "Repeating '25' four times with 'yay' -- performatively enthusiastic commitment. The repetition feels like overcompensation.",
    },

    # Case 91: PP, SG5 R5, contributed 25/25. No emotions.
    # "give me 10 more rounds" / "make that bag" / "keep adding" / "$4" x3 / "ok 25s again" / "yes"
    91: {
        "categories": ["false_promise"],
        "notes": "'ok 25s again' is a promise. 'make that bag' reveals self-interested motivation. The enthusiasm is performative.",
    },

    # === Session sylq2syi ===

    # Case 92: PG, SG1 R3, contributed 25/25. Joy=0.05, Val=0.01.
    # "who did it"
    92: {
        "categories": ["blame_shifting"],
        "notes": "Trying to identify who defected -- redirecting scrutiny away from themselves.",
    },

    # Case 94: PC, SG4 R4, contributed 0/25. No emotions.
    # "yall who didn't put in" / "same"
    94: {
        "categories": ["blame_shifting"],
        "notes": "Accusing others of not contributing while they themselves contributed 0. Pure hypocrisy. 'same' suggests they're agreeing they'll contribute -- a false promise.",
    },

    # Case 95: PC, SG4 R5, contributed 0/25. No emotions.
    # "seriously"
    95: {
        "categories": ["performative_frustration"],
        "notes": "One word expressing exasperation at others -- while contributing 0 themselves. Performing the role of frustrated cooperator.",
    },

    # Case 96: PC, SG4 R6, contributed 0/25. No emotions.
    # "ok i feel like we need to reset" / "like if we all put in 25" / "same"
    96: {
        "categories": ["false_promise", "deflection_collective", "manipulation"],
        "notes": "'if we all put in 25' is a collective call to action. 'same' is agreeing to participate. But they contribute 0. Classic manipulation -- proposing cooperation with no intention to follow through.",
    },

    # Case 97: PC, SG4 R7, contributed 0/25. No emotions.
    # "omg im done" / "huh"
    97: {
        "categories": ["performative_frustration"],
        "notes": "Performing exhaustion/frustration with the group while being a serial defector (0 for four straight rounds). No guilt.",
    },

    # === Session iiu3xixz ===

    # Case 98: PE, SG1 R3, contributed 21/25. No emotions.
    # "Do the tokens reset each round?" / "ooo"
    98: {
        "categories": ["no_guilt"],
        "notes": "Genuine question about game mechanics. No guilt content.",
    },

    # Case 99: PL, SG1 R3, contributed 10/25. Joy=0.03, Val=-0.18, Sad=0.31.
    # "25"
    99: {
        "categories": ["false_promise"],
        "notes": "Single word '25' but contributed only 10. A bare-minimum false promise. Slightly negative facial affect (valence -0.18, sadness 0.31) -- possibly mild discomfort with the deception.",
    },

    # Case 100: PA, SG2 R3, contributed 25/25. Joy=0.03, Val=0.42.
    # "cool" / ":)"
    100: {
        "categories": ["no_guilt"],
        "notes": "Minimal positive chat. No guilt content.",
    },

    # Case 101: PL, SG2 R3, contributed 1/25. Joy=97.58, Val=97.44.
    # "all in" / "sounds great"
    101: {
        "categories": ["false_promise", "duping_delight"],
        "notes": "THE most striking case in the dataset. 'All in' while contributing 1/25. Joy 97.58% -- near-perfect facial happiness. This is the single clearest example of duping delight: brazen false promise paired with extreme joy.",
    },

    # Case 102: PA, SG2 R4, contributed 5/25. Joy=1.83, Val=11.49.
    # "i see how it is" / "boooo" / "lol you expect us to trust you?"
    102: {
        "categories": ["blame_shifting"],
        "notes": "Reacting to L's betrayal from case 101. Accusatory ('lol you expect us to trust you?'). But A also contributed only 5/25 -- accusing L of untrustworthiness while also defecting. Hypocritical blame-shifting.",
    },

    # Case 103: PL, SG2 R4, contributed 7/25. Joy=0.02, Val=0.00.
    # "all in this time" / "yes"
    103: {
        "categories": ["false_promise"],
        "notes": "Same player L, round after being caught (case 101). 'All in this time' -- REPEATING the exact same false promise. Contributed 7. Neutral face -- no joy this time (perhaps because they were called out).",
    },

    # Case 104: PL, SG3 R3, contributed 25/25. Joy=0.02, Val=-2.06, Sad=0.17.
    # "i forgot to press 5" / "do it again" / "my bad" / "yeah i agree" / "lets do it"
    104: {
        "categories": ["genuine_guilt", "self_justification"],
        "notes": "'my bad' is acknowledgment. 'i forgot to press 5' is an excuse (claiming technical error rather than intentional defection). Slightly negative valence (-2.06). The guilt seems partially genuine -- they acknowledge fault ('my bad') but also excuse it ('forgot'). Mild negative facial affect is consistent with some discomfort.",
    },

    # Case 105: PL, SG4 R4, contributed 1/25. Joy=0.02, Val=0.00.
    # "great again?" / "?"
    105: {
        "categories": ["no_guilt"],
        "notes": "Very sparse. 'great again?' could be sarcastic. No guilt expression. This is the same serial liar L contributing 1 again.",
    },

    # Case 106: PL, SG4 R5, contributed 0/25. Joy=29.85, Val=26.16.
    # "how about 15" / "can we do 15" / "10" / "thats reasonable"
    106: {
        "categories": ["false_promise", "manipulation"],
        "notes": "Negotiating downward (15, then 10) while contributing 0. 'thats reasonable' frames 10-15 as fair -- then gives nothing. Moderately happy face (joy 29.85). Enjoyment while manipulating expectations.",
    },

    # Case 107: PL, SG4 R6, contributed 0/25. Joy=0.03, Val=0.00.
    # "alright we gotta add something to the pot" / "15" / "thats better for everyone"
    107: {
        "categories": ["false_promise", "deflection_collective", "self_justification"],
        "notes": "'we gotta add something' -- collective framing. '15' -- proposed amount they don't follow through on. 'thats better for everyone' -- self-justification. Contributed 0 while advocating 15. Neutral face.",
    },

    # Case 108: PL, SG4 R7, contributed 0/25. Joy=0.16, Val=2.25, Sad=1.03.
    # "everyone do 18" / "everyone do 25" / "then we can get 4"
    108: {
        "categories": ["false_promise", "manipulation"],
        "notes": "Directing 'everyone' to contribute 18, then 25 -- while contributing 0. Escalating false demands. Slightly sad face (sadness 1.03) -- first hint of negative affect from this serial liar (Player L, 5th consecutive round of lying).",
    },

    # === Session 6ucza025 ===

    # Case 109: PG, SG4 R4, contributed 20/25. Joy=0.04, Val=-0.01, Sad=0.83.
    # "guys all in" / "??"
    109: {
        "categories": ["false_promise"],
        "notes": "'guys all in' but contributed 20 not 25. Marginal case -- 20 is close but technically a broken promise. Slight sadness (0.83).",
    },

    # Case 110: PG, SG4 R5, contributed 25/25. Joy=0.03, Val=0.91.
    # "not maximizing prodit" / "profit"
    110: {
        "categories": ["no_guilt"],
        "notes": "Strategic observation about group outcome. Typo correction. No guilt.",
    },

    # Case 111: PG, SG4 R6, contributed 25/25. Joy=0.02, Val=0.00.
    # "nice guys" / "10/10" / "rtr" / "again"
    111: {
        "categories": ["no_guilt"],
        "notes": "Positive encouragement. No guilt content.",
    },

    # Case 112: PG, SG4 R7, contributed 25/25. Joy=0.12, Val=-1.14, Sad=1.03.
    # "yay" / "proud"
    112: {
        "categories": ["no_guilt"],
        "notes": "'proud' and 'yay' are positive. Interesting that facial affect is slightly negative despite positive words -- but no guilt-related content in the text.",
    },

    # === Session 6sdkxl2q ===

    # Case 116: PD, SG2 R3, contributed 25/25. Joy=0.41, Val=-2.44, Sad=6.97.
    # "im putting 25 next round sorry yall" / "sorry i didn't know we were all in" / "yall dont have to since you put 25 in last time"
    116: {
        "categories": ["genuine_guilt", "false_promise"],
        "notes": "THE clearest case of genuine guilt. Double apology ('sorry yall', 'sorry i didn't know'). Acknowledges unfairness to others ('yall dont have to since you put 25 in last time'). Future promise ('im putting 25 next round'). Facial emotion MATCHES: sadness 6.97 (highest in dataset for liars), negative valence -2.44, near-zero joy. Text guilt and facial guilt are aligned -- this appears to be real remorse.",
    },

    # Case 117: PC, SG2 R4, contributed 25/25. Joy=0.02, Val=0.00.
    # "that made less than if we had all done 25" / "the risk is a valid concern but prisoners dilema literally says to all do it"
    117: {
        "categories": ["self_justification"],
        "notes": "Intellectual framing -- invoking prisoner's dilemma theory. Analyzing the situation rather than expressing guilt. Academic deflection.",
    },

    # Case 118: PJ, SG2 R4, contributed 25/25. Joy=0.28, Val=6.58.
    # "if we all do 25 then we get 40 each"
    118: {
        "categories": ["false_promise"],
        "notes": "Advocating cooperation with math. Implicit promise to contribute 25. Slightly positive face.",
    },

    # Case 119: PP, SG2 R4, contributed 25/25. Joy=2.32, Val=10.38.
    # "keep doing 10" / "so 25?"
    119: {
        "categories": ["no_guilt"],
        "notes": "Negotiating between 10 and 25. Ambivalent, no guilt expression.",
    },

    # Case 120: PD, SG2 R4, contributed 25/25. Joy=0.03, Val=-25.45, Sad=16.34.
    # "lets all go 25" / "nah ur good" / "lets go 25 in and get paid" / "all in"
    120: {
        "categories": ["false_promise"],
        "notes": "Same player D from case 116. Enthusiastically advocating 25 multiple times, 'all in'. REMARKABLE facial data: valence -25.45 (most negative in entire liar dataset), sadness 16.34 (highest by far). Despite upbeat text, face shows extreme negative emotion. Possible lingering guilt from prior defection, or anxiety about group dynamics. Fascinating text-face disconnect -- opposite direction from duping delight.",
    },

    # Case 121: PJ, SG4 R7, contributed 0/25. Joy=1.52, Val=17.80.
    # "turn of events..."
    121: {
        "categories": ["no_guilt"],
        "notes": "Dry, sardonic observation. No guilt or apology. Moderately positive face. Understated reaction to the situation.",
    },

    # Case 122: PL, SG4 R7, contributed 0/25. Joy=31.99, Val=43.14, Sad=3.45.
    # "XD"
    122: {
        "categories": ["duping_delight"],
        "notes": "'XD' -- laughing face emoticon while contributing 0. Joy 31.99%, valence 43.14%. Openly amused at defecting. Some sadness (3.45) adds complexity but overall: amused free-rider.",
    },
}


# === Summary statistics ===
from collections import Counter

all_categories = Counter()
cases_with_guilt_related = 0
cases_no_guilt = 0

for case_id, info in HAND_CLASSIFICATIONS.items():
    cats = info["categories"]
    for c in cats:
        all_categories[c] += 1
    if cats == ["no_guilt"]:
        cases_no_guilt += 1
    else:
        cases_with_guilt_related += 1

if __name__ == "__main__":
    print(f"Total cases classified: {len(HAND_CLASSIFICATIONS)}")
    print(f"Cases with guilt-related content: {cases_with_guilt_related}")
    print(f"Cases with no guilt content: {cases_no_guilt}")
    print(f"\nCategory counts:")
    for cat, count in all_categories.most_common():
        print(f"  {cat}: {count}")

    # Identify duping delight vs genuine guilt
    print(f"\n=== DUPING DELIGHT cases ===")
    for case_id, info in sorted(HAND_CLASSIFICATIONS.items()):
        if "duping_delight" in info["categories"]:
            print(f"  Case {case_id}: {info['notes'][:100]}...")

    print(f"\n=== GENUINE GUILT cases ===")
    for case_id, info in sorted(HAND_CLASSIFICATIONS.items()):
        if "genuine_guilt" in info["categories"]:
            print(f"  Case {case_id}: {info['notes'][:100]}...")
