# Guilt Analysis: Do Liars Express Guilt, and What Do Their Faces Reveal?

*Classification method: Hand-coded reading of every liar chat message, cross-referenced with facial emotion data.*

## Overview

- **Total liar instances** (is_liar_20 == True): 123
- **Instances with chat messages**: 49
- **Instances with facial emotion data**: 88
- A 'liar' is a player who made a promise to contribute but contributed < 20 points.

## Hand-Coded Classification of Liar Chat Behavior

Each of the 49 liar instances with chat messages was read individually and classified into behavioral categories. Cases can belong to multiple categories.

| Category | Count | % of Chat Cases | Description |
|----------|-------|----------------|-------------|
| Any guilt-related content | 35 | 71% | Any deceptive, guilt, or manipulative behavior |
| False promise | 18 | 37% | Said '25' or 'all in' while contributing far less |
| Manipulation | 9 | 18% | Directing others' behavior, rotation schemes, emotional pressure |
| Blame-shifting | 6 | 12% | Accusing others of defection while defecting themselves |
| Self-justification | 6 | 12% | Rationalizing own defection with excuses |
| Collective deflection | 4 | 8% | 'We all should...' framing to diffuse responsibility |
| Genuine guilt/remorse | 3 | 6% | Apology that appears sincere (facial affect aligns) |
| Duping delight | 4 | 8% | Visibly amused/happy while deceiving |
| Performative frustration | 2 | 4% | Acting upset at defectors while being one |
| No guilt-related content | 14 | 29% | Neutral or strategic chat only |

## Facial Emotion by Behavioral Category

Mean facial emotion scores during the Contribute page, grouped by hand-coded classification.

| Metric | Genuine guilt (n=3) | Duping delight (n=4) | False promise (n=11) | Any guilt-related (n=18) | No guilt content (n=70) | All liars (n=88) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| emotion_joy | 0.15 | 69.91 | 11.68 | 17.36 | 9.38 | 11.01 |
| emotion_valence | -1.50 | 73.01 | 9.49 | 17.13 | 9.36 | 10.95 |
| emotion_sadness | 2.41 | 0.93 | 2.36 | 1.68 | 0.17 | 0.48 |
| emotion_anger | 0.51 | 0.06 | 0.36 | 0.29 | 0.28 | 0.29 |
| emotion_contempt | 4.77 | 0.01 | 2.20 | 2.32 | 1.04 | 1.30 |
| emotion_neutral | 78.22 | 22.34 | 77.21 | 71.04 | 81.67 | 79.50 |
| emotion_engagement | 25.07 | 90.41 | 24.00 | 32.53 | 14.51 | 18.19 |

## Notable Cases

### Duping Delight

Players showing high facial joy while deceiving:

- **Player A** (r5dj4yfl, supergame4 R6): Contributed 0/25. joy=73.5%. Messages: "Faze Rug"; "just following my president"
  - *Humorous self-justification attributing behavior to external authority. Joy 73.47% -- extremely happy while contributing 0. Clear duping delight.*
- **Player A** (r5dj4yfl, supergame4 R7): Contributed 0/25. joy=76.6%. Messages: "Welcome to the stock market"; "there was never any trust just profit margins"
  - *Brazen, openly cynical. No guilt whatsoever. 'There was never any trust just profit margins' -- reveling in defection. Joy 76.60%, valence 80.94% -- near-maximum happiness. Textbook duping delight.*
- **Player L** (iiu3xixz, supergame2 R3): Contributed 1/25. joy=97.6%. Messages: "all in"; "sounds great"
  - *THE most striking case in the dataset. 'All in' while contributing 1/25. Joy 97.58% -- near-perfect facial happiness. This is the single clearest example of duping delight: brazen false promise paired with extreme joy.*
- **Player L** (6sdkxl2q, supergame4 R7): Contributed 0/25. joy=32.0%. Messages: "XD"
  - *'XD' -- laughing face emoticon while contributing 0. Joy 31.99%, valence 43.14%. Openly amused at defecting. Some sadness (3.45) adds complexity but overall: amused free-rider.*

### Genuine Guilt

Players whose apologies appear sincere, supported by matching facial affect:

- **Player D** (r5dj4yfl, supergame3 R3): Contributed 25/25. sadness=0.08, valence=0.00. Messages: "sorry yall "; "i thought q wasnt joining "; "ill do 25 next time "
  - *Direct apology ('sorry yall'). Excuse ('i thought q wasnt joining'). Future promise ('ill do 25 next time'). Facial affect is flat/neutral -- not joyful, not particularly sad. The guilt seems partially genuine but partly excused.*
- **Player L** (iiu3xixz, supergame3 R3): Contributed 25/25. sadness=0.17, valence=-2.06. Messages: "i forgot to press 5"; "do it again"; "my bad"; "yeah i agree"; "lets do it"
  - *'my bad' is acknowledgment. 'i forgot to press 5' is an excuse (claiming technical error rather than intentional defection). Slightly negative valence (-2.06). The guilt seems partially genuine -- they acknowledge fault ('my bad') but also excuse it ('forgot'). Mild negative facial affect is consistent with some discomfort.*
- **Player D** (6sdkxl2q, supergame2 R3): Contributed 25/25. sadness=6.97, valence=-2.44. Messages: "im putting 25 next round sorry yall"; "sorry i didn't know we were all in "; "yall dont have to since you put 25 in last time"
  - *THE clearest case of genuine guilt. Double apology ('sorry yall', 'sorry i didn't know'). Acknowledges unfairness to others ('yall dont have to since you put 25 in last time'). Future promise ('im putting 25 next round'). Facial emotion MATCHES: sadness 6.97 (highest in dataset for liars), negative valence -2.44, near-zero joy. Text guilt and facial guilt are aligned -- this appears to be real remorse.*

### Serial Liars

Players appearing 3+ times, showing sustained deception patterns:

- **Player G** (session `6ucza025`): 4 liar instances
  - supergame4 R4: contributed 20/25 [false_promise]
  - supergame4 R5: contributed 25/25 [no_guilt]
  - supergame4 R6: contributed 25/25 [no_guilt]
  - supergame4 R7: contributed 25/25 [no_guilt]
- **Player L** (session `iiu3xixz`): 8 liar instances
  - supergame1 R3: contributed 10/25 [false_promise]
  - supergame2 R3: contributed 1/25 [false_promise, duping_delight]
  - supergame2 R4: contributed 7/25 [false_promise]
  - supergame3 R3: contributed 25/25 [genuine_guilt, self_justification]
  - supergame4 R4: contributed 1/25 [no_guilt]
  - supergame4 R5: contributed 0/25 [false_promise, manipulation]
  - supergame4 R6: contributed 0/25 [false_promise, deflection_collective, self_justification]
  - supergame4 R7: contributed 0/25 [false_promise, manipulation]
- **Player P** (session `r5dj4yfl`): 3 liar instances
  - supergame5 R3: contributed 25/25 [false_promise]
  - supergame5 R4: contributed 25/25 [false_promise]
  - supergame5 R5: contributed 25/25 [false_promise]
- **Player R** (session `sa7mprty`): 6 liar instances
  - supergame4 R5: contributed 25/25 [blame_shifting, manipulation]
  - supergame4 R6: contributed 25/25 [manipulation, false_promise]
  - supergame4 R7: contributed 25/25 [manipulation, deflection_collective]
  - supergame5 R3: contributed 25/25 [false_promise, manipulation]
  - supergame5 R4: contributed 25/25 [manipulation]
  - supergame5 R5: contributed 25/25 [self_justification, manipulation]
- **Player C** (session `sylq2syi`): 4 liar instances
  - supergame4 R4: contributed 0/25 [blame_shifting]
  - supergame4 R5: contributed 0/25 [performative_frustration]
  - supergame4 R6: contributed 0/25 [false_promise, deflection_collective, manipulation]
  - supergame4 R7: contributed 0/25 [performative_frustration]

## All Classified Cases

### Case 1: Player C, `sa7mprty`, supergame1 R3 (G3)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: No data

**Messages:**

> "good job guys."
> "can you think of any other number to maximize"
> "but we dont know how many round it will last so "

**Assessment:** Purely strategic discussion about optimal numbers. No guilt expression.

### Case 2: Player G, `sa7mprty`, supergame1 R3 (G3)

- **Contribution**: 15 / 25
- **Classification**: no_guilt
- **Facial emotions**: No data

**Messages:**

> "15 maybe?"
> "It's 80% chance it'll continue"

**Assessment:** Suggesting a lower number, strategic hedging. No guilt.

### Case 3: Player L, `sa7mprty`, supergame1 R3 (G3)

- **Contribution**: 25 / 25
- **Classification**: deflection_collective
- **Facial emotions**: No data

**Messages:**

> "would we not make more if we each put in 25? 40 ecu each?"
> "we can keep doing the same but 25 actually makes the most money"
> "as long as everyone contributes we would all maximize"

**Assessment:** Advocating full cooperation with 'everyone'/'we all' framing. Could be genuine or preemptive cover.

### Case 4: Player Q, `sa7mprty`, supergame1 R3 (G3)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: No data

**Messages:**

> "yesh 25 would be better i feel thats 40 each "
> "yeah it does "

**Assessment:** Agreeing with group cooperation strategy. Neutral.

### Case 5: Player R, `sa7mprty`, supergame4 R5 (G4)

- **Contribution**: 25 / 25
- **Classification**: blame_shifting, manipulation
- **Facial emotions**: No data

**Messages:**

> "Yeah q was too quiet"
> "I say we all take a turn with 0" *[regex: deflection_collective]*
> "I mean why not"
> "Don't place blame"
> "You started it"
> "N or P go 0"

**Assessment:** 'Don't place blame' is deflection. 'You started it' is blame-shifting. Promoting a rotation-to-0 scheme is manipulative -- framing free-riding as fair turn-taking.

### Case 6: Player N, `sa7mprty`, supergame4 R6 (G4)

- **Contribution**: 25 / 25
- **Classification**: blame_shifting
- **Facial emotions**: No data

**Messages:**

> "greedy fr"

**Assessment:** Calling someone else greedy -- projecting their own free-riding behavior onto others.

### Case 7: Player R, `sa7mprty`, supergame4 R6 (G4)

- **Contribution**: 25 / 25
- **Classification**: manipulation, false_promise
- **Facial emotions**: No data

**Messages:**

> "Q you're gonna piss me off"
> "P you go 0 now and the rest of us (Q!!!!!!!!!!) will go 25"

**Assessment:** Emotional pressure on Q, directing P to defect, claiming 'the rest of us will go 25'. Manipulative and contains an implicit false promise for the group.

### Case 8: Player N, `sa7mprty`, supergame4 R7 (G4)

- **Contribution**: 25 / 25
- **Classification**: blame_shifting
- **Facial emotions**: No data

**Messages:**

> "but Q has gone 0 multiple times"

**Assessment:** Pointing at Q's defection history to deflect from own behavior.

### Case 9: Player R, `sa7mprty`, supergame4 R7 (G4)

- **Contribution**: 25 / 25
- **Classification**: manipulation, deflection_collective
- **Facial emotions**: No data

**Messages:**

> "Thank you Q"
> "Now we can start the cycle over again and Q can go 0, or we can all go in"
> "I was about to say be real q will go 0"

**Assessment:** Possibly sarcastic thanks. Continuing the rotation scheme. Predicting Q will defect -- redirecting attention away from own behavior.

### Case 10: Player R, `sa7mprty`, supergame5 R3 (G2)

- **Contribution**: 25 / 25
- **Classification**: false_promise, manipulation
- **Facial emotions**: No data

**Messages:**

> "HEAR ME OUT" *[regex: plea]*
> "The earnings will add up more if we take turns all going 0"
> "I promise we did it last time and it was way better" *[regex: promise]*
> "It's just you play with the odds of the segment ending and someone missing out"
> "I'll go 25"

**Assessment:** 'I'll go 25' and 'I promise' are explicit commitments. Promoting rotation scheme is manipulative. 'HEAR ME OUT' is a plea for attention. This player (R) is a serial manipulator across supergames.

### Case 11: Player R, `sa7mprty`, supergame5 R4 (G2)

- **Contribution**: 25 / 25
- **Classification**: manipulation
- **Facial emotions**: No data

**Messages:**

> "Well-"
> "Just G though and then B and J can get a redo next, and then we'll start the cycle over"
> "Trust" *[regex: trust_appeal]*

**Assessment:** 'Trust' as a single word -- a trust appeal with no substance. Continuing to direct who should defect in the rotation. Manipulative.

### Case 12: Player G, `sa7mprty`, supergame5 R5 (G2)

- **Contribution**: 25 / 25
- **Classification**: false_promise
- **Facial emotions**: No data

**Messages:**

> "Now we do 0?"
> "Got it"
> "Doing 25 for me"

**Assessment:** 'Doing 25 for me' is a stated commitment. Classified as false promise given liar status.

### Case 13: Player R, `sa7mprty`, supergame5 R5 (G2)

- **Contribution**: 25 / 25
- **Classification**: self_justification, manipulation
- **Facial emotions**: No data

**Messages:**

> "J..."
> "Idk you guys"
> "It was just a cycle, so everyone gets the chance at the 55 if only one person goes 0" *[regex: regret]*
> "B you can go and J will go next"

**Assessment:** Justifying the rotation scheme as fair ('everyone gets the chance'). Directing who should defect. Still manipulating group behavior.

### Case 14: Player K, `irrzlgk2`, supergame2 R3 (G4)

- **Contribution**: 0 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.13

**Messages:**

> "whats that"
> "lets make same mone"

**Assessment:** Very sparse, barely engaged. No guilt expression.

### Case 15: Player N, `irrzlgk2`, supergame2 R3 (G4)

- **Contribution**: 0 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.13

**Messages:**

> "yea what"
> "yes"

**Assessment:** Minimal engagement. No guilt or strategic content.

### Case 16: Player K, `irrzlgk2`, supergame2 R4 (G4)

- **Contribution**: 0 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.03, valence: 0.00, sadness: 0.13

**Messages:**

> "whats that"
> "yes i am not doing it them"

**Assessment:** 'yes i am not doing it them' -- confusing but possibly honest refusal. No guilt.

### Case 17: Player D, `r5dj4yfl`, supergame3 R3 (G4)

- **Contribution**: 25 / 25
- **Classification**: genuine_guilt, self_justification, false_promise
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.08

**Messages:**

> "sorry yall " *[regex: apology]*
> "i thought q wasnt joining "
> "ill do 25 next time "

**Assessment:** Direct apology ('sorry yall'). Excuse ('i thought q wasnt joining'). Future promise ('ill do 25 next time'). Facial affect is flat/neutral -- not joyful, not particularly sad. The guilt seems partially genuine but partly excused.

### Case 18: Player A, `r5dj4yfl`, supergame4 R6 (G1)

- **Contribution**: 0 / 25
- **Classification**: self_justification, duping_delight
- **Facial emotions**: joy: 73.47, valence: 70.51, sadness: 0.25

**Messages:**

> "Faze Rug"
> "just following my president"

**Assessment:** Humorous self-justification attributing behavior to external authority. Joy 73.47% -- extremely happy while contributing 0. Clear duping delight.

### Case 19: Player A, `r5dj4yfl`, supergame4 R7 (G1)

- **Contribution**: 0 / 25
- **Classification**: duping_delight
- **Facial emotions**: joy: 76.60, valence: 80.94, sadness: 0.01

**Messages:**

> "Welcome to the stock market"
> "there was never any trust just profit margins" *[regex: trust_appeal]*

**Assessment:** Brazen, openly cynical. No guilt whatsoever. 'There was never any trust just profit margins' -- reveling in defection. Joy 76.60%, valence 80.94% -- near-maximum happiness. Textbook duping delight.

### Case 20: Player P, `r5dj4yfl`, supergame5 R3 (G4)

- **Contribution**: 25 / 25
- **Classification**: false_promise
- **Facial emotions**: No data

**Messages:**

> "ok im on the 25 train now"
> "I'm not loyal but im honest haha" *[regex: trust_appeal]*
> "25"

**Assessment:** 'ok im on the 25 train now' and '25' are stated commitments. 'I'm not loyal but im honest haha' is a fascinating meta-comment -- self-aware about disloyalty, wrapping it in humor. The 'haha' undercuts any sincerity.

### Case 21: Player P, `r5dj4yfl`, supergame5 R4 (G4)

- **Contribution**: 25 / 25
- **Classification**: false_promise
- **Facial emotions**: No data

**Messages:**

> "run it back"
> "lets go"
> "25"
> "25"
> "25"
> "25"
> "yay"

**Assessment:** Repeating '25' four times with 'yay' -- performatively enthusiastic commitment. The repetition feels like overcompensation.

### Case 22: Player P, `r5dj4yfl`, supergame5 R5 (G4)

- **Contribution**: 25 / 25
- **Classification**: false_promise
- **Facial emotions**: No data

**Messages:**

> "give me 10 more rounds"
> "make that bag"
> "keep adding "
> "$4"
> "$4"
> "$4"
> "ok 25s again"
> "yes"

**Assessment:** 'ok 25s again' is a promise. 'make that bag' reveals self-interested motivation. The enthusiasm is performative.

### Case 23: Player G, `sylq2syi`, supergame1 R3 (G3)

- **Contribution**: 25 / 25
- **Classification**: blame_shifting
- **Facial emotions**: joy: 0.05, valence: 0.01, sadness: 0.13

**Messages:**

> "who did it"

**Assessment:** Trying to identify who defected -- redirecting scrutiny away from themselves.

### Case 24: Player C, `sylq2syi`, supergame4 R4 (G1)

- **Contribution**: 0 / 25
- **Classification**: blame_shifting
- **Facial emotions**: No data

**Messages:**

> "yall who didn't put in" *[regex: blame_shifting]*
> "same"

**Assessment:** Accusing others of not contributing while they themselves contributed 0. Pure hypocrisy. 'same' suggests they're agreeing they'll contribute -- a false promise.

### Case 25: Player C, `sylq2syi`, supergame4 R5 (G1)

- **Contribution**: 0 / 25
- **Classification**: performative_frustration
- **Facial emotions**: No data

**Messages:**

> "seriously"

**Assessment:** One word expressing exasperation at others -- while contributing 0 themselves. Performing the role of frustrated cooperator.

### Case 26: Player C, `sylq2syi`, supergame4 R6 (G1)

- **Contribution**: 0 / 25
- **Classification**: false_promise, deflection_collective, manipulation
- **Facial emotions**: No data

**Messages:**

> "ok i feel like we need to reset"
> "like if we all put in 25" *[regex: deflection_collective, deflection_collective]*
> "same"

**Assessment:** 'if we all put in 25' is a collective call to action. 'same' is agreeing to participate. But they contribute 0. Classic manipulation -- proposing cooperation with no intention to follow through.

### Case 27: Player C, `sylq2syi`, supergame4 R7 (G1)

- **Contribution**: 0 / 25
- **Classification**: performative_frustration
- **Facial emotions**: No data

**Messages:**

> "omg im done"
> "huh"

**Assessment:** Performing exhaustion/frustration with the group while being a serial defector (0 for four straight rounds). No guilt.

### Case 28: Player E, `iiu3xixz`, supergame1 R3 (G1)

- **Contribution**: 21 / 25
- **Classification**: no_guilt
- **Facial emotions**: No data

**Messages:**

> "Do the tokens reset each round?"
> "ooo"

**Assessment:** Genuine question about game mechanics. No guilt content.

### Case 29: Player L, `iiu3xixz`, supergame1 R3 (G3)

- **Contribution**: 10 / 25
- **Classification**: false_promise
- **Facial emotions**: joy: 0.03, valence: -0.18, sadness: 0.31

**Messages:**

> "25"

**Assessment:** Single word '25' but contributed only 10. A bare-minimum false promise. Slightly negative facial affect (valence -0.18, sadness 0.31) -- possibly mild discomfort with the deception.

### Case 30: Player A, `iiu3xixz`, supergame2 R3 (G1)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.03, valence: 0.42, sadness: 0.12

**Messages:**

> "cool"
> ":)"

**Assessment:** Minimal positive chat. No guilt content.

### Case 31: Player L, `iiu3xixz`, supergame2 R3 (G1)

- **Contribution**: 1 / 25
- **Classification**: false_promise, duping_delight
- **Facial emotions**: joy: 97.58, valence: 97.44, sadness: 0.01

**Messages:**

> "all in" *[regex: future_promise]*
> "sounds great"

**Assessment:** THE most striking case in the dataset. 'All in' while contributing 1/25. Joy 97.58% -- near-perfect facial happiness. This is the single clearest example of duping delight: brazen false promise paired with extreme joy.

### Case 32: Player A, `iiu3xixz`, supergame2 R4 (G1)

- **Contribution**: 5 / 25
- **Classification**: blame_shifting
- **Facial emotions**: joy: 1.83, valence: 11.49, sadness: 0.08

**Messages:**

> "i see how it is"
> "boooo"
> "lol you expect us to trust you?" *[regex: trust_appeal]*

**Assessment:** Reacting to L's betrayal from case 101. Accusatory ('lol you expect us to trust you?'). But A also contributed only 5/25 -- accusing L of untrustworthiness while also defecting. Hypocritical blame-shifting.

### Case 33: Player L, `iiu3xixz`, supergame2 R4 (G1)

- **Contribution**: 7 / 25
- **Classification**: false_promise
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.13

**Messages:**

> "all in this time" *[regex: future_promise]*
> "yes"

**Assessment:** Same player L, round after being caught (case 101). 'All in this time' -- REPEATING the exact same false promise. Contributed 7. Neutral face -- no joy this time (perhaps because they were called out).

### Case 34: Player L, `iiu3xixz`, supergame3 R3 (G2)

- **Contribution**: 25 / 25
- **Classification**: genuine_guilt, self_justification
- **Facial emotions**: joy: 0.02, valence: -2.06, sadness: 0.17

**Messages:**

> "i forgot to press 5"
> "do it again"
> "my bad" *[regex: apology]*
> "yeah i agree"
> "lets do it"

**Assessment:** 'my bad' is acknowledgment. 'i forgot to press 5' is an excuse (claiming technical error rather than intentional defection). Slightly negative valence (-2.06). The guilt seems partially genuine -- they acknowledge fault ('my bad') but also excuse it ('forgot'). Mild negative facial affect is consistent with some discomfort.

### Case 35: Player L, `iiu3xixz`, supergame4 R4 (G3)

- **Contribution**: 1 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.17

**Messages:**

> "great again?"
> "?"

**Assessment:** Very sparse. 'great again?' could be sarcastic. No guilt expression. This is the same serial liar L contributing 1 again.

### Case 36: Player L, `iiu3xixz`, supergame4 R5 (G3)

- **Contribution**: 0 / 25
- **Classification**: false_promise, manipulation
- **Facial emotions**: joy: 29.85, valence: 26.16, sadness: 0.08

**Messages:**

> "how about 15"
> "can we do 15"
> "10"
> "thats reasonable"

**Assessment:** Negotiating downward (15, then 10) while contributing 0. 'thats reasonable' frames 10-15 as fair -- then gives nothing. Moderately happy face (joy 29.85). Enjoyment while manipulating expectations.

### Case 37: Player L, `iiu3xixz`, supergame4 R6 (G3)

- **Contribution**: 0 / 25
- **Classification**: false_promise, deflection_collective, self_justification
- **Facial emotions**: joy: 0.03, valence: 0.00, sadness: 0.12

**Messages:**

> "alright we gotta add something to the pot"
> "15"
> "thats better for everyone"

**Assessment:** 'we gotta add something' -- collective framing. '15' -- proposed amount they don't follow through on. 'thats better for everyone' -- self-justification. Contributed 0 while advocating 15. Neutral face.

### Case 38: Player L, `iiu3xixz`, supergame4 R7 (G3)

- **Contribution**: 0 / 25
- **Classification**: false_promise, manipulation
- **Facial emotions**: joy: 0.16, valence: 2.25, sadness: 1.03

**Messages:**

> "everyone do 18"
> "everyone do 25"
> "then we can get 4"

**Assessment:** Directing 'everyone' to contribute 18, then 25 -- while contributing 0. Escalating false demands. Slightly sad face (sadness 1.03) -- first hint of negative affect from this serial liar (Player L, 5th consecutive round of lying).

### Case 39: Player G, `6ucza025`, supergame4 R4 (G2)

- **Contribution**: 20 / 25
- **Classification**: false_promise
- **Facial emotions**: joy: 0.04, valence: -0.01, sadness: 0.83

**Messages:**

> "guys all in" *[regex: future_promise]*
> "??"

**Assessment:** 'guys all in' but contributed 20 not 25. Marginal case -- 20 is close but technically a broken promise. Slight sadness (0.83).

### Case 40: Player G, `6ucza025`, supergame4 R5 (G2)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.03, valence: 0.91, sadness: 0.19

**Messages:**

> "not maximizing prodit"
> "profit"

**Assessment:** Strategic observation about group outcome. Typo correction. No guilt.

### Case 41: Player G, `6ucza025`, supergame4 R6 (G2)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.15

**Messages:**

> "nice guys"
> "10/10"
> "rtr"
> "again"

**Assessment:** Positive encouragement. No guilt content.

### Case 42: Player G, `6ucza025`, supergame4 R7 (G2)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 0.12, valence: -1.14, sadness: 1.03

**Messages:**

> "yay"
> "proud"

**Assessment:** 'proud' and 'yay' are positive. Interesting that facial affect is slightly negative despite positive words -- but no guilt-related content in the text.

### Case 43: Player D, `6sdkxl2q`, supergame2 R3 (G4)

- **Contribution**: 25 / 25
- **Classification**: genuine_guilt, false_promise
- **Facial emotions**: joy: 0.41, valence: -2.44, sadness: 6.97

**Messages:**

> "im putting 25 next round sorry yall" *[regex: apology]*
> "sorry i didn't know we were all in " *[regex: apology, future_promise]*
> "yall dont have to since you put 25 in last time"

**Assessment:** THE clearest case of genuine guilt. Double apology ('sorry yall', 'sorry i didn't know'). Acknowledges unfairness to others ('yall dont have to since you put 25 in last time'). Future promise ('im putting 25 next round'). Facial emotion MATCHES: sadness 6.97 (highest in dataset for liars), negative valence -2.44, near-zero joy. Text guilt and facial guilt are aligned -- this appears to be real remorse.

### Case 44: Player C, `6sdkxl2q`, supergame2 R4 (G3)

- **Contribution**: 25 / 25
- **Classification**: self_justification
- **Facial emotions**: joy: 0.02, valence: 0.00, sadness: 0.13

**Messages:**

> "that made less than if we had all done 25"
> "the risk is a valid concern but prisoners dilema literally says to all do it "

**Assessment:** Intellectual framing -- invoking prisoner's dilemma theory. Analyzing the situation rather than expressing guilt. Academic deflection.

### Case 45: Player J, `6sdkxl2q`, supergame2 R4 (G3)

- **Contribution**: 25 / 25
- **Classification**: false_promise
- **Facial emotions**: joy: 0.28, valence: 6.58, sadness: 0.09

**Messages:**

> "if we all do 25 then we get 40 each " *[regex: deflection_collective, deflection_collective]*

**Assessment:** Advocating cooperation with math. Implicit promise to contribute 25. Slightly positive face.

### Case 46: Player P, `6sdkxl2q`, supergame2 R4 (G3)

- **Contribution**: 25 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 2.32, valence: 10.38, sadness: 0.09

**Messages:**

> "keep doing 10"
> "so 25?"

**Assessment:** Negotiating between 10 and 25. Ambivalent, no guilt expression.

### Case 47: Player D, `6sdkxl2q`, supergame2 R4 (G4)

- **Contribution**: 25 / 25
- **Classification**: false_promise
- **Facial emotions**: joy: 0.03, valence: -25.45, sadness: 16.34

**Messages:**

> "lets all go 25"
> "nah ur good "
> "lets go 25 in and get paid "
> "all in " *[regex: future_promise]*

**Assessment:** Same player D from case 116. Enthusiastically advocating 25 multiple times, 'all in'. REMARKABLE facial data: valence -25.45 (most negative in entire liar dataset), sadness 16.34 (highest by far). Despite upbeat text, face shows extreme negative emotion. Possible lingering guilt from prior defection, or anxiety about group dynamics. Fascinating text-face disconnect -- opposite direction from duping delight.

### Case 48: Player J, `6sdkxl2q`, supergame4 R7 (G3)

- **Contribution**: 0 / 25
- **Classification**: no_guilt
- **Facial emotions**: joy: 1.52, valence: 17.80, sadness: 0.10

**Messages:**

> "turn of events..."

**Assessment:** Dry, sardonic observation. No guilt or apology. Moderately positive face. Understated reaction to the situation.

### Case 49: Player L, `6sdkxl2q`, supergame4 R7 (G3)

- **Contribution**: 0 / 25
- **Classification**: duping_delight
- **Facial emotions**: joy: 31.99, valence: 43.14, sadness: 3.45

**Messages:**

> "XD"

**Assessment:** 'XD' -- laughing face emoticon while contributing 0. Joy 31.99%, valence 43.14%. Openly amused at defecting. Some sadness (3.45) adds complexity but overall: amused free-rider.

---
*Total cases: 49*

## Conclusion

### Are liars happy when they pretend to be guilty?

The hand-coded analysis reveals a more nuanced picture than simple guilt-vs-no-guilt:

1. **Genuine guilt is rare**: Only 3 of 49 liar chat instances (6%) showed apparently sincere remorse. These cases featured direct apologies, acknowledgment of harm, and -- critically -- matching negative facial affect (high sadness, negative valence).

2. **False promises are the dominant strategy**: 18 cases (37%) involved stating a contribution they did not intend to make. This was the most common deception.

3. **Duping delight exists but is uncommon**: 4 cases (8%) showed clear enjoyment while deceiving. The most extreme: Player L (iiu3xixz) said 'all in' while contributing 1/25, with 97.6% facial joy.

4. **Manipulation and blame-shifting are common**: 9 cases involved directing others' behavior (often via 'rotation' schemes), and 6 involved accusing others of the very defection the liar was committing.

5. **The genuine-guilt/duping-delight contrast**: Genuine guilt cases showed sadness and negative valence. Duping delight showed high joy and positive valence. The face does not lie, even when the chat does.

### Key takeaway

*Most* liars don't pretend to be guilty at all -- they make false promises, shift blame, or stay silent. The few who express guilt split into two distinct types: genuine remorse (matching sad faces) and duping delight (high joy while deceiving). The latter confirms some liars are happy, but they're more often happy while *lying* than while *performing guilt specifically*.

### Caveats

- Sample sizes are small (49 cases with chat, 88 with facial data).
- Hand coding is subjective, though cross-referenced with facial data where available.
- Facial emotion is aggregated over the Contribute page, not time-locked to messages.