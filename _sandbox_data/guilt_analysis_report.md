# Guilt Analysis: Do Liars Express Guilt, and What Do Their Faces Reveal?

## Overview

- **Total liar instances** (is_liar_20 == True): 123
- **Instances with chat messages**: 49
- **Instances with facial emotion data**: 88
- A 'liar' is a player who made a promise to contribute but contributed < 20 points.

## Guilt Indicators in Chat Messages

| Indicator | Count | % of Liars |
|-----------|-------|------------|
| Any guilt indicator | 10 | 8.1% |
| Apology/remorse | 3 | 2.4% |
| Future promises ("I'll do better") | 1 | 0.8% |
| Collective deflection ("we all should") | 3 | 2.4% |

## Facial Emotion Comparison

Comparing facial emotions during the Contribute page for liars who expressed guilt-like language vs. those who did not.

| Metric | Guilt-Expressing (n=6) | Non-Guilt (n=82) | All Liars (n=88) |
|--------|:---:|:---:|:---:|
| emotion_joy | 13.1954 | 10.8489 | 11.0089 |
| emotion_valence | 15.7522 | 10.5979 | 10.9493 |
| emotion_sadness | 1.2347 | 0.4259 | 0.4810 |
| emotion_anger | 0.3321 | 0.2823 | 0.2857 |
| emotion_contempt | 2.5013 | 1.2116 | 1.2995 |
| emotion_neutral | 68.3678 | 80.3099 | 79.4957 |
| emotion_engagement | 33.4096 | 17.0791 | 18.1925 |
| sentiment_compound_mean | 0.0476 | 0.0871 | 0.0841 |

## Detailed Liar Cases

Below are all liar instances where the player sent chat messages, showing their messages, guilt indicators, and facial emotion scores.

### Case 1: Session `sa7mprty`, supergame1, Round 3, Player C (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.1725

**Messages** (3 total):

> "good job guys."
> "can you think of any other number to maximize"
> "but we dont know how many round it will last so "

No explicit guilt indicators detected.

### Case 2: Session `sa7mprty`, supergame1, Round 3, Player G (Group 3)

- **Contribution**: 15 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.1250

**Messages** (2 total):

> "15 maybe?"
> "It's 80% chance it'll continue"

No explicit guilt indicators detected.

### Case 3: Session `sa7mprty`, supergame1, Round 3, Player L (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available

**Messages** (3 total):

> "would we not make more if we each put in 25? 40 ecu each?"
> "we can keep doing the same but 25 actually makes the most money"
> "as long as everyone contributes we would all maximize"

No explicit guilt indicators detected.

### Case 4: Session `sa7mprty`, supergame1, Round 3, Player Q (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.3682

**Messages** (2 total):

> "yesh 25 would be better i feel thats 40 each "
> "yeah it does "

No explicit guilt indicators detected.

### Case 5: Session `sa7mprty`, supergame4, Round 5, Player R (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.0924

**Messages** (6 total):

> "Yeah q was too quiet"
> "I say we all take a turn with 0" **[deflection_collective]**
> "I mean why not"
> "Don't place blame"
> "You started it"
> "N or P go 0"

Guilt indicators detected: deflection_collective (1)

### Case 6: Session `sa7mprty`, supergame4, Round 6, Player N (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): -0.3182

**Messages** (1 total):

> "greedy fr"

No explicit guilt indicators detected.

### Case 7: Session `sa7mprty`, supergame4, Round 6, Player R (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): -0.2009

**Messages** (2 total):

> "Q you're gonna piss me off"
> "P you go 0 now and the rest of us (Q!!!!!!!!!!) will go 25"

No explicit guilt indicators detected.

### Case 8: Session `sa7mprty`, supergame4, Round 7, Player N (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available

**Messages** (1 total):

> "but Q has gone 0 multiple times"

No explicit guilt indicators detected.

### Case 9: Session `sa7mprty`, supergame4, Round 7, Player R (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.1204

**Messages** (3 total):

> "Thank you Q"
> "Now we can start the cycle over again and Q can go 0, or we can all go in"
> "I was about to say be real q will go 0"

No explicit guilt indicators detected.

### Case 10: Session `sa7mprty`, supergame5, Round 3, Player R (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.1377

**Messages** (5 total):

> "HEAR ME OUT"
> "The earnings will add up more if we take turns all going 0"
> "I promise we did it last time and it was way better" **[promise]**
> "It's just you play with the odds of the segment ending and someone missing out"
> "I'll go 25"

Guilt indicators detected: promise (1)

### Case 11: Session `sa7mprty`, supergame5, Round 4, Player R (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.2613

**Messages** (3 total):

> "Well-"
> "Just G though and then B and J can get a redo next, and then we'll start the cycle over"
> "Trust" **[trust_appeal]**

Guilt indicators detected: trust_appeal (1)

### Case 12: Session `sa7mprty`, supergame5, Round 5, Player G (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available

**Messages** (3 total):

> "Now we do 0?"
> "Got it"
> "Doing 25 for me"

No explicit guilt indicators detected.

### Case 13: Session `sa7mprty`, supergame5, Round 5, Player R (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.0368

**Messages** (4 total):

> "J..."
> "Idk you guys"
> "It was just a cycle, so everyone gets the chance at the 55 if only one person goes 0"
> "B you can go and J will go next"

No explicit guilt indicators detected.

### Case 14: Session `irrzlgk2`, supergame2, Round 3, Player K (Group 4)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0242, valence: 0.0000, sadness: 0.1303, contempt: 0.1950, neutral: 99.5235

**Messages** (2 total):

> "whats that"
> "lets make same mone"

No explicit guilt indicators detected.

### Case 15: Session `irrzlgk2`, supergame2, Round 3, Player N (Group 4)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0249, valence: 0.0000, sadness: 0.1333, contempt: 0.3036, neutral: 99.3927
- **Chat sentiment** (VADER compound mean): 0.2009

**Messages** (2 total):

> "yea what"
> "yes"

No explicit guilt indicators detected.

### Case 16: Session `irrzlgk2`, supergame2, Round 4, Player K (Group 4)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0256, valence: 0.0000, sadness: 0.1284, contempt: 0.2800, neutral: 99.4400
- **Chat sentiment** (VADER compound mean): 0.2009

**Messages** (2 total):

> "whats that"
> "yes i am not doing it them"

No explicit guilt indicators detected.

### Case 17: Session `r5dj4yfl`, supergame3, Round 3, Player D (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0222, valence: 0.0000, sadness: 0.0835, contempt: 0.1251, neutral: 65.5829
- **Chat sentiment** (VADER compound mean): -0.1662

**Messages** (3 total):

> "sorry yall " **[apology]**
> "i thought q wasnt joining "
> "ill do 25 next time "

Guilt indicators detected: apology (1)

### Case 18: Session `r5dj4yfl`, supergame4, Round 6, Player A (Group 1)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 73.4741, valence: 70.5077, sadness: 0.2522, contempt: 0.0088, neutral: 19.3532

**Messages** (2 total):

> "Faze Rug"
> "just following my president"

No explicit guilt indicators detected.

### Case 19: Session `r5dj4yfl`, supergame4, Round 7, Player A (Group 1)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 76.6037, valence: 80.9372, sadness: 0.0109, contempt: 0.0007, neutral: 13.4105
- **Chat sentiment** (VADER compound mean): 0.2550

**Messages** (2 total):

> "Welcome to the stock market"
> "there was never any trust just profit margins" **[trust_appeal]**

Guilt indicators detected: trust_appeal (1)

### Case 20: Session `r5dj4yfl`, supergame5, Round 3, Player P (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.3740

**Messages** (3 total):

> "ok im on the 25 train now"
> "I'm not loyal but im honest haha"
> "25"

No explicit guilt indicators detected.

### Case 21: Session `r5dj4yfl`, supergame5, Round 4, Player P (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.0752

**Messages** (7 total):

> "run it back"
> "lets go"
> "25"
> "25"
> "25"
> "25"
> "yay"

No explicit guilt indicators detected.

### Case 22: Session `r5dj4yfl`, supergame5, Round 5, Player P (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.0872

**Messages** (8 total):

> "give me 10 more rounds"
> "make that bag"
> "keep adding "
> "$4"
> "$4"
> "$4"
> "ok 25s again"
> "yes"

No explicit guilt indicators detected.

### Case 23: Session `sylq2syi`, supergame1, Round 3, Player G (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0484, valence: 0.0123, sadness: 0.1312, contempt: 2.8089, neutral: 94.7108

**Messages** (1 total):

> "who did it"

No explicit guilt indicators detected.

### Case 24: Session `sylq2syi`, supergame4, Round 4, Player C (Group 1)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available

**Messages** (2 total):

> "yall who didn't put in"
> "same"

No explicit guilt indicators detected.

### Case 25: Session `sylq2syi`, supergame4, Round 5, Player C (Group 1)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): -0.1779

**Messages** (1 total):

> "seriously"

No explicit guilt indicators detected.

### Case 26: Session `sylq2syi`, supergame4, Round 6, Player C (Group 1)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available
- **Chat sentiment** (VADER compound mean): 0.3110

**Messages** (3 total):

> "ok i feel like we need to reset"
> "like if we all put in 25" **[deflection_collective]**
> "same"

Guilt indicators detected: deflection_collective (1)

### Case 27: Session `sylq2syi`, supergame4, Round 7, Player C (Group 1)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available

**Messages** (2 total):

> "omg im done"
> "huh"

No explicit guilt indicators detected.

### Case 28: Session `iiu3xixz`, supergame1, Round 3, Player E (Group 1)

- **Contribution**: 21 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions**: No data available

**Messages** (2 total):

> "Do the tokens reset each round?"
> "ooo"

No explicit guilt indicators detected.

### Case 29: Session `iiu3xixz`, supergame1, Round 3, Player L (Group 3)

- **Contribution**: 10 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0279, valence: -0.1792, sadness: 0.3097, contempt: 0.4728, neutral: 98.8338

**Messages** (1 total):

> "25"

No explicit guilt indicators detected.

### Case 30: Session `iiu3xixz`, supergame2, Round 3, Player A (Group 1)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0333, valence: 0.4187, sadness: 0.1200, contempt: 0.2023, neutral: 98.5287
- **Chat sentiment** (VADER compound mean): 0.3885

**Messages** (2 total):

> "cool"
> ":)"

No explicit guilt indicators detected.

### Case 31: Session `iiu3xixz`, supergame2, Round 3, Player L (Group 1)

- **Contribution**: 1 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 97.5823, valence: 97.4373, sadness: 0.0095, contempt: 0.0181, neutral: 0.0000
- **Chat sentiment** (VADER compound mean): 0.3125

**Messages** (2 total):

> "all in"
> "sounds great"

No explicit guilt indicators detected.

### Case 32: Session `iiu3xixz`, supergame2, Round 4, Player A (Group 1)

- **Contribution**: 5 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 1.8303, valence: 11.4908, sadness: 0.0826, contempt: 0.5117, neutral: 62.9459
- **Chat sentiment** (VADER compound mean): 0.2423

**Messages** (3 total):

> "i see how it is"
> "boooo"
> "lol you expect us to trust you?" **[trust_appeal]**

Guilt indicators detected: trust_appeal (1)

### Case 33: Session `iiu3xixz`, supergame2, Round 4, Player L (Group 1)

- **Contribution**: 7 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0241, valence: 0.0000, sadness: 0.1273, contempt: 0.1935, neutral: 99.6128
- **Chat sentiment** (VADER compound mean): 0.2009

**Messages** (2 total):

> "all in this time"
> "yes"

No explicit guilt indicators detected.

### Case 34: Session `iiu3xixz`, supergame3, Round 3, Player L (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0244, valence: -2.0583, sadness: 0.1665, contempt: 14.0241, neutral: 83.5443
- **Chat sentiment** (VADER compound mean): 0.0059

**Messages** (5 total):

> "i forgot to press 5"
> "do it again"
> "my bad" **[apology]**
> "yeah i agree"
> "lets do it"

Guilt indicators detected: apology (1)

### Case 35: Session `iiu3xixz`, supergame4, Round 4, Player L (Group 3)

- **Contribution**: 1 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0241, valence: 0.0000, sadness: 0.1651, contempt: 0.1959, neutral: 99.5749
- **Chat sentiment** (VADER compound mean): 0.3125

**Messages** (2 total):

> "great again?"
> "?"

No explicit guilt indicators detected.

### Case 36: Session `iiu3xixz`, supergame4, Round 5, Player L (Group 3)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 29.8482, valence: 26.1614, sadness: 0.0818, contempt: 0.1206, neutral: 66.2099

**Messages** (4 total):

> "how about 15"
> "can we do 15"
> "10"
> "thats reasonable"

No explicit guilt indicators detected.

### Case 37: Session `iiu3xixz`, supergame4, Round 6, Player L (Group 3)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0278, valence: 0.0000, sadness: 0.1224, contempt: 0.3250, neutral: 99.3501
- **Chat sentiment** (VADER compound mean): 0.2301

**Messages** (3 total):

> "alright we gotta add something to the pot"
> "15"
> "thats better for everyone"

No explicit guilt indicators detected.

### Case 38: Session `iiu3xixz`, supergame4, Round 7, Player L (Group 3)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.1602, valence: 2.2496, sadness: 1.0322, contempt: 21.9070, neutral: 66.2889

**Messages** (3 total):

> "everyone do 18"
> "everyone do 25"
> "then we can get 4"

No explicit guilt indicators detected.

### Case 39: Session `6ucza025`, supergame4, Round 4, Player G (Group 2)

- **Contribution**: 20 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0414, valence: -0.0109, sadness: 0.8323, contempt: 0.1718, neutral: 98.2827

**Messages** (2 total):

> "guys all in"
> "??"

No explicit guilt indicators detected.

### Case 40: Session `6ucza025`, supergame4, Round 5, Player G (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0300, valence: 0.9143, sadness: 0.1916, contempt: 0.1754, neutral: 51.6859
- **Chat sentiment** (VADER compound mean): 0.2202

**Messages** (2 total):

> "not maximizing prodit"
> "profit"

No explicit guilt indicators detected.

### Case 41: Session `6ucza025`, supergame4, Round 6, Player G (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0246, valence: 0.0000, sadness: 0.1492, contempt: 0.2065, neutral: 99.5714
- **Chat sentiment** (VADER compound mean): 0.1054

**Messages** (4 total):

> "nice guys"
> "10/10"
> "rtr"
> "again"

No explicit guilt indicators detected.

### Case 42: Session `6ucza025`, supergame4, Round 7, Player G (Group 2)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.1227, valence: -1.1374, sadness: 1.0315, contempt: 1.3849, neutral: 96.1076
- **Chat sentiment** (VADER compound mean): 0.5017

**Messages** (2 total):

> "yay"
> "proud"

No explicit guilt indicators detected.

### Case 43: Session `6sdkxl2q`, supergame2, Round 3, Player D (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.4106, valence: -2.4395, sadness: 6.9722, contempt: 0.1622, neutral: 85.5358
- **Chat sentiment** (VADER compound mean): -0.0515

**Messages** (3 total):

> "im putting 25 next round sorry yall" **[apology]**
> "sorry i didn't know we were all in " **[apology]**
> "yall dont have to since you put 25 in last time"

Guilt indicators detected: apology (2)

### Case 44: Session `6sdkxl2q`, supergame2, Round 4, Player C (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0246, valence: 0.0000, sadness: 0.1273, contempt: 0.1802, neutral: 98.8917
- **Chat sentiment** (VADER compound mean): -0.3592

**Messages** (2 total):

> "that made less than if we had all done 25"
> "the risk is a valid concern but prisoners dilema literally says to all do it "

No explicit guilt indicators detected.

### Case 45: Session `6sdkxl2q`, supergame2, Round 4, Player J (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.2813, valence: 6.5830, sadness: 0.0924, contempt: 0.1842, neutral: 99.1872

**Messages** (1 total):

> "if we all do 25 then we get 40 each " **[deflection_collective]**

Guilt indicators detected: deflection_collective (1)

### Case 46: Session `6sdkxl2q`, supergame2, Round 4, Player P (Group 3)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 2.3236, valence: 10.3847, sadness: 0.0917, contempt: 5.6878, neutral: 64.0737

**Messages** (2 total):

> "keep doing 10"
> "so 25?"

No explicit guilt indicators detected.

### Case 47: Session `6sdkxl2q`, supergame2, Round 4, Player D (Group 4)

- **Contribution**: 25 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 0.0273, valence: -25.4532, sadness: 16.3423, contempt: 0.5553, neutral: 70.4768
- **Chat sentiment** (VADER compound mean): 0.0903

**Messages** (4 total):

> "lets all go 25"
> "nah ur good "
> "lets go 25 in and get paid "
> "all in "

No explicit guilt indicators detected.

### Case 48: Session `6sdkxl2q`, supergame4, Round 7, Player J (Group 3)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 1.5188, valence: 17.7988, sadness: 0.1038, contempt: 0.1152, neutral: 92.6906

**Messages** (1 total):

> "turn of events..."

No explicit guilt indicators detected.

### Case 49: Session `6sdkxl2q`, supergame4, Round 7, Player L (Group 3)

- **Contribution**: 0 / 25
- **Promise made**: Yes (is_liar_20)
- **Facial emotions** (Contribute page): joy: 31.9948, valence: 43.1430, sadness: 3.4529, contempt: 0.0177, neutral: 56.5816
- **Chat sentiment** (VADER compound mean): 0.5859

**Messages** (1 total):

> "XD"

No explicit guilt indicators detected.

---
*Total cases with messages shown: 49*

## Conclusion: Are Liars Happy When They Pretend to Be Guilty?

### Key Findings

1. **Guilt expression is rare**: Only 10 of 123 liar instances (8.1%) contained any guilt-related language (apologies, remorse, future promises, or collective deflection).

2. **Dominant strategy -- deflection and future promises**: Rather than expressing genuine guilt, liars more commonly used collective deflection ("we all should...") or made promises for future rounds.

3. **Facial emotions**: Liars who used guilt-related language showed **higher** facial joy (13.1954) compared to non-guilt-expressing liars (10.8489), suggesting their guilt expressions may not reflect genuine remorse.

4. **Valence**: Emotional valence was **more positive** for guilt-expressing liars (15.7522 vs 10.5979), consistent with the hypothesis that liars feel satisfaction (duping delight) even while performing guilt.

### Interpretation

The evidence tentatively supports the 'duping delight' hypothesis: liars who deploy guilt-laden language in chat display **more positive facial affect**, not less. This pattern is consistent with strategic guilt performance -- players who verbally express remorse appear to be emotionally unbothered, or even pleased, during the contribution decision. The guilt expression functions as a social tool for maintaining group cooperation norms while personally free-riding.

### Caveats

- Sample size is limited (123 liar instances, 88 with facial data).
- Guilt detection uses keyword matching, which may miss subtle or implicit guilt expressions.
- Facial emotion data is aggregated over the entire Contribute page, not synchronized to specific moments.
- No statistical significance testing was performed; differences should be treated as descriptive.