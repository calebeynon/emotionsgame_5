---
title: "Liar Flag: Cumulative vs Round-Specific"
type: method
tags: [liar, behavior-classification, dot-plots, methodology]
summary: "Comparison of cumulative (is_liar_20) and round-specific (lied_this_round_20) liar flag approaches for analysis"
status: draft
last_verified: "2026-04-06"
---

## Summary

Two approaches exist for flagging "liar" behavior in the public goods game: a cumulative flag (`is_liar_20`) that marks a player as a liar from the first instance onward, and a round-specific flag (`lied_this_round_20`) that only marks rounds where the player actively lied. Both have been implemented as R dot plots, and the choice of which to use in the paper depends on the research question being addressed.

## Key Points

- **Cumulative (`is_liar_20`)**: Once a player lies (promises high contribution but contributes below threshold), they are flagged as a liar for all subsequent rounds. Captures the persistent "reputation" or behavioral type.
- **Round-specific (`lied_this_round_20`)**: Only flags the specific rounds where lying occurred. Captures the act of lying as a time-varying behavior.
- Both versions use the threshold of 20 (out of 25 endowment) to define "high contribution" promises.

## Details

The distinction matters for different analytical goals:

- **Type-based analysis**: Use cumulative flag when asking "how do liars behave over time?" — treating liar as a player characteristic.
- **Event-based analysis**: Use round-specific flag when asking "what happens in rounds where lying occurs?" — treating lying as a situational behavior.

Both versions have been implemented in the R dot plot scripts. Compare outputs to decide which version(s) to include in the final paper.

## Related

- [Analysis Pipeline](../tools/analysis-pipeline.md)
