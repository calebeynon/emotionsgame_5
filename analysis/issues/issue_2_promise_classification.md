# Issue #2: Promise Classification

## Description
Implement automated promise classification for chat messages in the public goods game experiment. Promises are commitments to contribute a certain amount to the group account, and identifying them is crucial for analyzing how communication affects cooperation behavior.

## Implementation Approach
- Use OpenAI GPT-5-mini for LLM-based classification with context-dependent rules
- Context-aware classification:
  - Proposals only count as promises if accepted by others in the conversation
  - Affirmations ("ok", "yes") only count if responding to proposals
  - "Same"/"me too" only count if responding to commitments
- Parallel processing with 20 workers for efficient batch classification
- Output at player-round level with promise counts and percentages

## Files Modified
- `analysis/analysis/classify_promises.py` - Main classification script
- `analysis/analysis/dual_classifier.py` - Dual LLM classification (deprecated, kept for reference)
- `analysis/analysis/llm_clients/` - OpenAI, Anthropic, and prompt template modules
- `analysis/tests/` - Test suite for all classification components
- `pyproject.toml` - Added openai, anthropic, python-dotenv dependencies

## Expected Outcomes
- CSV file at `analysis/datastore/derived/promise_classifications.csv` with:
  - 2,897 player-rounds classified
  - 5,944 messages analyzed
  - Promise counts and percentages per player-round
  - Individual message classifications
- Enable analysis of promise-keeping behavior, cooperation dynamics, and communication patterns

## Testing Plan
- Unit tests for all LLM client modules
- Manual review of sample classifications against behavioral expectations
- Comparison between OpenAI and Anthropic classifications (84.9% agreement achieved)
- Final system uses only OpenAI due to superior accuracy

## Results
- 21.6% promise rate (1,281 promises out of 5,944 messages)
- Classification cost: ~$0.21 using GPT-5-mini
- Context-dependent rules improved classification accuracy significantly

## Related Issues
- Relates to broader experiment data analysis workflow
- Output data will be used for regression analysis of cooperation behavior
