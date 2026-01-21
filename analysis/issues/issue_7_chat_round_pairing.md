# Issue #7: Chat-Round Pairing Bug

## Problem
Chat messages were incorrectly paired with the same round's contribution decision instead of the next round's contribution decision (which the chat actually influenced).

In the oTree experiment, the sequence within each round is:
1. Contribution decision is made
2. Chat occurs (after seeing results)

This means round N's chat discussion influences round N+1's contribution, not round N's (since chat happens after contribution in round N).

## Impact
- Analysis incorrectly associated chat content with decisions made before the chat occurred
- Promise classification and sentiment analysis paired with wrong behavioral outcomes
- Violated temporal causality in data structure

## Solution
- Updated `experiment_data.py` to pair chat from round N with round N+1's contribution
- Added `orphan_chats` property to `Segment` class to handle last round chat (which doesn't influence any subsequent decision)
- Round 1 now correctly has no chat (no prior round to influence it)
- Updated all analysis scripts to use corrected pairing
- Moved `classify_promises.py` and `llm_clients/` to `analysis/derived/` for better organization
- Added comprehensive test coverage with 6 new chat pairing tests

## Testing
- All 122 tests pass
- New tests verify correct pairing using raw CSV data (parsing channel format)
- Tests confirm orphan chat handling
- Verified with 3 traced examples from raw data

## Files Modified
- `analysis/experiment_data.py` - Core fix with round shift logic
- `analysis/analysis/analysis_plots.py` - Simplified sentiment correlation
- `analysis/analysis/multi_session_analysis.py` - Added orphan support
- `analysis/derived/classify_promises.py` - Moved and updated
- `analysis/derived/llm_clients/` - Moved package
- `analysis/tests/conftest.py` - Added chat_pairing_helper fixture
- `analysis/tests/test_chat.py` - Added 6 new chat pairing tests
- `analysis/tests/test_integration.py` - Added integration tests
- `analysis/tests/test_output_data_integrity.py` - Updated for new semantics
- `CLAUDE.md` - Updated documentation
