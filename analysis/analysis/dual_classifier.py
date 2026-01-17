"""
Dual classifier that runs both OpenAI and Anthropic classifiers and determines consensus.

Provides functions to classify messages using both LLM providers simultaneously,
detect agreement/disagreement, and calculate overall agreement rates for batches.

Author: Claude Code
Date: 2026-01-16
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from llm_clients import classify_promise_openai, classify_promise_anthropic


# =====
# Main classification function
# =====
def classify_message_dual(message: str, context: Optional[list] = None) -> dict:
    """
    Classify a message using both OpenAI and Anthropic in parallel, returning consensus.

    Args:
        message: The text to classify
        context: Optional list of prior message dicts with 'sender' and 'body' keys

    Returns:
        Dict with 'openai', 'anthropic', 'consensus', 'disputed', 'openai_raw', 'anthropic_raw'
    """
    context = context or []

    with ThreadPoolExecutor(max_workers=2) as executor:
        openai_future = executor.submit(classify_promise_openai, message, context)
        anthropic_future = executor.submit(classify_promise_anthropic, message, context)
        openai_result = openai_future.result()
        anthropic_result = anthropic_future.result()

    return _build_dual_result(openai_result, anthropic_result)


# =====
# Batch processing
# =====
def classify_batch_parallel(messages_with_context: list, max_workers: int = 20) -> list:
    """
    Classify a batch of messages in parallel with progress logging.

    Args:
        messages_with_context: List of dicts with 'message' and 'context' keys
        max_workers: Maximum concurrent API calls (default 20)

    Returns:
        List of classification result dicts in original order
    """
    total = len(messages_with_context)
    results = [None] * total
    completed = [0]

    def classify_item(idx_item):
        idx, item = idx_item
        msg = item.get('message', '')
        ctx = item.get('context', [])
        return idx, classify_message_dual(msg, ctx)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_item, (i, item)): i
                   for i, item in enumerate(messages_with_context)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed[0] += 1
            _log_progress(completed[0], total)

    _log_final_stats(results)
    return results


def classify_batch(messages_with_context: list) -> list:
    """
    Classify a batch of messages sequentially (legacy function).

    Args:
        messages_with_context: List of dicts with 'message' and 'context' keys

    Returns:
        List of classification result dicts
    """
    return classify_batch_parallel(messages_with_context, max_workers=1)


# =====
# Agreement rate calculation
# =====
def calculate_agreement_rate(results: list) -> float:
    """
    Calculate the agreement rate between classifiers.

    Args:
        results: List of classification result dicts from classify_message_dual

    Returns:
        Float from 0.0 to 1.0 representing agreement percentage
    """
    if not results:
        return 0.0

    agreed_count = sum(1 for r in results if not r.get('disputed', True))
    return agreed_count / len(results)


# =====
# Helper functions
# =====
def _build_dual_result(openai_result: dict, anthropic_result: dict) -> dict:
    """Build the combined result dict from both classifier outputs."""
    openai_class = openai_result.get('classification', 0)
    anthropic_class = anthropic_result.get('classification', 0)

    disputed = openai_class != anthropic_class
    consensus = None if disputed else openai_class

    return {
        'openai': openai_class,
        'anthropic': anthropic_class,
        'consensus': consensus,
        'disputed': disputed,
        'openai_raw': openai_result.get('raw_response', ''),
        'anthropic_raw': anthropic_result.get('raw_response', ''),
    }


def _log_progress(current: int, total: int) -> None:
    """Log progress every 100 messages."""
    if current % 100 == 0 or current == total:
        pct = current / total * 100
        print(f"Processed {current}/{total} messages ({pct:.1f}%)", flush=True)


def _log_final_stats(results: list) -> None:
    """Print agreement statistics at end of batch."""
    agreement_rate = calculate_agreement_rate(results)
    print(f"Agreement rate: {agreement_rate:.1%} ({len(results)} messages)", flush=True)
