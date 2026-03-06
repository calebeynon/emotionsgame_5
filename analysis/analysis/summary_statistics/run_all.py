"""
Purpose: Run all summary statistics modules and report generated output files.
Author: Caleb Eynon
Date: 2026-03-02
"""

import sys
import time
import traceback
from pathlib import Path

# Allow imports from this package when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ss_common import OUTPUT_DIR

# FILE PATHS
_SCRIPT_DIR = Path(__file__).resolve().parent


# =====
# Main function
# =====

def main():
    """Run all summary statistics modules and print output summary."""
    modules = _get_modules()
    results = []
    for name, func in modules:
        success, elapsed = run_module(name, func)
        results.append((name, success, elapsed))
    print_summary(results)
    if not all(success for _, success, _ in results):
        sys.exit(1)


def _get_modules():
    """Import and return list of (name, main_func) for all ss_* modules."""
    from ss_behavior import main as behavior_main
    from ss_chat import main as chat_main
    from ss_contributions import main as contributions_main
    from ss_demographics import main as demographics_main
    from ss_experiment_totals import main as experiment_totals_main
    from ss_groups import main as groups_main
    from ss_payoffs import main as payoffs_main
    from ss_sentiment import main as sentiment_main
    return [
        ('ss_contributions', contributions_main),
        ('ss_chat', chat_main),
        ('ss_sentiment', sentiment_main),
        ('ss_behavior', behavior_main),
        ('ss_payoffs', payoffs_main),
        ('ss_groups', groups_main),
        ('ss_demographics', demographics_main),
        ('ss_experiment_totals', experiment_totals_main),
    ]


# =====
# Module execution
# =====

def run_module(name, func):
    """Run a single module's main() and return (success, elapsed_seconds)."""
    print(f'Running {name}...')
    start = time.time()
    try:
        func()
        elapsed = round(time.time() - start, 2)
        print(f'  Done ({elapsed}s)')
        return True, elapsed
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f'  FAILED ({elapsed}s): {e}')
        traceback.print_exc()
        return False, elapsed


def print_summary(results):
    """Print run results and list all output files."""
    print('\n' + '=' * 50)
    print('Summary Statistics Run Complete')
    print('=' * 50)
    _print_module_results(results)
    _print_output_files()


def _print_module_results(results):
    """Print pass/fail status for each module."""
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f'\nModules: {passed}/{total} succeeded')
    for name, success, elapsed in results:
        status = 'PASS' if success else 'FAIL'
        print(f'  [{status}] {name} ({elapsed}s)')


def _print_output_files():
    """List all files in the output directory."""
    if not OUTPUT_DIR.exists():
        print('\nNo output directory found.')
        return
    files = sorted(OUTPUT_DIR.iterdir())
    print(f'\nOutput files ({len(files)}):')
    for f in files:
        size_kb = round(f.stat().st_size / 1024, 1)
        print(f'  {f.name} ({size_kb}KB)')


# %%
if __name__ == "__main__":
    main()
