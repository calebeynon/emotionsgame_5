"""
Manually-verified test cases for the issue_72_panel (lying-contagion panel).

Each case was traced by inspecting lied_this_round_20 values in
behavior_classifications.csv for the exact (session, segment, group, label).
Constants are imported by analysis/tests/test_issue_72_panel.py.

Author: Claude Code
Date: 2026-04-20
"""

REQUIRED_COLUMNS = [
    "session_code", "treatment", "segment", "round", "group", "label",
    "lied", "self_lied_lag", "group_lied_lag",
    "any_self_lied_prior", "any_group_lied_prior",
    "cluster_group", "label_session",
]
BINARY_COLUMNS = [
    "lied", "group_lied_lag", "self_lied_lag",
    "any_group_lied_prior", "any_self_lied_prior",
]

# CASE A — sole liar in prior round, tests self-exclusion.
# 6sdkxl2q / supergame2 / round 2 / group 4: only D lied in round 2.
# In round 3, D's group_lied_lag should be 0 (self-excluded);
# G, K, N's group_lied_lag should be 1 (D, who is not them, lied).
SOLE_LIAR_CASE_A = {
    "session_code": "6sdkxl2q",
    "segment": "supergame2",
    "round": 3,
    "group": 4,
    "sole_liar_label": "D",
    "other_labels": ("G", "K", "N"),
}

# CASE B — second sole-liar case for robustness.
# iiu3xixz / supergame1 / round 2 / group 3: only L lied in round 2.
SOLE_LIAR_CASE_B = {
    "session_code": "iiu3xixz",
    "segment": "supergame1",
    "round": 3,
    "group": 3,
    "sole_liar_label": "L",
    "other_labels": ("C", "G", "Q"),
}

# CASE C — never-liar: B in 6sdkxl2q lied 0 times across all 22 rounds.
# Every panel row for B in 6sdkxl2q must have any_self_lied_prior == 0.
NEVER_LIAR_CASE = {
    "session_code": "6sdkxl2q",
    "label": "B",
}

# CASE D — bug-regression case for self-exclusion in any_group_lied_prior.
# sa7mprty / supergame4 / group 4: R lied in round 4; N lied in round 5.
# For R in round 6, any_group_lied_prior must be 1 (N lied prior and is NOT R).
# With the original max-minus-self bug this was 0 because R had also lied.
BUG_REGRESSION_CASE = {
    "session_code": "sa7mprty",
    "segment": "supergame4",
    "group": 4,
    "round": 6,
    "label": "R",
    "expected_any_self_lied_prior": 1,
    "expected_any_group_lied_prior": 1,
    "expected_group_lied_lag": 1,
    "expected_self_lied_lag": 0,
    "expected_lied": 0,
}

# CASE E — same-round group_lied_lag bug-regression.
# iiu3xixz / supergame2 / group 1: A AND L both lied in round 2.
# For L in round 3: self_lied_lag=1 AND group_lied_lag=1 (A also lied).
# With the original max-minus-self bug this collapsed to 0 (group_max=1, self_lag=1).
SAME_ROUND_BOTH_LIED_CASE = {
    "session_code": "iiu3xixz",
    "segment": "supergame2",
    "group": 1,
    "round": 3,
    "label": "L",
    "expected_lied": 1,
    "expected_self_lied_lag": 1,
    "expected_group_lied_lag": 1,
    "expected_any_self_lied_prior": 1,
    "expected_any_group_lied_prior": 1,
}

# CASE F — all four groupmates lied together.
# sa7mprty / supergame1 / group 3 round 2: C, G, L, Q all lied simultaneously.
# In round 3, each of the four must have:
#   self_lied_lag=1, group_lied_lag=1, any_self_lied_prior=1, any_group_lied_prior=1.
# Strongest test of sum-minus-self arithmetic: even when every group member lied
# (including self), each player must still see a groupmate's lie.
ALL_GROUP_LIED_CASE = {
    "session_code": "sa7mprty",
    "segment": "supergame1",
    "group": 3,
    "round": 3,
    "all_labels": ("C", "G", "L", "Q"),
}
