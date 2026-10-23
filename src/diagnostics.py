
"""Statistical diagnostics for Value-at-Risk backtests."""

import numpy as np
from scipy.stats import chi2


def kupiec_test(breaches, confidence=0.95):
    """Kupiec unconditional-coverage test.

    Tests whether the observed breach rate differs from the expected
    rate of 1 - confidence.

    Returns the LR statistic, p-value, observed breach rate, and
    expected breach rate.
    """
    breaches = np.asarray(breaches, dtype=bool)

    if breaches.ndim != 1 or breaches.size == 0:
        raise ValueError("Breaches must be a non-empty 1-D array.")

    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be strictly between 0 and 1.")

    n = breaches.size
    x = int(breaches.sum())
    expected_rate = 1.0 - confidence
    observed_rate = x / n

    # Avoid log(0) when there are zero or all breaches.
    observed_term = 0.0
    if x > 0:
        observed_term += x * np.log(observed_rate / expected_rate)

    if x < n:
        observed_term += (n - x) * np.log(
            (1.0 - observed_rate) / (1.0 - expected_rate)
        )

    lr_statistic = 2.0 * observed_term
    p_value = float(chi2.sf(lr_statistic, df=1))

    return {
        "observations": int(n),
        "breaches": int(x),
        "observed_rate": float(observed_rate),
        "expected_rate": float(expected_rate),
        "lr_statistic": float(lr_statistic),
        "p_value": p_value,
    }


def breach_clustering(breaches):
    """Summarize consecutive VaR breaches and breach dependence."""
    breaches = np.asarray(breaches, dtype=bool)

    if breaches.ndim != 1 or breaches.size == 0:
        raise ValueError("Breaches must be a non-empty 1-D array.")

    breach_count = int(breaches.sum())

    # Count transitions from a non-breach to a breach.
    starts = breaches & np.concatenate(([True], ~breaches[:-1]))
    clusters = int(starts.sum())

    # Count adjacent breach pairs.
    adjacent_pairs = int(np.sum(breaches[:-1] & breaches[1:]))

    return {
        "breaches": breach_count,
        "breach_clusters": clusters,
        "adjacent_breach_pairs": adjacent_pairs,
        "max_consecutive_breaches": _max_consecutive(breaches),
    }


def _max_consecutive(values):
    """Return the longest consecutive run of True values."""
    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)