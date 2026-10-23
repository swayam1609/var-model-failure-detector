
"""VaR models for daily market-loss returns.

All outputs represent positive loss thresholds.
Returns and volatility must use the same units.
"""

import numpy as np
from scipy.stats import norm, t


def _validate_returns(returns, min_observations=2):
    """Return finite 1-D observations or raise ValueError."""
    values = np.asarray(returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("Returns must be a one-dimensional array.")

    values = values[np.isfinite(values)]

    if values.size < min_observations:
        raise ValueError(
            f"At least {min_observations} finite observations are required."
        )

    return values


def historical_var(returns, confidence=0.95):
    """Historical-simulation VaR as a positive loss threshold."""
    values = _validate_returns(returns)
    _validate_confidence(confidence)

    losses = -values
    return float(np.quantile(losses, confidence, method="higher"))


def gaussian_var(returns, confidence=0.95):
    """Parametric Gaussian VaR, using sample standard deviation."""
    values = _validate_returns(returns)
    _validate_confidence(confidence)

    mean = np.mean(values)
    volatility = np.std(values, ddof=1)
    loss_quantile = -(mean + norm.ppf(1.0 - confidence) * volatility)

    return float(loss_quantile)


def student_t_var(returns, confidence=0.95):
    """Student-t VaR with degrees of freedom fitted to standardized returns.

    The fitted Student-t distribution is rescaled to match the sample mean
    and sample standard deviation. Requires finite variance (df > 2).
    """
    values = _validate_returns(returns, min_observations=10)
    _validate_confidence(confidence)

    mean = np.mean(values)
    volatility = np.std(values, ddof=1)

    if volatility == 0:
        return float(-mean)

    standardized = (values - mean) / volatility

    degrees_freedom, _, _ = t.fit(standardized, floc=0, fscale=1)

    if not np.isfinite(degrees_freedom) or degrees_freedom <= 2:
        raise ValueError(
            "Fitted Student-t degrees of freedom must exceed 2."
        )

    # Standardize t to unit variance before matching sample volatility.
    scale = volatility * np.sqrt((degrees_freedom - 2) / degrees_freedom)
    loss_quantile = -(mean + scale * t.ppf(1.0 - confidence, degrees_freedom))

    return float(loss_quantile)


def _validate_confidence(confidence):
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be strictly between 0 and 1.")