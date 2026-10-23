
"""Rolling out-of-sample VaR backtesting."""

import numpy as np
import pandas as pd

from src.risk_models import (
    historical_var,
    gaussian_var,
    student_t_var,
)


def rolling_var_backtest(
    returns,
    window=500,
    confidence=0.95,
):
    """Forecast next-day VaR using a rolling window of prior returns.

    Parameters
    ----------
    returns : array-like
        Daily simple or log returns, expressed consistently.
    window : int
        Number of past observations used for each forecast.
    confidence : float
        VaR confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    pandas.DataFrame
        Date/index, realized return/loss, and VaR forecasts.
        VaR values are positive loss thresholds. A breach occurs
        when realized loss exceeds the forecast VaR.
    """
    values = np.asarray(returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("Returns must be one-dimensional.")

    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be strictly between 0 and 1.")

    if not isinstance(window, int) or window < 10:
        raise ValueError("window must be an integer of at least 10.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Returns must not contain missing or infinite values.")

    if len(values) <= window:
        raise ValueError("Returns length must exceed the rolling window.")

    records = []

    for i in range(window, len(values)):
        training_returns = values[i - window:i]
        realized_return = values[i]
        realized_loss = -realized_return

        records.append(
            {
                "position": i,
                "realized_return": realized_return,
                "realized_loss": realized_loss,
                "historical_var": historical_var(
                    training_returns, confidence
                ),
                "gaussian_var": gaussian_var(
                    training_returns, confidence
                ),
                "student_t_var": student_t_var(
                    training_returns, confidence
                ),
            }
        )

    result = pd.DataFrame(records)
    result["confidence"] = confidence

    for model in ("historical", "gaussian", "student_t"):
        result[f"{model}_breach"] = (
            result["realized_loss"] > result[f"{model}_var"]
        )

    return result