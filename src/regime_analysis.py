
"""Market-regime analysis for VaR model diagnostics."""

import numpy as np
import pandas as pd


def analyze_volatility_regimes(
    returns,
    window=20,
    stress_quantile=0.90,
):
    """Classify calm and high-volatility periods using past returns.

    Rolling volatility uses only observations available up to each date.
    The stress threshold is the requested quantile of valid rolling
    volatility estimates across the supplied sample.
    """
    values = np.asarray(returns, dtype=float)

    if values.ndim != 1:
        raise ValueError("Returns must be one-dimensional.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Returns must not contain missing or infinite values.")

    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2.")

    if len(values) < window:
        raise ValueError("Returns length must be at least the rolling window.")

    if not 0.0 < stress_quantile < 1.0:
        raise ValueError("stress_quantile must be strictly between 0 and 1.")

    series = pd.Series(values, name="return")
    rolling_volatility = series.rolling(window=window).std(ddof=1)

    valid_volatility = rolling_volatility.dropna()
    threshold = float(valid_volatility.quantile(stress_quantile))

    result = pd.DataFrame({
        "realized_return": series,
        "rolling_volatility": rolling_volatility,
    })

    result["volatility_threshold"] = threshold
    result["regime"] = np.where(
        result["rolling_volatility"].isna(),
        "insufficient_history",
        np.where(
            result["rolling_volatility"] > threshold,
            "high_volatility",
            "normal_volatility",
        ),
    )

    return result


def summarize_breaches_by_regime(regime_data, backtest_data):
    """Summarize model breach rates by volatility regime.

    The two inputs must be aligned row-for-row and have the same length.
    """
    if len(regime_data) != len(backtest_data):
        raise ValueError("Regime and backtest data must have equal lengths.")

    if "regime" not in regime_data:
        raise ValueError("regime_data must contain a 'regime' column.")

    breach_columns = [
        column for column in backtest_data.columns
        if column.endswith("_breach")
    ]

    if not breach_columns:
        raise ValueError("No breach columns were found in backtest_data.")

    combined = regime_data[["regime"]].reset_index(drop=True).join(
        backtest_data[breach_columns].reset_index(drop=True)
    )

    combined = combined[
        combined["regime"] != "insufficient_history"
    ]

    summary = combined.groupby("regime")[breach_columns].mean()
    summary = summary.rename(
        columns=lambda column: column.replace("_breach", "_breach_rate")
    )

    return summary.reset_index()