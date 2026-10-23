"""Run the VaR Model Failure Detector research pipeline."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Support saving plots without a graphical display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting import rolling_var_backtest
from src.diagnostics import kupiec_test, breach_clustering
from src.regime_analysis import (
    analyze_volatility_regimes,
    summarize_breaches_by_regime,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "market_prices.csv"
REPORTS_DIR = ROOT / "reports"

ROLLING_WINDOW = 500
CONFIDENCE_LEVEL = 0.95
VOLATILITY_WINDOW = 20
STRESS_QUANTILE = 0.90


def create_research_plots(backtest, regimes, reports_dir):
    """Save visual diagnostics for VaR forecasts and market regimes."""

    # Plot 1: Realized losses versus VaR thresholds.
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        backtest["Date"],
        backtest["realized_loss"] * 100,
        label="Realized daily loss",
        color="black",
        linewidth=0.65,
        alpha=0.55,
    )

    for model, label in (
        ("historical", "Historical VaR"),
        ("gaussian", "Gaussian VaR"),
        ("student_t", "Student-t VaR"),
    ):
        ax.plot(
            backtest["Date"],
            backtest[f"{model}_var"] * 100,
            label=label,
            linewidth=0.9,
            alpha=0.85,
        )

    ax.set_title("Realized Daily Losses vs Rolling 95% VaR")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss / VaR threshold (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports_dir / "var_forecasts.png", dpi=160)
    plt.close(fig)

    # Plot 2: Observed breach rates against the expected rate.
    models = ("historical", "gaussian", "student_t")
    rates = [
        backtest[f"{model}_breach"].mean() * 100
        for model in models
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, rates)

    ax.axhline(
        (1 - CONFIDENCE_LEVEL) * 100,
        linestyle="--",
        label=f"Expected rate ({1 - CONFIDENCE_LEVEL:.0%})",
    )
    ax.set_title("Out-of-Sample VaR Breach Rates")
    ax.set_xlabel("VaR model")
    ax.set_ylabel("Breach rate (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    fig.savefig(reports_dir / "breach_rates.png", dpi=160)
    plt.close(fig)

    # Plot 3: Rolling volatility and retrospective stress threshold.
    valid = regimes.dropna(subset=["rolling_volatility"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        valid["Date"],
        valid["rolling_volatility"] * 100,
        label=f"{VOLATILITY_WINDOW}-day rolling volatility",
        linewidth=0.8,
    )
    ax.axhline(
        valid["volatility_threshold"].iloc[0] * 100,
        linestyle="--",
        label=f"Full-sample {STRESS_QUANTILE:.0%} threshold",
    )

    ax.set_title("Market Volatility Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(reports_dir / "volatility_regimes.png", dpi=160)
    plt.close(fig)

    print("\nSaved research plots:")
    print("  - var_forecasts.png")
    print("  - breach_rates.png")
    print("  - volatility_regimes.png")


def main():
    print("=" * 65)
    print("WHEN DOES VaR STOP WORKING?")
    print("A Diagnostic Framework for Market-Risk Model Failure")
    print("=" * 65)

    # 1. Load and validate market prices.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Market data not found: {DATA_PATH}")

    prices = pd.read_csv(DATA_PATH)
    required_columns = {"Date", "Close"}

    if not required_columns.issubset(prices.columns):
        raise ValueError("CSV must contain Date and Close columns.")

    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    prices = prices.dropna(subset=["Date", "Close"])
    prices = prices[np.isfinite(prices["Close"]) & (prices["Close"] > 0)]
    prices = prices.sort_values("Date")
    prices = prices.drop_duplicates(subset="Date", keep="last")
    prices = prices.reset_index(drop=True)

    if len(prices) <= ROLLING_WINDOW + 1:
        raise ValueError(
            "Not enough valid price observations for the rolling window."
        )

    # 2. Calculate daily simple returns.
    prices["return"] = prices["Close"].pct_change()
    valid = prices["return"].notna()

    returns = prices.loc[valid, "return"].reset_index(drop=True)
    return_dates = prices.loc[valid, "Date"].reset_index(drop=True)

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns contain non-finite values.")

    if (returns <= -1).any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    print(f"\nPrice observations: {len(prices):,}")
    print(f"First date:         {prices['Date'].min().date()}")
    print(f"Last date:          {prices['Date'].max().date()}")
    print(f"Daily returns:      {len(returns):,}")
    print(f"Rolling window:     {ROLLING_WINDOW} observations")
    print(f"VaR confidence:     {CONFIDENCE_LEVEL:.0%}")

    # 3. Generate rolling out-of-sample VaR forecasts.
    print("\nRunning rolling VaR backtest...")

    backtest = rolling_var_backtest(
        returns.to_numpy(),
        window=ROLLING_WINDOW,
        confidence=CONFIDENCE_LEVEL,
    )

    forecast_dates = return_dates.iloc[ROLLING_WINDOW:].reset_index(drop=True)
    backtest.insert(0, "Date", forecast_dates)

    # 4. Calculate coverage and breach-clustering diagnostics.
    diagnostic_rows = []

    for model in ("historical", "gaussian", "student_t"):
        breaches = backtest[f"{model}_breach"].to_numpy()

        coverage = kupiec_test(
            breaches,
            confidence=CONFIDENCE_LEVEL,
        )
        clustering = breach_clustering(breaches)

        diagnostic_rows.append({
            "model": model,
            **coverage,
            **{
                f"cluster_{key}": value
                for key, value in clustering.items()
                if key != "breaches"
            },
        })

    diagnostics = pd.DataFrame(diagnostic_rows)

    # 5. Analyze volatility regimes.
    print("Analyzing volatility regimes...")

    regimes = analyze_volatility_regimes(
        returns.to_numpy(),
        window=VOLATILITY_WINDOW,
        stress_quantile=STRESS_QUANTILE,
    )
    regimes.insert(0, "Date", return_dates)

    # Align regime rows to the out-of-sample forecast dates.
    aligned_regimes = regimes.iloc[ROLLING_WINDOW:].reset_index(drop=True)

    regime_summary = summarize_breaches_by_regime(
        aligned_regimes,
        backtest,
    )

    # 6. Save tabular results and plots.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    backtest.to_csv(
        REPORTS_DIR / "var_backtest_results.csv",
        index=False,
    )
    diagnostics.to_csv(
        REPORTS_DIR / "diagnostics_summary.csv",
        index=False,
    )
    regime_summary.to_csv(
        REPORTS_DIR / "regime_breach_summary.csv",
        index=False,
    )
    regimes.to_csv(
        REPORTS_DIR / "volatility_regimes.csv",
        index=False,
    )

    create_research_plots(backtest, regimes, REPORTS_DIR)

    # 7. Print summary diagnostics.
    print("\n" + "=" * 65)
    print("BACKTEST DIAGNOSTICS")
    print("=" * 65)

    display_columns = [
        "model",
        "observations",
        "breaches",
        "observed_rate",
        "expected_rate",
        "p_value",
        "cluster_max_consecutive_breaches",
    ]

    print(
        diagnostics[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nBREACH RATES BY VOLATILITY REGIME")
    print(
        regime_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nInterpretation guide:")
    print("- Kupiec p-value below 0.05: reject correct unconditional coverage.")
    print("- A high breach rate during stress may indicate regime sensitivity.")
    print("- Breach frequency alone does not establish overall model quality.")
    print("- The volatility threshold is retrospective and uses the full sample.")

    print("\nSaved results in:")
    print(f"  {REPORTS_DIR}")
    for filename in (
        "var_backtest_results.csv",
        "diagnostics_summary.csv",
        "regime_breach_summary.csv",
        "volatility_regimes.csv",
        "var_forecasts.png",
        "breach_rates.png",
        "volatility_regimes.png",
    ):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()