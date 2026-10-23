# When Does VaR Stop Working?
### A Diagnostic Framework for Market-Risk Model Failure

**Quantitative Finance · Market Risk · Statistical Backtesting · Python**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance-16324F)](#)
[![Focus](https://img.shields.io/badge/Focus-Market%20Risk-2E8B57)](#)

---

## 1. The Research Problem

Financial institutions use risk models to estimate how much they could lose over a given time horizon. These estimates influence trading limits, capital allocation, portfolio construction, and risk management.

One of the most widely used measures is **Value at Risk (VaR)**. At a 95% confidence level, a daily VaR estimate represents a loss threshold that the model expects to be exceeded on approximately 5% of days, under the model's assumptions.

But financial markets do not always behave as models expect.

Volatility changes, extreme returns occur, and losses may cluster during periods of market stress. A model that appears reliable under ordinary conditions may produce too many breaches when market conditions change. Conversely, a model that produces very few breaches may be excessively conservative.

This project investigates a practical question:

> **When does a VaR model stop representing the risk observed in the market, and how can backtesting diagnostics reveal that deterioration?**

Using daily Microsoft (MSFT) closing-price data from February 2000 to December 2025, the project compares three VaR approaches through rolling out-of-sample forecasts, statistical coverage testing, breach-clustering diagnostics, and volatility-regime analysis.

## 2. Why This Concept Matters

VaR is a threshold, not a prediction of the exact loss on a particular day. Its usefulness depends on whether the observed losses behave consistently with the model's assumptions.

Consider a 95% daily VaR of 2%. The model is saying that, according to its estimated return distribution, the daily loss should exceed 2% on approximately 5% of days.

Three situations are possible:

- **Too many breaches:** the model may underestimate risk.
- **Approximately the expected number:** the model's overall breach frequency is consistent with its target.
- **Too few breaches:** the model may be conservative, although low breach frequency alone does not prove this.

Even when the overall breach rate is close to 5%, the model can still be unreliable if breaches cluster during market stress. Therefore, a complete evaluation must examine both how often breaches occur and how they are distributed over time.

## 3. Methodology

### 3.1 Data and Return Construction

The framework uses historical daily closing prices to calculate daily simple returns. Let \(P_t\) denote the closing price at time \(t\). The daily return is

$$
r_t=\frac{P_t-P_{t-1}}{P_{t-1}}
$$

The corresponding loss is defined as

$$
L_t=-r_t
$$

Positive losses represent negative returns, so a Value at Risk (VaR) forecast can be interpreted as a loss threshold.

The analysis uses a rolling estimation window of \(W=500\) observations. At each forecast date, the models use historical observations available before that date to estimate the next-period VaR.

**Data limitation:** The input price series uses closing prices rather than adjusted closing prices. Corporate actions, such as stock splits and dividends, may therefore affect calculated returns. The results should be interpreted with this limitation in mind.

### 3.2 Experimental Design

The framework compares three VaR approaches:

1. **Historical Simulation:** estimates the loss quantile empirically.
2. **Gaussian Parametric VaR:** assumes normally distributed returns.
3. **Student-t Parametric VaR:** accommodates heavier-tailed returns.

The models are evaluated using rolling out-of-sample forecasts at a 95% confidence level, with a 500-observation estimation window. Each forecast is compared with the subsequent realized loss.

The expected breach probability at confidence level \(\alpha\) is \(1-\alpha\). For a 95% VaR forecast, the expected breach probability is 5%.

The evaluation includes breach frequency, the Kupiec unconditional-coverage test, descriptive breach-clustering statistics, and comparisons across volatility regimes.

### 3.3 Historical Simulation VaR

Historical Simulation estimates VaR from the empirical distribution of losses in the rolling estimation window:

$$
\widehat{\mathrm{VaR}}_{\alpha,t}
=
Q_{\alpha}\left(-r_{t-W},\ldots,-r_{t-1}\right)
$$

Here, \(W=500\) observations and \(Q_{\alpha}\) denotes the empirical loss quantile corresponding to confidence level \(\alpha\).

**Why use it?** Historical Simulation does not require returns to follow a Gaussian distribution. Its limitations include sensitivity to the selected window and dependence on historical observations being representative of future market conditions.

### 3.4 Gaussian Parametric VaR

The Gaussian model assumes that returns follow a normal distribution. Let \(\hat{\mu}_t\) denote the estimated mean return and \(\hat{\sigma}_t\) the estimated standard deviation of returns in the rolling window.

The VaR forecast is

$$
\widehat{\mathrm{VaR}}_{\alpha,t}
=
-\left[
\hat{\mu}_t+
\hat{\sigma}_t\Phi^{-1}(1-\alpha)
\right]
$$

Here, \(\Phi^{-1}\) is the inverse cumulative distribution function of the standard normal distribution.

**Why use it?** The Gaussian model is a simple, computationally efficient benchmark. Its performance can deteriorate when returns exhibit heavy tails, skewness, or changing volatility.

### 3.5 Student-t Parametric VaR

The Student-t model provides a heavier-tailed alternative to the Gaussian model. Let \(\nu_t>2\) denote the fitted degrees of freedom.

The implementation fits a Student-t distribution to standardized returns and rescales it to match the estimated return mean and standard deviation. The scale parameter is

$$
s_t
=
\hat{\sigma}_t
\sqrt{\frac{\nu_t-2}{\nu_t}}
$$

The resulting VaR forecast is

$$
\widehat{\mathrm{VaR}}_{\alpha,t}
=
-\left[
\hat{\mu}_t+
s_tT_{\nu_t}^{-1}(1-\alpha)
\right]
$$

Here, \(T_{\nu_t}^{-1}\) denotes the inverse cumulative distribution function of the Student-t distribution with \(\nu_t\) degrees of freedom. Requiring \(\nu_t>2\) ensures that the fitted Student-t distribution has finite variance.

**Why use it?** The Student-t model can better represent heavy-tailed returns than a Gaussian model. However, a heavier-tailed distribution does not guarantee accurate forecasts when market conditions change.

### 3.6 Breach Frequency and Expected Coverage

A VaR breach occurs when the realized loss exceeds the forecast VaR. Define the breach indicator as

$$
I_t=
\begin{cases}
1, & L_t>\widehat{\mathrm{VaR}}_{\alpha,t},\\
0, & \text{otherwise}.
\end{cases}
$$

For \(N\) evaluated forecasts, the observed breach rate is

$$
\widehat{p}
=
\frac{1}{N}\sum_{t=1}^{N}I_t
$$

The expected breach probability under correct unconditional coverage is

$$
p_0=1-\alpha
$$

At 95% confidence, the target breach rate is 5%.

The Kupiec unconditional-coverage test evaluates whether the observed breach frequency is statistically consistent with the expected breach probability. A small p-value provides evidence against the target unconditional coverage rate. A non-significant result does not prove that the model is correct or that it performs well in every market condition.

The framework also reports descriptive breach-clustering statistics, including the number of breach clusters and the maximum consecutive breach run. These statistics describe the temporal pattern of breaches but do not constitute a formal independence or conditional-coverage test.

### 3.7 Volatility Regimes

The framework calculates rolling volatility using the standard deviation of the previous 20 daily returns, including the current observation:

$$
\hat{\sigma}_{20,t}
=
\mathrm{SD}(r_{t-19},\ldots,r_t)
$$

It classifies observations above the 90th percentile of the full sample's valid rolling-volatility estimates as high-volatility periods. The remaining observations are classified as normal-volatility periods. Breach rates are then compared between these regimes for each VaR model.

**Interpretation note:** Because the volatility threshold is estimated using the full sample, this classification is retrospective and descriptive rather than a strictly real-time regime signal. The regime comparison helps identify whether model performance differs across observed volatility conditions, but it does not establish causality.
## 4. Research Design

| Component | Implementation |
|---|---|
| Asset | Microsoft (MSFT) |
| Price observations | 6,518 |
| Sample period | 2000-02-02 to 2025-12-31 |
| Return measure | Daily simple returns |
| Forecast horizon | One trading observation |
| Estimation window | Rolling 500 observations |
| VaR confidence | 95% |
| Models | Historical, Gaussian, Student-t |
| Coverage diagnostic | Kupiec unconditional-coverage test |
| Additional diagnostics | Consecutive and adjacent breaches |
| Regime analysis | 20-observation volatility; retrospective 90th-percentile threshold |

Each model generates a forecast using the previous 500 returns. That threshold is compared with the next realized loss, after which the window rolls forward. This creates an out-of-sample sequence rather than evaluating VaR on the same observations used to estimate it.

The resulting 6,017 forecasts per model are used for the reported backtesting results.

## 5. Results

### 5.1 Out-of-Sample Backtesting

| Model | Breaches | Observed breach rate | Expected rate | Kupiec p-value |
|---|---:|---:|---:|---:|
| Historical VaR | 307 | 5.10% | 5.00% | 0.7169 |
| Gaussian VaR | 258 | 4.29% | 5.00% | 0.0094 |
| Student-t VaR | 269 | 4.47% | 5.00% | 0.0552 |

![Observed breach rates for the three VaR models](reports/breach_rates.png)

**Interpretation at the 5% significance level:**

- **Historical VaR:** The test does not reject the target unconditional breach frequency.
- **Gaussian VaR:** The test rejects the target unconditional breach frequency. The observed rate is lower than 5%, indicating fewer breaches than expected.
- **Student-t VaR:** The test does not reject the target at the 5% level, although its p-value is close to the cutoff.

These are statistical findings about unconditional breach frequency, not proof that one model is universally superior. In particular, fewer breaches can reflect conservative thresholds rather than better overall risk estimation.

### 5.2 Breach Rates Across Volatility Regimes

| Regime | Historical VaR | Gaussian VaR | Student-t VaR |
|---|---:|---:|---:|
| High volatility | 13.29% | 12.35% | 12.82% |
| Normal volatility | 4.47% | 3.67% | 3.83% |

![Rolling volatility and the retrospective high-volatility threshold](reports/volatility_regimes.png)

All three models have higher observed breach rates in the high-volatility regime than in the normal-volatility regime. The increase is particularly pronounced during high-volatility periods.

This supports the motivation for regime-aware risk analysis: a model's overall breach rate can hide periods in which losses exceed forecasts much more frequently. However, the regime labels use a retrospective threshold, so these comparisons should be treated as exploratory evidence rather than a deployable real-time stress signal.

### 5.3 Realized Losses and VaR Forecasts

![Realized losses compared with rolling VaR forecasts](reports/var_forecasts.png)

The forecast plot compares the realized daily loss series with the positive VaR loss thresholds from all three models. Days when realized losses exceed a threshold are the model's breaches.

The time series helps investigate when breaches occur, whether they cluster around turbulent periods, and how the three thresholds respond to market conditions. It complements the aggregate statistical tests rather than replacing them.

### 5.4 Breach Clustering

The implementation also records the number of breach clusters, adjacent breach pairs, and the longest consecutive breach run.

Clustering matters because two models with the same total number of breaches may behave differently. A model whose breaches occur close together during a stress episode may be less dependable during that episode than one whose breaches are dispersed.

These summary statistics are descriptive; they are not a formal test of breach independence. A future extension would implement the Christoffersen independence and conditional-coverage tests.

## 6. Main Findings

1. Historical simulation produced a 5.10% breach rate, close to the 5% target, and the Kupiec test did not reject unconditional coverage.
2. Gaussian VaR produced a 4.29% breach rate, and the Kupiec test rejected the target frequency at the 5% significance level.
3. Student-t VaR produced a 4.47% breach rate; its Kupiec test did not reject the target at the 5% level.
4. All three models experienced substantially higher breach rates during the retrospectively identified high-volatility regime.
5. Coverage statistics alone are insufficient to establish overall model quality. Breach timing, volatility changes, tail behavior, and the assumptions behind the model also matter.

## 7. Limitations and Next Research Steps

### Current limitations

- The dataset contains closing prices rather than adjusted closing prices. Stock splits and dividends can affect calculated returns.
- The analysis uses one asset, so results cannot automatically be generalized to other stocks or portfolios.
- The retrospective volatility threshold uses the full sample and is not suitable as a real-time signal without further development.
- The Kupiec test assesses unconditional coverage, not breach independence or conditional coverage.
- VaR does not quantify the magnitude of losses beyond the threshold; Expected Shortfall can help evaluate tail severity.
- Financial returns may be serially dependent and heteroskedastic. A rolling window does not eliminate these properties.

### Future extensions

1. Use adjusted prices and extend the study to multiple assets.
2. Implement Christoffersen conditional-coverage and independence tests.
3. Compare VaR models at both 95% and 99% confidence levels.
4. Evaluate Expected Shortfall alongside VaR.
5. Build a strictly past-data-based volatility threshold for real-time regime detection.
6. Examine how model performance changes around independently defined stress events.

## 8. Technology Stack

- Python
- NumPy and pandas
- SciPy
- Matplotlib
- pytest

## 9. Run the Project

Clone the repository, enter the project directory, and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python main.py
```

Run the current unit tests:

```bash
python -m pytest tests/test_risk_models.py -v
```

## 10. Repository Structure

```text
var-model-failure-detector/
├── data/
│   └── market_prices.csv
├── src/
│   ├── risk_models.py
│   ├── backtesting.py
│   ├── diagnostics.py
│   └── regime_analysis.py
├── tests/
│   └── test_risk_models.py
├── notebooks/
│   └── risk_analysis.ipynb
├── reports/
│   ├── var_backtest_results.csv
│   ├── diagnostics_summary.csv
│   ├── regime_breach_summary.csv
│   ├── volatility_regimes.csv
│   ├── var_forecasts.png
│   ├── breach_rates.png
│   └── volatility_regimes.png
├── main.py
├── requirements.txt
└── README.md
```

## Conclusion

This project builds a reproducible framework for investigating the reliability of market-risk forecasts. By combining three VaR models with out-of-sample backtesting, coverage diagnostics, breach-clustering summaries, and volatility-regime comparisons, it demonstrates why risk-model evaluation must consider both statistical performance and changing market conditions.

**The central research lesson is that a VaR model should not be judged by its estimated threshold alone. Its forecasts must be tested against realized losses, and its limitations must be understood under different market conditions.**