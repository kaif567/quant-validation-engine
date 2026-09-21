# Quant Validation Engine

A lightweight, general-purpose quantitative backtesting library designed to rigorously validate trading strategies. This engine forces out-of-sample validation, prevents look-ahead bias, and uses statistical metrics to expose overfitting.

## Why this exists
Most retail backtesting tools (like TradingView) suffer from severe flaws:
1. **Look-Ahead Bias:** Assuming you can execute at the exact high/low of a candle.
2. **Overfitting:** Fitting parameters to the entire dataset, creating an illusion of profitability.
3. **Statistical Noise:** Accepting a Profit Factor > 1.0 without checking if it's dependent on a few lucky trades.

This engine solves these issues by automating:
* **Walk-Forward Validation:** Dynamically splits data into expanding Train/Test windows.
* **Realistic Simulation:** Executes trades strictly on the *next* candle's open with configurable exchange fees.
* **Statistical Rigor (95% CI):** Uses Monte Carlo / Bootstrap resampling to calculate the 95% Confidence Interval lower-bound of your strategy's edge.
* **Robustness Checking:** Automatically removes the top most profitable trades to verify the edge is consistent, not a lottery ticket.

## Installation
Clone this repository and ensure you have `pandas` and `numpy` installed.
```bash
git clone <your-repo-url>
cd quant-validation-engine
pip install pandas numpy
```

## How to Use
You only need to write a single function that returns `1` (Buy), `-1` (Sell), or `0` (Hold) in a DataFrame column named `signal`. The engine handles the rest.

Check out `example_usage.py` for a complete working template.
