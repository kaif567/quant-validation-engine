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

## Zero Look-Ahead Bias Guarantee
One of the most common pitfalls in quantitative research is accidentally executing a trade at the closing price of the candle that generated the signal. This engine strictly enforces execution on the **NEXT candle's Open price**. 

*Example:* If your RSI drops below 30 at 14:15, the signal is generated. The engine will not fill your order at the 14:15 Close. It will fill it at the 14:30 Open, simulating real-world latency and preventing impossible executions.

## Disclaimer
**This software is for educational and quantitative research purposes only.**
It does not constitute financial advice. The models and testing frameworks provided are designed to validate statistical robustness, but past performance (even mathematically validated out-of-sample performance) is absolutely no guarantee of future returns. No profit guarantee is implied. Use at your own risk.

## Institutional-Grade Accuracy Features
This simulator goes beyond basic backtesting by implementing features standard in institutional quant desks:

1. **Pessimistic Intra-Candle Resolution:** In traditional backtesters (like TradingView), if a single candle's wick hits both your Take Profit and your Stop Loss, the engine often assumes the TP was hit first (yielding fake profits). This engine employs *Pessimistic Resolution*: if both are breached within the same candle, it **always** assumes the Stop Loss was hit first.
2. **Dynamic Slippage Modeling:** Not all executions are perfect. Limit orders (like Take Profit) get filled exactly. But Market orders (Entries, Stop Losses, Signal Reversals) are subjected to a configurable `slippage_rate` (default 0.05%) to accurately reflect order-book liquidity consumption.
