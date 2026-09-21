# How It Works: Quant Validation Engine

This document explains the architecture and internal mechanics of the `quant-validation-engine`. Unlike standard retail backtesters (which often suffer from look-ahead bias and optimistic execution), this engine is built on **institutional-grade strictness**.

## 🏗️ Architecture Overview

The engine is highly modular. It completely separates the **Trading Logic** (your strategy) from the **Execution Environment** (the simulator) and the **Statistical Analysis**. 

Here is the step-by-step pipeline of how a strategy is evaluated:

### 1. Strategy Plugin Interface (`user_strategy`)
You provide a simple Python function that takes a Pandas DataFrame and parameters, calculates indicators, and returns the DataFrame with a strict `signal` column:
* `1` = STRONG BUY (Enter Long / Close Short)
* `-1` = STRONG SELL (Enter Short / Close Long)
* `0` = HOLD (Do nothing)

### 2. Walk-Forward Optimizer (`validation_engine/walk_forward.py`)
To prevent curve-fitting (over-optimization), the engine **never** tests on the same data it trains on.
1. It slices the historical data into an expanding or rolling window.
2. **Train Window:** It tests all combinations of your `param_grid` (e.g., EMA 10 vs 20, SL 2% vs 3%) and finds the best Profit Factor.
3. **Test (Out-of-Sample) Window:** It applies the *best* parameters from the Train window to the unseen Test window. 
4. The trades generated in this strictly unseen Test window are collected for the final report.

### 3. The Execution Simulator (`validation_engine/simulator.py`)
This is the core "reality check" of the engine. It takes the signals and simulates trades with extreme pessimism:
* **Zero Look-Ahead Bias:** If a signal is generated on candle `i`, the trade is executed at the `open` price of candle `i+1`.
* **Dynamic Slippage:** Market orders (Entries, Stop Losses, Signal Reversals) are penalized with a `slippage_rate` (default 0.05%) to simulate order book liquidity consumption.
* **Gap Handling:** If the market gaps past your Stop Loss or Take Profit overnight/between candles, you are filled at the *worse* opening price of the gap, not your ideal price.
* **Pessimistic Intra-Candle Resolution:** If a single candle's high hits your Take Profit, and its low hits your Stop Loss, standard engines guess which happened first. **This engine always assumes the Stop Loss was hit first.** This mathematically guarantees you never overestimate returns.

### 4. Statistical Validation (`validation_engine/stats.py`)
Once the out-of-sample trades are collected, they are subjected to rigorous statistical analysis:
* **Standard Metrics:** Win Rate, Profit Factor, Total Trades.
* **Robustness Check (Outlier Removal):** It temporarily deletes your Top 2 most profitable trades and recalculates the Profit Factor. If your strategy turns negative without those 2 lucky trades, the edge is deemed "FRAGILE".
* **95% Confidence Interval (Bootstrapping):** It runs a Monte Carlo simulation (resampling your trades 1,000 times) to find the mathematical lower bound of your Profit Factor. If the lower bound is < 1.0, the strategy is statistically insignificant (just noise).

## 🚀 How to Audit a Strategy (Example)

```python
import pandas as pd
from validation_engine.protocol import multi_window_protocol

# 1. Define your strategy rules
def my_rsi_strategy(df, rsi_period=14, sl=0.02, tp=0.04):
    df['rsi'] = calculate_rsi(df, rsi_period)
    df['signal'] = 0
    df.loc[df['rsi'] < 30, 'signal'] = 1  # Buy
    df.loc[df['rsi'] > 70, 'signal'] = -1 # Sell
    return df

# 2. Define the parameter grid to test
param_grid = {
    'rsi_period': [10, 14, 21],
    'sl': [0.01, 0.02],
    'tp': [0.03, 0.05]
}

# 3. Pass data and let the engine audit it
data_dict = {"BTC_2023": btc_df, "ETH_2023": eth_df}
audit_report = multi_window_protocol(my_rsi_strategy, param_grid, data_dict)

print(audit_report)
```

## 🎯 Why this matters for B2B Clients
When auditing external strategies (from TradingView or retail developers), this engine strips away the illusions. It provides clients with a mathematically sound reality check before they risk live capital.
