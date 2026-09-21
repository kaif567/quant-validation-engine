import pandas as pd
import numpy as np
from validation_engine.protocol import multi_window_protocol

# 1. Define your custom strategy (Plug-in Interface)
def mean_reversion_strategy(df, rsi_period=14, rsi_lower=30, rsi_upper=70):
    """
    User-defined strategy. Must return a DataFrame with a 'signal' column.
    1 = Buy
    -1 = Sell
    0 = Hold
    """
    # Simulate a simple RSI calculation (using dummy data for example)
    # In reality, you'd use TA-Lib or Pandas-TA
    df['rsi'] = pd.Series(np.random.uniform(0, 100, size=len(df)))
    
    df['signal'] = 0
    df.loc[df['rsi'] < rsi_lower, 'signal'] = 1
    df.loc[df['rsi'] > rsi_upper, 'signal'] = -1
    
    return df

# 2. Setup your dummy data (Replace with real Binance/historical data)
dates = pd.date_range('2024-01-01', periods=5000, freq='15T')
dummy_df_1 = pd.DataFrame({'open_time': dates, 'open': np.random.randn(5000).cumsum() + 100})
dummy_df_1['close'] = dummy_df_1['open'] + np.random.randn(5000)

dummy_df_2 = pd.DataFrame({'open_time': dates, 'open': np.random.randn(5000).cumsum() + 200})
dummy_df_2['close'] = dummy_df_2['open'] + np.random.randn(5000)

datasets = {
    'BTC_Bull_Market': dummy_df_1,
    'ETH_Chop_Market': dummy_df_2
}

# 3. Define parameters to Walk-Forward Optimize
param_grid = {
    'rsi_period': [10, 14],
    'rsi_lower': [25, 30],
    'rsi_upper': [70, 75]
}

# 4. Run the Institutional-Grade Validation Protocol
if __name__ == "__main__":
    print("Starting Validation Engine...")
    results = multi_window_protocol(
        strategy_function=mean_reversion_strategy,
        param_grid=param_grid,
        data_dict=datasets
    )
    
    for dataset, metrics in results.items():
        print(f"\n--- {dataset} ---")
        print(f"Total OOS Trades: {metrics['total_trades']}")
        print(f"OOS Profit Factor: {metrics['oos_profit_factor']:.2f}")
        print(f"95% CI Lower Bound: {metrics['ci_lower_bound']:.2f}")
        print(f"Robust PF (Top 2 Removed): {metrics['robust_profit_factor']:.2f}")
        print(f"VERDICT: {'✅ PASSED' if metrics['passed'] else '❌ FAILED'}")
