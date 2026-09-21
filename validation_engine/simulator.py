import pandas as pd
import numpy as np

def simulate_trades(df_with_signals, fee_rate=0.001):
    """
    Takes a DataFrame containing a 'signal' column (1 for Buy, -1 for Sell/Close, 0 for Hold).
    Executes trades strictly on the NEXT candle's open to prevent look-ahead bias.
    """
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    
    # Ensure signal column exists
    if 'signal' not in df_with_signals.columns:
        raise ValueError("DataFrame must contain a 'signal' column.")
        
    for i in range(len(df_with_signals) - 1):
        row = df_with_signals.iloc[i]
        next_row = df_with_signals.iloc[i + 1]
        
        signal = row['signal']
        
        if not in_position and signal == 1:
            # Enter trade on NEXT candle's open
            in_position = True
            entry_price = next_row['open']
            entry_time = next_row['open_time'] if 'open_time' in next_row else next_row.name
            
        elif in_position and signal == -1:
            # Exit trade on NEXT candle's open
            exit_price = next_row['open']
            exit_time = next_row['open_time'] if 'open_time' in next_row else next_row.name
            
            # Calculate PnL with fees
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - (2 * fee_rate)  # Fee on entry and exit
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'net_return': net_return,
                'profit_factor_weight': max(net_return, 0) / abs(min(net_return, 0)) if min(net_return, 0) < 0 else (net_return if net_return > 0 else 0)
            })
            in_position = False
            
    # Close open position at end of data if still in position
    if in_position:
        last_row = df_with_signals.iloc[-1]
        exit_price = last_row['close']
        exit_time = last_row['open_time'] if 'open_time' in last_row else last_row.name
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - (2 * fee_rate)
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'net_return': net_return
        })
        
    return trades
