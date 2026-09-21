import pandas as pd

def simulate_trades(df_with_signals, fee_rate=0.001, slippage_rate=0.0005, stop_loss=None, take_profit=None):
    """
    Simulates trades with Institutional-Grade Accuracy:
    1. Zero Look-Ahead Bias (Next-Open execution).
    2. Dynamic Slippage applied to market orders (Entry and Stop-Loss).
    3. Pessimistic Intra-Candle Resolution: If a single candle hits BOTH TP and SL, 
       the engine always assumes SL was hit first to prevent overestimating returns.
    """
    trades = []
    position = None
    entry_price = 0.0
    entry_time = None
    
    if 'signal' not in df_with_signals.columns:
        raise ValueError("DataFrame must contain a 'signal' column.")
        
    for i in range(len(df_with_signals) - 1):
        current_row = df_with_signals.iloc[i]
        next_row = df_with_signals.iloc[i + 1]
        
        signal = current_row['signal']
        
        # 1. Exit Logic
        if position is not None:
            exit_price = None
            exit_reason = None
            applied_slippage = 0.0
            
            if stop_loss is not None and take_profit is not None:
                if position == 'LONG':
                    sl_price = entry_price * (1 - stop_loss)
                    tp_price = entry_price * (1 + take_profit)
                    
                    # Gap conditions
                    if next_row['open'] <= sl_price:
                        exit_price, exit_reason = next_row['open'], "STOP_LOSS_GAP"
                        applied_slippage = slippage_rate # Market sell on gap
                    elif next_row['open'] > tp_price:
                        exit_price, exit_reason = tp_price, "TAKE_PROFIT_GAP"
                        applied_slippage = 0.0 # Limit fill
                        
                    # Intra-candle conditions
                    else:
                        hit_sl = next_row['low'] <= sl_price
                        hit_tp = next_row['high'] > tp_price
                        
                        # PESSIMISTIC RESOLUTION: If both hit, assume Stop Loss triggered first
                        if hit_sl and hit_tp:
                            exit_price, exit_reason = sl_price, "STOP_LOSS_PESSIMISTIC"
                            applied_slippage = slippage_rate
                        elif hit_sl:
                            exit_price, exit_reason = sl_price, "STOP_LOSS"
                            applied_slippage = slippage_rate
                        elif hit_tp:
                            exit_price, exit_reason = tp_price, "TAKE_PROFIT"
                            applied_slippage = 0.0 # Limit order gets exact fill
                            
                elif position == 'SHORT':
                    sl_price = entry_price * (1 + stop_loss)
                    tp_price = entry_price * (1 - take_profit)
                    
                    if next_row['open'] >= sl_price:
                        exit_price, exit_reason = next_row['open'], "STOP_LOSS_GAP"
                        applied_slippage = slippage_rate
                    elif next_row['open'] < tp_price:
                        exit_price, exit_reason = tp_price, "TAKE_PROFIT_GAP"
                        applied_slippage = 0.0
                    else:
                        hit_sl = next_row['high'] >= sl_price
                        hit_tp = next_row['low'] < tp_price
                        
                        if hit_sl and hit_tp:
                            exit_price, exit_reason = sl_price, "STOP_LOSS_PESSIMISTIC"
                            applied_slippage = slippage_rate
                        elif hit_sl:
                            exit_price, exit_reason = sl_price, "STOP_LOSS"
                            applied_slippage = slippage_rate
                        elif hit_tp:
                            exit_price, exit_reason = tp_price, "TAKE_PROFIT"
                            applied_slippage = 0.0
            
            # Explicit Exit Signal Reversal
            if exit_price is None:
                if (position == 'LONG' and signal == -1) or (position == 'SHORT' and signal == 1):
                    exit_price = next_row['open']
                    exit_reason = "SIGNAL_REVERSAL"
                    applied_slippage = slippage_rate # Market order
                    
            if exit_price is not None:
                # Apply slippage to exit if it was a market order
                actual_exit = exit_price * (1 - applied_slippage) if position == 'LONG' else exit_price * (1 + applied_slippage)
                
                gross_return = (actual_exit - entry_price) / entry_price if position == 'LONG' else (entry_price - actual_exit) / entry_price
                net_return = gross_return - (2 * fee_rate)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': next_row['open_time'] if 'open_time' in next_row else next_row.name,
                    'entry_price': entry_price,
                    'exit_price': actual_exit,
                    'net_return': net_return,
                    'exit_reason': exit_reason
                })
                position = None
                
        # 2. Entry Logic
        if position is None:
            if signal == 1:
                position = 'LONG'
                # Apply slippage to entry
                entry_price = next_row['open'] * (1 + slippage_rate) 
                entry_time = next_row['open_time'] if 'open_time' in next_row else next_row.name
            elif signal == -1:
                position = 'SHORT'
                entry_price = next_row['open'] * (1 - slippage_rate)
                entry_time = next_row['open_time'] if 'open_time' in next_row else next_row.name
                
    if position is not None:
        last_row = df_with_signals.iloc[-1]
        exit_price = last_row['close']
        actual_exit = exit_price * (1 - slippage_rate) if position == 'LONG' else exit_price * (1 + slippage_rate)
        
        gross_return = (actual_exit - entry_price) / entry_price if position == 'LONG' else (entry_price - actual_exit) / entry_price
        net_return = gross_return - (2 * fee_rate)
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': last_row['open_time'] if 'open_time' in last_row else last_row.name,
            'entry_price': entry_price,
            'exit_price': actual_exit,
            'net_return': net_return,
            'exit_reason': 'BOUNDARY_CLOSE'
        })
        
    return trades
