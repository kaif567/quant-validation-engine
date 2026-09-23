import pandas as pd
import numpy as np

def simulate_trades(
    df_with_signals, 
    mode='base', 
    fee_rate=0.001, 
    entry_slippage=0.0005, 
    stop_slippage=0.001, 
    stop_loss=None, 
    take_profit=None,
    delisted_info=None,
    liquidity_config=None,
    timeframe='1h',
    intended_quantity=1.0
):
    trades = []
    rejected_events = []
    
    if liquidity_config is None:
        liquidity_config = {
            "liquidity_tiering_enabled": False,
            "capacity_constraint_enabled": False
        }
        
    bars_per_day = 24
    if timeframe == '1d': bars_per_day = 1
    elif timeframe == '4h': bars_per_day = 6
    
    df = df_with_signals.copy()
    if 'quote_volume' not in df.columns:
        if 'volume' in df.columns and 'close' in df.columns:
            df['quote_volume'] = df['volume'] * df['close']
        else:
            df['quote_volume'] = 0.0
            
    df['prior_24h_quote_volume'] = df['quote_volume'].rolling(bars_per_day).sum()
    df['prior_30d_avg_daily_quote_volume'] = df['quote_volume'].rolling(30 * bars_per_day).sum() / 30.0
    if 'high' in df.columns and 'low' in df.columns:
        df['prior_intrabar_range_fraction'] = (df['high'] - df['low']) / df['close']
    else:
        df['prior_intrabar_range_fraction'] = 0.0
    
    position = None
    entry_price = 0.0
    entry_time = None
    entry_fill_obj = None
    
    if 'signal' not in df.columns:
        raise ValueError("DataFrame must contain a 'signal' column.")
        
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        
        signal = current_row['signal']
        
        if delisted_info is not None and next_row.name >= delisted_info['delisted_time']:
            if position is not None:
                exit_price = float(current_row['close'])
                exit_reason = "FORCED_DELISTING_EXIT"
                exit_order_type = "UNRESOLVED"
                
                exit_fill_obj = {
                    'bar_index': i + 1,
                    'order_type': exit_order_type,
                    'requested_price': None,
                    'fill_price': exit_price,
                    'last_observed_mark': exit_price,
                    'mode': mode,
                    'gap_fill': False,
                    'reason': exit_reason,
                    'terminal_value_unverified': True,
                    'liquidity_diagnostics': {'exit_liquidity_modeled': False}
                }
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': next_row.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'net_return': None,
                    'unresolved_notional': exit_price,
                    'exit_reason': exit_reason,
                    'entry_fill': entry_fill_obj,
                    'exit_fill': exit_fill_obj
                })
                position = None
                entry_fill_obj = None
            break

        if position is not None:
            exit_price = None
            exit_reason = None
            exit_order_type = None
            gap_fill = False
            req_price = None
            applied_slippage = 0.0
            
            if stop_loss is not None and take_profit is not None:
                if position == 'LONG':
                    sl_price = entry_price * (1 - stop_loss)
                    tp_price = entry_price * (1 + take_profit)
                    
                    if next_row['open'] <= sl_price:
                        exit_price, exit_reason = next_row['open'], "STOP_LOSS_GAP"
                        applied_slippage = stop_slippage
                        exit_order_type = "STOP_MARKET"
                        gap_fill = True
                        req_price = sl_price
                    elif next_row['open'] >= tp_price:
                        exit_price, exit_reason = tp_price, "TAKE_PROFIT_GAP"
                        applied_slippage = 0.0
                        exit_order_type = "LIMIT"
                        gap_fill = True
                        req_price = tp_price
                    else:
                        hit_sl = next_row['low'] <= sl_price
                        hit_tp = next_row['high'] >= tp_price
                        
                        if hit_sl and hit_tp:
                            dist_sl = (next_row['open'] - next_row['low'])
                            dist_tp = (next_row['high'] - next_row['open'])
                            
                            if mode == 'pessimistic':
                                sl_first = True
                            elif mode == 'optimistic':
                                sl_first = False
                            else: 
                                sl_first = True if dist_sl <= dist_tp else False
                                
                            if sl_first:
                                exit_price = (sl_price - (current_row['atr'] * stop_slippage)) if 'atr' in current_row else (sl_price * (1 - stop_slippage))
                                exit_order_type = 'STOP_MARKET'
                                exit_reason = 'SL_TOUCH_BOTH'
                                req_price = sl_price
                            else:
                                exit_price = tp_price
                                exit_order_type = 'LIMIT'
                                exit_reason = 'TP_TOUCH_BOTH'
                                req_price = tp_price
                        elif hit_sl:
                            exit_price = (sl_price - (current_row['atr'] * stop_slippage)) if 'atr' in current_row else (sl_price * (1 - stop_slippage))
                            exit_order_type = 'STOP_MARKET'
                            exit_reason = 'STOP_LOSS'
                            req_price = sl_price
                        elif hit_tp:
                            exit_price = tp_price
                            exit_order_type = 'LIMIT'
                            exit_reason = 'TAKE_PROFIT'
                            req_price = tp_price
                            
                elif position == 'SHORT':
                    sl_price = entry_price * (1 + stop_loss)
                    tp_price = entry_price * (1 - take_profit)
                    
                    if next_row['open'] >= sl_price:
                        exit_price, exit_reason = next_row['open'], "STOP_LOSS_GAP"
                        applied_slippage = stop_slippage
                        exit_order_type = "STOP_MARKET"
                        gap_fill = True
                        req_price = sl_price
                    elif next_row['open'] <= tp_price:
                        exit_price, exit_reason = tp_price, "TAKE_PROFIT_GAP"
                        applied_slippage = 0.0
                        exit_order_type = "LIMIT"
                        gap_fill = True
                        req_price = tp_price
                    else:
                        hit_sl = next_row['high'] >= sl_price
                        hit_tp = next_row['low'] <= tp_price
                        
                        if hit_sl and hit_tp:
                            dist_sl = (next_row['high'] - next_row['open'])
                            dist_tp = (next_row['open'] - next_row['low'])
                            if mode == 'pessimistic': sl_first = True
                            elif mode == 'optimistic': sl_first = False
                            else: sl_first = True if dist_sl <= dist_tp else False
                                
                            if sl_first:
                                exit_price = (sl_price + (current_row['atr'] * stop_slippage)) if 'atr' in current_row else (sl_price * (1 + stop_slippage))
                                exit_order_type = 'STOP_MARKET'
                                exit_reason = 'SL_TOUCH_BOTH'
                                req_price = sl_price
                            else:
                                exit_price = tp_price
                                exit_order_type = 'LIMIT'
                                exit_reason = 'TP_TOUCH_BOTH'
                                req_price = tp_price
                        elif hit_sl:
                            exit_price = (sl_price + (current_row['atr'] * stop_slippage)) if 'atr' in current_row else (sl_price * (1 + stop_slippage))
                            exit_order_type = 'STOP_MARKET'
                            exit_reason = 'STOP_LOSS'
                            req_price = sl_price
                        elif hit_tp:
                            exit_price = tp_price
                            exit_order_type = 'LIMIT'
                            exit_reason = 'TAKE_PROFIT'
                            req_price = tp_price
            
            if exit_price is None:
                if (position == 'LONG' and signal == -1) or (position == 'SHORT' and signal == 1):
                    req_price = float(next_row['open'])
                    exit_price = req_price + (current_row['atr'] * entry_slippage) if 'atr' in current_row else req_price * (1 + entry_slippage)
                    if position == 'LONG': exit_price = req_price - (current_row['atr'] * entry_slippage) if 'atr' in current_row else req_price * (1 - entry_slippage)
                    exit_order_type = 'MARKET'
                    exit_reason = "SIGNAL_REVERSAL"
                elif signal == -2:
                    req_price = float(next_row['open'])
                    exit_price = req_price + (current_row['atr'] * entry_slippage) if 'atr' in current_row else req_price * (1 + entry_slippage)
                    if position == 'LONG': exit_price = req_price - (current_row['atr'] * entry_slippage) if 'atr' in current_row else req_price * (1 - entry_slippage)
                    exit_order_type = 'MARKET'
                    exit_reason = "SIGNAL_FORCE_CLOSE"
                    
            if exit_price is not None:
                gross_return = (exit_price - entry_price) / entry_price if position == 'LONG' else (entry_price - exit_price) / entry_price
                net_return = gross_return - (2 * fee_rate)
                
                exit_fill_obj = {
                    'bar_index': i + 1,
                    'order_type': exit_order_type,
                    'requested_price': req_price,
                    'fill_price': exit_price,
                    'last_observed_mark': None,
                    'mode': mode,
                    'gap_fill': gap_fill,
                    'reason': exit_reason,
                    'terminal_value_unverified': False,
                    'liquidity_diagnostics': {'exit_liquidity_modeled': False}
                }
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': next_row['open_time'] if 'open_time' in next_row else next_row.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'net_return': net_return,
                    'exit_reason': exit_reason,
                    'entry_fill': entry_fill_obj,
                    'exit_fill': exit_fill_obj
                })
                position = None
                entry_fill_obj = None
                
        if position is None and signal in [1, -1]:
            req_px = float(next_row['open'])
            tier_factor = 1.0
            capacity_cap = float('inf')
            missing_tier = False
            missing_cap = False
            
            tiering_enabled = liquidity_config.get('liquidity_tiering_enabled', False)
            capacity_enabled = liquidity_config.get('capacity_constraint_enabled', False)
            
            p_30d = float(current_row.get('prior_30d_avg_daily_quote_volume', float('nan')))
            p_24h = float(current_row.get('prior_24h_quote_volume', float('nan')))
            
            if pd.isna(p_30d) or i < 30 * bars_per_day - 1: missing_tier = True
            if pd.isna(p_24h) or i < bars_per_day - 1: missing_cap = True
                
            applied_policy = "NONE"
            rejected = False
            reject_reason = ""
            
            is_missing = (tiering_enabled and missing_tier) or (capacity_enabled and missing_cap)
            
            if is_missing:
                policy = liquidity_config.get('missing_data_policy', 'REJECT')
                if tiering_enabled and capacity_enabled and (missing_tier or missing_cap):
                    rejected = True
                    reject_reason = "REJECT_MISSING_DATA_CAP_AND_TIER"
                    applied_policy = "REJECT"
                elif tiering_enabled and not capacity_enabled and missing_tier:
                    if policy == 'REJECT':
                        rejected = True
                        reject_reason = "REJECT_MISSING_TIER_DATA"
                        applied_policy = "REJECT"
                    elif policy == 'CONSERVATIVE_LOW_TIER':
                        tier_factor = 3.0
                        applied_policy = "CONSERVATIVE_LOW_TIER"
                elif not tiering_enabled and capacity_enabled and missing_cap:
                    rejected = True
                    reject_reason = "REJECT_MISSING_CAP_DATA"
                    applied_policy = "REJECT"
                elif not tiering_enabled and not capacity_enabled:
                    applied_policy = "LEGACY_NO_LIMITS"
                    
            if tiering_enabled and not missing_tier:
                metric_val = p_30d if liquidity_config.get('tier_metric') == 'prior_30d_avg_daily_quote_volume' else p_24h
                tiers = liquidity_config.get('tiers', [])
                for t in sorted(tiers, key=lambda x: x['min_quote_volume'], reverse=True):
                    if metric_val >= t['min_quote_volume']:
                        tier_factor = float(t['tier_factor'])
                        break
                        
            if capacity_enabled and not missing_cap:
                part_rate = liquidity_config.get('capacity', {}).get('participation_rate_cap', 0.05)
                capacity_cap = part_rate * p_24h
                
            slip_amt = (current_row['atr'] * entry_slippage * tier_factor) if 'atr' in current_row else (req_px * entry_slippage * tier_factor)
            
            est_fill_px = req_px + slip_amt if signal == 1 else req_px - slip_amt
            if est_fill_px <= 0:
                rejected = True
                reject_reason = "INVALID_ESTIMATED_FILL_PRICE"
                
            req_ref_notional = intended_quantity * req_px
            req_exec_notional = intended_quantity * est_fill_px
            approved_notional = req_exec_notional
            
            if not rejected and capacity_enabled:
                if req_exec_notional > capacity_cap:
                    action = liquidity_config.get('capacity', {}).get('action', 'REJECT')
                    if action == 'REJECT':
                        rejected = True
                        reject_reason = "REQUESTED_EXECUTION_NOTIONAL_EXCEEDS_CAP"
                    elif action == 'RESIZE':
                        approved_notional = capacity_cap
                        
            diag_obj = {
                "timeframe_implied_bars_per_day": bars_per_day,
                "required_lookback_bars": 30 * bars_per_day if tiering_enabled else bars_per_day,
                "prior_30d_avg_daily_quote_volume": p_30d,
                "prior_24h_quote_volume": p_24h,
                "prior_intrabar_range_fraction": float(current_row.get('prior_intrabar_range_fraction', 0.0)),
                "requested_reference_notional": req_ref_notional,
                "requested_execution_notional": req_exec_notional,
                "estimated_causal_fill_price": est_fill_px,
                "maximum_notional": capacity_cap,
                "approved_notional": approved_notional,
                "participation_rate_cap": liquidity_config.get('capacity', {}).get('participation_rate_cap', 0.05) if capacity_enabled else None,
                "liquidity_tier_factor": tier_factor,
                "capacity_action": "REJECT" if rejected else liquidity_config.get('capacity', {}).get('action', 'ACCEPT'),
                "missing_data_policy_applied": applied_policy,
                "signal_bar_time": str(current_row.name),
                "execution_bar_time": str(next_row.name)
            }
            
            if rejected:
                rejected_events.append({
                    "event_type": "trade_rejected_missing_liquidity_data" if "MISSING" in reject_reason else "trade_rejected_liquidity_cap",
                    "payload": {
                        "reject_reason": reject_reason,
                        "causal_input_timestamps": {
                            "signal_bar_time": str(current_row.name),
                            "execution_bar_time": str(next_row.name)
                        },
                        "liquidity_inputs": diag_obj
                    }
                })
            else:
                position = 'LONG' if signal == 1 else 'SHORT'
                entry_price = est_fill_px
                entry_time = next_row['open_time'] if 'open_time' in next_row else next_row.name
                entry_fill_obj = {
                    'bar_index': i + 1,
                    'order_type': 'MARKET_ENTRY',
                    'requested_price': req_px,
                    'fill_price': est_fill_px,
                    'mode': mode,
                    'gap_fill': False,
                    'reason': 'SIGNAL_LONG' if signal == 1 else 'SIGNAL_SHORT',
                    'time': next_row.name,
                    'liquidity_diagnostics': diag_obj
                }
                
    if position is not None:
        last_row = df.iloc[-1]
        req_px = float(last_row['close'])
        exit_price = req_px - (last_row['atr'] * stop_slippage) if position == 'LONG' else req_px + (last_row['atr'] * stop_slippage)
        if 'atr' not in last_row:
            exit_price = req_px * (1 - stop_slippage) if position == 'LONG' else req_px * (1 + stop_slippage)
            
        gross_return = (exit_price - entry_price) / entry_price if position == 'LONG' else (entry_price - exit_price) / entry_price
        net_return = gross_return - (2 * fee_rate)
        
        exit_fill_obj = {
            'bar_index': len(df) - 1,
            'order_type': 'MARKET_FORCE_CLOSE',
            'requested_price': req_px,
            'fill_price': exit_price,
            'mode': mode,
            'gap_fill': False,
            'reason': 'BOUNDARY_CLOSE',
            'liquidity_diagnostics': {'exit_liquidity_modeled': False}
        }
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': last_row['open_time'] if 'open_time' in last_row else last_row.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'net_return': net_return,
            'exit_reason': 'BOUNDARY_CLOSE',
            'entry_fill': entry_fill_obj,
            'exit_fill': exit_fill_obj
        })
        
    return trades, rejected_events
