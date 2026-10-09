import pandas as pd
import numpy as np
import math
from decimal import Decimal, ROUND_HALF_UP
from validation_engine.trial_ledger import normalize_payload, _build_liquidity_diag_normalized

MONEY_Q = Decimal("0.00000001")
RETURN_Q = Decimal("0.000000000000000001")


class ShadowCapitalLedgerTracker:
    def __init__(self, shadow_ledger_config=None, configured_quantity=1.0):
        config = shadow_ledger_config or {}
        embargo_close_price = config.get('embargo_close_price')
        qty = config.get('configured_quantity', configured_quantity)
        margin = config.get('fee_coverage_margin', Decimal("0.005"))

        self.valid = True
        self.invalid_reason = None
        self.e_initial = None
        self.shadow_cash = None
        self.shadow_qty = Decimal("0")
        self.shadow_direction = 'cash'
        self.bar_equities = []
        self.bar_returns = []
        self.bar_timestamps = []
        self.trade_count = 0

        self.embargo_close_price = None
        self.embargo_close_timestamp = config.get('embargo_close_timestamp')
        self.configured_qty = None
        self.fee_coverage_margin = margin
        self.effective_fee_rate = config.get('fee_rate')
        self.effective_entry_slippage = config.get('entry_slippage')
        self.effective_stop_slippage = config.get('stop_slippage')

        if embargo_close_price is None or qty is None or margin is None:
            self.valid = False
            self.invalid_reason = 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE'
            return

        if isinstance(embargo_close_price, float) and (math.isnan(embargo_close_price) or math.isinf(embargo_close_price)):
            self.valid = False
            self.invalid_reason = 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE'
            return

        if isinstance(qty, float) and (math.isnan(qty) or math.isinf(qty)):
            self.valid = False
            self.invalid_reason = 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE'
            return

        try:
            p_close = Decimal(str(embargo_close_price))
            q_val = Decimal(str(qty))
            m_val = Decimal(str(margin))
            if (
                p_close.is_nan() or p_close.is_infinite() or p_close <= Decimal("0") or
                q_val.is_nan() or q_val.is_infinite() or q_val <= Decimal("0") or
                m_val.is_nan() or m_val.is_infinite() or m_val < Decimal("0")
            ):
                self.valid = False
                self.invalid_reason = 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE'
                return
            self.embargo_close_price = p_close
            self.configured_qty = q_val
            self.e_initial = p_close * q_val * (Decimal("1") + m_val)
            self.shadow_cash = self.e_initial
        except Exception:
            self.valid = False
            self.invalid_reason = 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE'

    def record_entry(self, fill_price: Decimal, quantity: Decimal, fee: Decimal, direction: str):
        if not self.valid:
            return
        if direction.lower() != 'long':
            self.valid = False
            self.invalid_reason = 'INVALID_UNSUPPORTED_SHORT_DIRECTION'
            return
        notional = fill_price * quantity
        self.shadow_cash -= (notional + fee)
        if self.shadow_cash < Decimal("0"):
            self.valid = False
            self.invalid_reason = 'INVALID_SHADOW_CASH_DEPLETED'
        self.shadow_qty = quantity
        self.shadow_direction = 'long'
        self.trade_count += 1

    def record_exit(self, fill_price: Decimal, quantity: Decimal, fee: Decimal, direction: str):
        if not self.valid:
            return
        if direction.lower() != 'long' or self.shadow_direction != 'long':
            self.valid = False
            self.invalid_reason = 'INVALID_UNSUPPORTED_SHORT_DIRECTION'
            return
        notional = fill_price * quantity
        self.shadow_cash += (notional - fee)
        if self.shadow_cash < Decimal("0"):
            self.valid = False
            self.invalid_reason = 'INVALID_SHADOW_CASH_DEPLETED'
        self.shadow_qty = Decimal("0")
        self.shadow_direction = 'cash'

    def record_bar_close(self, close_price, timestamp=None):
        ts_str = str(timestamp) if timestamp is not None else ""
        if not self.valid:
            self.bar_equities.append(Decimal("0"))
            self.bar_returns.append(Decimal("0.0"))
            self.bar_timestamps.append(ts_str)
            return

        if close_price is None or (isinstance(close_price, float) and (math.isnan(close_price) or math.isinf(close_price))):
            self.valid = False
            self.invalid_reason = 'INVALID_NONPOSITIVE_OR_NONFINITE_EQUITY'
            self.bar_equities.append(Decimal("0"))
            self.bar_returns.append(Decimal("0.0"))
            self.bar_timestamps.append(ts_str)
            return

        try:
            close_dec = Decimal(str(close_price))
            if close_dec.is_nan() or close_dec.is_infinite() or close_dec <= Decimal("0"):
                self.valid = False
                self.invalid_reason = 'INVALID_NONPOSITIVE_OR_NONFINITE_EQUITY'
                self.bar_equities.append(Decimal("0"))
                self.bar_returns.append(Decimal("0.0"))
                self.bar_timestamps.append(ts_str)
                return
        except Exception:
            self.valid = False
            self.invalid_reason = 'INVALID_NONPOSITIVE_OR_NONFINITE_EQUITY'
            self.bar_equities.append(Decimal("0"))
            self.bar_returns.append(Decimal("0.0"))
            self.bar_timestamps.append(ts_str)
            return

        if self.shadow_direction == 'long':
            mtm = self.shadow_cash + (self.shadow_qty * close_dec)
        else:
            mtm = self.shadow_cash

        if mtm.is_nan() or mtm.is_infinite() or mtm <= Decimal("0"):
            self.valid = False
            self.invalid_reason = 'INVALID_NONPOSITIVE_OR_NONFINITE_EQUITY'
            self.bar_equities.append(mtm if not (mtm.is_nan() or mtm.is_infinite()) else Decimal("0"))
            self.bar_returns.append(Decimal("0.0"))
            self.bar_timestamps.append(ts_str)
            return

        prev_e = self.e_initial if len(self.bar_equities) == 0 else self.bar_equities[-1]
        if prev_e > Decimal("0"):
            r_bar = ((mtm - prev_e) / prev_e).quantize(RETURN_Q, rounding=ROUND_HALF_UP)
        else:
            r_bar = Decimal("0.0")

        self.bar_equities.append(mtm)
        self.bar_returns.append(r_bar)
        self.bar_timestamps.append(ts_str)

    def get_report(self, official_trades: list) -> dict:
        from validation_engine.walk_forward import official_completed_trade_pnl
        official_pnl_sum = sum((official_completed_trade_pnl(t) for t in official_trades), Decimal("0"))
        if self.valid and self.e_initial is not None and len(self.bar_equities) > 0:
            final_equity = self.bar_equities[-1]
            delta_cash = final_equity - self.e_initial
            monetary_residual = abs(delta_cash - official_pnl_sum)
        else:
            final_equity = None
            delta_cash = None
            monetary_residual = None

        return {
            'model': 'OBSERVATION-ONLY SHADOW-CAPITAL LEDGER',
            'status': 'VALID' if self.valid else self.invalid_reason,
            'valid': self.valid,
            'invalid_reason': self.invalid_reason,
            'embargo_close_price': self.embargo_close_price,
            'embargo_close_timestamp': self.embargo_close_timestamp,
            'configured_quantity': self.configured_qty,
            'fee_coverage_margin': self.fee_coverage_margin,
            'effective_fee_rate': self.effective_fee_rate,
            'effective_entry_slippage': self.effective_entry_slippage,
            'effective_stop_slippage': self.effective_stop_slippage,
            'e_initial': self.e_initial,
            'e_final': final_equity,
            'delta_shadow_cash': delta_cash,
            'official_monetary_pnl': official_pnl_sum,
            'monetary_reconciliation_residual': monetary_residual,
            'bar_count': len(self.bar_returns),
            'bar_equities': self.bar_equities,
            'bar_returns': self.bar_returns,
            'bar_timestamps': self.bar_timestamps,
            'trade_count': self.trade_count,
        }


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
    intended_quantity=1.0,
    initial_cash=1000000000.0,
    track_shadow_ledger=False,
    shadow_ledger_config=None
):
    entry_slippage = Decimal(str(entry_slippage))
    stop_slippage = Decimal(str(stop_slippage))

    requires_atr = (
        entry_slippage != Decimal("0")
        or stop_slippage != Decimal("0")
    )

    if requires_atr and "atr" not in df_with_signals.columns:
        raise ValueError(
            "ATR column required when entry_slippage or stop_slippage is non-zero"
        )
    trades = []
    rejected_events = []
    decision_traces = []   # Legacy G4 two-stage decision traces (plain dicts with Decimals)
    current_cash = Decimal(str(initial_cash))
    current_gross_exposure = Decimal("0")
    current_position_count = 0
    
    shadow_tracker = None
    if track_shadow_ledger:
        shadow_tracker = ShadowCapitalLedgerTracker(
            shadow_ledger_config=shadow_ledger_config,
            configured_quantity=intended_quantity
        )

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
        raise ValueError("ATR column required when entry_slippage or stop_slippage is non-zero")
        
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
                                exit_price = sl_price if stop_slippage == Decimal("0") else sl_price - (float(Decimal(str(current_row['atr'])) * stop_slippage))
                                exit_order_type = 'STOP_MARKET'
                                exit_reason = 'SL_TOUCH_BOTH'
                                req_price = sl_price
                            else:
                                exit_price = tp_price
                                exit_order_type = 'LIMIT'
                                exit_reason = 'TP_TOUCH_BOTH'
                                req_price = tp_price
                        elif hit_sl:
                            exit_price = sl_price if stop_slippage == Decimal("0") else sl_price - (float(Decimal(str(current_row['atr'])) * stop_slippage))
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
                                exit_price = sl_price if stop_slippage == Decimal("0") else sl_price + (float(Decimal(str(current_row['atr'])) * stop_slippage))
                                exit_order_type = 'STOP_MARKET'
                                exit_reason = 'SL_TOUCH_BOTH'
                                req_price = sl_price
                            else:
                                exit_price = tp_price
                                exit_order_type = 'LIMIT'
                                exit_reason = 'TP_TOUCH_BOTH'
                                req_price = tp_price
                        elif hit_sl:
                            exit_price = sl_price if stop_slippage == Decimal("0") else sl_price + (float(Decimal(str(current_row['atr'])) * stop_slippage))
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
                    exit_price = req_price if entry_slippage == Decimal("0") else req_price + (float(Decimal(str(current_row['atr'])) * entry_slippage))
                    if position == 'LONG': exit_price = req_price if entry_slippage == Decimal("0") else req_price - (float(Decimal(str(current_row['atr'])) * entry_slippage))
                    exit_order_type = 'MARKET'
                    exit_reason = "SIGNAL_REVERSAL"
                elif signal == -2:
                    req_price = float(next_row['open'])
                    exit_price = req_price if entry_slippage == Decimal("0") else req_price + (float(Decimal(str(current_row['atr'])) * entry_slippage))
                    if position == 'LONG': exit_price = req_price if entry_slippage == Decimal("0") else req_price - (float(Decimal(str(current_row['atr'])) * entry_slippage))
                    exit_order_type = 'MARKET'
                    exit_reason = "SIGNAL_FORCE_CLOSE"
                    
            if exit_price is not None:
                gross_return = (exit_price - entry_price) / entry_price if position == 'LONG' else (entry_price - exit_price) / entry_price
                net_return = gross_return - (2 * fee_rate)
                
                exit_fee_amount = exit_price * intended_quantity * fee_rate
                exit_fill_obj = {
                    'bar_index': i + 1,
                    'order_type': exit_order_type,
                    'requested_price': req_price,
                    'fill_price': exit_price,
                    'fee': exit_fee_amount,
                    'quantity': intended_quantity,
                    'last_observed_mark': None,
                    'mode': mode,
                    'gap_fill': gap_fill,
                    'reason': exit_reason,
                    'terminal_value_unverified': False,
                    'liquidity_diagnostics': {'exit_liquidity_modeled': False}
                }
                
                entry_notional = Decimal(str(entry_fill_obj['quantity'])) * Decimal(str(entry_fill_obj['fill_price']))
                exit_notional = Decimal(str(intended_quantity)) * Decimal(str(exit_price))
                if entry_current_position_direction.lower() == 'long':
                    gross_pnl_exit = exit_notional - entry_notional
                else:
                    gross_pnl_exit = entry_notional - exit_notional
                current_cash += entry_notional + gross_pnl_exit - Decimal(str(exit_fee_amount))
                current_gross_exposure = Decimal("0")
                current_position_count = 0

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': next_row['open_time'] if 'open_time' in next_row else next_row.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': intended_quantity,
                    'direction': 'long' if entry_current_position_direction.lower() == 'long' else 'short',
                    'entry_fee': entry_fill_obj.get('fee', 0.0),
                    'exit_fee': exit_fee_amount,
                    'net_return': net_return,
                    'exit_reason': exit_reason,
                    'entry_fill': entry_fill_obj,
                    'exit_fill': exit_fill_obj
                })
                position = None
                entry_fill_obj = None
                if shadow_tracker:
                    _exit_dir = 'long' if entry_current_position_direction.lower() == 'long' else 'short'
                    shadow_tracker.record_exit(
                        fill_price=Decimal(str(exit_price)),
                        quantity=Decimal(str(intended_quantity)),
                        fee=Decimal(str(exit_fee_amount)),
                        direction=_exit_dir
                    )
                
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
                
            slip_amt = 0.0 if entry_slippage == Decimal("0") else (float(Decimal(str(current_row['atr'])) * entry_slippage) * tier_factor)
            
            est_fill_px = req_px + slip_amt if signal == 1 else req_px - slip_amt
            if math.isnan(est_fill_px) or est_fill_px <= 0:
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
                        
            diag_obj = _build_liquidity_diag_normalized(
                bars_per_day=bars_per_day,
                required_lookback_bars=30 * bars_per_day if tiering_enabled else bars_per_day,
                p_30d=p_30d,
                p_24h=p_24h,
                prior_intrabar_range_fraction=float(current_row.get('prior_intrabar_range_fraction', 0.0)),
                req_ref_notional=req_ref_notional,
                req_exec_notional=req_exec_notional,
                est_fill_px=est_fill_px,
                capacity_cap=capacity_cap,
                approved_notional=approved_notional,
                participation_rate_cap=liquidity_config.get('capacity', {}).get('participation_rate_cap', 0.05) if capacity_enabled else None,
                tier_factor=tier_factor,
                capacity_action="REJECT" if rejected else liquidity_config.get('capacity', {}).get('action', 'ACCEPT'),
                missing_data_policy_applied=applied_policy,
                signal_bar_time=current_row.name,
                execution_bar_time=next_row.name,
                tiering_enabled=tiering_enabled,
                capacity_enabled=capacity_enabled,
                missing_tier=missing_tier,
                missing_cap=missing_cap,
                reject_reason=reject_reason,
            )
            
            # Legacy G4 Stage A: compute decision trace BEFORE any mutation (pure approval)
            _req_qty_dec = Decimal(str(intended_quantity))
            _req_px_dec = Decimal(str(req_px))
            _req_notional_dec = _req_qty_dec * _req_px_dec
            _slip_dec = Decimal(str(slip_amt))
            _est_fill_px_dec = Decimal(str(est_fill_px))
            _req_exec_notional_dec = _req_qty_dec * _est_fill_px_dec

            free_cash_before = current_cash
            gross_exposure_before = current_gross_exposure
            position_count_before = current_position_count

            # Check cash constraint
            if not rejected:
                if _est_fill_px_dec.is_nan() or _est_fill_px_dec <= Decimal("0"):
                    rejected = True
                    reject_reason = "INVALID_ESTIMATED_FILL_PRICE"
                elif not _req_exec_notional_dec.is_nan() and free_cash_before < _req_exec_notional_dec:
                    rejected = True
                    reject_reason = "INSUFFICIENT_CASH"

            if rejected:
                # Stage A produces immutable decision; zero out approval/fees and assert no state mutation
                _g4_trace = {
                    "attempt_id": f"sim_{i}_{str(current_row.name)}",
                    "timestamp": str(current_row.name),
                    "symbol": "single_asset",
                    "requested_qty": _req_qty_dec,
                    "requested_notional": _req_notional_dec,
                    "approved_qty": Decimal("0"),
                    "approved_notional": Decimal("0"),
                    "fill_price": Decimal("0"),
                    "slippage_amount": _slip_dec,
                    "entry_fee": Decimal("0"),
                    "exit_fee_reserve": Decimal("0"),
                    "free_cash_before": free_cash_before,
                    "free_cash_after": free_cash_before,
                    "gross_exposure_before": gross_exposure_before,
                    "gross_exposure_after": gross_exposure_before,
                    "position_count_before": position_count_before,
                    "position_count_after": position_count_before,
                    "status": "REJECTED",
                    "rejection_reason": reject_reason,
                }
                decision_traces.append(_g4_trace)

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
                # Stage B - Apply only if ACCEPTED
                # Compute fees only inside the ACCEPTED path
                _approved_qty_dec = _req_qty_dec
                _approved_notional_dec = _approved_qty_dec * _est_fill_px_dec
                fee_rate_dec = Decimal(str(fee_rate))
                _entry_fee_dec = _approved_notional_dec * fee_rate_dec
                _exit_fee_reserve_dec = _entry_fee_dec

                # In legacy single-asset mode, exit_fee_reserve is calculated and recorded
                # in the decision trace for diagnostic transparency and risk visibility,
                # but is NOT deducted from free_cash_after at entry.
                # Cash deducted at entry = _approved_notional_dec + _entry_fee_dec.
                # The actual exit fee is settled upon position closure on exit (line 261),
                # preventing duplicate deduction of the exit fee.
                free_cash_after = free_cash_before - _approved_notional_dec - _entry_fee_dec
                gross_exposure_after = gross_exposure_before + _approved_notional_dec
                position_count_after = position_count_before + 1

                _g4_trace = {
                    "attempt_id": f"sim_{i}_{str(current_row.name)}",
                    "timestamp": str(current_row.name),
                    "symbol": "single_asset",
                    "requested_qty": _req_qty_dec,
                    "requested_notional": _req_notional_dec,
                    "approved_qty": _approved_qty_dec,
                    "approved_notional": _approved_notional_dec,
                    "fill_price": _est_fill_px_dec,
                    "slippage_amount": _slip_dec,
                    "entry_fee": _entry_fee_dec,
                    "exit_fee_reserve": _exit_fee_reserve_dec,
                    "free_cash_before": free_cash_before,
                    "free_cash_after": free_cash_after,
                    "gross_exposure_before": gross_exposure_before,
                    "gross_exposure_after": gross_exposure_after,
                    "position_count_before": position_count_before,
                    "position_count_after": position_count_after,
                    "status": "ACCEPTED",
                    "rejection_reason": None,
                }
                decision_traces.append(_g4_trace)

                # Mutate portfolio state
                current_cash = free_cash_after
                current_gross_exposure = gross_exposure_after
                current_position_count = position_count_after

                position = 'LONG' if signal == 1 else 'SHORT'
                entry_price = est_fill_px
                entry_time = next_row['open_time'] if 'open_time' in next_row else next_row.name
                entry_qty = intended_quantity
                entry_fee_amount = float(_entry_fee_dec)
                entry_fill_obj = {
                    'bar_index': i + 1,
                    'order_type': 'MARKET_ENTRY',
                    'requested_price': req_px,
                    'fill_price': est_fill_px,
                    'fee': entry_fee_amount,
                    'quantity': entry_qty,
                    'mode': mode,
                    'gap_fill': False,
                    'reason': 'SIGNAL_LONG' if signal == 1 else 'SIGNAL_SHORT',
                    'time': next_row.name,
                    'liquidity_diagnostics': diag_obj
                }
                entry_current_position_direction = 'long' if signal == 1 else 'short'
                if shadow_tracker:
                    _entry_dir = 'long' if signal == 1 else 'short'
                    shadow_tracker.record_entry(
                        fill_price=_est_fill_px_dec,
                        quantity=_approved_qty_dec,
                        fee=_entry_fee_dec,
                        direction=_entry_dir
                    )

        if shadow_tracker:
            shadow_tracker.record_bar_close(current_row['close'], timestamp=current_row.name)
                
    if position is not None:
        last_row = df.iloc[-1]
        req_px = float(last_row['close'])
        if position.lower() == 'long':
            exit_price = req_px if stop_slippage == Decimal("0") else req_px - (last_row['atr'] * float(stop_slippage))
        else:
            exit_price = req_px if stop_slippage == Decimal("0") else req_px + (last_row['atr'] * float(stop_slippage))
            
        gross_return = (exit_price - entry_price) / entry_price if position.lower() == 'long' else (entry_price - exit_price) / entry_price
        net_return = gross_return - (2 * fee_rate)
        
        bndry_exit_fee = exit_price * intended_quantity * fee_rate
        exit_fill_obj = {
            'bar_index': len(df) - 1,
            'order_type': 'MARKET_FORCE_CLOSE',
            'requested_price': req_px,
            'fill_price': exit_price,
            'fee': bndry_exit_fee,
            'quantity': intended_quantity,
            'mode': mode,
            'gap_fill': False,
            'reason': 'BOUNDARY_CLOSE',
            'liquidity_diagnostics': {'exit_liquidity_modeled': False}
        }
        
        entry_notional = Decimal(str(entry_fill_obj['quantity'])) * Decimal(str(entry_fill_obj['fill_price']))
        exit_notional = Decimal(str(intended_quantity)) * Decimal(str(exit_price))
        if entry_current_position_direction.lower() == 'long':
            gross_pnl_exit = exit_notional - entry_notional
        else:
            gross_pnl_exit = entry_notional - exit_notional
        current_cash += entry_notional + gross_pnl_exit - Decimal(str(bndry_exit_fee))
        current_gross_exposure = Decimal("0")
        current_position_count = 0

        trades.append({
            'entry_time': entry_time,
            'exit_time': last_row['open_time'] if 'open_time' in last_row else last_row.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': intended_quantity,
            'direction': 'long' if entry_current_position_direction.lower() == 'long' else 'short',
            'entry_fee': entry_fill_obj.get('fee', 0.0) if entry_fill_obj else 0.0,
            'exit_fee': bndry_exit_fee,
            'net_return': net_return,
            'exit_reason': 'BOUNDARY_CLOSE',
            'entry_fill': entry_fill_obj,
            'exit_fill': exit_fill_obj
        })
        if shadow_tracker:
            _bndry_dir = 'long' if entry_current_position_direction.lower() == 'long' else 'short'
            shadow_tracker.record_exit(
                fill_price=Decimal(str(exit_price)),
                quantity=Decimal(str(intended_quantity)),
                fee=Decimal(str(bndry_exit_fee)),
                direction=_bndry_dir
            )
        
    if shadow_tracker and len(df) > 0:
        shadow_tracker.record_bar_close(df.iloc[-1]['close'], timestamp=df.iloc[-1].name)

    if track_shadow_ledger:
        shadow_report = shadow_tracker.get_report(trades) if shadow_tracker else None
        return trades, rejected_events, decision_traces, shadow_report

    return trades, rejected_events, decision_traces
