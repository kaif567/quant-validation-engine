import pandas as pd
import numpy as np
import itertools
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
from typing import Callable, Dict, Any, List, TypedDict, Optional
from validation_engine.simulator import simulate_trades
from validation_engine.stats import calculate_metrics

class LegacyWalkForwardResult(TypedDict):
    schema_version: str
    validation_scope: str   # must equal "LEGACY_SINGLE_ASSET"
    per_fold: list[dict]
    folds: list[dict]
    portfolio_gates: dict
    overall_pass: bool
    trade_records: list[dict]
    raw_trial_ledger_ref: Optional[str]
    failed_gates: list[str]
    bootstrap: Optional[Any]
    top5_share: Optional[Any]

def official_completed_trade_pnl(trade: dict) -> Decimal:
    """
    Extract official completed trade monetary PnL using Decimal arithmetic.
    Direction must be 'long' or 'short' (case-insensitive); otherwise raises ValueError.
    Requires fill_price, fee, and quantity to be present (no silent defaults).
    pnl = gross_pnl - entry_fee - exit_fee
    """
    if not isinstance(trade, dict):
        raise TypeError("Completed trade must be a dict")
        
    direction = trade.get('direction')
    if not direction or direction.lower() not in ('long', 'short'):
        raise ValueError("Completed trade has invalid direction")
    direction = direction.lower()
    
    entry_fill = trade.get('entry_fill')
    exit_fill = trade.get('exit_fill')
    if not isinstance(entry_fill, dict) or not isinstance(exit_fill, dict):
        raise ValueError("Completed trade missing entry_fill or exit_fill dict")
        
    if 'fill_price' not in entry_fill or entry_fill['fill_price'] is None:
        raise KeyError("entry_fill missing fill_price")
    if 'fill_price' not in exit_fill or exit_fill['fill_price'] is None:
        raise KeyError("exit_fill missing fill_price")
    if 'fee' not in entry_fill or entry_fill['fee'] is None:
        raise KeyError("entry_fill missing fee")
    if 'fee' not in exit_fill or exit_fill['fee'] is None:
        raise KeyError("exit_fill missing fee")
    if 'quantity' not in trade or trade['quantity'] is None:
        raise KeyError("trade missing quantity")
        
    entry_price = Decimal(str(entry_fill['fill_price']))
    exit_price = Decimal(str(exit_fill['fill_price']))
    entry_fee = Decimal(str(entry_fill['fee']))
    exit_fee = Decimal(str(exit_fill['fee']))
    quantity = Decimal(str(trade['quantity']))
    
    if direction == 'long':
        gross_pnl = (exit_price - entry_price) * quantity
    else:
        gross_pnl = (entry_price - exit_price) * quantity
        
    return gross_pnl - entry_fee - exit_fee

def validate_legacy_walk_forward_result(result: dict) -> None:
    """
    Runtime validator for LegacyWalkForwardResult.
    Raises ValueError or TypeError on missing keys, wrong types, or overall_pass mismatch.
    """
    if not isinstance(result, dict):
        raise TypeError(f"LegacyWalkForwardResult must be a dict, got {type(result)}")
        
    required_keys = {
        'schema_version': str,
        'validation_scope': str,
        'per_fold': list,
        'folds': list,
        'portfolio_gates': dict,
        'overall_pass': bool,
        'trade_records': list,
        'raw_trial_ledger_ref': (str, type(None)),
    }
    for key, expected_type in required_keys.items():
        if key not in result:
            raise ValueError(f"Missing required key in LegacyWalkForwardResult: {key}")
        if not isinstance(result[key], expected_type):
            raise TypeError(f"Key {key} must be of type {expected_type}, got {type(result[key])}")
            
    if result['validation_scope'] != "LEGACY_SINGLE_ASSET":
        raise ValueError(
            f"validation_scope must equal 'LEGACY_SINGLE_ASSET', got {result['validation_scope']!r}"
        )
        
    pg = result['portfolio_gates']
    if 'overall_pass' not in pg:
        raise ValueError("portfolio_gates missing 'overall_pass'")
    if not isinstance(pg['overall_pass'], bool):
        raise TypeError(f"portfolio_gates['overall_pass'] must be bool, got {type(pg['overall_pass'])}")
        
    if result['overall_pass'] != pg['overall_pass']:
        raise ValueError(
            f"overall_pass mismatch: result['overall_pass']={result['overall_pass']} "
            f"!= portfolio_gates['overall_pass']={pg['overall_pass']}"
        )

def walk_forward_test(
    strategy_function: Callable,
    param_grid: Dict[str, list],
    df: pd.DataFrame,
    **kwargs
) -> LegacyWalkForwardResult:
    
    initial_train_window = kwargs.get('initial_train_window', kwargs.get('train_window', 400))
    test_window = kwargs.get('test_window', 100)
    step = kwargs.get('step', 100)
    purge_bars = kwargs.get('purge_bars', 0)
    embargo_bars = kwargs.get('embargo_bars', 0)
    fee_rate = kwargs.get('fee_rate', 0.001)
    ledger = kwargs.get('ledger', None)
    run_id = kwargs.get('run_id', None)
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if ledger:
        ledger.start_run("legacy_camp", run_id or "default_run")
        from validation_engine.trial_ledger import dict_hash
    
    total_len = len(df)
    current_start = 0
    all_oos_trades = []
    fold_results = []
    
    while current_start + initial_train_window + purge_bars + test_window <= total_len:
        train_df = df.iloc[current_start:current_start + initial_train_window]
        
        test_start = current_start + initial_train_window + purge_bars + embargo_bars
        test_end = test_start + test_window
        test_df = df.iloc[test_start:test_end]
        
        best_params = None
        best_pf = -1.0
        
        evaluations = []
        for params in param_combinations:
            sim_train = strategy_function(train_df.copy(), **params)
            
            sim_kwargs = {
                'stop_loss': params.get('stop_loss', params.get('sl', None)),
                'take_profit': params.get('take_profit', params.get('tp', None)),
                'fee_rate': params.get('fee_rate', fee_rate),
                'entry_slippage': kwargs.get('entry_slippage', 0.0),
                'stop_slippage': kwargs.get('stop_slippage', 0.0)
            }
            sim_kwargs = {k: v for k, v in sim_kwargs.items() if v is not None}
            
            trades, _, _ = simulate_trades(sim_train, **sim_kwargs)
            metrics = calculate_metrics(trades)
            evaluations.append((params, metrics))

            if metrics['profit_factor'] > best_pf:
                best_pf = metrics['profit_factor']
                best_params = params
        
        if ledger:
            for params in param_combinations:
                hash_params = params.copy()
                hash_params["fee_rate"] = kwargs.get("fee_rate", fee_rate)
                hash_params["entry_slippage"] = kwargs.get("entry_slippage", 0.0)
                hash_params["stop_slippage"] = kwargs.get("stop_slippage", 0.0)
                identity = {
                    "strategy_family": "strategy",
                    "universe_hash": "univ_hash",
                    "param_hash": dict_hash(hash_params)
                }
                plan_data = {
                    'fold_definition_hash': '0'*64,
                    "identity": identity,
                    "fold_id": len(fold_results),
                    "status": "planned"
                }
                ledger.record_evaluation(plan_data)

            for params, metrics in evaluations:
                hash_params = params.copy()
                hash_params["fee_rate"] = kwargs.get("fee_rate", fee_rate)
                hash_params["entry_slippage"] = kwargs.get("entry_slippage", 0.0)
                hash_params["stop_slippage"] = kwargs.get("stop_slippage", 0.0)
                identity = {
                    "strategy_family": "strategy",
                    "universe_hash": "univ_hash",
                    "param_hash": dict_hash(hash_params)
                }
                evaluation_data = {
                    'fold_definition_hash': '0'*64,
                    "identity": identity,
                    "fold_id": len(fold_results),
                    "is_metrics": {"profit_factor": metrics['profit_factor']},
                    "is_champion_selected": (params == best_params),
                    "oos_metrics": {},
                    "status": "completed",
                    "objective_value": metrics['profit_factor']
                }
                ledger.record_evaluation(evaluation_data)
                
        if best_params is not None:
            if ledger:
                ledger.record_event('selection', {'fold_id': len(fold_results), 'selected': True, 'objective_value': best_pf})
            sim_test = strategy_function(test_df.copy(), **best_params)
            
            sim_kwargs = {
                'stop_loss': best_params.get('stop_loss', best_params.get('sl', None)),
                'take_profit': best_params.get('take_profit', best_params.get('tp', None)),
                'fee_rate': best_params.get('fee_rate', fee_rate),
                'entry_slippage': kwargs.get('entry_slippage', 0.0),
                'stop_slippage': kwargs.get('stop_slippage', 0.0)
            }
            sim_kwargs = {k: v for k, v in sim_kwargs.items() if v is not None}
            
            oos_trades, _, _ = simulate_trades(sim_test, **sim_kwargs)
            all_oos_trades.extend(oos_trades)
            
        fold_results.append({'fold_id': len(fold_results), 'best_params': best_params, 'train_objective': best_pf})
        current_start += step
        
    if ledger:
        ledger.finish_run()
        
    baseline_failed_gates = []
    failed_gates = []
    overall_pass = True
    
    if all_oos_trades:
        net = calculate_metrics(all_oos_trades).get('mean_net_return', 0)
        if net < 0:
            baseline_failed_gates.extend(['G1', 'G2', 'G3'])
            failed_gates.extend(['G1', 'G2', 'G3'])
            overall_pass = False
    
    # G4 Cost Shock Replay
    if all_oos_trades:
        all_shock_trades = []
        current_start_shock = 0
        idx = 0
        while current_start_shock + initial_train_window + purge_bars + test_window <= total_len:
            test_start = current_start_shock + initial_train_window + purge_bars + embargo_bars
            test_end = test_start + test_window
            test_df = df.iloc[test_start:test_end].copy()
            if idx < len(fold_results):
                best_p = fold_results[idx]['best_params']
                if best_p:
                    sim_test = strategy_function(test_df.copy(), **best_p)
                    shock_kwargs = {
                        'stop_loss': best_p.get('stop_loss', best_p.get('sl', None)),
                        'take_profit': best_p.get('take_profit', best_p.get('tp', None)),
                        'fee_rate': best_p.get('fee_rate', fee_rate) * 2,
                        'entry_slippage': kwargs.get('entry_slippage', 0.0) * 2,
                        'stop_slippage': kwargs.get('stop_slippage', 0.0) * 2
                    }
                    shock_kwargs = {k: v for k, v in shock_kwargs.items() if v is not None}
                    shock_trades, _, _ = simulate_trades(sim_test, **shock_kwargs)
                    all_shock_trades.extend(shock_trades)
            current_start_shock += step
            idx += 1
            
        g4_metrics = calculate_metrics(all_shock_trades)
        if g4_metrics.get('profit_factor', 0) < 1.0 or g4_metrics.get('net_profit', 0) < 0:
            failed_gates.append('G4')
            overall_pass = False

    # G5 Concentration — Decimal official completed-trade PnL
    # Uses actual entry/exit fill prices, fees, and quantity (NOT net_return shortcut)
    min_trades_for_g5 = kwargs.get('min_trades_for_g5', 10)
    max_share = Decimal(str(kwargs.get('max_top5_share', 1.0)))

    official_pnls = []
    g5_completed_trade_count = 0
    g5_profitable_trade_count = 0

    for t in all_oos_trades:
        # Skip unresolved terminal trades (net_return is None)
        if t.get('net_return') is None:
            continue
        g5_completed_trade_count += 1
        pnl = official_completed_trade_pnl(t)
        official_pnls.append(pnl)
        if pnl > Decimal("0"):
            g5_profitable_trade_count += 1

    profitable_pnls = sorted([p for p in official_pnls if p > Decimal("0")], reverse=True)

    if g5_completed_trade_count < min_trades_for_g5:
        g5_status = 'INSUFFICIENT_TRADES'
        g5_top_count = 0
        g5_top_profit = None
        g5_total_positive_profit = None
        g5_top5_share = None
    elif g5_profitable_trade_count == 0:
        g5_status = 'NO_PROFITABLE_TRADES'
        g5_top_count = 0
        g5_top_profit = Decimal("0")
        g5_total_positive_profit = Decimal("0")
        g5_top5_share = None
    else:
        raw_top = (Decimal(str(g5_profitable_trade_count)) * Decimal("0.05")).to_integral_value(rounding=ROUND_CEILING)
        g5_top_count = max(1, int(raw_top))
        g5_top_profit = sum(profitable_pnls[:g5_top_count])
        g5_total_positive_profit = sum(profitable_pnls)
        if g5_total_positive_profit > Decimal("0"):
            g5_top5_share = g5_top_profit / g5_total_positive_profit
        else:
            g5_top5_share = Decimal("0")

        if g5_top5_share > max_share:
            g5_status = 'FAIL'
            baseline_failed_gates.append('G5')
            failed_gates.append('G5')
        else:
            g5_status = 'PASS'

    if len(all_oos_trades) == 0:
        overall_pass = False
        if 'G4' not in failed_gates:
            failed_gates.append('G4')

    if failed_gates:
        overall_pass = False

    portfolio_gates = {
        'overall_pass': overall_pass,
        'failed_gates': failed_gates,
        'baseline_failed_gates': baseline_failed_gates,
        'institutional_pass_eligible': overall_pass,
        'evaluation_equity_complete': True,
        'G4': 'PASS' if 'G4' not in failed_gates and all_oos_trades else 'FAIL',
        'G5': g5_status,
        'g5_status': g5_status,
        'completed_trade_count': g5_completed_trade_count,
        'profitable_trade_count': g5_profitable_trade_count,
        'top_count': g5_top_count,
        'top_profit': g5_top_profit,
        'total_positive_profit': g5_total_positive_profit,
        'top5_share': g5_top5_share,
        'G5b': 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER',
        'g5b_status': 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER',
        'G6': 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER',
        'g6_status': 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER',
        'bootstrap': None,
    }

    result: LegacyWalkForwardResult = {
        'schema_version': 'legacy_walk_forward_v1',
        'validation_scope': 'LEGACY_SINGLE_ASSET',
        'per_fold': fold_results,
        'folds': fold_results,
        'portfolio_gates': portfolio_gates,
        'failed_gates': failed_gates,
        'overall_pass': overall_pass,
        'trade_records': all_oos_trades,
        'raw_trial_ledger_ref': run_id,
        'bootstrap': None,
        'top5_share': None
    }

    validate_legacy_walk_forward_result(result)
    return result
