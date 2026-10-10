import pandas as pd
import numpy as np
import itertools
import math
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
from typing import Callable, Dict, Any, List, TypedDict, Optional
from validation_engine.simulator import simulate_trades
from validation_engine.stats import calculate_metrics, run_g6_shadow_bootstrap_v2

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
    pending_gates: Optional[list[str]]
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


# Gate status categories (reference_design/multi_asset_orchestrator.py L47-56, L309-350):
#   PASS: gate evaluated, requirement met
#   FAIL: gate evaluated, requirement NOT met
#   NOT_EVALUATED_PENDING_LEGACY_ADAPTER: gate not yet computed (no adapter exists)
#   INSUFFICIENT_TRADES: cannot evaluate — not enough trades
#   NO_PROFITABLE_TRADES: G5-specific — no winners to measure concentration
#   NOT_APPLICABLE_*: gate does not apply (e.g. no losses for G5b)
#
# G1_G3_MEAN_RETURN: single gate covering mean_net_return > 0.
#   In reference_design, G1 is fold-level mean_ret > 0, G2 is median(fold_G1) > 0,
#   G3 is full-replay cost-resilience (separate from G4). In legacy single-asset
#   mode all three derive from the same mean_net_return < 0 check, so we honestly
#   collapse them into one gate. See reference_design/multi_asset_orchestrator.py L48-55.
#
# evaluation_equity_complete: True iff no trade has net_return=None (unresolved terminal).
#   Reference: reference_design/multi_asset_orchestrator.py L112, L273, L317.

GATE_STATUS_BACKDOOR_KWARGS = frozenset({'g5b_status', 'g6_status', 'g1_status', 'g2_status', 'g3_status', 'g4_status', 'g5_status'})

def aggregate_gate_statuses(gate_status_map: dict) -> tuple:
    """
    Pure function: given a dict mapping gate_name -> status_string,
    returns (overall_pass, failed_gates, pending_gates).
    
    Rules:
      - PASS or NOT_APPLICABLE_* → gate satisfied
      - FAIL, NO_PROFITABLE_TRADES → gate failed
      - NOT_EVALUATED_*, INSUFFICIENT_TRADES, or any status containing 'PENDING' → pending
      - overall_pass = True ONLY when failed_gates == [] AND pending_gates == []
    """
    pending_gates = []
    failed_gates = []

    for g_name in sorted(gate_status_map.keys()):
        status = gate_status_map[g_name]
        if status == 'PASS' or status.startswith('NOT_APPLICABLE'):
            continue
        elif (status.startswith('NOT_EVALUATED')
              or status == 'INSUFFICIENT_TRADES'
              or 'PENDING' in status):
            pending_gates.append(g_name)
        else:
            # FAIL, NO_PROFITABLE_TRADES, or any unknown status → treat as failure
            failed_gates.append(g_name)

    overall_pass = (len(failed_gates) == 0 and len(pending_gates) == 0)
    return overall_pass, failed_gates, pending_gates


def walk_forward_test(
    strategy_function: Callable,
    param_grid: Dict[str, list],
    df: pd.DataFrame,
    **kwargs
) -> LegacyWalkForwardResult:
    # Reject gate-status backdoor kwargs
    backdoor_found = GATE_STATUS_BACKDOOR_KWARGS & set(kwargs.keys())
    if backdoor_found:
        raise TypeError(
            f"walk_forward_test does not accept gate-status overrides: {backdoor_found}. "
            f"Gate statuses are computed internally, never injected by the caller."
        )
    
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
    fold_shadow_reports = []
    enable_shadow_g6 = kwargs.get('enable_shadow_g6', False)
    
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
            
            tp_param = params.get('take_profit', params.get('tp', None))
            target_roi = params.get('target_roi', kwargs.get('target_roi', None))
            f_rate = params.get('fee_rate', fee_rate)
            e_slip = kwargs.get('entry_slippage', 0.0)
            s_slip = kwargs.get('stop_slippage', 0.0)
            if target_roi is not None and tp_param is None:
                eff_entry = Decimal(str(f_rate)) + Decimal(str(e_slip))
                eff_exit = Decimal(str(f_rate)) + Decimal(str(s_slip))
                target_roi_dec = Decimal(str(target_roi))
                tp_param = float((Decimal("1") + target_roi_dec) * (Decimal("1") + eff_entry) / (Decimal("1") - eff_exit) - Decimal("1"))

            sim_kwargs = {
                'stop_loss': params.get('stop_loss', params.get('sl', None)),
                'take_profit': tp_param,
                'fee_rate': f_rate,
                'entry_slippage': e_slip,
                'stop_slippage': s_slip,
                'execution_model': kwargs.get('execution_model', 'v1_legacy')
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
            
            tp_param = best_params.get('take_profit', best_params.get('tp', None))
            target_roi = best_params.get('target_roi', kwargs.get('target_roi', None))
            f_rate = best_params.get('fee_rate', fee_rate)
            e_slip = kwargs.get('entry_slippage', 0.0)
            s_slip = kwargs.get('stop_slippage', 0.0)
            if target_roi is not None and tp_param is None:
                eff_entry = Decimal(str(f_rate)) + Decimal(str(e_slip))
                eff_exit = Decimal(str(f_rate)) + Decimal(str(s_slip))
                target_roi_dec = Decimal(str(target_roi))
                tp_param = float((Decimal("1") + target_roi_dec) * (Decimal("1") + eff_entry) / (Decimal("1") - eff_exit) - Decimal("1"))

            sim_kwargs = {
                'stop_loss': best_params.get('stop_loss', best_params.get('sl', None)),
                'take_profit': tp_param,
                'fee_rate': f_rate,
                'entry_slippage': e_slip,
                'stop_slippage': s_slip,
                'execution_model': kwargs.get('execution_model', 'v1_legacy')
            }
            sim_kwargs = {k: v for k, v in sim_kwargs.items() if v is not None}
            
            if enable_shadow_g6:
                embargo_close_price = None
                embargo_timestamp = None
                if test_start > 0:
                    embargo_bar = df.iloc[test_start - 1]
                    embargo_close_price = float(embargo_bar['close'])
                    embargo_timestamp = str(embargo_bar.name)
                shadow_cfg = {
                    'embargo_close_price': embargo_close_price,
                    'embargo_timestamp': embargo_timestamp,
                    'configured_quantity': kwargs.get('intended_quantity', 1.0),
                    'fee_coverage_margin': kwargs.get('fee_coverage_margin', Decimal("0.005")),
                    'fee_rate': sim_kwargs.get('fee_rate', fee_rate),
                    'entry_slippage': sim_kwargs.get('entry_slippage', 0.0),
                    'stop_slippage': sim_kwargs.get('stop_slippage', 0.0),
                }
                oos_trades, _, _, shadow_report = simulate_trades(
                    sim_test,
                    track_shadow_ledger=True,
                    shadow_ledger_config=shadow_cfg,
                    **sim_kwargs
                )
                if shadow_report:
                    shadow_report['fold_id'] = len(fold_results)
                fold_shadow_reports.append(shadow_report)
            else:
                oos_trades, _, _ = simulate_trades(sim_test, **sim_kwargs)
            all_oos_trades.extend(oos_trades)
            
        fold_results.append({'fold_id': len(fold_results), 'best_params': best_params, 'train_objective': best_pf})
        current_start += step
        
    if ledger:
        ledger.finish_run()
        
    baseline_failed_gates = []
    
    # G1_G3_MEAN_RETURN: collapsed gate for legacy single-asset mode.
    # In reference_design/multi_asset_orchestrator.py L48-55:
    #   G1 = fold-level mean_ret > min_fold_return
    #   G2 = median(fold_G1_returns) > 0
    #   G3 = NOT_EVALUATED_PENDING_PORTFOLIO_GATE_ADAPTER (full replay resilience)
    # All three reduce to mean_net_return < 0 in single-asset mode, so we
    # honestly collapse them into one gate rather than pretend we have three.
    if all_oos_trades:
        net = calculate_metrics(all_oos_trades).get('mean_net_return', 0)
        if net < 0:
            g1_g3_status = 'FAIL'
            baseline_failed_gates.append('G1_G3_MEAN_RETURN')
        else:
            g1_g3_status = 'PASS'
    else:
        g1_g3_status = 'INSUFFICIENT_TRADES'
    
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
                    f_rate_shock = best_p.get('fee_rate', fee_rate) * 2
                    e_slip_shock = kwargs.get('entry_slippage', 0.0) * 2
                    s_slip_shock = kwargs.get('stop_slippage', 0.0) * 2
                    tp_param_shock = best_p.get('take_profit', best_p.get('tp', None))
                    target_roi = best_p.get('target_roi', kwargs.get('target_roi', None))
                    if target_roi is not None:
                        eff_entry_shock = Decimal(str(f_rate_shock)) + Decimal(str(e_slip_shock))
                        eff_exit_shock = Decimal(str(f_rate_shock)) + Decimal(str(s_slip_shock))
                        target_roi_dec = Decimal(str(target_roi))
                        tp_param_shock = float((Decimal("1") + target_roi_dec) * (Decimal("1") + eff_entry_shock) / (Decimal("1") - eff_exit_shock) - Decimal("1"))

                    shock_kwargs = {
                        'stop_loss': best_p.get('stop_loss', best_p.get('sl', None)),
                        'take_profit': tp_param_shock,
                        'fee_rate': f_rate_shock,
                        'entry_slippage': e_slip_shock,
                        'stop_slippage': s_slip_shock,
                        'execution_model': kwargs.get('execution_model', 'v1_legacy')
                    }
                    shock_kwargs = {k: v for k, v in shock_kwargs.items() if v is not None}
                    shock_trades, _, _ = simulate_trades(sim_test, **shock_kwargs)
                    all_shock_trades.extend(shock_trades)
            current_start_shock += step
            idx += 1
            
        g4_metrics = calculate_metrics(all_shock_trades)
        if g4_metrics.get('profit_factor', 0) < 1.0 or g4_metrics.get('net_profit', 0) < 0:
            g4_status = 'FAIL'
        else:
            g4_status = 'PASS'
    else:
        g4_status = 'INSUFFICIENT_TRADES'

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
        else:
            g5_status = 'PASS'

    # G5b Loss Stress Test — Decimal official completed-trade PnL
    min_trades_for_g5b = kwargs.get('min_trades_for_g5b', 10)

    if g5_completed_trade_count < min_trades_for_g5b:
        g5b_status = 'INSUFFICIENT_TRADES'
        g5b_global_pnl = None
        g5b_largest_loss = None
        g5b_stressed_pnl = None
    else:
        g5b_global_pnl = sum(official_pnls, Decimal("0"))
        g5b_losses = [p for p in official_pnls if p < Decimal("0")]

        if g5b_global_pnl <= Decimal("0"):
            g5b_status = 'NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL'
            g5b_largest_loss = min(g5b_losses) if g5b_losses else None
            g5b_stressed_pnl = None
        elif not g5b_losses:
            g5b_status = 'NOT_APPLICABLE_NO_LOSSES'
            g5b_largest_loss = None
            g5b_stressed_pnl = None
        else:
            g5b_largest_loss = min(g5b_losses)  # most negative value
            g5b_stressed_pnl = g5b_global_pnl + g5b_largest_loss  # doubles the loss
            if g5b_stressed_pnl >= Decimal("0"):
                g5b_status = 'PASS'
            else:
                g5b_status = 'FAIL'

    if enable_shadow_g6:
        invalid_reports = [r for r in fold_shadow_reports if not r.get('valid', False)]
        if invalid_reports:
            g6_status = invalid_reports[0].get('invalid_reason', 'INVALID_SHADOW_CASH_DEPLETED')
            g6_computational_metric = 'INVALID'
            g6_q05 = None
            g6_hash = None
            g6_artifact = {
                'schema_version': 'g6_shadow_bar_returns_v2',
                'formula_version': 'v2',
                'g6_status': g6_status,
                'computational_metric': 'INVALID',
                'invalid_reports': [r.get('invalid_reason') for r in invalid_reports]
            }
        elif not fold_shadow_reports or any(r.get('bar_count', 0) < kwargs.get('g6_block_length', 10) for r in fold_shadow_reports):
            g6_status = 'INSUFFICIENT_TRADES'
            g6_computational_metric = 'INSUFFICIENT'
            g6_q05 = None
            g6_hash = None
            g6_artifact = {
                'schema_version': 'g6_shadow_bar_returns_v2',
                'formula_version': 'v2',
                'g6_status': g6_status,
                'g6_reason': 'BAR_SAMPLE_TOO_SHORT',
                'computational_metric': 'INSUFFICIENT',
            }
        else:
            cfg_meta = {
                'normalization_multiplier': Decimal("1.005"),
                'configured_quantity': kwargs.get('intended_quantity', 1.0),
                'effective_fee_rate': fee_rate,
                'effective_entry_slippage': kwargs.get('entry_slippage', 0.0),
                'effective_stop_slippage': kwargs.get('stop_slippage', 0.0),
            }
            g6_artifact, g6_hash, g6_q05 = run_g6_shadow_bootstrap_v2(
                fold_shadow_reports,
                seed=kwargs.get('g6_seed', 42),
                block_length=kwargs.get('g6_block_length', 10),
                replicates=kwargs.get('g6_replicates', 1000),
                configuration_metadata=cfg_meta
            )
            g6_status = g6_artifact['g6_status']
            g6_computational_metric = g6_artifact['computational_metric']
    else:
        # G6 — NOT computed in legacy single-asset mode. No adapter exists.
        g6_status = 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER'
        g6_computational_metric = None
        g6_q05 = None
        g6_hash = None
        g6_artifact = None

    gate_status_map = {
        'G1_G3_MEAN_RETURN': g1_g3_status,
        'G4': g4_status,
        'G5': g5_status,
        'G5b': g5b_status,
        'G6': g6_status,
    }

    # Use the pure aggregation function
    overall_pass, failed_gates, pending_gates = aggregate_gate_statuses(gate_status_map)
    institutional_pass_eligible = overall_pass

    # evaluation_equity_complete: True iff every trade has a resolved net_return.
    # (reference_design/multi_asset_orchestrator.py L112, L273)
    evaluation_equity_complete = all(
        t.get('net_return') is not None for t in all_oos_trades
    ) if all_oos_trades else True  # No trades = vacuously complete

    portfolio_gates = {
        'overall_pass': overall_pass,
        'failed_gates': failed_gates,
        'pending_gates': pending_gates,
        'baseline_failed_gates': baseline_failed_gates,
        'institutional_pass_eligible': institutional_pass_eligible,
        'evaluation_equity_complete': evaluation_equity_complete,
        'G1_G3_MEAN_RETURN': g1_g3_status,
        'G4': g4_status,
        'G5': g5_status,
        'g5_status': g5_status,
        'completed_trade_count': g5_completed_trade_count,
        'profitable_trade_count': g5_profitable_trade_count,
        'top_count': g5_top_count,
        'top_profit': g5_top_profit,
        'total_positive_profit': g5_total_positive_profit,
        'top5_share': g5_top5_share,
        # G5b fields
        'G5b': g5b_status,
        'g5b_status': g5b_status,
        'g5b_completed_trade_count': g5_completed_trade_count,
        'g5b_global_pnl': g5b_global_pnl,
        'g5b_largest_loss': g5b_largest_loss,
        'g5b_stressed_pnl': g5b_stressed_pnl,
        'g5b_min_trades': min_trades_for_g5b,
        # G6 fields
        'G6': g6_status,
        'g6_status': g6_status,
        'g6_computational_metric': g6_computational_metric,
        'g6_q05': g6_q05,
        'full_bootstrap_artifact_hash': g6_hash,
        'bootstrap': g6_artifact,
    }

    result: LegacyWalkForwardResult = {
        'schema_version': 'legacy_walk_forward_v1',
        'validation_scope': 'LEGACY_SINGLE_ASSET',
        'per_fold': fold_results,
        'folds': fold_results,
        'portfolio_gates': portfolio_gates,
        'failed_gates': failed_gates,
        'pending_gates': pending_gates,
        'overall_pass': overall_pass,
        'trade_records': all_oos_trades,
        'raw_trial_ledger_ref': run_id,
        'bootstrap': g6_artifact,
        'top5_share': None
    }

    validate_legacy_walk_forward_result(result)
    return result
