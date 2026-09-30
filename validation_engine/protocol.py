import subprocess
from .walk_forward import walk_forward_test, validate_legacy_walk_forward_result
from .stats import calculate_metrics, calculate_confidence_interval, check_robustness

def get_git_revision_hash() -> str:
    """Gets the current git commit hash for the audit trail."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def evaluate_protocol_metrics(oos_trades, remove_top_pct=0.05, seed=None):
    """
    Evaluates out-of-sample trades using metrics, bootstrap confidence interval,
    and robustness checking with optional seed.
    """
    metrics = calculate_metrics(oos_trades)
    ci_lower = calculate_confidence_interval(oos_trades, seed=seed)
    
    # Percentage-based robustness
    robust_metrics = check_robustness(oos_trades, remove_top_pct=remove_top_pct)
    
    return {
        'total_trades': metrics['total_trades'],
        'oos_profit_factor': metrics['profit_factor'],
        'ci_lower_bound': ci_lower,
        'robust_profit_factor': robust_metrics['profit_factor'],
        'passed': bool(ci_lower >= 1.0 and robust_metrics['profit_factor'] >= 1.0)
    }

def multi_window_protocol(strategy_function, param_grid, data_dict, purge_bars=50, remove_top_pct=0.05, seed=None):
    """
    Evaluates the strategy strictly across multiple provided datasets (coins/windows).
    Generates a cryptographically auditable report structure.
    """
    git_hash = get_git_revision_hash()
    
    results = {
        "_audit_trail": {
            "engine_version_hash": git_hash,
            "purge_bars_applied": purge_bars,
            "robustness_pct_removed": remove_top_pct
        },
        "windows": {}
    }
    
    for name, df in data_dict.items():
        print(f"Running Walk-Forward Protocol for: {name} (Purging {purge_bars} bars)...")
        oos_result = walk_forward_test(strategy_function, param_grid, df, purge_bars=purge_bars)
        validate_legacy_walk_forward_result(oos_result)
        oos_trades = oos_result['trade_records']
        results["windows"][name] = evaluate_protocol_metrics(oos_trades, remove_top_pct=remove_top_pct, seed=seed)
        
    return results
