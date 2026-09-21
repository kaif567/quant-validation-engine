from .walk_forward import walk_forward_test
from .stats import calculate_metrics, calculate_confidence_interval, check_robustness

def multi_window_protocol(strategy_function, param_grid, data_dict):
    """
    Evaluates the strategy strictly across multiple provided datasets (coins/windows).
    """
    results = {}
    
    for name, df in data_dict.items():
        print(f"Running Walk-Forward Protocol for: {name}...")
        oos_trades = walk_forward_test(strategy_function, param_grid, df)
        
        metrics = calculate_metrics(oos_trades)
        ci_lower = calculate_confidence_interval(oos_trades)
        robust_metrics = check_robustness(oos_trades, remove_top_n=2)
        
        results[name] = {
            'total_trades': metrics['total_trades'],
            'oos_profit_factor': metrics['profit_factor'],
            'ci_lower_bound': ci_lower,
            'robust_profit_factor': robust_metrics['profit_factor'],
            'passed': ci_lower >= 1.0 and robust_metrics['profit_factor'] >= 1.0
        }
        
    return results
