import itertools
from .simulator import simulate_trades
from .stats import calculate_metrics

def walk_forward_test(strategy_function, param_grid, df, initial_train_window=1000, test_window=200, step=200):
    """
    Executes a walk-forward optimization across the DataFrame.
    """
    total_len = len(df)
    current_start = 0
    all_oos_trades = []
    
    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    while current_start + initial_train_window + test_window <= total_len:
        train_end = current_start + initial_train_window
        test_end = train_end + test_window
        
        train_df = df.iloc[current_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        
        best_pf = -1
        best_params = None
        
        # 1. Optimize on Train Window
        for params in param_combinations:
            sim_train = strategy_function(train_df.copy(), **params)
            trades = simulate_trades(sim_train)
            metrics = calculate_metrics(trades)
            
            if metrics['profit_factor'] > best_pf:
                best_pf = metrics['profit_factor']
                best_params = params
                
        # 2. Execute on Out-Of-Sample Test Window
        if best_params is not None:
            sim_test = strategy_function(test_df.copy(), **best_params)
            oos_trades = simulate_trades(sim_test)
            all_oos_trades.extend(oos_trades)
            
        current_start += step
        
    return all_oos_trades
