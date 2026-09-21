import itertools
from .simulator import simulate_trades
from .stats import calculate_metrics

def walk_forward_test(strategy_function, param_grid, df, initial_train_window=1000, test_window=200, step=200, purge_bars=50):
    """
    Executes a walk-forward optimization across the DataFrame.
    Includes 'purge_bars' to prevent data leakage between Train and Test windows 
    (Lopez de Prado purging for indicator lookbacks).
    """
    total_len = len(df)
    current_start = 0
    all_oos_trades = []
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    while current_start + initial_train_window + purge_bars + test_window <= total_len:
        train_end = current_start + initial_train_window
        
        # Purging: Skip overlapping lookback bars to prevent leakage into Out-Of-Sample
        test_start = train_end + purge_bars 
        test_end = test_start + test_window
        
        train_df = df.iloc[current_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        best_pf = -1
        best_params = None
        
        # 1. Optimize on Train Window
        for params in param_combinations:
            sim_train = strategy_function(train_df.copy(), **params)
            
            sim_kwargs = {
                'stop_loss': params.get('stop_loss', params.get('sl', None)),
                'take_profit': params.get('take_profit', params.get('tp', None)),
                'fee_rate': params.get('fee_rate', 0.001),
                'slippage_rate': params.get('slippage_rate', 0.0005)
            }
            sim_kwargs = {k: v for k, v in sim_kwargs.items() if v is not None}
            
            trades = simulate_trades(sim_train, **sim_kwargs)
            metrics = calculate_metrics(trades)
            
            if metrics['profit_factor'] > best_pf:
                best_pf = metrics['profit_factor']
                best_params = params
                
        # 2. Execute on Out-Of-Sample Test Window
        if best_params is not None:
            sim_test = strategy_function(test_df.copy(), **best_params)
            
            sim_kwargs = {
                'stop_loss': best_params.get('stop_loss', best_params.get('sl', None)),
                'take_profit': best_params.get('take_profit', best_params.get('tp', None)),
                'fee_rate': best_params.get('fee_rate', 0.001),
                'slippage_rate': best_params.get('slippage_rate', 0.0005)
            }
            sim_kwargs = {k: v for k, v in sim_kwargs.items() if v is not None}
            
            oos_trades = simulate_trades(sim_test, **sim_kwargs)
            all_oos_trades.extend(oos_trades)
            
        current_start += step
        
    return all_oos_trades
