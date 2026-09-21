import numpy as np

def calculate_metrics(trades):
    if not trades:
        return {'profit_factor': 0.0, 'win_rate': 0.0, 'total_trades': 0}
        
    returns = np.array([t['net_return'] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    
    pf = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    win_rate = len(wins) / len(returns)
    
    return {
        'profit_factor': pf,
        'win_rate': win_rate,
        'total_trades': len(trades)
    }

def calculate_confidence_interval(trades, n_iterations=1000, ci_level=95):
    """
    Calculates the lower bound of the Confidence Interval for the Profit Factor using Bootstrap resampling.
    NOTE: Currently uses random sampling (IID assumption). Block-bootstrapping planned for Phase 3 to account for serial correlation.
    """
    if len(trades) < 5:
        return 0.0  
        
    returns = np.array([t['net_return'] for t in trades])
    simulated_pfs = []
    
    for _ in range(n_iterations):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        wins = sample[sample > 0]
        losses = sample[sample < 0]
        
        gp = wins.sum() if len(wins) > 0 else 0
        gl = abs(losses.sum()) if len(losses) > 0 else 0
        
        if gl > 0:
            simulated_pfs.append(gp / gl)
        elif gp > 0:
            simulated_pfs.append(gp)
        else:
            simulated_pfs.append(0)
            
    lower_percentile = (100 - ci_level) / 2
    return np.percentile(simulated_pfs, lower_percentile)

def check_robustness(trades, remove_top_pct=0.05):
    """
    Profit Concentration Analysis (Claude Audit Fix).
    Removes the top X% most profitable trades instead of a fixed number,
    making the robustness check sample-size agnostic.
    """
    if not trades:
        return calculate_metrics([])
        
    n_to_remove = int(len(trades) * remove_top_pct)
    
    # Ensure we remove at least 1 trade if the sample size is small but meaningful (>10 trades)
    if n_to_remove == 0 and len(trades) >= 10:
        n_to_remove = 1
        
    if len(trades) <= n_to_remove:
        return calculate_metrics([])
        
    sorted_trades = sorted(trades, key=lambda x: x['net_return'], reverse=True)
    robust_trades = sorted_trades[n_to_remove:]
    return calculate_metrics(robust_trades)
