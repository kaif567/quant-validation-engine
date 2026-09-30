import numpy as np
import pandas as pd

def calculate_metrics(trades, n_bars=None):
    if not trades:
        return {
            'profit_factor': 0.0,
            'profit_factor_status': 'ZERO_NO_TRADES',
            'net_profit': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'mean_net_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }

    returns = np.array([t['net_return'] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0

    if gross_loss > 0:
        pf = float(gross_profit / gross_loss)
        pf_status = 'FINITE'
    elif gross_profit > 0:
        pf = float('inf')
        pf_status = 'INFINITE_NO_LOSSES'
    else:
        pf = 0.0
        pf_status = 'ZERO_NO_PROFIT'

    win_rate = len(wins) / len(returns)
    net_profit = float(returns.sum())

    mean_net = returns.mean()
    std_net = returns.std()
    sharpe = (mean_net / std_net) if std_net > 0 else 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

    skew_val = 0.0
    kurt_val = 0.0

    # Calculate Skewness & Kurtosis (DSR requirement)
    if len(returns) > 2:
        s = pd.Series(returns)
        skew_val = s.skew()
        kurt_val = s.kurt()
        if pd.isna(skew_val): skew_val = 0.0
        if pd.isna(kurt_val): kurt_val = 0.0

    return {
        'profit_factor': float(pf),
        'profit_factor_status': pf_status,
        'net_profit': float(net_profit),
        'win_rate': float(win_rate),
        'total_trades': len(trades),
        'mean_net_return': float(mean_net),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'skewness': float(skew_val),
        'kurtosis': float(kurt_val)
    }

def block_bootstrap(returns, block_length, n_reps=300, seed=None):
    n = len(returns)
    if n == 0:
        return 0.0
    if block_length >= n:
        block_length = max(1, n // 2)
        if block_length == 0:
            block_length = 1

    n_blocks = int(np.ceil(n / block_length))
    valid_starts = n - block_length + 1
    if valid_starts <= 0:
        valid_starts = 1

    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        rng = seed
    elif seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    simulated_means = []
    for _ in range(n_reps):
        if hasattr(rng, 'integers'):
            starts = rng.integers(0, valid_starts, size=n_blocks)
        else:
            starts = rng.randint(0, valid_starts, size=n_blocks)
        sample = []
        for s in starts:
            sample.extend(returns[s : s + block_length])
        sample = sample[:n]
        simulated_means.append(np.mean(sample))

    return float(np.percentile(simulated_means, 5))

def calculate_confidence_interval(trades, n_iterations=1000, ci_level=95, seed=None):
    """
    Calculates the lower bound of the Confidence Interval for the Profit Factor using Bootstrap resampling.
    NOTE: Currently uses random sampling (IID assumption). Block-bootstrapping planned for Phase 3 to account for serial correlation.
    """
    if len(trades) < 5:
        return 0.0  
        
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        rng = seed
    elif seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = None

    returns = np.array([t['net_return'] for t in trades])
    simulated_pfs = []
    
    for _ in range(n_iterations):
        if rng is not None:
            if hasattr(rng, 'choice'):
                sample = rng.choice(returns, size=len(returns), replace=True)
            else:
                sample = rng.choice(returns, size=len(returns), replace=True)
        else:
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
    return float(np.percentile(simulated_pfs, lower_percentile))

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

