import numpy as np
import pandas as pd
import json
import hashlib
import sys
import math
from decimal import Decimal, ROUND_HALF_UP, getcontext, localcontext

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


CONFIGURED_G6_DECIMAL_PRECISION = 28
RETURN_Q = Decimal("0.000000000000000001")
MONEY_Q = Decimal("0.00000001")


def q_return(value: Decimal) -> Decimal:
    return value.quantize(RETURN_Q, rounding=ROUND_HALF_UP)


def q_money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_Q, rounding=ROUND_HALF_UP)


def to_canonical_decimal_str(val) -> str:
    if val is None:
        return "0.0"
    if isinstance(val, (int, str)):
        val = Decimal(str(val))
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Non-finite value cannot be serialized: {val}")
        val = Decimal(str(val))
    elif not isinstance(val, Decimal):
        val = Decimal(str(val))

    if val.is_nan() or val.is_infinite():
        raise ValueError(f"Non-finite Decimal cannot be serialized: {val}")

    return format(val, "f")


def get_runtime_environment_metadata() -> dict:
    import subprocess
    git_commit = None
    is_dirty = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status) > 0
    except Exception:
        pass
    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": sys.platform,
        "git_commit_hash": git_commit,
        "git_is_dirty": is_dirty,
    }


def run_g6_shadow_bootstrap_v2(
    folds_input: list,
    seed: int = 42,
    block_length: int = 10,
    replicates: int = 1000,
    configuration_metadata: dict = None
) -> tuple:
    """
    Circular block bootstrap under versioned contract:
      formula_version = "v2"
      schema_version = "g6_shadow_bar_returns_v2"
      block_length = 10
      replicates = 1000
      seed = 42
      boundary_policy = "CIRCULAR_WRAPAROUND"
      fold_weighting_policy = "EQUALLY_WEIGHTED_ARITHMETIC_MEAN"
      metric = Decimal compounded terminal return W - 1
      lower_bound = exact rank sorted_stats[int(0.05 * replicates)]
      strictness = q05 > Decimal("0")
    """
    environment_metadata = get_runtime_environment_metadata()

    if not folds_input:
        artifact = {
            "schema_version": "g6_shadow_bar_returns_v2",
            "formula_version": "v2",
            "g6_status": "INSUFFICIENT_TRADES",
            "g6_reason": "BAR_SAMPLE_TOO_SHORT",
            "computational_metric": "INSUFFICIENT",
            "q05": None,
            "full_bootstrap_artifact_hash": None,
            "canonical_payload_bytes": None,
            "full_statistics_artifact": None,
            "environment_metadata": environment_metadata,
        }
        return artifact, None, None

    cfg = configuration_metadata or {}
    folds_returns = []
    canonical_folds = []

    if isinstance(folds_input[0], dict):
        first_r = folds_input[0]
        norm_mult = cfg.get('normalization_multiplier', Decimal("1.005"))
        conf_qty = cfg.get('configured_quantity', first_r.get('configured_quantity', Decimal("1.0")))
        fee_rt = cfg.get('effective_fee_rate', first_r.get('effective_fee_rate', Decimal("0.001")))
        ent_slip = cfg.get('effective_entry_slippage', first_r.get('effective_entry_slippage', Decimal("0.0")))
        stp_slip = cfg.get('effective_stop_slippage', first_r.get('effective_stop_slippage', Decimal("0.0")))

        for idx, r in enumerate(folds_input):
            fr = r.get('bar_returns', [])
            fr_dec = [q_return(Decimal(str(x)) if not isinstance(x, Decimal) else x) for x in fr]
            folds_returns.append(fr_dec)

            p_emb = r.get('embargo_close_price')
            ts_emb = r.get('embargo_close_timestamp')
            c_init = r.get('e_initial')
            ts_obs = r.get('bar_timestamps', [])

            canonical_folds.append({
                "fold_id": r.get('fold_id', idx),
                "bar_count": len(fr_dec),
                "embargo_close_price": to_canonical_decimal_str(p_emb),
                "embargo_close_timestamp": str(ts_emb) if ts_emb is not None else "",
                "initial_shadow_capital": to_canonical_decimal_str(c_init),
                "observation_timestamps": [str(ts) for ts in ts_obs],
                "bar_returns": [format(val, "f") for val in fr_dec],
            })
    else:
        norm_mult = cfg.get('normalization_multiplier', Decimal("1.005"))
        conf_qty = cfg.get('configured_quantity', Decimal("1.0"))
        fee_rt = cfg.get('effective_fee_rate', Decimal("0.001"))
        ent_slip = cfg.get('effective_entry_slippage', Decimal("0.0"))
        stp_slip = cfg.get('effective_stop_slippage', Decimal("0.0"))

        for idx, raw_fold in enumerate(folds_input):
            fr_dec = [q_return(Decimal(str(x)) if not isinstance(x, Decimal) else x) for x in raw_fold]
            folds_returns.append(fr_dec)
            ts_obs = cfg.get('observation_timestamps')
            if ts_obs is None or len(ts_obs) != len(fr_dec):
                ts_obs = [f"bar_{i}" for i in range(len(fr_dec))]

            canonical_folds.append({
                "fold_id": idx,
                "bar_count": len(fr_dec),
                "embargo_close_price": to_canonical_decimal_str(cfg.get('embargo_close_price', Decimal("100.0"))),
                "embargo_close_timestamp": str(cfg.get('embargo_close_timestamp', "")),
                "initial_shadow_capital": to_canonical_decimal_str(cfg.get('initial_shadow_capital', Decimal("100.5"))),
                "observation_timestamps": [str(ts) for ts in ts_obs],
                "bar_returns": [format(val, "f") for val in fr_dec],
            })

    if any(len(fr) < block_length for fr in folds_returns):
        artifact = {
            "schema_version": "g6_shadow_bar_returns_v2",
            "formula_version": "v2",
            "g6_status": "INSUFFICIENT_TRADES",
            "g6_reason": "BAR_SAMPLE_TOO_SHORT",
            "computational_metric": "INSUFFICIENT",
            "q05": None,
            "full_bootstrap_artifact_hash": None,
            "canonical_payload_bytes": None,
            "full_statistics_artifact": None,
            "environment_metadata": environment_metadata,
        }
        return artifact, None, None

    with localcontext() as ctx:
        ctx.prec = CONFIGURED_G6_DECIMAL_PRECISION
        ctx.rounding = ROUND_HALF_UP

        assert getcontext().prec == CONFIGURED_G6_DECIMAL_PRECISION
        assert getcontext().rounding == ROUND_HALF_UP

        rng = np.random.RandomState(seed)
        full_bootstrap_statistics = []

        for _ in range(replicates):
            fold_statistics = []
            for fold_returns in folds_returns:
                n_f = len(fold_returns)
                sampled_returns = []
                while len(sampled_returns) < n_f:
                    start_idx = rng.randint(0, n_f)
                    block = [fold_returns[(start_idx + i) % n_f] for i in range(block_length)]
                    sampled_returns.extend(block)
                sampled_returns = sampled_returns[:n_f]

                fold_statistic = Decimal("1")
                for sr in sampled_returns:
                    fold_statistic = q_return(fold_statistic * (Decimal("1") + sr))
                fold_statistic = q_return(fold_statistic - Decimal("1"))
                fold_statistics.append(fold_statistic)

            aggregate_statistic = q_return(
                sum(fold_statistics, Decimal("0")) / Decimal(len(fold_statistics))
            )
            full_bootstrap_statistics.append(aggregate_statistic)

        sorted_stats = sorted(full_bootstrap_statistics)
        q05 = sorted_stats[int(0.05 * replicates)]
        computational_metric_pass = (q05 > Decimal("0"))
        g6_status = (
            "COMPUTATIONAL_PASS_PENDING_STATISTICAL_AUDIT"
            if computational_metric_pass
            else "COMPUTATIONAL_FAIL_NONPOSITIVE_LOWER_BOUND"
        )

        full_statistics_artifact = {
            "schema_version": "g6_shadow_bar_returns_v2",
            "formula_version": "v2",
            "policy_identifiers": {
                "scope": "LONG_CASH_ONLY",
                "sizing": "FIXED_EXECUTED_QUANTITY_NO_RESIZING",
                "signal_timing": "NEXT_OPEN_EXECUTION_BAR_CLOSE_MTM",
                "boundary_policy": "CIRCULAR_WRAPAROUND",
                "fold_weighting_policy": "EQUALLY_WEIGHTED_ARITHMETIC_MEAN",
                "rng_algorithm": "NUMPY_RANDOM_STATE_MT19937",
                "rng_call_order": "PER_REPLICATE_PER_FOLD_SEQUENTIAL_BLOCKS",
                "threshold_policy": "STRICT_GREATER_THAN_ZERO",
            },
            "configuration": {
                "block_length": block_length,
                "resample_count": len(full_bootstrap_statistics),
                "rng_seed": seed,
                "decimal_precision": getcontext().prec,
                "decimal_rounding": "ROUND_HALF_UP",
                "return_quantizer": format(RETURN_Q, "f"),
                "normalization_multiplier": to_canonical_decimal_str(norm_mult),
                "configured_quantity": to_canonical_decimal_str(conf_qty),
                "effective_fee_rate": to_canonical_decimal_str(fee_rt),
                "effective_entry_slippage": to_canonical_decimal_str(ent_slip),
                "effective_stop_slippage": to_canonical_decimal_str(stp_slip),
            },
            "folds": canonical_folds,
            "statistics": [format(val.quantize(RETURN_Q, rounding=ROUND_HALF_UP), "f") for val in full_bootstrap_statistics],
            "q05": format(q05.quantize(RETURN_Q, rounding=ROUND_HALF_UP), "f"),
        }

        serialized_bytes = json.dumps(
            full_statistics_artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False
        ).encode("utf-8")
        full_hash = hashlib.sha256(serialized_bytes).hexdigest()

        artifact = {
            "schema_version": "g6_shadow_bar_returns_v2",
            "formula_version": "v2",
            "g6_status": g6_status,
            "computational_metric": "PASS" if computational_metric_pass else "FAIL",
            "q05": q05,
            "full_bootstrap_artifact_hash": full_hash,
            "canonical_payload_bytes": serialized_bytes,
            "full_statistics_artifact": full_statistics_artifact,
            "environment_metadata": environment_metadata,
        }

        return artifact, full_hash, q05

