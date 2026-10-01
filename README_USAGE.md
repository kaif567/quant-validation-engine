# Quant Validation Engine (`quant-validation-engine`)

Institutional quantitative validation engine for systematic trading strategies.

## Scope Notice
> **Validation Scope**: `LEGACY_SINGLE_ASSET`  
> This package provides strict single-asset walk-forward validation and institutional gating (G1–G5). It does not include multi-asset portfolio or canonical R2 orchestration.

## Installation
Install in editable mode:
```bash
pip install -e .
```

## Minimal Example

```python
import pandas as pd
from validation_engine.walk_forward import (
    walk_forward_test,
    validate_legacy_walk_forward_result
)

# 1. Define strategy function
def moving_average_crossover(df, short_window=10, long_window=30):
    df = df.copy()
    df['fast'] = df['close'].rolling(short_window).mean()
    df['slow'] = df['close'].rolling(long_window).mean()
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1
    df.loc[df['fast'] < df['slow'], 'signal'] = -1
    return df

# 2. Run walk-forward validation
result = walk_forward_test(
    strategy_function=moving_average_crossover,
    param_grid={'short_window': [5, 10], 'long_window': [20, 30]},
    df=df_ohlcv,
    train_window=1000,
    test_window=200,
    step=200,
    purge_bars=50,
    embargo_bars=1,
    fee_rate=0.00075
)

# 3. Validate result contract
validate_legacy_walk_forward_result(result)
print("Overall pass:", result['overall_pass'])
print("Failed gates:", result['portfolio_gates']['failed_gates'])
```

## Result Schema Fields (`LegacyWalkForwardResult`)

The returned dictionary conforms strictly to the `LegacyWalkForwardResult` typed contract:

- `schema_version`: String identifier (e.g. `"legacy_wf_v1"`).
- `validation_scope`: Strict string `"LEGACY_SINGLE_ASSET"`.
- `per_fold`: List of fold summaries containing `fold_id`, `best_params`, and `train_objective`.
- `folds`: Total number of executed walk-forward folds.
- `portfolio_gates`: Dictionary of gate evaluation statuses:
  - `overall_pass`: Boolean whether all active institutional gates passed.
  - `failed_gates`: List of gate names that failed (`['G1']`, `['G4']`, `['G5']`, etc.).
  - `baseline_failed_gates`: List of gate failures under baseline fee/slippage conditions.
  - `institutional_pass_eligible`: Boolean eligibility flag.
  - `evaluation_equity_complete`: Integrity flag.
  - `G1`–`G6`: Detailed statuses per gate (`'PASS'`, `'FAIL'`, `'INSUFFICIENT_TRADES'`, etc.).
  - `completed_trade_count`, `profitable_trade_count`, `top5_share`, `top_profit`, `total_positive_profit`.
- `overall_pass`: Top-level boolean identical to `portfolio_gates['overall_pass']`.
- `trade_records`: Full list of out-of-sample executed trade dictionaries.
- `raw_trial_ledger_ref`: Local path string or reference to the run's `TrialLedgerV2` JSONL ledger file.

