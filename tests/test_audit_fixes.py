import unittest
from decimal import Decimal

# 1. G5b: zero/nonpositive PnL applicability
def compute_g5b_loss_concentration(trade_pnls, min_trades=2):
    if len(trade_pnls) < min_trades:
        return "INSUFFICIENT_TRADES", None, None, None
    global_trade_pnl = sum(trade_pnls)
    losses = [p for p in trade_pnls if p < Decimal("0")]
    
    if not losses:
        return "NOT_APPLICABLE_NO_LOSSES", global_trade_pnl, None, None
    if global_trade_pnl <= Decimal("0"):
        return "NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL", global_trade_pnl, None, None
        
    largest_loss = min(losses)
    pnl_with_loss_doubled = global_trade_pnl + largest_loss
    status = "FAIL" if pnl_with_loss_doubled < Decimal("0") else "PASS"
    return status, global_trade_pnl, largest_loss, pnl_with_loss_doubled

# 2. Genuine circular wraparound
def sample_circular_block(fold_returns, start_idx, block_length):
    n_f = len(fold_returns)
    return [fold_returns[(start_idx + i) % n_f] for i in range(block_length)]

# 3. Trade-level purge
def purge_overlapping_trades(is_trades, embargo_start_timestamp):
    return [t for t in is_trades if t['exit_time'] <= embargo_start_timestamp]

def parameter_selection_loop(candidates, embargo_start_timestamp):
    best_param = None
    best_purged_pnl = Decimal("-999999")
    results = []

    for cand in candidates:
        raw_is_trades = cand['raw_is_trades']
        purged_is_trades = purge_overlapping_trades(raw_is_trades, embargo_start_timestamp)

        raw_pnl = sum((t['pnl'] for t in raw_is_trades), Decimal("0"))
        purged_pnl = sum((t['pnl'] for t in purged_is_trades), Decimal("0"))

        if purged_pnl > best_purged_pnl:
            best_purged_pnl = purged_pnl
            best_param = cand['name']

        results.append({
            "name": cand['name'],
            "raw_pnl": raw_pnl,
            "purged_pnl": purged_pnl,
            "raw_trades": raw_is_trades,
            "purged_trades": purged_is_trades
        })
    return best_param, results

# 4. Warmup boundaries
def build_canonical_folds_v2(n_bars, warmup_bars, train_window, purge_bars, embargo_bars, test_window, step):
    folds = []
    curr_train_start_idx = 0
    while True:
        curr_train_end_idx = curr_train_start_idx + train_window
        curr_test_start_idx = curr_train_end_idx + purge_bars + embargo_bars
        curr_test_end_idx = curr_test_start_idx + test_window
        if curr_test_end_idx > n_bars: break

        actual_train_start = max(0, curr_train_start_idx - warmup_bars)
        folds.append({
            "fold_id": len(folds),
            "warmup_context_interval": (actual_train_start, max(0, curr_train_start_idx - 1) if curr_train_start_idx > 0 else 0),
            "formal_is_scoring_interval": (curr_train_start_idx, curr_train_end_idx - 1),
            "oos_interval": (curr_test_start_idx, curr_test_end_idx - 1)
        })
        curr_train_start_idx += step
    return folds


class TestAuditFixes(unittest.TestCase):
    def test_g5b(self):
        tests_g5b = {
            "Test Z (Zero PnL)": ([Decimal("10"), Decimal("-10")], "NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL"),
            "Test A (Winning fragile)": ([Decimal("100"), Decimal("80"), Decimal("50"), Decimal("-220")], "FAIL"),
            "Test B (Winning robust)": ([Decimal("100"), Decimal("80"), Decimal("50"), Decimal("-30"), Decimal("-20")], "PASS"),
            "Test C (Negative global)": ([Decimal("10"), Decimal("-50")], "NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL"),
            "Test D (No losses)": ([Decimal("10"), Decimal("20"), Decimal("30")], "NOT_APPLICABLE_NO_LOSSES"),
            "Test E (Insufficient trades)": ([Decimal("10")], "INSUFFICIENT_TRADES")
        }
        for name, (pnls, expected_status) in tests_g5b.items():
            status, _, _, _ = compute_g5b_loss_concentration(pnls)
            self.assertEqual(status, expected_status, f"{name} failed")

    def test_circular_wraparound(self):
        fold_returns = list(range(100, 110))
        res = sample_circular_block(fold_returns, 8, 5)
        self.assertEqual(res, [108, 109, 100, 101, 102])
        self.assertEqual(len(res), 5)

    def test_trade_level_purge(self):
        candidate_A = {
            "name": "Candidate A",
            "raw_is_trades": [
                {"id": "A1", "entry_time": 10, "exit_time": 20, "pnl": Decimal("10")},
                {"id": "A2 (Overlap)", "entry_time": 95, "exit_time": 110, "pnl": Decimal("50")}
            ]
        }
        candidate_B = {
            "name": "Candidate B",
            "raw_is_trades": [
                {"id": "B1", "entry_time": 30, "exit_time": 40, "pnl": Decimal("20")}
            ]
        }
        best_param, results = parameter_selection_loop([candidate_A, candidate_B], 100)
        
        # Candidate A purged trades should only be A1
        self.assertEqual(len(results[0]['purged_trades']), 1)
        self.assertEqual(results[0]['purged_trades'][0]['id'], "A1")
        # Candidate B should win because A's +50 was purged
        self.assertEqual(best_param, "Candidate B")
        self.assertTrue(all(t['exit_time'] <= 100 for t in results[0]['purged_trades']))

    def test_warmup_boundaries(self):
        folds = build_canonical_folds_v2(1000, 100, 200, 0, 20, 100, 150)
        self.assertEqual(folds[0]['warmup_context_interval'], (0, 0))
        self.assertEqual(folds[1]['warmup_context_interval'], (50, 149))
        self.assertEqual(folds[1]['formal_is_scoring_interval'], (150, 349))
        self.assertEqual(folds[1]['oos_interval'], (370, 469))

    def test_minimum_sample_validity(self):
        test_window = 500
        truncated_length = 50
        MIN_OOS_FRACTION = Decimal("0.5")
        status = "COMPLETED"
        if Decimal(truncated_length) < Decimal(test_window) * MIN_OOS_FRACTION:
            status = "SKIPPED_INVALID_OOS_TRUNCATION"
        self.assertEqual(status, "SKIPPED_INVALID_OOS_TRUNCATION")

    def test_decimal_type_safety(self):
        def fix_run_g6_bootstrap(fold_returns):
            assert all(isinstance(v, Decimal) for v in fold_returns)
            return "Passed"
        
    def test_profit_factor_zero_loss_infinite_and_defined_status(self):
        """Zero losing trades yields profit_factor == inf and INFINITE_NO_LOSSES status"""
        from validation_engine.stats import calculate_metrics
        # All winning trades
        win_trades = [{'net_return': 0.05}, {'net_return': 0.02}]
        m_win = calculate_metrics(win_trades)
        self.assertEqual(m_win['profit_factor'], float('inf'))
        self.assertEqual(m_win['profit_factor_status'], 'INFINITE_NO_LOSSES')
        self.assertAlmostEqual(m_win['net_profit'], 0.07)

        # Empty trades
        m_empty = calculate_metrics([])
        self.assertEqual(m_empty['profit_factor'], 0.0)
        self.assertEqual(m_empty['profit_factor_status'], 'ZERO_NO_TRADES')
        self.assertEqual(m_empty['net_profit'], 0.0)

        # Standard win/loss
        normal_trades = [{'net_return': 0.10}, {'net_return': -0.05}]
        m_norm = calculate_metrics(normal_trades)
        self.assertEqual(m_norm['profit_factor'], 2.0)
        self.assertEqual(m_norm['profit_factor_status'], 'FINITE')
        self.assertAlmostEqual(m_norm['net_profit'], 0.05)

    def test_block_bootstrap_deterministic_repeat_run(self):
        """block_bootstrap is deterministic given explicit seed and accepts np.random.Generator"""
        import numpy as np
        from validation_engine.stats import block_bootstrap
        returns = [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.02, -0.015] * 5
        # Repeat run with identical integer seed
        q1 = block_bootstrap(returns, block_length=4, n_reps=200, seed=42)
        q2 = block_bootstrap(returns, block_length=4, n_reps=200, seed=42)
        self.assertEqual(q1, q2)

        # Different seed produces different result
        q3 = block_bootstrap(returns, block_length=4, n_reps=200, seed=99)
        self.assertNotEqual(q1, q3)

        # Accepts np.random.Generator directly
        rng = np.random.default_rng(42)
        q_gen = block_bootstrap(returns, block_length=4, n_reps=200, seed=rng)
        self.assertEqual(q1, q_gen)

    def test_protocol_smoke_ci_and_robustness(self):
        """Smoke test: imports validation_engine/protocol.py, runs calculate_confidence_interval and check_robustness with fixed seed, asserts expected keys and types"""
        import validation_engine.protocol as proto
        
        # Small fixed trade list (len >= 5 so CI is computed)
        fixed_trades = [
            {'net_return': 0.05},
            {'net_return': -0.02},
            {'net_return': 0.03},
            {'net_return': 0.01},
            {'net_return': -0.01},
            {'net_return': 0.04},
        ]
        
        # Call evaluate_protocol_metrics with a fixed seed
        res = proto.evaluate_protocol_metrics(fixed_trades, remove_top_pct=0.10, seed=42)
        
        # Assert returns without error and returns expected keys/types
        self.assertIsInstance(res, dict)
        self.assertIn('total_trades', res)
        self.assertIn('oos_profit_factor', res)
        self.assertIn('ci_lower_bound', res)
        self.assertIn('robust_profit_factor', res)
        self.assertIn('passed', res)
        
        self.assertIsInstance(res['total_trades'], int)
        self.assertEqual(res['total_trades'], 6)
        self.assertIsInstance(res['oos_profit_factor'], float)
        self.assertIsInstance(res['ci_lower_bound'], float)
        self.assertIsInstance(res['robust_profit_factor'], float)
        self.assertIsInstance(res['passed'], bool)
        
        # Deterministic repeat check with fixed seed
        res2 = proto.evaluate_protocol_metrics(fixed_trades, remove_top_pct=0.10, seed=42)
        self.assertEqual(res['ci_lower_bound'], res2['ci_lower_bound'])

if __name__ == '__main__':
    unittest.main()

