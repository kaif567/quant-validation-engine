import unittest
import pandas as pd
import numpy as np
import json
import hashlib
from decimal import Decimal
from validation_engine.simulator import simulate_trades, ShadowCapitalLedgerTracker
from validation_engine.stats import run_g6_shadow_bootstrap_v2
from validation_engine.walk_forward import (
    walk_forward_test,
    aggregate_gate_statuses,
    official_completed_trade_pnl,
)


class TestShadowCapitalLedgerG6(unittest.TestCase):

    def _make_sample_df(self, n_bars=75, base_price=100.0):
        dates = pd.date_range("2026-01-01", periods=n_bars, freq="4h", tz="UTC")
        df = pd.DataFrame({
            'open': [base_price] * n_bars,
            'high': [base_price + 2.0] * n_bars,
            'low': [base_price - 2.0] * n_bars,
            'close': [base_price] * n_bars,
            'signal': [0] * n_bars,
            'atr': [1.0] * n_bars,
        }, index=dates)
        return df

    # --- SECTION A & E: Monetary Reconciliation & Numeric Precision ---

    def test_cash_only_fold(self):
        """No trades: all 75 bars flat in cash, returns 0.0, zero monetary residual"""
        df = self._make_sample_df(75, 100.0)
        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}
        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertEqual(len(trades), 0)
        self.assertTrue(report['valid'])
        self.assertEqual(report['status'], 'VALID')
        self.assertEqual(report['bar_count'], 75)
        self.assertEqual(report['trade_count'], 0)
        self.assertEqual(report['e_initial'], Decimal("100.0") * Decimal("1.005"))
        self.assertEqual(report['e_final'], report['e_initial'])
        self.assertEqual(report['delta_shadow_cash'], Decimal("0"))
        self.assertEqual(report['official_monetary_pnl'], Decimal("0"))
        self.assertEqual(report['monetary_reconciliation_residual'], Decimal("0"))
        self.assertTrue(all(r == Decimal("0.0") for r in report['bar_returns']))

    def test_winning_and_losing_trades_and_reconciliation(self):
        """Winning and losing trades: delta shadow cash matches sum of official PnL with zero residual"""
        df = self._make_sample_df(75, 100.0)
        df.loc[df.index[1], 'signal'] = 1   # Signal at bar 1 close -> enter bar 2 open
        df.loc[df.index[9], 'signal'] = -2  # Signal at bar 9 close -> exit bar 10 open
        df.loc[df.index[10], 'open'] = 110.0
        df.loc[df.index[10], 'close'] = 110.0

        df.loc[df.index[19], 'signal'] = 1  # Signal at bar 19 close -> enter bar 20 open
        df.loc[df.index[20], 'open'] = 110.0
        df.loc[df.index[29], 'signal'] = -2 # Signal at bar 29 close -> exit bar 30 open
        df.loc[df.index[30], 'open'] = 95.0
        df.loc[df.index[30], 'close'] = 95.0

        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}
        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertEqual(len(trades), 2)
        self.assertTrue(report['valid'])
        self.assertEqual(report['monetary_reconciliation_residual'], Decimal("0"))

        # Pure Decimal return-product fixture tolerance comparison
        compounded = Decimal("1")
        for r in report['bar_returns']:
            compounded *= (Decimal("1") + r)
        expected_ratio = report['e_final'] / report['e_initial']
        diff = abs(compounded - expected_ratio)
        self.assertLess(diff, Decimal("1e-15"))

    def test_entry_exit_fees_exactness(self):
        """Entry and exit fees deduct cleanly on execution bars without leaks"""
        df = self._make_sample_df(75, 100.0)
        df.loc[df.index[5], 'signal'] = 1
        df.loc[df.index[10], 'signal'] = -2
        df.loc[df.index[11], 'open'] = 100.0

        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}
        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, entry_slippage=0.0, stop_slippage=0.0,
            track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertEqual(len(trades), 1)
        t = trades[0]
        expected_entry_fee = Decimal(str(t['entry_fill']['fee']))
        expected_exit_fee = Decimal(str(t['exit_fill']['fee']))
        total_fees = expected_entry_fee + expected_exit_fee

        self.assertEqual(report['delta_shadow_cash'], -total_fees)
        self.assertEqual(report['official_monetary_pnl'], -total_fees)
        self.assertEqual(report['monetary_reconciliation_residual'], Decimal("0"))

    def test_non_unit_quantity(self):
        """Configured quantity 2.5 scales notional, fees, and reconciles to official PnL"""
        df = self._make_sample_df(75, 100.0)
        df.loc[df.index[5], 'signal'] = 1
        df.loc[df.index[10], 'signal'] = -2
        df.loc[df.index[11], 'open'] = 120.0

        qty = 2.5
        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': qty}
        trades, rejected, traces, report = simulate_trades(
            df, intended_quantity=qty, fee_rate=0.0015, entry_slippage=0.0, stop_slippage=0.0,
            track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(report['e_initial'], Decimal("100.0") * Decimal("2.5") * Decimal("1.005"))
        self.assertEqual(report['monetary_reconciliation_residual'], Decimal("0"))

    def test_open_position_mark_to_market(self):
        """Open position marks to market causal close without fee leakage"""
        df = self._make_sample_df(75, 100.0)
        df.loc[df.index[1], 'signal'] = 1
        df.loc[df.index[2], 'open'] = 100.0
        df.loc[df.index[2], 'close'] = 105.0
        df.loc[df.index[3], 'close'] = 110.0

        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}
        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, entry_slippage=0.0, stop_slippage=0.0,
            track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        equities = report['bar_equities']
        fee_entry = Decimal("100.0") * Decimal("0.0015")
        cash_after_entry = report['e_initial'] - Decimal("100.0") - fee_entry
        self.assertEqual(equities[2], cash_after_entry + Decimal("105.0"))
        self.assertEqual(equities[3], cash_after_entry + Decimal("110.0"))

    def test_forced_boundary_exit_reconciliation(self):
        """Trade remaining open through bar 74 force-closes with zero double counting"""
        df = self._make_sample_df(75, 100.0)
        df.loc[df.index[70], 'signal'] = 1
        df.loc[df.index[74], 'close'] = 115.0

        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}
        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['exit_reason'], 'BOUNDARY_CLOSE')
        self.assertTrue(report['valid'])
        self.assertEqual(report['monetary_reconciliation_residual'], Decimal("0"))

    def test_negative_shadow_cash_depletion(self):
        """Depleting shadow cash marks INVALID_SHADOW_CASH_DEPLETED and fails gate"""
        df = self._make_sample_df(75, 1000.0)
        df.loc[df.index[5], 'signal'] = 1
        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}

        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertFalse(report['valid'])
        self.assertEqual(report['status'], 'INVALID_SHADOW_CASH_DEPLETED')
        self.assertEqual(report['invalid_reason'], 'INVALID_SHADOW_CASH_DEPLETED')

    def test_unsupported_short_direction_rejected(self):
        """Short signal while flat rejects with INVALID_UNSUPPORTED_SHORT_DIRECTION"""
        df = self._make_sample_df(75, 100.0)
        df.loc[df.index[5], 'signal'] = -1

        shadow_cfg = {'embargo_close_price': 100.0, 'configured_quantity': 1.0}
        trades, rejected, traces, report = simulate_trades(
            df, fee_rate=0.0015, track_shadow_ledger=True, shadow_ledger_config=shadow_cfg
        )

        self.assertFalse(report['valid'])
        self.assertEqual(report['status'], 'INVALID_UNSUPPORTED_SHORT_DIRECTION')

    # --- SECTION B: Artifact Export and Sensitivity ---

    def test_artifact_bytes_export_reproduces_sha256(self):
        """Exported canonical bytes independently reproduce the recorded SHA-256 digest"""
        fold_returns = [
            [Decimal("0.001"), Decimal("-0.002"), Decimal("0.003")] * 5,
        ]
        art, full_hash, q05 = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100)

        raw_bytes = art['canonical_payload_bytes']
        self.assertIsInstance(raw_bytes, bytes)
        computed_hash = hashlib.sha256(raw_bytes).hexdigest()
        self.assertEqual(computed_hash, full_hash)
        self.assertEqual(computed_hash, art['full_bootstrap_artifact_hash'])

        # Verify exact re-serialization from dictionary
        re_serialized = json.dumps(
            art['full_statistics_artifact'],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False
        ).encode("utf-8")
        self.assertEqual(re_serialized, raw_bytes)

    def test_canonical_hash_identity_same_inputs(self):
        """Same inputs, parameters, and environment yield identical canonical bytes and hash"""
        fold_returns = [
            [Decimal("0.002"), Decimal("-0.001"), Decimal("0.001")] * 5,
            [Decimal("-0.001"), Decimal("0.001"), Decimal("-0.002")] * 5,
        ]
        art1, hash1, q1 = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100)
        art2, hash2, q2 = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100)

        self.assertEqual(art1['canonical_payload_bytes'], art2['canonical_payload_bytes'])
        self.assertEqual(hash1, hash2)
        self.assertEqual(q1, q2)

    def test_hash_sensitivity_changing_input_return_by_1e18(self):
        """Perturbing a bound input return by exactly 1e-18 changes input provenance and the hash"""
        fold_returns1 = [
            [Decimal("0.001"), Decimal("-0.002"), Decimal("0.003")] * 5,
        ]
        fold_returns2 = [
            [Decimal("0.001") + Decimal("0.000000000000000001"), Decimal("-0.002"), Decimal("0.003")] +
            [Decimal("0.001"), Decimal("-0.002"), Decimal("0.003")] * 4,
        ]

        art1, hash1, q1 = run_g6_shadow_bootstrap_v2(fold_returns1, seed=42, block_length=10, replicates=100)
        art2, hash2, q2 = run_g6_shadow_bootstrap_v2(fold_returns2, seed=42, block_length=10, replicates=100)

        self.assertNotEqual(hash1, hash2)
        self.assertNotEqual(art1['canonical_payload_bytes'], art2['canonical_payload_bytes'])
        # The input return in canonical payload is explicitly changed
        self.assertNotEqual(
            art1['full_statistics_artifact']['folds'][0]['bar_returns'][0],
            art2['full_statistics_artifact']['folds'][0]['bar_returns'][0]
        )

    def test_hash_sensitivity_rounding_policy_below_quantizer(self):
        """Perturbations below RETURN_Q obey ROUND_HALF_UP rounding policy"""
        # Base value with 19 digits: 0.0010000000000000000
        base_val = Decimal("0.0010000000000000000")
        # Below 0.5 * 1e-18: add 0.0000000000000000004 -> rounds down under ROUND_HALF_UP
        val_down = base_val + Decimal("0.0000000000000000004")
        # Above 0.5 * 1e-18: add 0.0000000000000000006 -> rounds up under ROUND_HALF_UP
        val_up = base_val + Decimal("0.0000000000000000006")

        f_base = [[base_val] * 15]
        f_down = [[val_down] + [base_val] * 14]
        f_up = [[val_up] + [base_val] * 14]

        art_base, hash_base, _ = run_g6_shadow_bootstrap_v2(f_base, seed=42, block_length=10, replicates=100)
        art_down, hash_down, _ = run_g6_shadow_bootstrap_v2(f_down, seed=42, block_length=10, replicates=100)
        art_up, hash_up, _ = run_g6_shadow_bootstrap_v2(f_up, seed=42, block_length=10, replicates=100)

        self.assertEqual(hash_base, hash_down)
        self.assertNotEqual(hash_base, hash_up)

    def test_hash_sensitivity_configuration_change(self):
        """Changing effective capital or configuration is bound by the canonical hash"""
        fold_returns = [[Decimal("0.001")] * 15]
        cfg1 = {'effective_fee_rate': Decimal("0.0015"), 'embargo_close_price': Decimal("100.0")}
        cfg2 = {'effective_fee_rate': Decimal("0.0020"), 'embargo_close_price': Decimal("100.0")}
        cfg3 = {'effective_fee_rate': Decimal("0.0015"), 'embargo_close_price': Decimal("105.0")}

        art1, hash1, _ = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100, configuration_metadata=cfg1)
        art2, hash2, _ = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100, configuration_metadata=cfg2)
        art3, hash3, _ = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100, configuration_metadata=cfg3)

        self.assertNotEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)

    def test_hash_invariance_environment_metadata(self):
        """Changing outer environment metadata alone does NOT change the canonical payload or hash"""
        fold_returns = [[Decimal("0.001")] * 15]
        art, hash1, _ = run_g6_shadow_bootstrap_v2(fold_returns, seed=42, block_length=10, replicates=100)

        # Mutate environment metadata in outer wrapper
        art['environment_metadata']['python_version'] = "custom_python_test"
        art['environment_metadata']['git_commit_hash'] = "custom_commit_hash"

        # Recompute hash of canonical payload
        payload_bytes = json.dumps(
            art['full_statistics_artifact'],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False
        ).encode("utf-8")
        self.assertEqual(hashlib.sha256(payload_bytes).hexdigest(), hash1)

    # --- SECTION C: Invalid-Input Coverage ---

    def test_invalid_initial_capital_nan_infinity_nonpositive(self):
        """Initial capital handling gracefully rejects NaN, Inf, negative, zero, and bad types"""
        invalid_cases = [
            None,
            float('nan'),
            float('inf'),
            float('-inf'),
            -10.0,
            0.0,
            "NaN",
            "Infinity",
            "-Infinity",
            "invalid_price_string",
        ]
        for val in invalid_cases:
            tracker = ShadowCapitalLedgerTracker({'embargo_close_price': val})
            self.assertFalse(tracker.valid, f"Failed to invalidate for embargo_close_price={val}")
            self.assertEqual(tracker.invalid_reason, 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE')

        # Invalid quantity
        for q in [None, 0.0, -1.0, float('nan'), float('inf')]:
            tracker = ShadowCapitalLedgerTracker({'embargo_close_price': 100.0, 'configured_quantity': q})
            self.assertFalse(tracker.valid, f"Failed to invalidate for configured_quantity={q}")
            self.assertEqual(tracker.invalid_reason, 'INVALID_CAPITAL_INPUT_MISSING_OR_NONPOSITIVE')

    def test_record_bar_close_nan_infinity_nonpositive(self):
        """record_bar_close gracefully catches NaN, Inf, negative, zero, and bad types without uncaught exceptions"""
        invalid_prices = [
            None,
            float('nan'),
            float('inf'),
            float('-inf'),
            0.0,
            -50.0,
            "NaN",
            "Infinity",
            "-Infinity",
            "not_a_number",
        ]
        for px in invalid_prices:
            tracker = ShadowCapitalLedgerTracker({'embargo_close_price': 100.0, 'configured_quantity': 1.0})
            self.assertTrue(tracker.valid)
            # Must catch before unsafe Decimal comparisons
            tracker.record_bar_close(px)
            self.assertFalse(tracker.valid, f"Failed to invalidate on close_price={px}")
            self.assertEqual(tracker.invalid_reason, 'INVALID_NONPOSITIVE_OR_NONFINITE_EQUITY')

    def test_short_bar_sample_produces_insufficient_trades(self):
        """Fewer observations than block length returns INSUFFICIENT_TRADES with BAR_SAMPLE_TOO_SHORT"""
        short_folds = [
            [Decimal("0.01"), Decimal("-0.01"), Decimal("0.02")],
        ]
        artifact, full_hash, q05 = run_g6_shadow_bootstrap_v2(short_folds, block_length=10)
        self.assertEqual(artifact['g6_status'], 'INSUFFICIENT_TRADES')
        self.assertEqual(artifact['g6_reason'], 'BAR_SAMPLE_TOO_SHORT')
        self.assertIsNone(full_hash)
        self.assertIsNone(q05)

    # --- SECTION D: Walk-Forward Integration & Routing ---

    def test_walk_forward_enabled_integration_and_side_by_side(self):
        """Integration test: enable_shadow_g6=True preserves fills, trades, and gates vs disabled"""
        dates = pd.date_range("2026-01-01", periods=160, freq="4h", tz="UTC")
        df = pd.DataFrame({
            'open': [100.0] * 160,
            'high': [105.0] * 160,
            'low': [95.0] * 160,
            'close': [100.0] * 160,
            'signal': [0] * 160,
            'atr': [1.0] * 160,
        }, index=dates)

        # Put positive trades every 8 bars in fold 1 (70..110) and fold 2 (120..160)
        for start in [71, 79, 87, 95, 121, 129, 137, 145]:
            df.loc[df.index[start], 'signal'] = 1
            df.loc[df.index[start+1], 'open'] = 100.0
            df.loc[df.index[start+4], 'signal'] = -2
            df.loc[df.index[start+5], 'open'] = 105.0
            df.loc[df.index[start+5], 'close'] = 105.0

        def dummy_strategy(data, p=1):
            return data

        res_enabled = walk_forward_test(
            dummy_strategy, {'p': [1]}, df,
            train_window=50, purge_bars=10, embargo_bars=10, test_window=40, step=50,
            enable_shadow_g6=True, fee_rate=0.0015
        )
        res_disabled = walk_forward_test(
            dummy_strategy, {'p': [1]}, df,
            train_window=50, purge_bars=10, embargo_bars=10, test_window=40, step=50,
            enable_shadow_g6=False, fee_rate=0.0015
        )

        # Fills, quantities, fees, completed trades must remain completely unchanged
        self.assertEqual(len(res_enabled['trade_records']), len(res_disabled['trade_records']))
        for t_en, t_dis in zip(res_enabled['trade_records'], res_disabled['trade_records']):
            self.assertEqual(t_en['entry_price'], t_dis['entry_price'])
            self.assertEqual(t_en['exit_price'], t_dis['exit_price'])
            self.assertEqual(t_en['quantity'], t_dis['quantity'])
            self.assertEqual(t_en['entry_fee'], t_dis['entry_fee'])
            self.assertEqual(t_en['exit_fee'], t_dis['exit_fee'])
            self.assertEqual(t_en['net_return'], t_dis['net_return'])

        # Pre-existing gate statuses must remain unchanged
        pg_en = res_enabled['portfolio_gates']
        pg_dis = res_disabled['portfolio_gates']
        self.assertEqual(pg_en['G1_G3_MEAN_RETURN'], pg_dis['G1_G3_MEAN_RETURN'])
        self.assertEqual(pg_en['G4'], pg_dis['G4'])
        self.assertEqual(pg_en['G5'], pg_dis['G5'])
        self.assertEqual(pg_en['G5b'], pg_dis['G5b'])

        # Disabled preserves legacy state
        self.assertEqual(pg_dis['G6'], 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER')
        self.assertIn('G6', pg_dis['pending_gates'])
        self.assertIsNone(pg_dis['bootstrap'])

        # Enabled routes computational pass to PENDING and overall_pass is False
        self.assertEqual(pg_en['G6'], 'COMPUTATIONAL_PASS_PENDING_STATISTICAL_AUDIT')
        self.assertIn('G6', pg_en['pending_gates'])
        self.assertFalse(pg_en['overall_pass'])
        self.assertIsNotNone(pg_en['bootstrap'])
        self.assertIsNotNone(pg_en['full_bootstrap_artifact_hash'])
        self.assertGreater(pg_en['g6_q05'], Decimal("0"))

        # Verify causal embargo metadata was populated in the canonical artifact
        canonical = pg_en['bootstrap']['full_statistics_artifact']
        self.assertEqual(len(canonical['folds']), 2)
        self.assertEqual(canonical['folds'][0]['bar_count'], 40)
        self.assertEqual(canonical['folds'][1]['bar_count'], 40)
        self.assertEqual(canonical['folds'][0]['embargo_close_price'], "100.0")
        self.assertEqual(canonical['folds'][1]['embargo_close_price'], "100.0")

    def test_walk_forward_enabled_insufficient_bars(self):
        """Walk-forward test with test window < 10 bars yields INSUFFICIENT_TRADES"""
        dates = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC")
        df = pd.DataFrame({
            'open': [100.0] * 100, 'high': [105.0] * 100, 'low': [95.0] * 100,
            'close': [100.0] * 100, 'signal': [0] * 100, 'atr': [1.0] * 100,
        }, index=dates)

        res = walk_forward_test(
            lambda d, p=1: d, {'p': [1]}, df,
            train_window=30, purge_bars=5, embargo_bars=5, test_window=8, step=30,
            enable_shadow_g6=True, g6_block_length=10
        )
        pg = res['portfolio_gates']
        self.assertEqual(pg['G6'], 'INSUFFICIENT_TRADES')
        self.assertIn('G6', pg['pending_gates'])
        self.assertFalse(pg['overall_pass'])

    def test_walk_forward_enabled_invalid_close_price_blocks_promotion(self):
        """Walk-forward test encountering nonpositive/nonfinite close marks INVALID and fails"""
        dates = pd.date_range("2026-01-01", periods=100, freq="4h", tz="UTC")
        df = pd.DataFrame({
            'open': [100.0] * 100, 'high': [105.0] * 100, 'low': [95.0] * 100,
            'close': [100.0] * 100, 'signal': [0] * 100, 'atr': [1.0] * 100,
        }, index=dates)
        # Corrupt a bar in the test window
        df.loc[df.index[45], 'close'] = float('nan')

        res = walk_forward_test(
            lambda d, p=1: d, {'p': [1]}, df,
            train_window=30, purge_bars=5, embargo_bars=5, test_window=20, step=30,
            enable_shadow_g6=True
        )
        pg = res['portfolio_gates']
        self.assertEqual(pg['G6'], 'INVALID_NONPOSITIVE_OR_NONFINITE_EQUITY')
        self.assertIn('G6', pg['failed_gates'])
        self.assertFalse(pg['overall_pass'])

    def test_gate_routing_computational_pass_held_in_pending(self):
        """COMPUTATIONAL_PASS routes into pending_gates and overall_pass remains False"""
        status_map = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'PASS',
            'G6': 'COMPUTATIONAL_PASS_PENDING_STATISTICAL_AUDIT',
        }
        overall_pass, failed_gates, pending_gates = aggregate_gate_statuses(status_map)
        self.assertFalse(overall_pass)
        self.assertEqual(failed_gates, [])
        self.assertIn('G6', pending_gates)

    def test_gate_routing_computational_fail_in_failed_gates(self):
        """COMPUTATIONAL_FAIL routes into failed_gates and overall_pass remains False"""
        status_map = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'PASS',
            'G6': 'COMPUTATIONAL_FAIL_NONPOSITIVE_LOWER_BOUND',
        }
        overall_pass, failed_gates, pending_gates = aggregate_gate_statuses(status_map)
        self.assertFalse(overall_pass)
        self.assertIn('G6', failed_gates)
        self.assertEqual(pending_gates, [])

    def test_gate_routing_invalid_in_failed_gates(self):
        """INVALID_* statuses route into failed_gates"""
        status_map = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'PASS',
            'G6': 'INVALID_SHADOW_CASH_DEPLETED',
        }
        overall_pass, failed_gates, pending_gates = aggregate_gate_statuses(status_map)
        self.assertFalse(overall_pass)
        self.assertIn('G6', failed_gates)

    def test_default_disabled_execution_compatibility(self):
        """Default walk_forward_test without enable_shadow_g6 preserves legacy NOT_EVALUATED_PENDING_LEGACY_ADAPTER"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            return df
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=100, test_window=100, step=100, purge_bars=0, embargo_bars=0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['G6'], 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER')
        self.assertIn('G6', pg['pending_gates'])
        self.assertIsNone(pg['bootstrap'])


if __name__ == '__main__':
    unittest.main()
