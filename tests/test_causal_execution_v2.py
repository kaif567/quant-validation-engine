import unittest
import pandas as pd
import numpy as np
from decimal import Decimal
from validation_engine.simulator import simulate_trades


class TestCausalExecutionV2(unittest.TestCase):
    def setUp(self):
        # Base DataFrame fixture helper
        dates = pd.date_range('2024-01-01', periods=5, freq='1h')
        self.df_base = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'high': [100.0, 100.0, 100.0, 100.0, 100.0],
            'low': [100.0, 100.0, 100.0, 100.0, 100.0],
            'close': [100.0, 100.0, 100.0, 100.0, 100.0],
            'volume': [10.0, 10.0, 10.0, 10.0, 10.0],
            'signal': [0, 0, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0, 1.0]
        }, index=dates)

    def test_invalid_execution_model_rejected(self):
        """Reject unknown execution model identifiers."""
        with self.assertRaises(ValueError) as ctx:
            simulate_trades(self.df_base, execution_model='v3_future')
        self.assertIn("Unknown execution_model: 'v3_future'", str(ctx.exception))

    def test_v1_legacy_default_preserved(self):
        """Default execution_model is v1_legacy and matches explicit call."""
        df = self.df_base.copy()
        df.loc[df.index[0], 'signal'] = 1
        df.loc[df.index[2], 'signal'] = -2

        trades_default, _, _ = simulate_trades(df)
        trades_v1, _, _ = simulate_trades(df, execution_model='v1_legacy')

        self.assertEqual(len(trades_default), len(trades_v1))
        self.assertEqual(trades_default[0]['entry_price'], trades_v1[0]['entry_price'])
        self.assertEqual(trades_default[0]['exit_price'], trades_v1[0]['exit_price'])
        self.assertEqual(trades_default[0]['exit_reason'], trades_v1[0]['exit_reason'])

    def test_pending_exit_vs_same_bar_stop(self):
        """
        Pending open exit must execute at Open and cannot be overridden
        by later same-bar intrabar extremes.
        """
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 90.0, 80.0],
            'high': [100.0, 100.0, 90.0, 80.0],
            'low': [100.0, 100.0, 70.0, 80.0],    # Bar 2 low drops to 70.0 (below stop 75.0)
            'close': [100.0, 100.0, 75.0, 80.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, -2, 0, 0],              # Bar 0 buys at Bar 1 open; Bar 1 emits force-close for Bar 2 open
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        # Under v1_legacy: Bar 2 low (70.0 <= 75.0) triggers STOP_LOSS before checking signal=-2
        trades_v1, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v1_legacy'
        )
        self.assertEqual(len(trades_v1), 1)
        self.assertEqual(trades_v1[0]['exit_reason'], 'STOP_LOSS')
        self.assertAlmostEqual(trades_v1[0]['exit_price'], 75.0, places=7)

        # Under v2_causal: Pending market exit executes at Bar 2 Open=90.0 before intrabar low occurs
        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        self.assertEqual(trades_v2[0]['exit_reason'], 'SIGNAL_FORCE_CLOSE')
        self.assertAlmostEqual(trades_v2[0]['exit_price'], 90.0, places=7)

    def test_entry_bar_stop(self):
        """Entry-bar intrabar stop-loss must be evaluated on the bar of entry."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 80.0, 80.0],
            'high': [100.0, 101.0, 85.0, 80.0],
            'low': [100.0, 70.0, 78.0, 80.0],     # Bar 1 drops to 70.0 (below stop 75.0) on the entry bar
            'close': [100.0, 80.0, 82.0, 80.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],               # Signal on Bar 0 -> enters at Bar 1 open
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        # Under v1_legacy: Bar 1 entry skips intrabar checks on Bar 1; exits at boundary close
        trades_v1, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v1_legacy'
        )
        self.assertEqual(len(trades_v1), 1)
        self.assertEqual(trades_v1[0]['exit_reason'], 'BOUNDARY_CLOSE')

        # Under v2_causal: Intrabar stop-loss triggers on Bar 1 at 75.0
        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        self.assertEqual(trades_v2[0]['exit_reason'], 'STOP_LOSS')
        self.assertAlmostEqual(trades_v2[0]['exit_price'], 75.0, places=7)

    def test_entry_bar_take_profit(self):
        """Entry-bar intrabar take-profit must be evaluated on the bar of entry."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 105.0, 105.0],
            'high': [100.0, 115.0, 106.0, 105.0],  # Bar 1 reaches 115.0 (above TP 110.0)
            'low': [100.0, 99.0, 104.0, 105.0],
            'close': [100.0, 105.0, 105.0, 105.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.10,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        self.assertEqual(trades_v2[0]['exit_reason'], 'TAKE_PROFIT')
        self.assertAlmostEqual(trades_v2[0]['exit_price'], 110.0, places=7)

    def test_both_sl_tp_touched_stop_first(self):
        """When both SL and TP are touched on same bar, Stop-Loss takes priority."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 100.0, 100.0],
            'high': [100.0, 100.0, 115.0, 100.0],  # Bar 2: High 115 >= 110 (TP), Low 70 <= 75 (SL)
            'low': [100.0, 100.0, 70.0, 100.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.10,
            mode='pessimistic',
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        self.assertEqual(trades_v2[0]['exit_reason'], 'SL_TOUCH_BOTH')
        self.assertAlmostEqual(trades_v2[0]['exit_price'], 75.0, places=7)

    def test_gap_through_stop(self):
        """Long gap-through-stop must fill at the executable Open price."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 60.0, 60.0],   # Bar 2 opens at 60.0 (gapped below stop 75.0)
            'high': [100.0, 100.0, 65.0, 60.0],
            'low': [100.0, 100.0, 55.0, 60.0],
            'close': [100.0, 100.0, 60.0, 60.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        self.assertEqual(trades_v2[0]['exit_reason'], 'STOP_LOSS_GAP')
        self.assertAlmostEqual(trades_v2[0]['exit_price'], 60.0, places=7)

    def test_target_gap_behavior(self):
        """Gap above limit target must fill at limit price, not unrequested price."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 120.0, 120.0],  # Bar 2 opens at 120.0 (gapped above TP 110.0)
            'high': [100.0, 100.0, 125.0, 120.0],
            'low': [100.0, 100.0, 118.0, 120.0],
            'close': [100.0, 100.0, 122.0, 120.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.10,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        self.assertEqual(trades_v2[0]['exit_reason'], 'TAKE_PROFIT_GAP')
        self.assertAlmostEqual(trades_v2[0]['exit_price'], 110.0, places=7)

    def test_boundary_closure_and_fee_count(self):
        """Unclosed position at fold end must close at final close with exactly two fees."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 102.0, 103.0],
            'high': [100.0, 101.0, 103.0, 105.0],
            'low': [100.0, 99.0, 101.0, 102.0],
            'close': [100.0, 102.0, 103.0, 105.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_v2, _, _ = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_v2), 1)
        t = trades_v2[0]
        self.assertEqual(t['exit_reason'], 'BOUNDARY_CLOSE')
        self.assertAlmostEqual(t['exit_price'], 105.0, places=7)
        self.assertAlmostEqual(t['entry_fee'], 100.0 * 1.0 * 0.0015, places=7)
        self.assertAlmostEqual(t['exit_fee'], 105.0 * 1.0 * 0.0015, places=7)

    def test_shadow_reconciliation_v2(self):
        """Shadow tracker operates synchronously with v2_causal execution."""
        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 105.0, 105.0],
            'high': [100.0, 115.0, 106.0, 105.0],
            'low': [100.0, 99.0, 104.0, 105.0],
            'close': [100.0, 105.0, 105.0, 105.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        shadow_cfg = {
            'embargo_close_price': 100.0,
            'embargo_close_timestamp': '2023-12-31 23:00:00',
            'configured_quantity': 1.0,
            'fee_coverage_margin': Decimal("0.005"),
            'fee_rate': 0.0015,
            'entry_slippage': Decimal("0"),
            'stop_slippage': Decimal("0")
        }

        trades, _, _, report = simulate_trades(
            df,
            stop_loss=0.25,
            take_profit=0.10,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            track_shadow_ledger=True,
            shadow_ledger_config=shadow_cfg,
            execution_model='v2_causal'
        )
        self.assertIsNotNone(report)
        self.assertTrue(report['valid'])
        self.assertEqual(report['trade_count'], 1)
        self.assertEqual(len(report['bar_returns']), 4)

    def test_fee_inclusive_target_roi_baseline_and_shocked(self):
        """
        Verify source-inspired fee-inclusive net ROI target formula:
        tp_fraction = (1 + target_roi) * (1 + f_entry) / (1 - f_exit) - 1
        Realized net ROI on invested capital must equal target_roi under both
        baseline fees (0.0015) and shocked fees (0.0030).
        """
        target_roi = Decimal("0.10")

        # Case 1: Baseline fees (0.0015 each side)
        f_entry = Decimal("0.0015")
        f_exit = Decimal("0.0015")
        tp_fraction_base = float(
            (Decimal("1") + target_roi) * (Decimal("1") + f_entry) / (Decimal("1") - f_exit) - Decimal("1")
        )
        # Expected fraction is ~0.1033049574
        self.assertAlmostEqual(tp_fraction_base, 0.1033049574, places=6)

        entry_price = 100.0
        expected_target_price = entry_price * (1.0 + tp_fraction_base)

        dates = pd.date_range('2024-01-01', periods=4, freq='1h')
        df_base = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 105.0, 105.0],
            'high': [100.0, 101.0, expected_target_price + 1.0, 105.0],  # Bar 2 reaches target
            'low': [100.0, 99.0, 100.0, 105.0],
            'close': [100.0, 100.0, expected_target_price, 105.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_base, _, _ = simulate_trades(
            df_base,
            stop_loss=0.25,
            take_profit=tp_fraction_base,
            fee_rate=float(f_entry),
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_base), 1)
        tb = trades_base[0]
        self.assertEqual(tb['exit_reason'], 'TAKE_PROFIT')
        self.assertAlmostEqual(tb['exit_price'], expected_target_price, places=7)

        # Realized Net ROI on invested capital = [exit_price*(1-f_exit)] / [entry_price*(1+f_entry)] - 1
        invested_cap_base = Decimal(str(tb['entry_price'])) * (Decimal("1") + f_entry)
        exit_proceeds_base = Decimal(str(tb['exit_price'])) * (Decimal("1") - f_exit)
        realized_roi_base = (exit_proceeds_base / invested_cap_base) - Decimal("1")
        self.assertAlmostEqual(float(realized_roi_base), 0.10, places=6)

        # Case 2: Shocked fees (0.0030 each side)
        f_entry_shock = Decimal("0.0030")
        f_exit_shock = Decimal("0.0030")
        tp_fraction_shock = float(
            (Decimal("1") + target_roi) * (Decimal("1") + f_entry_shock) / (Decimal("1") - f_exit_shock) - Decimal("1")
        )
        # Expected shocked fraction is ~0.1066198596
        self.assertAlmostEqual(tp_fraction_shock, 0.1066198596, places=6)
        expected_target_price_shock = entry_price * (1.0 + tp_fraction_shock)

        df_shock = pd.DataFrame({
            'open_time': dates,
            'open': [100.0, 100.0, 105.0, 105.0],
            'high': [100.0, 101.0, expected_target_price_shock + 1.0, 105.0],
            'low': [100.0, 99.0, 100.0, 105.0],
            'close': [100.0, 100.0, expected_target_price_shock, 105.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=dates)

        trades_shock, _, _ = simulate_trades(
            df_shock,
            stop_loss=0.25,
            take_profit=tp_fraction_shock,
            fee_rate=float(f_entry_shock),
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_shock), 1)
        ts = trades_shock[0]
        self.assertEqual(ts['exit_reason'], 'TAKE_PROFIT')
        self.assertAlmostEqual(ts['exit_price'], expected_target_price_shock, places=7)

        invested_cap_shock = Decimal(str(ts['entry_price'])) * (Decimal("1") + f_entry_shock)
        exit_proceeds_shock = Decimal(str(ts['exit_price'])) * (Decimal("1") - f_exit_shock)
        realized_roi_shock = (exit_proceeds_shock / invested_cap_shock) - Decimal("1")
        self.assertAlmostEqual(float(realized_roi_shock), 0.10, places=6)

    def test_target_roi_forwarded_in_walk_forward_g4(self):
        """
        Verify walk_forward_test calculates baseline take_profit from target_roi
        and recalculates shocked take_profit dynamically under G4 cost shock.
        """
        from validation_engine.walk_forward import walk_forward_test

        def dummy_strategy(df_in, **params):
            df_out = df_in.copy()
            df_out['signal'] = 0
            # Emit buy signal at bar 10
            if len(df_out) > 15:
                df_out.iloc[10, df_out.columns.get_loc('signal')] = 1
            return df_out

        # Create a small dataset with enough bars for 1 fold
        # train=200, test=100 -> total 300
        n_bars = 350
        dates = pd.date_range('2024-01-01', periods=n_bars, freq='1h')
        prices = [100.0] * n_bars
        # In test window (bars 200..300), let price spike to reach target
        for b in range(215, 230):
            prices[b] = 115.0

        df_wf = pd.DataFrame({
            'open_time': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [10.0] * n_bars,
            'atr': [1.0] * n_bars
        }, index=dates)

        wf_res = walk_forward_test(
            strategy_function=dummy_strategy,
            param_grid={'target_roi': [0.10], 'stop_loss': [0.25]},
            df=df_wf,
            train_window=200,
            test_window=100,
            step=100,
            fee_rate=0.0015,
            execution_model='v2_causal'
        )
        self.assertIsNotNone(wf_res)
        self.assertIn('folds', wf_res)
        self.assertGreaterEqual(len(wf_res['folds']), 1)

    def test_gap_pending_order_timing_and_open_fill(self):
        """
        When consecutive available bars have an exchange outage gap (e.g. 12:00 to 14:00 UTC),
        pending orders execute at the first tradable Open after the gap (14:00:00),
        NOT during the halted period.
        """
        # Timestamps mimicking Binance 2023-03-24 outage: 12:00 then 14:00 (13:00 missing)
        timestamps = [
            pd.Timestamp('2023-03-24 11:00:00+00:00'),
            pd.Timestamp('2023-03-24 12:00:00+00:00'),
            pd.Timestamp('2023-03-24 14:00:00+00:00'),  # 13:00 missing
            pd.Timestamp('2023-03-24 15:00:00+00:00'),
        ]
        df_gap = pd.DataFrame({
            'open_time': timestamps,
            'open': [28039.71, 28080.00, 28079.99, 27989.07],
            'high': [28091.03, 28080.00, 28253.01, 28076.82],
            'low': [27963.84, 28080.00, 27835.00, 27843.41],
            'close': [28080.00, 28080.00, 27989.06, 28018.04],
            'volume': [1267.4, 0.0, 8983.2, 5198.2],
            'signal': [0, 1, 0, 0],   # Signal at 12:00 close -> pending entry for next open
            'atr': [50.0, 50.0, 50.0, 50.0]
        }, index=timestamps)

        trades_gap, _, _ = simulate_trades(
            df_gap,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_gap), 1)
        tg = trades_gap[0]
        # Must enter at Bar 2 Open (14:00 UTC), not 13:00
        self.assertEqual(str(tg['entry_time']), '2023-03-24 14:00:00+00:00')
        self.assertAlmostEqual(tg['entry_price'], 28079.99, places=2)

    def test_gap_through_stop_across_missing_bars(self):
        """
        When a position is open going into an outage gap, and the market re-opens
        below the stop-loss level, it must fill at the first available Open (STOP_LOSS_GAP),
        not at the unexecutable stop-loss price.
        """
        timestamps = [
            pd.Timestamp('2023-03-24 11:00:00+00:00'),
            pd.Timestamp('2023-03-24 12:00:00+00:00'),
            pd.Timestamp('2023-03-24 14:00:00+00:00'),  # 13:00 missing
            pd.Timestamp('2023-03-24 15:00:00+00:00'),
        ]
        # Enters at 12:00 open (100.0). SL = 25% (75.0).
        # At 14:00, market gaps open to 60.0 (below 75.0).
        df_gap = pd.DataFrame({
            'open_time': timestamps,
            'open': [100.0, 100.0, 60.0, 60.0],
            'high': [100.0, 100.0, 65.0, 60.0],
            'low': [100.0, 95.0, 55.0, 60.0],
            'close': [100.0, 95.0, 60.0, 60.0],
            'volume': [10.0, 10.0, 10.0, 10.0],
            'signal': [1, 0, 0, 0],   # Signal at 11:00 -> entry at 12:00
            'atr': [1.0, 1.0, 1.0, 1.0]
        }, index=timestamps)

        trades_gap, _, _ = simulate_trades(
            df_gap,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            execution_model='v2_causal'
        )
        self.assertEqual(len(trades_gap), 1)
        tg = trades_gap[0]
        self.assertEqual(tg['exit_reason'], 'STOP_LOSS_GAP')
        self.assertEqual(str(tg['exit_time']), '2023-03-24 14:00:00+00:00')
        self.assertAlmostEqual(tg['exit_price'], 60.0, places=7)

    def test_gap_shadow_returns_spans_elapsed_hours(self):
        """
        Shadow ledger does not create synthetic bars for missing hours.
        Return on the post-gap bar directly spans the multi-hour elapsed price transition.
        """
        timestamps = [
            pd.Timestamp('2023-03-24 12:00:00+00:00'),
            pd.Timestamp('2023-03-24 14:00:00+00:00'),  # 13:00 missing
        ]
        df_gap = pd.DataFrame({
            'open_time': timestamps,
            'open': [100.0, 110.0],
            'high': [100.0, 112.0],
            'low': [100.0, 108.0],
            'close': [100.0, 110.0],
            'volume': [10.0, 10.0],
            'signal': [0, 0],
            'atr': [1.0, 1.0]
        }, index=timestamps)

        shadow_cfg = {
            'embargo_close_price': 100.0,
            'embargo_close_timestamp': '2023-03-24 11:00:00+00:00',
            'configured_quantity': 1.0,
            'fee_coverage_margin': Decimal("0.005"),
            'fee_rate': 0.0015,
            'entry_slippage': Decimal("0"),
            'stop_slippage': Decimal("0")
        }

        _, _, _, report = simulate_trades(
            df_gap,
            stop_loss=0.25,
            take_profit=0.50,
            fee_rate=0.0015,
            entry_slippage=0.0,
            stop_slippage=0.0,
            track_shadow_ledger=True,
            shadow_ledger_config=shadow_cfg,
            execution_model='v2_causal'
        )
        self.assertIsNotNone(report)
        self.assertTrue(report['valid'])
        # Exactly 2 bars recorded (no synthetic 13:00 bar)
        self.assertEqual(len(report['bar_returns']), 2)
        self.assertEqual(len(report['bar_timestamps']), 2)
        self.assertIn('2023-03-24 12:00:00', report['bar_timestamps'][0])
        self.assertIn('2023-03-24 14:00:00', report['bar_timestamps'][1])
        # Cash-only (no open position), so returns are 0.0
        self.assertEqual(report['bar_returns'][0], Decimal("0.0"))
        self.assertEqual(report['bar_returns'][1], Decimal("0.0"))

if __name__ == '__main__':
    unittest.main()
