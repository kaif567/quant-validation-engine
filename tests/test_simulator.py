import unittest
import pandas as pd
from validation_engine.simulator import simulate_trades

class TestSimulatorFillPhysics(unittest.TestCase):
    def get_test_df(self, data):
        df = pd.DataFrame(data)
        return df

    def test_a_signal_i_no_fill_on_i(self):
        """signal on i does not fill on i"""
        df = self.get_test_df([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 110, 'high': 110, 'low': 110, 'close': 110, 'signal': 0},
            {'open': 120, 'high': 120, 'low': 120, 'close': 120, 'signal': -1},
            {'open': 130, 'high': 130, 'low': 130, 'close': 130, 'signal': 0},
        ])
        trades, _, _ = simulate_trades(df, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        self.assertGreaterEqual(len(trades), 1)
        entry = trades[0]['entry_fill']
        self.assertEqual(entry['bar_index'], 1)  # bar 1, not bar 0
        self.assertAlmostEqual(entry['fill_price'], 110.0)
        
    def test_b_open_gaps_past_sl(self):
        """Open gaps past SL -> fill at Open"""
        df = self.get_test_df([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 0},
            {'open': 80, 'high': 100, 'low': 70, 'close': 90, 'signal': 0}, 
        ])
        trades, _, _ = simulate_trades(df, stop_loss=0.1, take_profit=0.5, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        exit = trades[0]['exit_fill']
        self.assertEqual(exit['reason'], 'STOP_LOSS_GAP')
        self.assertTrue(exit['gap_fill'])
        self.assertAlmostEqual(exit['fill_price'], 80.0)

    def test_c_both_sides_touched_base_vs_pessimistic(self):
        """both sides touched, Open much closer to TP -> base fills TP; pessimistic SL"""
        df = self.get_test_df([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 0},
            {'open': 108, 'high': 111, 'low': 89, 'close': 100, 'signal': 0},
        ])
        
        trades_base, _, _ = simulate_trades(df, mode='base', stop_loss=0.1, take_profit=0.1, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        self.assertEqual(trades_base[0]['exit_fill']['reason'], 'TP_TOUCH_BOTH')
        self.assertAlmostEqual(trades_base[0]['exit_fill']['fill_price'], 110.0)
        
        trades_pess, _, _ = simulate_trades(df, mode='pessimistic', stop_loss=0.1, take_profit=0.1, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        self.assertEqual(trades_pess[0]['exit_fill']['reason'], 'SL_TOUCH_BOTH')
        self.assertAlmostEqual(trades_pess[0]['exit_fill']['fill_price'], 90.0)

    def test_d_high_equals_tp_no_fill(self):
        """High == TP exactly, High never > TP -> TP does NOT fill"""
        df = self.get_test_df([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 0},
            {'open': 100, 'high': 110, 'low': 95, 'close': 100, 'signal': 0},
            {'open': 100, 'high': 120, 'low': 95, 'close': 100, 'signal': 0},
        ])
        trades, _, _ = simulate_trades(df, stop_loss=0.2, take_profit=0.1, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        exit = trades[0]['exit_fill']
        self.assertEqual(exit['bar_index'], 3)
        self.assertAlmostEqual(exit['fill_price'], 110.0)
        
    def test_e_high_greater_tp_fills(self):
        """High > TP -> TP fills at TP"""
        df = self.get_test_df([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 0},
            {'open': 100, 'high': 115, 'low': 95, 'close': 100, 'signal': 0},
        ])
        trades, _, _ = simulate_trades(df, stop_loss=0.2, take_profit=0.1, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        exit = trades[0]['exit_fill']
        self.assertEqual(exit['bar_index'], 2)
        self.assertAlmostEqual(exit['fill_price'], 110.0)
        self.assertEqual(exit['order_type'], 'LIMIT')

    def test_f_deterministic(self):
        """same bars + same config -> identical fills"""
        df = self.get_test_df([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 0},
            {'open': 105, 'high': 115, 'low': 80, 'close': 100, 'signal': 0},
        ])
        t1, _, _ = simulate_trades(df, mode='base', stop_loss=0.1, take_profit=0.1, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        t2, _, _ = simulate_trades(df, mode='base', stop_loss=0.1, take_profit=0.1, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.0)
        self.assertEqual(t1[0]['entry_fill']['fill_price'], t2[0]['entry_fill']['fill_price'])


    def test_atr_validation_zero_slippage_no_atr(self):
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({'signal': [1, 0], 'close': [100.0, 105.0], 'open': [100.0, 105.0], 'high': [100.0, 105.0], 'low': [100.0, 105.0]})
        trades, _, _ = simulate_trades(df, entry_slippage=0.0, stop_slippage=0.0)
        self.assertEqual(len(trades), 1)

    def test_atr_validation_nonzero_entry_no_atr(self):
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({'signal': [1, 0], 'close': [100.0, 105.0], 'open': [100.0, 105.0], 'high': [100.0, 105.0], 'low': [100.0, 105.0]})
        with self.assertRaisesRegex(ValueError, "ATR column required when entry_slippage or stop_slippage is non-zero"):
            simulate_trades(df, entry_slippage=0.001, stop_slippage=0.0)

    def test_atr_validation_nonzero_stop_no_atr(self):
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({'signal': [1, 0], 'close': [100.0, 105.0], 'open': [100.0, 105.0], 'high': [100.0, 105.0], 'low': [100.0, 105.0]})
        with self.assertRaisesRegex(ValueError, "ATR column required when entry_slippage or stop_slippage is non-zero"):
            simulate_trades(df, entry_slippage=0.0, stop_slippage=0.001)

    def test_atr_validation_nonzero_slippage_with_atr(self):
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({'signal': [1, 0], 'close': [100.0, 105.0], 'open': [100.0, 105.0], 'high': [100.0, 105.0], 'low': [100.0, 105.0], 'atr': [1.0, 1.0]})
        trades, _, _ = simulate_trades(df, entry_slippage=0.001, stop_slippage=0.001)
        self.assertEqual(len(trades), 1)


    def test_g4_legacy_decision_trace_accepted(self):
        """Legacy G4 Stage A/B: accepted order produces correct Decimal trace dict and mutates state"""
        from decimal import Decimal
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({
            'open': [100.0, 100.0],
            'high': [100.0, 100.0],
            'low': [100.0, 100.0],
            'close': [100.0, 100.0],
            'signal': [1, 0],
        })
        trades, rejected, traces = simulate_trades(df, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.001)
        accepted = [t for t in traces if t['status'] == 'ACCEPTED']
        self.assertGreater(len(accepted), 0)
        tr = accepted[0]
        self.assertEqual(tr['status'], 'ACCEPTED')
        self.assertIsNone(tr['rejection_reason'])
        self.assertIsInstance(tr['approved_qty'], Decimal)
        self.assertIsInstance(tr['approved_notional'], Decimal)
        self.assertIsInstance(tr['entry_fee'], Decimal)
        self.assertIsInstance(tr['exit_fee_reserve'], Decimal)
        self.assertIsInstance(tr['free_cash_before'], Decimal)
        self.assertIsInstance(tr['free_cash_after'], Decimal)
        self.assertIsInstance(tr['gross_exposure_before'], Decimal)
        self.assertIsInstance(tr['gross_exposure_after'], Decimal)
        self.assertGreater(tr['approved_qty'], Decimal("0"))
        self.assertGreater(tr['approved_notional'], Decimal("0"))
        self.assertGreater(tr['entry_fee'], Decimal("0"))
        self.assertGreater(tr['exit_fee_reserve'], Decimal("0"))
        self.assertEqual(tr['free_cash_after'], tr['free_cash_before'] - tr['approved_notional'] - tr['entry_fee'])
        self.assertEqual(tr['gross_exposure_after'], tr['gross_exposure_before'] + tr['approved_notional'])
        self.assertEqual(tr['position_count_after'], tr['position_count_before'] + 1)

    def test_g4_legacy_decision_trace_rejected_invariants(self):
        """Legacy G4 Stage A: rejected trace has zero approved_qty/fee/notional, no state mutation"""
        from decimal import Decimal
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({
            'open': [100.0, 100.0, 100.0],
            'high': [100.0, 100.0, 100.0],
            'low': [100.0, 100.0, 100.0],
            'close': [100.0, 100.0, 100.0],
            'volume': [1.0] * 3,
            'quote_volume': [1.0] * 3,
            'signal': [1, 0, 0],
        })
        liquidity_config = {
            'liquidity_tiering_enabled': False,
            'capacity_constraint_enabled': True,
            'capacity': {'participation_rate_cap': 0.00001, 'action': 'REJECT'},
        }
        trades, rejected, traces = simulate_trades(
            df, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.001,
            liquidity_config=liquidity_config, intended_quantity=10000.0,
        )
        rejected_traces = [t for t in traces if t['status'] == 'REJECTED']
        self.assertGreater(len(rejected_traces), 0)
        tr = rejected_traces[0]
        self.assertEqual(tr['status'], 'REJECTED')
        self.assertIsNotNone(tr['rejection_reason'])
        self.assertEqual(tr['approved_qty'], Decimal("0"))
        self.assertEqual(tr['approved_notional'], Decimal("0"))
        self.assertEqual(tr['entry_fee'], Decimal("0"))
        self.assertEqual(tr['exit_fee_reserve'], Decimal("0"))
        self.assertEqual(tr['free_cash_after'], tr['free_cash_before'])
        self.assertEqual(tr['gross_exposure_after'], tr['gross_exposure_before'])
        self.assertEqual(tr['position_count_after'], tr['position_count_before'])

    def test_g4_real_run_order1_accepted_order2_rejected_binding_constraint(self):
        """Fixture: a real run where order 1 is ACCEPTED and order 2 is REJECTED by binding constraint"""
        from decimal import Decimal
        from validation_engine.simulator import simulate_trades
        df = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000.0] * 50,
            'quote_volume': [1000.0] * 50,
            'signal': [0] * 50,
        })
        # Order 1 signal at bar 24 (after 24-hour volume lookback)
        df.loc[24, 'signal'] = 1
        # Bar 26 high hits take profit -> position closes cleanly
        df.loc[26, 'high'] = 120.0

        # Quote volume drops to 1.0 after bar 24 so 24h rolling volume at bar 48 drops to 24.0
        for b in range(25, 50):
            df.loc[b, 'quote_volume'] = 1.0

        # Order 2 signal at bar 48 (requested notional = 200.0, but cap = 0.05 * 24.0 = 1.2)
        df.loc[48, 'signal'] = 1

        liquidity_config = {
            'liquidity_tiering_enabled': False,
            'capacity_constraint_enabled': True,
            'capacity': {'participation_rate_cap': 0.05, 'action': 'REJECT'},
        }
        trades, rejected, traces = simulate_trades(
            df, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.001,
            take_profit=0.05, stop_loss=0.05,
            liquidity_config=liquidity_config, intended_quantity=2.0,
            initial_cash=50000.0,
        )

        print("\n--- ACTIVE G4 REAL RUN DECISION TRACES ---")
        for tr in traces:
            print(f"attempt_id={tr['attempt_id']} status={tr['status']} approved_qty={tr['approved_qty']} "
                  f"approved_notional={tr['approved_notional']} free_cash_before={tr['free_cash_before']} "
                  f"free_cash_after={tr['free_cash_after']} gross_exposure_after={tr['gross_exposure_after']} "
                  f"rejection_reason={tr['rejection_reason']}")

        self.assertEqual(len(traces), 2)
        tr1 = traces[0]
        tr2 = traces[1]

        # Order 1: ACCEPTED
        self.assertEqual(tr1['status'], 'ACCEPTED')
        self.assertIsNone(tr1['rejection_reason'])
        self.assertEqual(tr1['approved_qty'], Decimal("2.0"))
        self.assertEqual(tr1['approved_notional'], Decimal("200.0"))
        self.assertGreater(tr1['entry_fee'], Decimal("0"))
        self.assertEqual(tr1['free_cash_after'], tr1['free_cash_before'] - tr1['approved_notional'] - tr1['entry_fee'])
        self.assertEqual(tr1['gross_exposure_after'], Decimal("200.0"))
        self.assertEqual(tr1['position_count_after'], 1)

        # Order 2: REJECTED by binding capacity cap (p_24h dropped, cap < requested)
        self.assertEqual(tr2['status'], 'REJECTED')
        self.assertEqual(tr2['rejection_reason'], 'REQUESTED_EXECUTION_NOTIONAL_EXCEEDS_CAP')
        self.assertEqual(tr2['approved_qty'], Decimal("0"))
        self.assertEqual(tr2['approved_notional'], Decimal("0"))
        self.assertEqual(tr2['entry_fee'], Decimal("0"))
        self.assertEqual(tr2['exit_fee_reserve'], Decimal("0"))
        self.assertEqual(tr2['free_cash_after'], tr2['free_cash_before'])
        self.assertEqual(tr2['gross_exposure_after'], tr2['gross_exposure_before'])
        self.assertEqual(tr2['position_count_after'], tr2['position_count_before'])
        self.assertEqual(tr2['free_cash_after'], tr2['free_cash_before'])
        self.assertEqual(tr2['gross_exposure_after'], tr2['gross_exposure_before'])
        self.assertEqual(tr2['position_count_after'], tr2['position_count_before'])


class TestDeterministicRepeatRun(unittest.TestCase):
    """Requirement I / Item 7: Full deterministic comparison of two identical runs"""
    def _make_df(self):
        return pd.DataFrame([
            {'open': 100, 'high': 100, 'low': 100, 'close': 100, 'signal': 1},
            {'open': 110, 'high': 115, 'low': 108, 'close': 112, 'signal': 0},
            {'open': 112, 'high': 118, 'low': 105, 'close': 106, 'signal': -1},
            {'open': 106, 'high': 107, 'low': 100, 'close': 102, 'signal': 0},
        ])

    def test_full_deterministic_repeat_run(self):
        """Two identical runs must produce identical trade and trace output on all fields including cash/exposure"""
        from validation_engine.simulator import simulate_trades
        df = self._make_df()
        kw = dict(entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.001, mode='base', initial_cash=10000.0)
        trades1, rejected1, traces1 = simulate_trades(df.copy(), **kw)
        trades2, rejected2, traces2 = simulate_trades(df.copy(), **kw)
        self.assertEqual(len(trades1), len(trades2), "Trade count mismatch")
        for i, (t1, t2) in enumerate(zip(trades1, trades2)):
            self.assertEqual(t1['entry_time'], t2['entry_time'], f"Trade {i} entry_time")
            self.assertEqual(t1['exit_time'], t2['exit_time'], f"Trade {i} exit_time")
            self.assertEqual(t1['entry_fill']['fill_price'], t2['entry_fill']['fill_price'], f"Trade {i} entry fill_price")
            self.assertEqual(t1['exit_fill']['fill_price'], t2['exit_fill']['fill_price'], f"Trade {i} exit fill_price")
            self.assertEqual(t1.get('quantity'), t2.get('quantity'), f"Trade {i} quantity")
            self.assertEqual(t1.get('direction'), t2.get('direction'), f"Trade {i} direction")
            self.assertEqual(t1.get('entry_fee'), t2.get('entry_fee'), f"Trade {i} entry_fee")
            self.assertEqual(t1.get('exit_fee'), t2.get('exit_fee'), f"Trade {i} exit_fee")
            self.assertEqual(t1.get('net_return'), t2.get('net_return'), f"Trade {i} net_return")
            self.assertEqual(t1.get('exit_reason'), t2.get('exit_reason'), f"Trade {i} exit_reason")
            self.assertEqual(t1['entry_fill'].get('fee'), t2['entry_fill'].get('fee'), f"Trade {i} entry fill fee")
            self.assertEqual(t1['exit_fill'].get('fee'), t2['exit_fill'].get('fee'), f"Trade {i} exit fill fee")

        self.assertEqual(len(traces1), len(traces2), "Decision trace count mismatch")
        for i, (tr1, tr2) in enumerate(zip(traces1, traces2)):
            self.assertEqual(tr1['attempt_id'], tr2['attempt_id'], f"Trace {i} attempt_id")
            self.assertEqual(tr1['timestamp'], tr2['timestamp'], f"Trace {i} timestamp")
            self.assertEqual(tr1['symbol'], tr2['symbol'], f"Trace {i} symbol")
            self.assertEqual(tr1['requested_qty'], tr2['requested_qty'], f"Trace {i} requested_qty")
            self.assertEqual(tr1['requested_notional'], tr2['requested_notional'], f"Trace {i} requested_notional")
            self.assertEqual(tr1['approved_qty'], tr2['approved_qty'], f"Trace {i} approved_qty")
            self.assertEqual(tr1['approved_notional'], tr2['approved_notional'], f"Trace {i} approved_notional")
            self.assertEqual(tr1['fill_price'], tr2['fill_price'], f"Trace {i} fill_price")
            self.assertEqual(tr1['slippage_amount'], tr2['slippage_amount'], f"Trace {i} slippage_amount")
            self.assertEqual(tr1['entry_fee'], tr2['entry_fee'], f"Trace {i} entry_fee")
            self.assertEqual(tr1['exit_fee_reserve'], tr2['exit_fee_reserve'], f"Trace {i} exit_fee_reserve")
            self.assertEqual(tr1['free_cash_before'], tr2['free_cash_before'], f"Trace {i} free_cash_before")
            self.assertEqual(tr1['free_cash_after'], tr2['free_cash_after'], f"Trace {i} free_cash_after")
            self.assertEqual(tr1['gross_exposure_before'], tr2['gross_exposure_before'], f"Trace {i} gross_exposure_before")
            self.assertEqual(tr1['gross_exposure_after'], tr2['gross_exposure_after'], f"Trace {i} gross_exposure_after")
            self.assertEqual(tr1['position_count_before'], tr2['position_count_before'], f"Trace {i} position_count_before")
            self.assertEqual(tr1['position_count_after'], tr2['position_count_after'], f"Trace {i} position_count_after")
            self.assertEqual(tr1['status'], tr2['status'], f"Trace {i} status")
            self.assertEqual(tr1['rejection_reason'], tr2['rejection_reason'], f"Trace {i} rejection_reason")

if __name__ == '__main__':
    unittest.main()
