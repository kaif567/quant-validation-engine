import unittest
import pandas as pd
import numpy as np
import re
from decimal import Decimal
from validation_engine.walk_forward import (
    walk_forward_test,
    aggregate_gate_statuses,
    official_completed_trade_pnl,
)
from validation_engine.stats import block_bootstrap

class TestValidationGates(unittest.TestCase):

    def test_c1_negative_test_and_cliff(self):
        df = pd.DataFrame({'open': [100.0]*100, 'high': [105.0]*100, 'low': [95.0]*100, 'close': [100.0]*100})
        df.loc[11, 'open'] = 100.0
        df.loc[13, 'open'] = 110.0 # p=1 train wins
        df.loc[61, 'open'] = 100.0
        df.loc[63, 'open'] = 90.0 # test loses
        
        def strategy(df, p):
            df['signal'] = 0
            if df.index[0] < 50:
                if p == 1:
                    df.loc[10, 'signal'] = 1; df.loc[12, 'signal'] = -1
                else:
                    df.loc[10, 'signal'] = -1; df.loc[12, 'signal'] = 1 
            else:
                df.loc[60, 'signal'] = 1; df.loc[62, 'signal'] = -1
            return df
            
        res = walk_forward_test(strategy, {'p': [1, 2]}, df, train_window=50, test_window=50, step=50, purge_bars=0, embargo_bars=0, fee_rate=0.0, entry_slippage=0.0, stop_slippage=0.0, min_trades_for_g5=0)
        
        self.assertFalse(res['portfolio_gates']['overall_pass'])
        self.assertIn('G1_G3_MEAN_RETURN', res['portfolio_gates']['failed_gates'])

    def test_aggregate_gate_statuses_all_pass(self):
        """Pure aggregation function: all gates PASS -> overall_pass True, no failed or pending gates"""
        status_map = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'PASS',
            'G6': 'PASS',
        }
        overall_pass, failed_gates, pending_gates = aggregate_gate_statuses(status_map)
        self.assertTrue(overall_pass)
        self.assertEqual(failed_gates, [])
        self.assertEqual(pending_gates, [])

    def test_c2_all_positive_walk_forward_has_pending_gates(self):
        """In real walk_forward_test, positive trades pass G1_G3, G4, G5, G5b is NOT_APPLICABLE_NO_LOSSES (satisfied), but uncomputed G6 remains pending so overall_pass is False"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        
        def strategy(df, p):
            df['signal'] = 0
            for i in range(10, len(df)-5, 10):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df
            
        for i in range(10, 200-5, 10):
            df.loc[i+1, 'open'] = 100.0
            df.loc[i+3, 'open'] = 110.0 
            
        res = walk_forward_test(strategy, {'p': [1, 2]}, df, train_window=100, test_window=100, step=100, purge_bars=0, embargo_bars=0, max_top5_share=0.9, min_trades_for_g5=0, entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['failed_gates'], [])
        self.assertEqual(pg['G1_G3_MEAN_RETURN'], 'PASS')
        self.assertEqual(pg['G4'], 'PASS')
        self.assertEqual(pg['G5'], 'PASS')
        self.assertEqual(pg['G5b'], 'NOT_APPLICABLE_NO_LOSSES')
        self.assertEqual(pg['G6'], 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER')
        self.assertNotIn('G5b', pg['pending_gates'])
        self.assertIn('G6', pg['pending_gates'])
        self.assertFalse(pg['overall_pass'])
        self.assertFalse(pg['institutional_pass_eligible'])

    def test_g6_not_evaluated_blocks_pass(self):
        """G6 NOT_EVALUATED -> overall_pass False, institutional_pass_eligible False, pending_gates contains 'G6'"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            for i in range(10, len(df)-5, 10):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df
        for i in range(10, 200-5, 10):
            df.loc[i+1, 'open'] = 100.0
            df.loc[i+3, 'open'] = 110.0
        res = walk_forward_test(strategy, {'p': [1, 2]}, df, train_window=100, test_window=100, step=100, purge_bars=0, embargo_bars=0, max_top5_share=0.9, min_trades_for_g5=0, entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertFalse(res['overall_pass'])
        self.assertFalse(pg['overall_pass'])
        self.assertFalse(pg['institutional_pass_eligible'])
        self.assertIn('G6', pg['pending_gates'])
        self.assertEqual(pg['G6'], 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER')

    def test_backdoor_gate_status_injection_raises_type_error(self):
        """Calling walk_forward_test with g6_status or other gate status kwargs raises TypeError"""
        df = pd.DataFrame({'open': [100.0]*100, 'high': [105.0]*100, 'low': [95.0]*100, 'close': [100.0]*100})
        def strategy(df, p):
            df['signal'] = 0
            return df
        with self.assertRaisesRegex(TypeError, "walk_forward_test does not accept gate-status overrides"):
            walk_forward_test(strategy, {'p': [1]}, df, g6_status='PASS')
        with self.assertRaisesRegex(TypeError, "walk_forward_test does not accept gate-status overrides"):
            walk_forward_test(strategy, {'p': [1]}, df, g5b_status='PASS')

    def test_c3_cost_shock_fails_g4(self):
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            for i in range(10, len(df)-5, 10):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df
            
        for i in range(10, 200-5, 10):
            df.loc[i+1, 'open'] = 100.0
            df.loc[i+3, 'open'] = 100.1 
            
        res = walk_forward_test(strategy, {'p': [1, 2]}, df, train_window=100, test_window=100, step=100, purge_bars=0, embargo_bars=0, fee_rate=0.0004, max_top5_share=0.99, min_trades_for_g5=0, entry_slippage=0.0, stop_slippage=0.0)
        
        # Assert baseline failed_gates == [] and shocked failed_gates == ['G4'] using actual gate outputs
        self.assertEqual(res['portfolio_gates']['baseline_failed_gates'], [])
        self.assertEqual(res['portfolio_gates']['failed_gates'], ['G4'])
        self.assertFalse(res['portfolio_gates']['overall_pass']) 
    def test_g4_cost_inputs_end_to_end(self):
        """
        Captures simulator arguments for baseline and G4 shocked replay,
        asserting exact 2x doubling of fee_rate, entry_slippage, and stop_slippage.
        """
        from unittest.mock import patch
        from decimal import Decimal
        import validation_engine.walk_forward as wf_mod

        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200, 'atr': [1.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            for i in range(10, len(df)-5, 10):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df

        captured_calls = []
        orig_simulate = wf_mod.simulate_trades

        def intercept_simulate(sim_df, **sim_kwargs):
            captured_calls.append(sim_kwargs.copy())
            return orig_simulate(sim_df, **sim_kwargs)

        with patch.object(wf_mod, 'simulate_trades', side_effect=intercept_simulate):
            wf_mod.walk_forward_test(
                strategy,
                {'p': [1]},
                df,
                train_window=50,
                test_window=50,
                step=50,
                fee_rate=0.001,
                entry_slippage=0.0005,
                stop_slippage=0.0005,
                min_trades_for_g5=0
            )

        baseline_call = [c for c in captured_calls if c.get('fee_rate') == 0.001][0]
        shocked_call = [c for c in captured_calls if c.get('fee_rate') == 0.002][0]

        # Assert baseline cost inputs
        self.assertEqual(Decimal(str(baseline_call['fee_rate'])), Decimal("0.001"))
        self.assertEqual(Decimal(str(baseline_call['entry_slippage'])), Decimal("0.0005"))
        self.assertEqual(Decimal(str(baseline_call['stop_slippage'])), Decimal("0.0005"))

        # Assert G4 shocked replay cost inputs (exact 2x)
        self.assertEqual(Decimal(str(shocked_call['fee_rate'])), Decimal("0.002"))
        self.assertEqual(Decimal(str(shocked_call['entry_slippage'])), Decimal("0.0010"))
        self.assertEqual(Decimal(str(shocked_call['stop_slippage'])), Decimal("0.0010"))

    def test_c4_concentration_only_fails_g5(self):
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        
        def strategy(df, p):
            df['signal'] = 0
            for i in range(2, len(df)-2, 4):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df
            
        for i in range(2, 200-2, 4):
            df.loc[i+1, 'open'] = 100.0
            df.loc[i+3, 'open'] = 101.0 # +1% win
            
        df.loc[83, 'open'] = 100.0
        df.loc[85, 'open'] = 200.0 # +100% win
        
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=40, test_window=80, step=80, purge_bars=0, embargo_bars=0, max_top5_share=0.50, fee_rate=0.0, entry_slippage=0.0, stop_slippage=0.0, min_trades_for_g5=0)
        
        self.assertEqual(res['portfolio_gates']['failed_gates'], ['G5'])

    def test_c5_no_dsr_keys(self):
        df = pd.DataFrame({'open': [100.0]*100, 'high': [105.0]*100, 'low': [95.0]*100, 'close': [100.0]*100})
        def strategy(df, p):
            df['signal'] = 0
            return df
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=50, test_window=50, purge_bars=0, embargo_bars=0, min_trades_for_g5=0, entry_slippage=0.0, stop_slippage=0.0)
        
        res_str = str(res)
        self.assertFalse(bool(re.search(r'dsr|deflated_sharpe|robustness_score', res_str, re.IGNORECASE)))



    def test_g5_insufficient_trades(self):
        df = pd.DataFrame({'open': [100.0]*100, 'high': [105.0]*100, 'low': [95.0]*100, 'close': [100.0]*100})
        def strategy(df, p):
            df['signal'] = 0
            if df.index[-1] > 50:
                df.iloc[-20, df.columns.get_loc('signal')] = 1
                df.iloc[-10, df.columns.get_loc('signal')] = -1
            return df
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=50, test_window=50, step=50, purge_bars=0, embargo_bars=0, min_trades_for_g5=10, entry_slippage=0.0, stop_slippage=0.0)
        self.assertEqual(res['portfolio_gates']['G5'], 'INSUFFICIENT_TRADES')

    def test_g5_concentrated_fails(self):
        """Concentrated profits fail G5"""
        from decimal import Decimal
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            for i in range(2, len(df)-2, 4):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df
        for i in range(2, 200-2, 4):
            df.loc[i+1, 'open'] = 100.0
            df.loc[i+3, 'open'] = 101.0
        df.loc[83, 'open'] = 100.0
        df.loc[85, 'open'] = 200.0
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=40, test_window=80, step=80, purge_bars=0, embargo_bars=0, max_top5_share=0.50, fee_rate=0.0, entry_slippage=0.0, stop_slippage=0.0, min_trades_for_g5=5)
        self.assertEqual(res['portfolio_gates']['G5'], 'FAIL')
        self.assertEqual(res['portfolio_gates']['g5_status'], 'FAIL')
        self.assertIn('G5', res['portfolio_gates']['failed_gates'])

    def test_g5_diversified_passes(self):
        """Diversified profits pass G5"""
        from decimal import Decimal
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            for i in range(2, len(df)-2, 4):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i+2], 'signal'] = -1
            return df
        for i in range(2, 200-2, 4):
            df.loc[i+1, 'open'] = 100.0
            df.loc[i+3, 'open'] = 101.0
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=40, test_window=80, step=80, purge_bars=0, embargo_bars=0, max_top5_share=0.50, fee_rate=0.0, entry_slippage=0.0, stop_slippage=0.0, min_trades_for_g5=5)
        self.assertEqual(res['portfolio_gates']['G5'], 'PASS')
        self.assertEqual(res['portfolio_gates']['g5_status'], 'PASS')
        self.assertNotIn('G5', res['portfolio_gates']['failed_gates'])

    def test_g5_invalid_direction_raises(self):
        """Invalid trade direction raises ValueError"""
        from validation_engine.walk_forward import official_completed_trade_pnl
        trade = {
            'direction': 'sideways',
            'entry_fill': {'fill_price': 100.0, 'fee': 0.1},
            'exit_fill': {'fill_price': 105.0, 'fee': 0.1},
            'quantity': 1.0
        }
        with self.assertRaisesRegex(ValueError, "Completed trade has invalid direction"):
            official_completed_trade_pnl(trade)

    def test_g5_missing_fields_raises(self):
        """Missing fill_price, fee, or quantity raises KeyError/ValueError"""
        from validation_engine.walk_forward import official_completed_trade_pnl
        t1 = {'direction': 'long', 'entry_fill': {'fee': 0.1}, 'exit_fill': {'fill_price': 105.0, 'fee': 0.1}, 'quantity': 1.0}
        with self.assertRaises(KeyError):
            official_completed_trade_pnl(t1)

        t2 = {'direction': 'long', 'entry_fill': {'fill_price': 100.0}, 'exit_fill': {'fill_price': 105.0, 'fee': 0.1}, 'quantity': 1.0}
        with self.assertRaises(KeyError):
            official_completed_trade_pnl(t2)

        t3 = {'direction': 'long', 'entry_fill': {'fill_price': 100.0, 'fee': 0.1}, 'exit_fill': {'fill_price': 105.0, 'fee': 0.1}}
        with self.assertRaises(KeyError):
            official_completed_trade_pnl(t3)

    def test_g5_pnl_semantics_fees_quantity_and_shortcut_failure(self):
        """Entry fee, exit fee, and quantity != 1 change PnL; old shortcut entry_price*net_return fails"""
        from decimal import Decimal
        from validation_engine.walk_forward import official_completed_trade_pnl
        trade = {
            'direction': 'long',
            'entry_fill': {'fill_price': 100.0, 'fee': 1.5},
            'exit_fill': {'fill_price': 110.0, 'fee': 2.0},
            'quantity': 3.0,
            'net_return': 0.065,
        }
        actual_pnl = official_completed_trade_pnl(trade)
        self.assertEqual(actual_pnl, Decimal("26.5"))
        shortcut_pnl = Decimal(str(trade['entry_fill']['fill_price'])) * Decimal(str(trade['net_return']))
        self.assertNotEqual(actual_pnl, shortcut_pnl, "Old shortcut entry_price*net_return would fail the fixture")

        # Short direction
        short_trade = {
            'direction': 'short',
            'entry_fill': {'fill_price': 100.0, 'fee': 1.0},
            'exit_fill': {'fill_price': 90.0, 'fee': 1.5},
            'quantity': 2.0,
        }
        self.assertEqual(official_completed_trade_pnl(short_trade), Decimal("17.5"))

    def test_g5_losers_and_zero_pnl_excluded_from_numerator_and_denominator(self):
        """Losers and zero-PnL excluded from numerator and denominator, but counted in completed_trade_count"""
        from decimal import Decimal
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            if len(df) > 100:
                for i in range(10, 160, 10):
                    if i + 2 < len(df):
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                        df.iloc[i+2, df.columns.get_loc('signal')] = -1
                        df.iloc[i+3, df.columns.get_loc('signal')] = 0
            return df
            
        for i in range(10, 160, 10):
            df.loc[i+1, 'open'] = 100.0
            if i < 110:
                df.loc[i+3, 'open'] = 100.0
            elif i in [110, 120, 130]:
                df.loc[i+3, 'open'] = 105.0
            elif i == 140:
                df.loc[i+3, 'open'] = 90.0
            else:
                df.loc[i+3, 'open'] = 120.0
                
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=20, test_window=180, step=180, purge_bars=0, embargo_bars=0, max_top5_share=0.99, min_trades_for_g5=10, entry_slippage=0.0, stop_slippage=0.0, fee_rate=0.01)
        
        pg = res['portfolio_gates']
        self.assertEqual(pg['completed_trade_count'], 30)
        self.assertGreater(pg['top_count'], 0)
        self.assertGreater(pg['total_positive_profit'], Decimal("0"))
        self.assertEqual(pg['top_profit'], Decimal("17.8"))

    # ── G5b Loss Stress Test Permanent Tests ─────────────────────────────────

    def test_g5b_insufficient_sample(self):
        """G5b: trade count below min_trades_for_g5b -> INSUFFICIENT_TRADES in pending_gates"""
        df = pd.DataFrame({'open': [100.0]*100, 'high': [105.0]*100, 'low': [95.0]*100, 'close': [100.0]*100})
        def strategy(df, p):
            df['signal'] = 0
            if df.index[-1] > 50:
                df.iloc[-20, df.columns.get_loc('signal')] = 1
                df.iloc[-10, df.columns.get_loc('signal')] = -1
            return df
        # Only 1 trade produced, default min_trades_for_g5b=10
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=50, test_window=50, step=50,
                                purge_bars=0, embargo_bars=0, min_trades_for_g5=0, min_trades_for_g5b=10,
                                entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['G5b'], 'INSUFFICIENT_TRADES')
        self.assertEqual(pg['g5b_status'], 'INSUFFICIENT_TRADES')
        self.assertIn('G5b', pg['pending_gates'])
        self.assertIsNone(pg['g5b_global_pnl'])
        self.assertIsNone(pg['g5b_largest_loss'])
        self.assertIsNone(pg['g5b_stressed_pnl'])
        self.assertEqual(pg['g5b_min_trades'], 10)

    def test_g5b_nonpositive_global_pnl(self):
        """G5b: global PnL <= 0 -> NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL, satisfied (not pending/failed)"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            # 2 trades in test window
            df.loc[110, 'signal'] = 1
            df.loc[120, 'signal'] = -2
            df.loc[130, 'signal'] = 1
            df.loc[140, 'signal'] = -2
            return df
        # trade 1: win (+5), trade 2: loss (-15) -> global PnL = -10 <= 0
        df.loc[111, 'open'] = 100.0
        df.loc[121, 'open'] = 105.0
        df.loc[131, 'open'] = 100.0
        df.loc[141, 'open'] = 85.0
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=100, test_window=100, step=100,
                                purge_bars=0, embargo_bars=0, fee_rate=0.0, min_trades_for_g5=0,
                                min_trades_for_g5b=2, entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['G5b'], 'NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL')
        self.assertEqual(pg['g5b_status'], 'NOT_APPLICABLE_NONPOSITIVE_GLOBAL_PNL')
        self.assertNotIn('G5b', pg['pending_gates'])
        self.assertNotIn('G5b', pg['failed_gates'])
        self.assertEqual(pg['g5b_global_pnl'], Decimal("-10.0"))
        self.assertEqual(pg['g5b_largest_loss'], Decimal("-15.0"))
        self.assertIsNone(pg['g5b_stressed_pnl'])

    def test_g5b_no_losses(self):
        """G5b: all trades winning -> NOT_APPLICABLE_NO_LOSSES, satisfied (not pending/failed)"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            df.loc[110, 'signal'] = 1
            df.loc[120, 'signal'] = -2
            df.loc[130, 'signal'] = 1
            df.loc[140, 'signal'] = -2
            return df
        df.loc[111, 'open'] = 100.0
        df.loc[121, 'open'] = 110.0
        df.loc[131, 'open'] = 100.0
        df.loc[141, 'open'] = 105.0
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=100, test_window=100, step=100,
                                purge_bars=0, embargo_bars=0, fee_rate=0.0, min_trades_for_g5=0,
                                min_trades_for_g5b=2, entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['G5b'], 'NOT_APPLICABLE_NO_LOSSES')
        self.assertEqual(pg['g5b_status'], 'NOT_APPLICABLE_NO_LOSSES')
        self.assertNotIn('G5b', pg['pending_gates'])
        self.assertNotIn('G5b', pg['failed_gates'])
        self.assertEqual(pg['g5b_global_pnl'], Decimal("15.0"))
        self.assertIsNone(pg['g5b_largest_loss'])
        self.assertIsNone(pg['g5b_stressed_pnl'])

    def test_g5b_robust_profitable_pass(self):
        """G5b: robust profitable case (stressed_pnl >= 0) -> PASS"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            # 3 trades in test window
            df.loc[110, 'signal'] = 1
            df.loc[120, 'signal'] = -2
            df.loc[130, 'signal'] = 1
            df.loc[140, 'signal'] = -2
            df.loc[150, 'signal'] = 1
            df.loc[160, 'signal'] = -2
            return df
        # trade 1: +10, trade 2: +10, trade 3: -3
        # global = 17, largest_loss = -3, stressed = 17 + (-3) = 14 >= 0 -> PASS
        df.loc[111, 'open'] = 100.0
        df.loc[121, 'open'] = 110.0
        df.loc[131, 'open'] = 100.0
        df.loc[141, 'open'] = 110.0
        df.loc[151, 'open'] = 100.0
        df.loc[161, 'open'] = 97.0
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=100, test_window=100, step=100,
                                purge_bars=0, embargo_bars=0, fee_rate=0.0, min_trades_for_g5=0,
                                min_trades_for_g5b=3, entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['G5b'], 'PASS')
        self.assertEqual(pg['g5b_status'], 'PASS')
        self.assertEqual(pg['g5b_global_pnl'], Decimal("17.0"))
        self.assertEqual(pg['g5b_largest_loss'], Decimal("-3.0"))
        self.assertEqual(pg['g5b_stressed_pnl'], Decimal("14.0"))
        self.assertNotIn('G5b', pg['failed_gates'])
        self.assertNotIn('G5b', pg['pending_gates'])

    def test_g5b_fragile_profitable_fail(self):
        """G5b: fragile profitable case (stressed_pnl < 0) -> FAIL, in failed_gates, overall_pass False"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200})
        def strategy(df, p):
            df['signal'] = 0
            df.loc[110, 'signal'] = 1
            df.loc[120, 'signal'] = -2
            df.loc[130, 'signal'] = 1
            df.loc[140, 'signal'] = -2
            return df
        # trade 1: +5, trade 2: -4
        # global = 1, largest_loss = -4, stressed = 1 + (-4) = -3 < 0 -> FAIL
        df.loc[111, 'open'] = 100.0
        df.loc[121, 'open'] = 105.0
        df.loc[131, 'open'] = 100.0
        df.loc[141, 'open'] = 96.0
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=100, test_window=100, step=100,
                                purge_bars=0, embargo_bars=0, fee_rate=0.0, min_trades_for_g5=0,
                                min_trades_for_g5b=2, entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        self.assertEqual(pg['G5b'], 'FAIL')
        self.assertEqual(pg['g5b_status'], 'FAIL')
        self.assertEqual(pg['g5b_global_pnl'], Decimal("1.0"))
        self.assertEqual(pg['g5b_largest_loss'], Decimal("-4.0"))
        self.assertEqual(pg['g5b_stressed_pnl'], Decimal("-3.0"))
        self.assertIn('G5b', pg['failed_gates'])
        self.assertFalse(pg['overall_pass'])
        self.assertFalse(res['overall_pass'])

    def test_g5b_fee_and_quantity_semantics(self):
        """G5b uses official completed-trade Decimal PnL: entry fee, exit fee, and quantity affect result"""
        # Unit verification with official_completed_trade_pnl
        trade_base = {
            'direction': 'long',
            'entry_fill': {'fill_price': 100.0, 'fee': 0.0},
            'exit_fill': {'fill_price': 105.0, 'fee': 0.0},
            'quantity': 1.0,
        }
        self.assertEqual(official_completed_trade_pnl(trade_base), Decimal("5.0"))

        # Entry fee reduces PnL
        trade_entry_fee = {
            'direction': 'long',
            'entry_fill': {'fill_price': 100.0, 'fee': 2.0},
            'exit_fill': {'fill_price': 105.0, 'fee': 0.0},
            'quantity': 1.0,
        }
        self.assertEqual(official_completed_trade_pnl(trade_entry_fee), Decimal("3.0"))

        # Exit fee reduces PnL
        trade_exit_fee = {
            'direction': 'long',
            'entry_fill': {'fill_price': 100.0, 'fee': 0.0},
            'exit_fill': {'fill_price': 105.0, 'fee': 1.5},
            'quantity': 1.0,
        }
        self.assertEqual(official_completed_trade_pnl(trade_exit_fee), Decimal("3.5"))

        # Quantity scales gross PnL
        trade_scaled_qty = {
            'direction': 'long',
            'entry_fill': {'fill_price': 100.0, 'fee': 0.5},
            'exit_fill': {'fill_price': 105.0, 'fee': 0.5},
            'quantity': 2.0,
        }
        # gross = (105 - 100) * 2 = 10, net = 10 - 0.5 - 0.5 = 9.0
        self.assertEqual(official_completed_trade_pnl(trade_scaled_qty), Decimal("9.0"))

    def test_g5b_portfolio_gates_fields_presence(self):
        """All 7 required G5b fields appear in portfolio_gates with Decimal / proper types"""
        df = pd.DataFrame({'open': [100.0]*100, 'high': [105.0]*100, 'low': [95.0]*100, 'close': [100.0]*100})
        def strategy(df, p):
            df['signal'] = 0
            return df
        res = walk_forward_test(strategy, {'p': [1]}, df, train_window=50, test_window=50, step=50,
                                entry_slippage=0.0, stop_slippage=0.0)
        pg = res['portfolio_gates']
        required_fields = [
            'G5b', 'g5b_status', 'g5b_completed_trade_count',
            'g5b_global_pnl', 'g5b_largest_loss', 'g5b_stressed_pnl', 'g5b_min_trades',
        ]
        for f in required_fields:
            self.assertIn(f, pg, f"Missing required G5b field: {f}")

    def test_g5b_aggregate_gate_statuses_pass_removes_pending(self):
        """G5b PASS removes G5b from pending_gates; G5b FAIL adds to failed_gates"""
        # All pass
        status_map_pass = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'PASS',
            'G6': 'PASS',
        }
        overall, failed, pending = aggregate_gate_statuses(status_map_pass)
        self.assertNotIn('G5b', pending)
        self.assertNotIn('G5b', failed)
        self.assertTrue(overall)

        # G5b fails
        status_map_fail = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'FAIL',
            'G6': 'PASS',
        }
        overall, failed, pending = aggregate_gate_statuses(status_map_fail)
        self.assertIn('G5b', failed)
        self.assertNotIn('G5b', pending)
        self.assertFalse(overall)

        # G5b insufficient
        status_map_insufficient = {
            'G1_G3_MEAN_RETURN': 'PASS',
            'G4': 'PASS',
            'G5': 'PASS',
            'G5b': 'INSUFFICIENT_TRADES',
            'G6': 'PASS',
        }
        overall, failed, pending = aggregate_gate_statuses(status_map_insufficient)
        self.assertIn('G5b', pending)
        self.assertNotIn('G5b', failed)
        self.assertFalse(overall)


if __name__ == '__main__':
    unittest.main()
