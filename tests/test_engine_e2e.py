import unittest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from validation_engine.walk_forward import walk_forward_test
from validation_engine.trial_ledger import TrialLedgerV2 as TrialLedger

class TestEngineE2E(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = os.path.join(self.temp_dir.name, 'test_ledger.jsonl')
        

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_e2e_pipeline(self):
        np.random.seed(42)
        base = 100.0
        closes = [base]
        for _ in range(999):
            closes.append(closes[-1] + np.random.normal(0, 1))
        
        df = pd.DataFrame({
            'open': closes,
            'high': [c + 2.0 for c in closes],
            'low': [c - 2.0 for c in closes],
            'close': closes
        })

        def breakout_strategy(df, fast, slow):
            df['sma_fast'] = df['close'].rolling(fast).mean()
            df['sma_slow'] = df['close'].rolling(slow).mean()
            df['signal'] = 0
            df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
            df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1
            return df
            
        param_grid = {'fast': [5, 10], 'slow': [20, 30]} # 2x2 grid = 4 combos
        ledger = TrialLedger(self.ledger_path)
        
        # 3. Real pipeline
        res = walk_forward_test(
            breakout_strategy, 
            param_grid, 
            df, 
            train_window=400, 
            test_window=200, 
            step=200, 
            purge_bars=0, 
            embargo_bars=1,
            fee_rate=0.001,
            ledger=ledger,
            run_id="e2e_run_1"
        , entry_slippage=0.0, stop_slippage=0.0)
        
        n_folds = 3 # 0-400, 200-600, 400-800
        n_trials = ledger.n_trials("e2e_run_1")
        
        self.assertGreaterEqual(n_trials, 4 * n_folds)
        
        self.assertIn('overall_pass', res)
        self.assertIn('failed_gates', res)
        self.assertIsNone(res['bootstrap'])
        self.assertIsNone(res['top5_share'])
        self.assertIn(res['portfolio_gates']['g5_status'], ['PASS', 'FAIL', 'INSUFFICIENT_TRADES'])
        self.assertEqual(res['portfolio_gates']['g6_status'], 'NOT_EVALUATED_PENDING_LEGACY_ADAPTER')
        
        
        
        
        has_start = False
        has_finish = False
        with open(self.ledger_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('event_type') == 'run_started': has_start = True
                if rec.get('event_type') == 'run_finished': has_finish = True
        
        self.assertTrue(has_start)
        self.assertTrue(has_finish)
        
        # Deterministic
        res2 = walk_forward_test(
            breakout_strategy, 
            param_grid, 
            df, 
            train_window=400, 
            test_window=200, 
            step=200, 
            purge_bars=0, 
            embargo_bars=1,
            fee_rate=0.001,
            ledger=ledger,
            run_id="e2e_run_2"
        , entry_slippage=0.0, stop_slippage=0.0)
        
        self.assertEqual(res['overall_pass'], res2['overall_pass'])
        self.assertEqual(res['failed_gates'], res2['failed_gates'])
        self.assertEqual(ledger.n_trials("e2e_run_1"), ledger.n_trials("e2e_run_2"))
        
        print("\n=== E2E RESULT ===")
        print({
            'overall_pass': res['overall_pass'],
            'failed_gates': res['failed_gates'],
            'n_trials': n_trials,
            'top5_share': res['top5_share'],
            'bootstrap.g6_pass': res['bootstrap'].get('g6_pass') if res.get('bootstrap') else None
        })

if __name__ == '__main__':
    unittest.main()
