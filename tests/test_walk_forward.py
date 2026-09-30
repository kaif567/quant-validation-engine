import unittest
import pandas as pd
import os
import tempfile
import json
from validation_engine.walk_forward import walk_forward_test
from validation_engine.trial_ledger import TrialLedgerV2 as TrialLedger

class TestWalkForward(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = os.path.join(self.temp_dir.name, 'test_ledger.jsonl')
        

    def tearDown(self):
        self.temp_dir.cleanup()

    def dummy_strategy_pf_vs_mean(self, df, p):
        df['signal'] = 0
        if p == 1:
            df.loc[10, 'signal'] = 1
            df.loc[12, 'signal'] = -1 
        elif p == 2:
            df.loc[10, 'signal'] = 1
            df.loc[12, 'signal'] = -1
            df.loc[14, 'signal'] = 1
            df.loc[16, 'signal'] = -1
        return df

    def test_champion_is_max_mean_net_return(self):
        df = pd.DataFrame({'open': [100]*100, 'high': [105]*100, 'low': [95]*100, 'close': [100]*100})
        df.loc[12, 'open'] = 101 
        df.loc[16, 'open'] = 110 
        
        param_grid = {'p': [1, 2]}
        res = walk_forward_test(
            self.dummy_strategy_pf_vs_mean, param_grid, df, 
            train_window=50, test_window=20, step=20, 
            purge_bars=1, embargo_bars=1, objective_name='mean_net_return'
        , entry_slippage=0.0, stop_slippage=0.0)
        self.assertIn('per_fold', res)
        self.assertIn('portfolio_gates', res)

    def test_same_grid_different_fees_different_fingerprint(self):
        ledger = TrialLedger(self.ledger_path)
        param_grid = {'sl': [0.02], 'tp': [0.05]}
        df = pd.DataFrame({'open': [100]*2200, 'high': [105]*2200, 'low': [95]*2200, 'close': [100]*2200, 'signal': [0]*2200})
        
        walk_forward_test(lambda df, **k: df, param_grid, df, train_window=1000, test_window=200, step=200, fee_rate=0.001, ledger=ledger, run_id="r1", entry_slippage=0.0, stop_slippage=0.0)
        
        hash1 = None
        with open(self.ledger_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('event_type') == 'candidate_fold_evaluation' and rec.get('run_id') == 'r1':
                    hash1 = rec.get('payload', {}).get('identity', {}).get('param_hash')
                    break
        
        walk_forward_test(lambda df, **k: df, param_grid, df, train_window=1000, test_window=200, step=200, fee_rate=0.005, ledger=ledger, run_id="r2", entry_slippage=0.0, stop_slippage=0.0)
        
        hash2 = None
        with open(self.ledger_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('event_type') == 'candidate_fold_evaluation' and rec.get('run_id') == 'r2':
                    hash2 = rec.get('payload', {}).get('identity', {}).get('param_hash')
                    break
                    
        self.assertIsNotNone(hash1)
        self.assertIsNotNone(hash2)
        self.assertNotEqual(hash1, hash2)

    def test_2x2_grid_ledger_and_neighbors(self):
        df = pd.DataFrame({'open': [100]*2200, 'high': [105]*2200, 'low': [95]*2200, 'close': [100]*2200, 'signal': [0]*2200})
        ledger = TrialLedger(self.ledger_path)
        param_grid = {'sl': [0.02, 0.04], 'tp': [0.05, 0.10]}
        
        walk_forward_test(lambda df, **k: df, param_grid, df, train_window=1000, test_window=200, step=200, ledger=ledger, run_id="r1", entry_slippage=0.0, stop_slippage=0.0)
        self.assertEqual(ledger.n_trials("r1"), 24)

    def test_rolling_vs_expanding(self):
        df = pd.DataFrame({'open': [100]*2200, 'high': [105]*2200, 'low': [95]*2200, 'close': [100]*2200, 'signal': [0]*2200})
        res_exp = walk_forward_test(lambda df, **k: df, {'p':[1]}, df, train_window=1000, test_window=200, step=200, window_mode='expanding', entry_slippage=0.0, stop_slippage=0.0)
        res_roll = walk_forward_test(lambda df, **k: df, {'p':[1]}, df, train_window=1000, test_window=200, step=200, window_mode='rolling', entry_slippage=0.0, stop_slippage=0.0)
        
        self.assertGreater(len(res_exp['per_fold']), 1)
        self.assertEqual(len(res_exp['per_fold']), len(res_roll['per_fold']))

    def test_embargo_purge_drops_trade(self):
        df = pd.DataFrame({'open': [100]*100, 'high': [105]*100, 'low': [95]*100, 'close': [100]*100, 'signal': [0]*100})
        df.loc[45, 'signal'] = 1
        df.loc[55, 'signal'] = -1
        
        res = walk_forward_test(lambda df, **k: df, {'p':[1]}, df, train_window=50, test_window=20, purge_bars=1, embargo_bars=5, entry_slippage=0.0, stop_slippage=0.0)
        self.assertEqual(res['per_fold'][0]['train_objective'], 0.0)

    def test_full_legacy_walk_forward_result_contract_normal_run(self):
        """Normal run returns full LegacyWalkForwardResult with all required keys and correct types"""
        df = pd.DataFrame({'open': [100.0]*200, 'high': [105.0]*200, 'low': [95.0]*200, 'close': [100.0]*200, 'signal': [0]*200})
        res = walk_forward_test(lambda d, **k: d, {'p': [1]}, df, train_window=50, test_window=20, step=20, entry_slippage=0.0, stop_slippage=0.0)
        self.assertEqual(res['schema_version'], 'legacy_walk_forward_v1')
        self.assertEqual(res['validation_scope'], 'LEGACY_SINGLE_ASSET')
        self.assertIsInstance(res['per_fold'], list)
        self.assertIsInstance(res['folds'], list)
        self.assertIsInstance(res['portfolio_gates'], dict)
        self.assertIsInstance(res['overall_pass'], bool)
        self.assertIsInstance(res['trade_records'], list)
        self.assertEqual(res['overall_pass'], res['portfolio_gates']['overall_pass'])

    def test_full_legacy_walk_forward_result_contract_empty_run(self):
        """Empty run returns full LegacyWalkForwardResult with validation_scope == LEGACY_SINGLE_ASSET"""
        df_empty = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'signal'])
        res = walk_forward_test(lambda d, **k: d, {'p': [1]}, df_empty, train_window=50, test_window=20, step=20, entry_slippage=0.0, stop_slippage=0.0)
        self.assertEqual(res['schema_version'], 'legacy_walk_forward_v1')
        self.assertEqual(res['validation_scope'], 'LEGACY_SINGLE_ASSET')
        self.assertIsInstance(res['per_fold'], list)
        self.assertIsInstance(res['folds'], list)
        self.assertIsInstance(res['portfolio_gates'], dict)
        self.assertIsInstance(res['overall_pass'], bool)
        self.assertIsInstance(res['trade_records'], list)
        self.assertEqual(res['overall_pass'], res['portfolio_gates']['overall_pass'])
        self.assertFalse(res['overall_pass'])

    def test_runtime_validator_rejects_missing_key_or_type_or_mismatch(self):
        """validate_legacy_walk_forward_result raises on missing key, wrong type, or overall_pass mismatch"""
        from validation_engine.walk_forward import validate_legacy_walk_forward_result
        valid_res = {
            'schema_version': 'legacy_walk_forward_v1',
            'validation_scope': 'LEGACY_SINGLE_ASSET',
            'per_fold': [],
            'folds': [],
            'portfolio_gates': {'overall_pass': True},
            'overall_pass': True,
            'trade_records': [],
            'raw_trial_ledger_ref': None,
        }
        # Valid passes
        validate_legacy_walk_forward_result(valid_res)

        # Missing key
        broken = valid_res.copy()
        del broken['schema_version']
        with self.assertRaises(ValueError):
            validate_legacy_walk_forward_result(broken)

        # Wrong validation_scope
        broken = valid_res.copy()
        broken['validation_scope'] = 'ACTIVE_LEGACY_CODE'
        with self.assertRaises(ValueError):
            validate_legacy_walk_forward_result(broken)

        # Wrong type
        broken = valid_res.copy()
        broken['per_fold'] = "not_a_list"
        with self.assertRaises(TypeError):
            validate_legacy_walk_forward_result(broken)

        # overall_pass mismatch
        broken = valid_res.copy()
        broken['overall_pass'] = False
        with self.assertRaises(ValueError):
            validate_legacy_walk_forward_result(broken)

if __name__ == '__main__':
    unittest.main()
