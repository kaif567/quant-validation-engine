import unittest
import os
import tempfile
import json
import pandas as pd
from validation_engine.trial_ledger import TrialLedgerV2, dict_hash
from validation_engine.walk_forward import walk_forward_test

class TestTrialLedger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ledger_path = os.path.join(self.temp_dir.name, 'test_ledger.jsonl')

    def tearDown(self):
        self.temp_dir.cleanup()

    def dummy_strategy(self, df, tp, sl):
        df['signal'] = 1
        return df

    def test_2x2_grid_xfolds_exact_count(self):
        """2x2 grid x folds exact count: 4 param combinations x 4 folds = 16 planned trials"""
        ledger = TrialLedgerV2(self.ledger_path)
        df = pd.DataFrame({'open': [100]*2000, 'high': [105]*2000, 'low': [95]*2000, 'close': [100]*2000, 'signal': [0]*2000})
        param_grid = {
            'tp': [0.05, 0.10],
            'sl': [0.02, 0.04]
        }
        walk_forward_test(
            self.dummy_strategy, 
            param_grid, 
            df, 
            initial_train_window=1000, 
            test_window=200, 
            step=200, 
            purge_bars=50, 
            ledger=ledger,
            run_id="run_1",
            entry_slippage=0.0,
            stop_slippage=0.0
        )
        self.assertEqual(ledger.n_trials(run_id="run_1"), 16)
        rows = ledger.trial_rows("run_1")
        self.assertEqual(len(rows), 16)

    def test_duplicate_status_update_no_count_change(self):
        """Duplicate status updates for same trial do not increase n_trials or trial_rows count"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "dup_status.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_dup", "run_dup")
            base = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "P"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(base)
            self.assertEqual(ledger.n_trials("run_dup"), 1)

            # Multiple status updates for the exact same planned trial
            for s in ["running", "completed", "completed"]:
                update = base.copy()
                update["status"] = s
                update["objective_value"] = 1.23
                ledger.record_evaluation(update)

            self.assertEqual(ledger.n_trials("run_dup"), 1)
            rows = ledger.trial_rows("run_dup")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["planned_trial"]["status"], "PLANNED")
            self.assertEqual(rows[0]["latest_status"]["status"], "COMPLETED")

    def test_duplicate_planned_key_does_not_add_plan(self):
        """Duplicate planned key event does not add a second plan to trial_rows or increase n_trials"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "dup_plan.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_dup_plan", "run_dp")
            plan_ev = {
                "identity": {"strategy_family": "SF", "universe_hash": "UH", "param_hash": "PH1"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(plan_ev)
            ledger.record_evaluation(plan_ev)  # exact duplicate planned event
            self.assertEqual(ledger.n_trials("run_dp"), 1)
            rows = ledger.trial_rows("run_dp")
            self.assertEqual(len(rows), 1)

    def test_all_status_types_count_once(self):
        """completed/skipped/failed/incomplete trials each count exactly once"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "all_statuses.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_statuses", "run_st")
            for i, status in enumerate(["completed", "skipped", "failed", "incomplete"]):
                ev_plan = {
                    "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": f"P{i}"},
                    "fold_id": 0,
                    "fold_definition_hash": "0" * 64,
                    "status": "planned",
                }
                ledger.record_evaluation(ev_plan)
                ev_update = {
                    "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": f"P{i}"},
                    "fold_id": 0,
                    "fold_definition_hash": "0" * 64,
                    "status": status,
                }
                ledger.record_evaluation(ev_update)
            self.assertEqual(ledger.n_trials("run_st"), 4)
            rows = ledger.trial_rows("run_st")
            self.assertEqual(len(rows), 4)
            statuses = [r["latest_status"]["status"] for r in rows]
            self.assertIn("COMPLETED", statuses)
            self.assertIn("SKIPPED", statuses)
            self.assertIn("FAILED", statuses)
            self.assertIn("INCOMPLETE", statuses)

    def test_planned_creation_order_preserved(self):
        """trial_rows preserves the initial planned creation order"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "order.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_order", "run_ord")
            hashes = ["order_C", "order_A", "order_B"]
            for h in hashes:
                ev = {
                    "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": h},
                    "fold_id": 0,
                    "fold_definition_hash": "0" * 64,
                    "status": "planned",
                }
                ledger.record_evaluation(ev)
            # Update the middle one
            upd = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "order_A"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "completed",
            }
            ledger.record_evaluation(upd)
            rows = ledger.trial_rows("run_ord")
            self.assertEqual(len(rows), 3)
            self.assertEqual([r["planned_trial"]["identity"]["param_hash"] for r in rows], hashes)

    def test_latest_status_correct(self):
        """latest_status accurately reflects the latest update and reason"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "latest.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_latest", "run_lt")
            ev = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "PX"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(ev)
            ev["status"] = "failed"
            ev["reason"] = "MemoryLimitExceeded"
            ledger.record_evaluation(ev)
            rows = ledger.trial_rows("run_lt")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["latest_status"]["status"], "FAILED")
            self.assertEqual(rows[0]["latest_status"]["reason"], "MemoryLimitExceeded")

    def test_planned_trial_status_always_planned(self):
        """planned_trial.status is always 'PLANNED', even after completed/failed status updates"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "immutable_plan.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_plan_imm", "run_pi")
            ev = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "PY"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(ev)
            ev["status"] = "completed"
            ledger.record_evaluation(ev)
            rows = ledger.trial_rows("run_pi")
            self.assertEqual(rows[0]["planned_trial"]["status"], "PLANNED")
            self.assertEqual(rows[0]["latest_status"]["status"], "COMPLETED")

    def test_raw_planned_jsonl_line_byte_identical_after_updates(self):
        """Raw planned JSONL event line must be byte-for-byte identical after multiple status updates"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "byte_identical.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_imm", "run_imm")
            planned = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "PA"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(planned)
            with open(path, "rb") as f:
                bytes_after_plan = f.read()

            for obj_val in [1.0, 2.0, 3.0]:
                update = planned.copy()
                update["status"] = "completed"
                update["objective_value"] = obj_val
                ledger.record_evaluation(update)

            with open(path, "rb") as f:
                bytes_after_updates = f.read()

            self.assertTrue(
                bytes_after_updates.startswith(bytes_after_plan),
                "Planned JSONL event line was mutated — file is not append-only"
            )

    def test_temporary_ledger_lifecycle(self):
        """Construction alone does not create file; start_run creates it; path is inside temp dir"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "lifecycle.jsonl")
            ledger = TrialLedgerV2(path)
            self.assertFalse(os.path.exists(path), "Construction alone must not create the file")
            ledger.start_run("camp_life", "run_life")
            self.assertTrue(os.path.exists(path), "start_run must create the file")
            self.assertTrue(path.startswith(td), "Path must be inside temporary directory")

    def test_trial_rows_identity_contains_run_id(self):
        """planned_trial identity must contain run_id, strategy_family, universe_hash, param_hash, fold_id"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ident.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_id", "run_ident_check")
            payload = {
                "identity": {"strategy_family": "FAM", "universe_hash": "UNIV", "param_hash": "PAR"},
                "fold_id": 3,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(payload)
            rows = ledger.trial_rows("run_ident_check")
            self.assertEqual(len(rows), 1)
            ident = rows[0]["planned_trial"]["identity"]
            self.assertEqual(ident["run_id"], "run_ident_check")
            self.assertEqual(ident["strategy_family"], "FAM")
            self.assertEqual(ident["universe_hash"], "UNIV")
            self.assertEqual(ident["param_hash"], "PAR")
            self.assertEqual(ident["fold_id"], 3)

    def test_append_only_multiple_runs(self):
        """Assert file is append-only by recording 2 runs and both still readable"""
        ledger = TrialLedgerV2(self.ledger_path)
        ledger.start_run("camp_1", "run_A")
        ledger.record_evaluation({'fold_definition_hash': '0'*64, 'identity': {'strategy_family': 'S1', 'universe_hash': 'U1', 'param_hash': 'P1'}, 'fold_id': 0, 'status': 'planned'})
        ledger.record_evaluation({'fold_definition_hash': '0'*64, 'identity': {'strategy_family': 'S1', 'universe_hash': 'U1', 'param_hash': 'P1'}, 'fold_id': 0, 'status': 'completed'})
        ledger.finish_run()

        ledger.start_run("camp_1", "run_B")
        ledger.record_evaluation({'fold_definition_hash': '0'*64, 'identity': {'strategy_family': 'S1', 'universe_hash': 'U1', 'param_hash': 'P1'}, 'fold_id': 0, 'status': 'planned'})
        ledger.record_evaluation({'fold_definition_hash': '0'*64, 'identity': {'strategy_family': 'S1', 'universe_hash': 'U1', 'param_hash': 'P1'}, 'fold_id': 0, 'status': 'completed'})
        ledger.finish_run()

        self.assertEqual(ledger.n_trials("run_A"), 1)
        self.assertEqual(ledger.n_trials("run_B"), 1)
        self.assertEqual(ledger.n_trials(None), 2)

    def test_orphan_status_update_does_not_create_plan_or_count_in_n_trials(self):
        """Orphan status update (no prior planned event) does not increment n_trials or create planned rows"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "orphan.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_orphan", "run_orphan")
            # Status update arrives for a trial that was NEVER planned
            orphan_ev = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "P_UNPLANNED"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "completed",
                "objective_value": 2.5
            }
            ledger.record_evaluation(orphan_ev)

            # n_trials must only count planned trials -> 0
            self.assertEqual(ledger.n_trials("run_orphan"), 0)
            # trial_rows must not create a planned trial row -> 0
            rows = ledger.trial_rows("run_orphan")
            self.assertEqual(len(rows), 0)
            # orphan must be recorded in orphan_status_events
            self.assertGreater(len(ledger.orphan_status_events), 0)

    def test_first_event_is_status_orphan(self):
        """First event being a status update (failed) without prior plan is rejected from planned trials"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "first_orphan.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_fo", "run_fo")
            first_ev = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "P_FIRST"},
                "fold_id": 1,
                "fold_definition_hash": "0" * 64,
                "status": "failed",
                "reason": "EarlyCrash"
            }
            ledger.record_evaluation(first_ev)
            self.assertEqual(ledger.n_trials("run_fo"), 0)
            self.assertEqual(len(ledger.trial_rows("run_fo")), 0)
            self.assertGreater(len(ledger.orphan_status_events), 0)

    def test_orphan_cross_instance_derivable_from_disk(self):
        """Write orphan with instance A, read with fresh instance B, orphan listed and n_trials == 0"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cross_orphan.jsonl")
            # Instance A writes orphan status update
            ledger_a = TrialLedgerV2(path)
            ledger_a.start_run("camp_cross", "run_cross")
            orphan_payload = {
                "identity": {"strategy_family": "SF", "universe_hash": "UH", "param_hash": "ORPHAN_PARAM"},
                "fold_id": 2,
                "fold_definition_hash": "0" * 64,
                "status": "completed",
                "objective_value": 3.14
            }
            ledger_a.record_evaluation(orphan_payload)
            ledger_a.finish_run()

            # Fresh Instance B reads from disk
            ledger_b = TrialLedgerV2(path)
            self.assertEqual(ledger_b.n_trials("run_cross"), 0)
            self.assertEqual(len(ledger_b.trial_rows("run_cross")), 0)
            self.assertGreater(len(ledger_b.orphan_status_events), 0)
            orphan_identities = [e.get('identity', {}).get('param_hash') for e in ledger_b.orphan_status_events]
            self.assertIn("ORPHAN_PARAM", orphan_identities)

    def test_profit_factor_inf_serialization_through_artifact_json(self):
        """Result with profit_factor=inf serializes through artifact/JSON path without error"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "pf_inf.jsonl")
            ledger = TrialLedgerV2(path)
            ledger.start_run("camp_inf", "run_inf")
            inf_eval = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "P_INF"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "planned",
            }
            ledger.record_evaluation(inf_eval)
            inf_update = {
                "identity": {"strategy_family": "S", "universe_hash": "U", "param_hash": "P_INF"},
                "fold_id": 0,
                "fold_definition_hash": "0" * 64,
                "status": "completed",
                "profit_factor": float('inf'),
                "profit_factor_status": "INFINITE_NO_LOSSES",
                "objective_value": float('inf'),
            }
            # Record evaluation with float('inf')
            ledger.record_evaluation(inf_update)

            # Commit artifact with float('inf') in metadata
            artifact_data = b"sample_artifact_content"
            meta = {
                "profit_factor": float('inf'),
                "profit_factor_status": "INFINITE_NO_LOSSES"
            }
            art_path, art_hash = ledger.commit_artifact(artifact_data, meta)
            self.assertTrue(os.path.exists(art_path))
            self.assertTrue(len(art_hash) == 64)

            # Verify chain integrity reconciles with zero errors
            reconcile_res = ledger.verify_chain_and_reconcile()
            self.assertEqual(reconcile_res["status"], "verified")
            self.assertTrue(reconcile_res["sequence_integrity"]["hash_chain_valid"])

    def test_deterministic_fingerprints(self):
        """Deterministic fingerprints: same config dict -> same identity hash"""
        hash_1 = dict_hash({'tp': [0.05], 'sl': [0.02]})
        hash_2 = dict_hash({'sl': [0.02], 'tp': [0.05]})
        self.assertEqual(hash_1, hash_2)

if __name__ == '__main__':
    unittest.main()
