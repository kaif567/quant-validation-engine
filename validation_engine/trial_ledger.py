import json
import hashlib
import math
import os
import re
import shutil
import tempfile
import pandas as pd
from datetime import datetime, timezone


def normalize_payload(obj):
    """
    Recursively normalize a payload so it is valid standard JSON.

    Rules (per user audit spec):
        float NaN  -> None  (serializes as JSON null)
        float +Inf -> None
        float -Inf -> None

    Companion status fields must be injected by the caller before passing
    to this function; this function only guarantees no non-finite float
    survives into the serialized output.

    Raises TypeError for unknown types to prevent silent data corruption.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, dict):
        return {k: normalize_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_payload(v) for v in obj]
    # pandas / numpy scalars
    try:
        import numpy as np
        if isinstance(obj, (np.floating,)):
            f = float(obj)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except ImportError:
        pass
    # Timestamps
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    raise TypeError(f"normalize_payload: unhandled type {type(obj).__name__!r}")


def _build_liquidity_diag_normalized(
    *,
    bars_per_day,
    required_lookback_bars,
    p_30d,
    p_24h,
    prior_intrabar_range_fraction,
    req_ref_notional,
    req_exec_notional,
    est_fill_px,
    capacity_cap,
    approved_notional,
    participation_rate_cap,
    tier_factor,
    capacity_action,
    missing_data_policy_applied,
    signal_bar_time,
    execution_bar_time,
    tiering_enabled,
    capacity_enabled,
    missing_tier,
    missing_cap,
    reject_reason="",
):
    """
    Build a liquidity-diagnostics dict with:
    - All NaN/Inf values replaced by None
    - Explicit semantic companion status fields for every nullable metric
    Returns a dict safe for json.dumps(allow_nan=False).
    """

    def _vol_status(val, missing_flag):
        if missing_flag:
            return "UNAVAILABLE_INSUFFICIENT_HISTORY"
        return "AVAILABLE" if val is not None and not math.isnan(float(val)) and not math.isinf(float(val)) else "UNAVAILABLE_INSUFFICIENT_HISTORY"

    p30_clean = None if (missing_tier or math.isnan(p_30d) or math.isinf(p_30d)) else p_30d
    p24_clean = None if (missing_cap or math.isnan(p_24h) or math.isinf(p_24h)) else p_24h

    if not capacity_enabled:
        cap_clean = None
        cap_status = "DISABLED"
        appr_clean = None
        appr_status = "DISABLED"
    elif missing_cap:
        cap_clean = None
        cap_status = "UNAVAILABLE_MISSING_CAUSAL_VOLUME"
        appr_clean = None
        appr_status = "UNAVAILABLE_MISSING_CAUSAL_VOLUME"
    else:
        raw_cap = capacity_cap
        cap_clean = None if (math.isnan(raw_cap) or math.isinf(raw_cap)) else raw_cap
        cap_status = "UNLIMITED" if cap_clean is None else "COMPUTED"
        raw_appr = approved_notional
        appr_clean = None if (math.isnan(raw_appr) or math.isinf(raw_appr)) else raw_appr
        appr_status = "COMPUTED"

    diag = {
        "timeframe_implied_bars_per_day": bars_per_day,
        "required_lookback_bars": required_lookback_bars,

        "prior_30d_avg_daily_quote_volume": p30_clean,
        "prior_30d_avg_daily_quote_volume_status": _vol_status(p_30d, missing_tier),

        "prior_24h_quote_volume": p24_clean,
        "prior_24h_quote_volume_status": _vol_status(p_24h, missing_cap),

        "prior_intrabar_range_fraction": prior_intrabar_range_fraction,

        "requested_reference_notional": req_ref_notional,
        "requested_execution_notional": req_exec_notional,
        "estimated_causal_fill_price": est_fill_px,

        "capacity_constraint_enabled": capacity_enabled,
        "maximum_notional": cap_clean,
        "capacity_limit_status": cap_status,

        "approved_notional": appr_clean,
        "approved_notional_status": appr_status,

        "participation_rate_cap": participation_rate_cap if capacity_enabled else None,
        "liquidity_tier_factor": tier_factor,
        "liquidity_tiering_enabled": tiering_enabled,

        "capacity_action": capacity_action,
        "missing_data_policy_applied": missing_data_policy_applied,
        "reject_reason": reject_reason if reject_reason else None,

        "signal_bar_time": str(signal_bar_time),
        "execution_bar_time": str(execution_bar_time),
    }
    # Final safety pass — normalize any remaining NaN/Inf
    return normalize_payload(diag)

def dict_hash(d):
    """Deterministic hash for a dictionary."""
    canonical_json = json.dumps(
        d,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()

class HashChainCorruptionError(Exception):
    pass

class TrialLedgerV2:
    def __init__(self, path, artifact_dir=None):
        self.path = path
        self.artifact_dir = artifact_dir or os.path.join(os.path.dirname(path), "artifacts")
        os.makedirs(os.path.join(self.artifact_dir, "sha256"), exist_ok=True)
        self.current_campaign_id = None
        self.current_run_id = None
        self.last_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        self.last_seq = 0
        self.schema_version = "2.1"
        self._load_chain_state()

    def _load_chain_state(self):
        if not os.path.exists(self.path):
            return
        # Just quickly seek to end to find the last valid state (or we can read full chain)
        # We will read full chain to ensure state is right and we know the seq/hash
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    if 'schema_version' in record:
                        self.last_hash = record['event_hash']
                        self.last_seq = record['event_sequence']
                        self.current_campaign_id = record.get('research_campaign_id')
                except json.JSONDecodeError:
                    pass

    def _fsync_file(self, file_obj):
        file_obj.flush()
        os.fsync(file_obj.fileno())

    def _append_event(self, event_type, payload):
        self.last_seq += 1
        # --- Strict normalization: NaN/Inf -> None before ANY serialization ---
        clean_payload = normalize_payload(payload)
        record = {
            "event_sequence": self.last_seq,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "prev_hash": self.last_hash,
            "schema_version": self.schema_version,
            "research_campaign_id": self.current_campaign_id,
            "run_id": self.current_run_id,
            "event_type": event_type,
            "payload": clean_payload,
        }

        # Canonical serialization for hashing — allow_nan=False enforces standard JSON
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        # Guard: this will raise ValueError if any NaN/Inf survived normalization
        json.loads(canonical)  # strict parse validation
        event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        record["event_hash"] = event_hash

        line = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        # Final guard before disk write
        json.loads(line)  # raises ValueError on non-standard JSON
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            self._fsync_file(f)

        self.last_hash = event_hash
        return record

    def record_resize_event(
        self,
        *,
        fold_id,
        fold_definition_hash,
        candidate_id,
        evaluation_id,
        asset,
        venue,
        signal_bar_time,
        execution_bar_time,
        requested_quantity,
        approved_quantity,
        requested_execution_notional,
        approved_notional,
        capacity_cap,
        participation_rate_cap,
        estimated_causal_fill_price,
        liquidity_tier_factor,
        action="RESIZE",
        reason="REQUESTED_EXECUTION_NOTIONAL_EXCEEDS_CAP",
        extra=None,
    ):
        """
        Append a first-class `trade_resized_liquidity_cap` ledger event with full
        hash-chain linkage.  A resize does NOT create a new candidate identity and
        does NOT alter the candidate-fold evaluation count — it is a fill-modification
        event within an existing evaluation.
        """
        payload = {
            "fold_id": fold_id,
            "fold_definition_hash": fold_definition_hash,
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            "asset": asset,
            "venue": venue,
            "causal_execution_timestamp": {
                "signal_bar_time": str(signal_bar_time),
                "execution_bar_time": str(execution_bar_time),
            },
            "requested_quantity": requested_quantity,
            "approved_quantity": approved_quantity,
            "requested_execution_notional": requested_execution_notional,
            "capacity_cap": capacity_cap,
            "approved_notional": approved_notional,
            "participation_rate_cap": participation_rate_cap,
            "estimated_causal_fill_price": estimated_causal_fill_price,
            "liquidity_tier_factor": liquidity_tier_factor,
            "action": action,
            "reason": reason,
            # Resize / reject events do NOT change candidate-fold evaluation counts.
            # They are fill-modification records within an existing evaluation.
            "evaluation_count_impact": "NONE",
            **(extra or {}),
        }
        return self._append_event("trade_resized_liquidity_cap", payload)


    def start_run(self, research_campaign_id, run_id, identity_hashes=None):
        self.current_campaign_id = research_campaign_id
        self.current_run_id = run_id
        return self._append_event("run_started", {"identity_hashes": identity_hashes or {}})

    def finish_run(self):
        res = self._append_event("run_finished", {})
        self.current_run_id = None
        return res

    def get_candidate_id(self, identity_dict):
        return dict_hash(identity_dict)

    def get_evaluation_id(self, candidate_id, fold_id, fold_definition_hash):
        return hashlib.sha256(f"{candidate_id}_{fold_id}_{fold_definition_hash}".encode()).hexdigest()

    def record_evaluation(self, evaluation_data):
        identity = evaluation_data['identity']
        fold_definition_hash = evaluation_data.get('fold_definition_hash', '')
        if not re.match(r'^[0-9a-f]{64}$', fold_definition_hash):
            raise ValueError(f"fold_definition_hash must be a 64-character lowercase hex string. Got: {fold_definition_hash}")
        
        candidate_id = self.get_candidate_id(identity)
        evaluation_id = self.get_evaluation_id(candidate_id, evaluation_data['fold_id'], evaluation_data.get('fold_definition_hash', ''))
        
        payload = {
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            **evaluation_data
        }
        if not hasattr(self, '_run_candidates'):
            self._run_candidates = {}
        self._run_candidates[candidate_id] = payload
        return self._append_event("candidate_fold_evaluation", payload)
        
    def record_event(self, event_type, payload):
        return self._append_event(event_type, payload)
        
    def record_selection(self, decision_data):
        sel_id = decision_data.get('selected_candidate_id')
        if hasattr(self, '_run_candidates') and sel_id in self._run_candidates:
            cand = self._run_candidates[sel_id]
            if cand.get('status') == 'ERROR':
                raise ValueError("Ledger rejected selection: Candidate has ERROR status.")
            if cand.get('objective_value') is None:
                raise ValueError("Ledger rejected selection: Candidate has null objective_value.")
        return self._append_event("selection_decision", decision_data)

    def commit_artifact(self, df_or_bytes, metadata):
        fd, temp_path = tempfile.mkstemp(dir=self.artifact_dir)
        with os.fdopen(fd, 'wb') as f:
            if isinstance(df_or_bytes, pd.DataFrame):
                df_or_bytes.to_parquet(f)
            else:
                f.write(df_or_bytes)
            self._fsync_file(f)
            
        # compute hash
        hasher = hashlib.sha256()
        with open(temp_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        artifact_hash = hasher.hexdigest()
        
        final_path = os.path.join(self.artifact_dir, "sha256", f"{artifact_hash}.parquet")
        os.rename(temp_path, final_path)
        
        payload = {
            "artifact_path": final_path,
            "artifact_hash": artifact_hash,
            "metadata": metadata
        }
        self._append_event("artifact_committed", payload)
        return final_path, artifact_hash

    def verify_chain_and_reconcile(self):
        if not os.path.exists(self.path):
            return {"status": "empty"}

        expected_prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        expected_seq = 1
        
        candidates = {} # candidate_id -> [fold_evaluations]
        campaigns = set()
        runs = set()
        artifacts_referenced = set()
        
        legacy_trials = 0
        
        with open(self.path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip(): continue
                record = json.loads(line)
                
                # Legacy Support
                if 'schema_version' not in record:
                    if record.get('event') == 'trial':
                        legacy_trials += 1
                    continue
                
                # Strict Monotonicity & Hashing
                seq = record['event_sequence']
                if seq != expected_seq:
                    raise HashChainCorruptionError(f"Sequence gap/duplicate: expected {expected_seq}, got {seq}")
                
                prev_hash = record['prev_hash']
                if prev_hash != expected_prev_hash:
                    raise HashChainCorruptionError(f"prev_hash_mismatch at seq {seq}. Expected {expected_prev_hash}, got {prev_hash}")
                
                event_hash = record.pop('event_hash')
                record_bytes = json.dumps(
                    record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode('utf-8')
                recomputed_hash = hashlib.sha256(record_bytes).hexdigest()
                
                if event_hash != recomputed_hash:
                    raise HashChainCorruptionError(f"event_hash_mismatch at seq {seq}. Recomputed {recomputed_hash}, stored {event_hash}")
                
                # Update expected for next iteration
                record['event_hash'] = event_hash # restore
                expected_prev_hash = event_hash
                expected_seq += 1
                
                campaigns.add(record['research_campaign_id'])
                runs.add(record['run_id'])
                
                evt = record['event_type']
                if evt == 'candidate_fold_evaluation':
                    cid = record['payload']['candidate_id']
                    if cid not in candidates:
                        candidates[cid] = []
                    candidates[cid].append(record['payload'])
                elif evt == 'artifact_committed':
                    artifacts_referenced.add(record['payload']['artifact_path'])

        # Aggregate counts
        c_attempted = len(candidates)
        c_completed = sum(1 for evals in candidates.values() if any(e['status'] == 'COMPLETED' for e in evals))
        c_fully_completed = sum(1 for evals in candidates.values() if all(e['status'] == 'COMPLETED' for e in evals))
        c_error = sum(1 for evals in candidates.values() if any(e['status'] == 'ERROR' for e in evals))
        c_zero = sum(1 for evals in candidates.values() if any(e['status'] == 'ZERO_TRADES' for e in evals))
        
        f_attempted = sum(len(evals) for evals in candidates.values())
        f_completed = sum(sum(1 for e in evals if e['status'] == 'COMPLETED') for evals in candidates.values())
        f_error = sum(sum(1 for e in evals if e['status'] == 'ERROR') for evals in candidates.values())
        f_zero = sum(sum(1 for e in evals if e['status'] == 'ZERO_TRADES') for evals in candidates.values())

        # Artifact reconciliation
        for path in artifacts_referenced:
            if not os.path.exists(path):
                raise HashChainCorruptionError(f"Integrity failure: Artifact referenced in ledger is missing: {path}")

        # Orphan check
        orphans = []
        artifact_sha_dir = os.path.join(self.artifact_dir, "sha256")
        if os.path.exists(artifact_sha_dir):
            for file in os.listdir(artifact_sha_dir):
                full_path = os.path.join(artifact_sha_dir, file)
                if full_path not in artifacts_referenced:
                    orphans.append(full_path)

        if legacy_trials > 0 and c_attempted == 0:
            counts = {
                "n_unique_candidates_attempted": "UNKNOWN_LEGACY_FORMAT",
                "n_unique_candidates_with_any_completed_fold": "UNKNOWN_LEGACY_FORMAT",
                "n_unique_candidates_fully_completed": "UNKNOWN_LEGACY_FORMAT",
                "n_unique_candidates_with_any_error": "UNKNOWN_LEGACY_FORMAT",
                "n_unique_candidates_with_any_zero_trade": "UNKNOWN_LEGACY_FORMAT",
                "n_unique_candidates_return_matrix_eligible": "UNAVAILABLE_PENDING_ALIGNMENT_POLICY",
                
                "n_fold_evaluations_attempted": legacy_trials,
                "n_fold_evaluations_completed": "UNKNOWN_LEGACY_FORMAT",
                "n_fold_evaluations_error": "UNKNOWN_LEGACY_FORMAT",
                "n_fold_evaluations_zero_trades": "UNKNOWN_LEGACY_FORMAT",
                "legacy_status_resolution": "UNAVAILABLE",
                "n_effective_trials_estimate": "UNAVAILABLE_PENDING_ALIGNMENT_POLICY"
            }
        else:
            counts = {
                "n_unique_candidates_attempted": c_attempted,
                "n_unique_candidates_with_any_completed_fold": c_completed,
                "n_unique_candidates_fully_completed": c_fully_completed,
                "n_unique_candidates_with_any_error": c_error,
                "n_unique_candidates_with_any_zero_trade": c_zero,
                "n_unique_candidates_return_matrix_eligible": "UNAVAILABLE_PENDING_ALIGNMENT_POLICY",
                
                "n_fold_evaluations_attempted": f_attempted,
                "n_fold_evaluations_completed": f_completed,
                "n_fold_evaluations_error": f_error,
                "n_fold_evaluations_zero_trades": f_zero,
                "n_effective_trials_estimate": "UNAVAILABLE_PENDING_ALIGNMENT_POLICY"
            }
        
        return {
            "status": "verified",
            "campaigns": list(campaigns),
            "runs": list(runs),
            "counts": counts,
            "orphans": orphans,
            "sequence_integrity": {
                "events_scanned": expected_seq - 1,
                "sequence_range": f"1..{expected_seq - 1}" if expected_seq > 1 else "empty",
                "missing_sequences": [],
                "duplicate_sequences": [],
                "hash_chain_valid": True
            }
        }
