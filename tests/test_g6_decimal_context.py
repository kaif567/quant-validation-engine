import unittest
import json
import hashlib
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, getcontext, localcontext

CONFIGURED_G6_DECIMAL_PRECISION = 28
RETURN_Q = Decimal("0.000000000000000001")

def q_return(value: Decimal) -> Decimal:
    return value.quantize(RETURN_Q, rounding=ROUND_HALF_UP)

def run_g6_bootstrap(folds_returns, rng_seed, block_length, resample_count):
    with localcontext() as ctx:
        ctx.prec = CONFIGURED_G6_DECIMAL_PRECISION
        ctx.rounding = ROUND_HALF_UP
        
        assert getcontext().prec == CONFIGURED_G6_DECIMAL_PRECISION
        assert getcontext().rounding == ROUND_HALF_UP
        
        rng = np.random.RandomState(rng_seed)
        
        full_bootstrap_statistics = []
        for _ in range(resample_count):
            fold_statistics = []
            for fold_returns in folds_returns:
                n_f = len(fold_returns)
                sampled_returns = []
                while len(sampled_returns) < n_f:
                    start_idx = rng.randint(0, n_f)
                    end_idx = min(n_f, start_idx + block_length)
                    sampled_returns.extend(fold_returns[start_idx:end_idx])
                sampled_returns = sampled_returns[:n_f]
                
                fold_statistic = Decimal("1")
                for sr in sampled_returns:
                    fold_statistic = q_return(fold_statistic * (Decimal("1") + sr))
                fold_statistic = q_return(fold_statistic - Decimal("1"))
                fold_statistics.append(fold_statistic)
                
            aggregate_statistic = q_return(
                sum(fold_statistics, Decimal("0")) / Decimal(len(fold_statistics))
            )
            full_bootstrap_statistics.append(aggregate_statistic)
            
        sorted_stats = sorted(full_bootstrap_statistics)
        q05 = sorted_stats[int(0.05 * resample_count)]
        g6_status = "PASS" if q05 > Decimal("0") else "FAIL"
        
        full_statistics_artifact = {
            "schema_version": "g6_full_statistics_v1",
            "formula_version": "v1",
            "resample_count": len(full_bootstrap_statistics),
            "rng_seed": rng_seed,
            "block_length": block_length,
            "fold_weighting_policy": "EQUALLY_WEIGHTED_ARITHMETIC_MEAN",
            "decimal_precision": getcontext().prec,
            "decimal_rounding": "ROUND_HALF_UP",
            "return_quantizer": format(RETURN_Q, "f"),
            "statistics": [format(value, "f") for value in full_bootstrap_statistics],
        }

        full_statistics_serialized = json.dumps(
            full_statistics_artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

        full_hash = hashlib.sha256(full_statistics_serialized).hexdigest()
        
        artifact = {
            "g6_status": g6_status,
            "full_bootstrap_artifact_hash": full_hash
        }
        
        return artifact, full_hash, q05

class TestG6DecimalContext(unittest.TestCase):
    def test_g6_decimal_reproducibility(self):
        fail_fold_returns = [
            [Decimal("-0.01"), Decimal("0.02"), Decimal("-0.03"), Decimal("-0.01")] * 10,
            [Decimal("-0.02"), Decimal("0.01"), Decimal("-0.04"), Decimal("-0.01")] * 10,
        ]
        
        art_fail_a, hash_fail_a, q05_fail_a = run_g6_bootstrap(fail_fold_returns, 42, 10, 1000)
        art_fail_b, hash_fail_b, q05_fail_b = run_g6_bootstrap(fail_fold_returns, 42, 10, 1000)
        art_fail_c, hash_fail_c, q05_fail_c = run_g6_bootstrap(fail_fold_returns, 43, 10, 1000)
        
        self.assertEqual(hash_fail_a, hash_fail_b)
        self.assertNotEqual(hash_fail_a, hash_fail_c)
        
        self.assertTrue(q05_fail_a < Decimal("0"))
        self.assertEqual(art_fail_a["g6_status"], "FAIL")

if __name__ == '__main__':
    unittest.main()
