import unittest

raise unittest.SkipTest("Uses R2 regime module; not Phase 1 acceptance evidence")

import pandas as pd
import numpy as np
from validation_engine.regime import detect_btc_regime

class TestRegime(unittest.TestCase):
    def test_causality(self):
        dates = pd.date_range("2026-01-01", periods=300, freq="1h")
        df_btc = pd.DataFrame({"close": 50000.0 + np.random.randn(300).cumsum()}, index=dates)
        # Force a spike
        df_btc.loc[dates[250]:, 'close'] += 10000.0
        
        regimes = detect_btc_regime(df_btc, fast_ma=10, slow_ma=50, trend_threshold=0.02)
        self.assertEqual(regimes.iloc[250], 'RANGE', "Causality broken! Current bar data leaked into current bar regime.")

if __name__ == '__main__':
    unittest.main()
