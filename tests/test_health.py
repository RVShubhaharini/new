import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from agents.fairness_agent import check_demographic_parity
from agents.drift_agent import detect_data_drift

class TestModelHealth(unittest.TestCase):

    def test_fairness_parity_perfect(self):
        # Mock model with perfect parity
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 1, 0, 0])
        
        # 4 samples: 2 age < 25, 2 age >= 25
        # We need a DataFrame with 'age'.
        data = pd.DataFrame({
            'age': [20, 22, 30, 40],
            'feature2': [1, 2, 3, 4]
        })
        
        # y_true not strictly used for parity but required by sig
        y_true = np.array([1, 1, 0, 0])
        
        # Predict is mocked, so inputs don't matter much to model
        ratio, is_fair, details = check_demographic_parity(mock_model, data, y_true)
        
        # Group 1 (Age<25): Indices 0,1. Preds: 1, 1. Prob = 1.0
        # Group 2 (Age>=25): Indices 2,3. Preds: 0, 0. Prob = 0.0
        # Wait, parity ratio = min/max. 0/1 = 0.
        # This should fail Fairness.
        
        self.assertEqual(ratio, 0.0)
        self.assertFalse(is_fair)

    def test_fairness_parity_fair(self):
        mock_model = MagicMock()
        # Grp 1: [1, 0] -> 0.5
        # Grp 2: [1, 0] -> 0.5
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
         
        data = pd.DataFrame({
            'age': [20, 22, 30, 40],
            'feature2': [1, 2, 3, 4]
        })
        y = np.array([1, 0, 1, 0])
        
        ratio, is_fair, details = check_demographic_parity(mock_model, data, y)
        
        self.assertEqual(ratio, 1.0)
        self.assertTrue(is_fair)

    def test_drift_identical(self):
        # Identical distribs should pass (p > 0.05)
        ref = np.random.normal(0, 1, (100, 2))
        curr = ref.copy()
        
        drift, p_val, idx = detect_data_drift(ref, curr)
        
        self.assertFalse(drift)
        self.assertGreater(p_val, 0.05)
        
    def test_drift_detected(self):
        # Shift mean significantly
        ref = np.random.normal(0, 1, (100, 2))
        curr = np.random.normal(5, 1, (100, 2)) # Huge drift
        
        drift, p_val, idx = detect_data_drift(ref, curr)
        
        self.assertTrue(drift)
        self.assertLess(p_val, 0.05)

    def test_fairness_with_pred_data(self):
        mock_model = MagicMock()
        # Mock predict
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        
        data = pd.DataFrame({'age': [20, 22, 30, 40]})
        pred_data = np.array([[1], [2], [3], [4]])
        
        # Call with pred_data
        ratio, is_fair, details = check_demographic_parity(mock_model, data, None, pred_data=pred_data)
        
        # Verify model was called with pred_data, NOT data
        mock_model.predict.assert_called_with(pred_data)
        self.assertEqual(ratio, 1.0)

if __name__ == '__main__':
    unittest.main()
