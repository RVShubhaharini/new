from scipy.stats import ks_2samp
import numpy as np
import pandas as pd

def detect_data_drift(reference_data, current_data, threshold_p=0.05):
    """
    Detects Data Drift using Kolmogorov-Smirnov (KS) Test.
    Returns: drift_detected (bool), p_value (min across features), failing_feature (int/str)
    """
    try:
        # Validations
        if reference_data is None or current_data is None:
            return False, 1.0, None
            
        # Ensure numpy
        if isinstance(reference_data, pd.DataFrame):
            ref = reference_data.values
        else:
            ref = reference_data
            
        if isinstance(current_data, pd.DataFrame):
            curr = current_data.values
        else:
            curr = current_data
            
        # Check dimensions
        if ref.shape[1] != curr.shape[1]:
            print("Drift Check Skipped: Feature mismatch.")
            return False, 1.0, None
            
        min_p = 1.0
        drift_feat = None
        
        # Iterate features
        for i in range(ref.shape[1]):
            stat, p_val = ks_2samp(ref[:, i], curr[:, i])
            
            if p_val < min_p:
                min_p = p_val
                drift_feat = i
                
        is_drift = min_p < threshold_p
        
        if is_drift:
            print(f"Drift Detected! Min P-Value: {min_p:.4f} at Feature {drift_feat}")
        else:
            print(f"Drift Check Passed. Min P-Value: {min_p:.4f}")
            
        return is_drift, min_p, drift_feat

    except Exception as e:
        print(f"Drift Check Error: {e}")
        return False, 1.0, None
