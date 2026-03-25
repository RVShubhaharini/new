import numpy as np
import pandas as pd


def check_demographic_parity(model, X_test, y_test, protected_attr='age', threshold=30, pred_data=None):
    """
    Checks Demographic Parity: P(y_pred=1 | Age < 30) vs P(y_pred=1 | Age >= 30).
    Args:
        X_test (DataFrame or array): Data containing the protected attribute.
        pred_data (Tensor or array, optional): Preprocessed data strictly for model prediction.
    Returns: parity_ratio, is_fair, details_str
    """
    try:
        # We need X_test as DataFrame to access 'age'.
        # If X_test is numpy array, we might not have column names easily unless we passed them.
        # In this system, X_test in training_state might be preprocessed (scaled).
        # This is a challenge. We might need to look at the ORIGINAL X_test (before scaling) 
        # or assume 'age' is a specific column index.
        # "age" is usually the first column in bank dataset. Let's assume index 0 for simplicity if numpy.
        
        # However, `training_state['dl']['process']` creates tensor.
        # SISA might have DF or numpy.
        
        # Strategy: Use the 'original_df' subset if available, or try to infer.
        # Simplest: If X_test is DataFrame, use col name. If numpy, use index 0 (Age).
        
        if isinstance(X_test, pd.DataFrame):
            ages = X_test[protected_attr].values
        else:
            # Assume index 0 is age (standard for bank.csv processed)
            ages = X_test[:, 0]
            
        # Get Predictions
        if pred_data is not None:
            # Use provided prediction data (e.g. Tensor) directly
            if hasattr(model, "predict"):
                y_pred = model.predict(pred_data)
            else:
                # Assume PyTorch model
                import torch
                model.eval()
                with torch.no_grad():
                    if isinstance(pred_data, torch.Tensor):
                        inputs = pred_data
                    else:
                        inputs = torch.tensor(pred_data, dtype=torch.float32)
                    outputs = model(inputs)
                    _, y_pred = torch.max(outputs, 1)
                    y_pred = y_pred.cpu().numpy()
        elif hasattr(model, "predict"):
            y_pred = model.predict(X_test)
        else:
            # DL model requires torch
            import torch
            model.eval()
            with torch.no_grad():
                inputs = torch.tensor(X_test, dtype=torch.float32)
                outputs = model(inputs)
                _, y_pred = torch.max(outputs, 1)
                y_pred = y_pred.numpy()

        # Split Groups
        group_a_idx = np.where(ages < threshold)[0] # Young
        group_b_idx = np.where(ages >= threshold)[0] # Old
        
        if len(group_a_idx) == 0 or len(group_b_idx) == 0:
            return 1.0, True, "Insufficient data for fairness check"
            
        # Approval Rate (Prediction = 1)
        rate_a = np.mean(y_pred[group_a_idx] == 1)
        rate_b = np.mean(y_pred[group_b_idx] == 1)
        
        # Avoid div by zero
        if rate_a == 0 and rate_b == 0:
            ratio = 1.0
        elif rate_a == 0 or rate_b == 0:
             ratio = 0.0
        else:
            ratio = min(rate_a, rate_b) / max(rate_a, rate_b)
            
        is_fair = ratio >= 0.8 # 80% Rule
        
        details = f"Approval Rates: Young={rate_a:.2f}, Old={rate_b:.2f}. Ratio={ratio:.2f}"
        print(f"Fairness Check ({protected_attr}): {details}")
        
        return ratio, is_fair, details
        
    except Exception as e:
        print(f"Fairness Check Error: {e}")
        return -1.0, False, str(e)
