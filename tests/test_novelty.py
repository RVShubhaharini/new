import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.compliance_agent import check_regulatory_retention
from agents.robustness_agent import run_ml_noise_attack
import numpy as np

def test_compliance():
    print("--- Testing Compliance Agent ---")
    
    # Test allowed ID
    allowed, reason = check_regulatory_retention(10001)
    print(f"ID 10001: Allowed={allowed}, Reason={reason}")
    assert allowed == True
    
    # Test Blocked ID (AML)
    blocked_aml, reason_aml = check_regulatory_retention(10099)
    print(f"ID 10099: Allowed={blocked_aml}, Reason={reason_aml}")
    assert blocked_aml == False
    assert "AML" in reason_aml
    
    # Test Blocked ID (Age)
    blocked_age, reason_age = check_regulatory_retention(95000)
    print(f"ID 95000: Allowed={blocked_age}, Reason={reason_age}")
    assert blocked_age == False
    assert "years old" in reason_age
    
    print("Compliance Tests PASSED.")

class MockModel:
    def predict(self, X):
        # Always return 0
        return np.zeros(X.shape[0])

def test_robustness():
    print("\n--- Testing Robustness Agent ---")
    
    # Mocking SISA state
    X_test = np.random.rand(10, 5)
    y_test = np.zeros(10) # Ground truth
    
    training_state = {
        "sisa": {
            "model": MockModel(),
            "X_test": X_test,
            "y_test": y_test
        }
    }
    
    score = run_ml_noise_attack(training_state, noise_level=0.1)
    print(f"Robustness Score: {score}")
    assert score >= 0.0 and score <= 1.0
    
    print("Robustness Tests PASSED.")

if __name__ == "__main__":
    test_compliance()
    test_robustness()
