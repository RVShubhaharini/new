import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from agents.certificate_agent import save_metrics_json
import json

def test_save():
    metrics = {
        "float_np": np.float32(0.85),
        "int_np": np.int64(42),
        "bool_np": np.bool_(True),
        "list_np": np.array([1, 2, 3]),
        "normal": "string",
        "nested": {"a": np.int32(1)}
    }
    
    print("Testing save with numpy types...")
    save_metrics_json(metrics, "dashboard/test_metrics.json")
    
    with open("dashboard/test_metrics.json", "r") as f:
        data = json.load(f)
        print("Loaded back successfully:", data)

if __name__ == "__main__":
    test_save()
