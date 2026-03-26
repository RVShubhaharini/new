import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models.sisa_model import OptimizedSISA
from models.dl_unlearning_model import train_dl, BankNet
import torch

NUM_SHARDS = 5
NUM_SLICES = 5
RANDOM_STATE = 42

import json
import os

# ... imports ...

DELETION_HISTORY_FILE = "deletion_history.json"

def load_deletion_history():
    if os.path.exists(DELETION_HISTORY_FILE):
        with open(DELETION_HISTORY_FILE, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list): # Backwards compatibility
                    return {'dl': set(data), 'ml': set()}
                return {k: set(v) for k, v in data.items()}
            except:
                return {'dl': set(), 'ml': set()}
    return {'dl': set(), 'ml': set()}

def save_deletion_history(indices, model_type="DL"):
    history = load_deletion_history()
    key = model_type.lower()
    if key not in history:
        history[key] = set()
    
    # helper to handle numpy types
    indices = [int(x) for x in indices]
    history[key].update(indices)
    
    # Save as dict of lists for JSON serialization
    serializable = {k: list(v) for k, v in history.items()}
    with open(DELETION_HISTORY_FILE, "w") as f:
        json.dump(serializable, f)
    return len(history[key])

def get_cumulative_deleted_count():
    # Return count of unique deleted incidents (just DL for simple metric, or sum)
    h = load_deletion_history()
    return len(h.get('dl', []))

def train_sisa(data_path="data/bank.csv", fit_model=True):

    df = pd.read_csv(data_path)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    X_df = df.drop(columns=["deposit"])
    if "customer_id" in X_df.columns:
        X_df = X_df.drop(columns=["customer_id"])
    # ... (rest of logic same) ...
    y = df["deposit"].map({"yes": 1, "no": 0}).values

    cat_cols = X_df.select_dtypes(include=["object"]).columns
    num_cols = X_df.select_dtypes(include=["number"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = encoder.fit_transform(X_df[cat_cols])
    X_num = X_df[num_cols].values

    X = np.hstack([X_num, X_cat])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    indices = np.arange(len(X))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )

    X_train = X[train_idx]
    X_test  = X[test_idx]

    sisa = OptimizedSISA(NUM_SHARDS, NUM_SLICES)
    
    if fit_model:
        sisa.fit(X_train, y_train)
        joblib.dump(sisa, "models/sisa_model_baseline.pkl")

    # Return total_deleted so we can report it
    return sisa, X_train, y_train, X_test, y_test, len(df)

def load_or_train_models():
    print("Checking for existing models...")
    
    # Path constants
    SISA_CURRENT = "models/sisa_model_current.pkl"
    DL_CURRENT = "models/dl_model_current.pth"
    SISA_BASELINE = "models/sisa_model_baseline.pkl"
    DL_BASELINE = "models/dl_model_baseline.pth"
    
    # SISA
    if os.path.exists(SISA_CURRENT):
        print(f"Loading existing SISA model from {SISA_CURRENT}...")
        _, X_train_sisa, y_train_sisa, X_test_sisa, y_test_sisa, total_sisa = train_sisa(fit_model=False)
        sisa_model = joblib.load(SISA_CURRENT)
    else:
        print("Loading Baseline SISA model explicitly to save RAM...")
        _, X_train_sisa, y_train_sisa, X_test_sisa, y_test_sisa, total_sisa = train_sisa(fit_model=False)
        if os.path.exists(SISA_BASELINE):
            sisa_model = joblib.load(SISA_BASELINE)
        else:
            print("Baseline not found. Forcing full cloud training (MAY CRASH OOM)...")
            sisa_model, *_ = train_sisa(fit_model=True)
        joblib.dump(sisa_model, SISA_CURRENT)
        
    # DL
    if os.path.exists(DL_CURRENT):
        print(f"Loading existing DL model from {DL_CURRENT}...")
        _, train_df_dl, test_loader_dl, process_dl, encoder_dl, scaler_dl, num_cols_dl, cat_cols_dl, total_dl = train_dl(fit_model=False)
        input_dim = len(num_cols_dl) + encoder_dl.transform(train_df_dl[cat_cols_dl]).shape[1]
        dl_model = BankNet(input_dim)
        dl_model.load_state_dict(torch.load(DL_CURRENT))
    else:
        print("Loading Baseline DL model explicitly to save RAM...")
        _, train_df_dl, test_loader_dl, process_dl, encoder_dl, scaler_dl, num_cols_dl, cat_cols_dl, total_dl = train_dl(fit_model=False)
        input_dim = len(num_cols_dl) + encoder_dl.transform(train_df_dl[cat_cols_dl]).shape[1]
        dl_model = BankNet(input_dim)
        if os.path.exists(DL_BASELINE):
            dl_model.load_state_dict(torch.load(DL_BASELINE))
        else:
            print("Baseline not found. Forcing full cloud training (MAY CRASH OOM)...")
            dl_model, *_ = train_dl(fit_model=True)
        torch.save(dl_model.state_dict(), DL_CURRENT)
    
    print("Models Ready (Stateful).")
    
    return {
        "sisa": {
            "model": sisa_model,
            "X_train": X_train_sisa,
            "y_train": y_train_sisa,
            "X_test": X_test_sisa,
            "y_test": y_test_sisa,
            "total_records": total_sisa
        },
        "dl": {
            "model": dl_model,
            "train_df": train_df_dl,
            "test_loader": test_loader_dl,
            "process": process_dl,
            "encoder": encoder_dl,
            "scaler": scaler_dl,
            "num_cols": num_cols_dl,
            "cat_cols": cat_cols_dl,
            "total_records": total_dl
        }
    }

def run_training():
    # Alias for compatibility, but logic is now stateful
    return load_or_train_models()
