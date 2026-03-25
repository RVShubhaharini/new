import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import joblib
from models.dl_unlearning_model import BankNet
from models.sisa_model import OptimizedSISA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unlearn_dl(train_df, process, encoder, scaler, num_cols, cat_cols, forget_indices_list):

    # Ensure we use the exact same feature set as training
    # We must exclude customer_id if it exists in train_df used for dimension calc
    temp_df = train_df.drop(columns=["deposit"], errors='ignore')
    if "customer_id" in temp_df.columns:
        temp_df = temp_df.drop(columns=["customer_id"])
        
    # Re-calculate inputs based on this clean df
    # Note: num_cols and cat_cols passed in might ALREADY assume dropped ID?
    # Let's check `training_agent.py`: 
    # `train_dl` returns `num_cols`, `cat_cols` derived AFTER dropping ID.
    # So `num_cols` should be safe.
    # However, `encoder.transform(train_df[cat_cols])`. 
    # If `cat_cols` does NOT contain `customer_id`, we are fine.
    # `customer_id` is int/numeric? Yes `range(10001...)`.
    # So it would be in `num_cols` if not dropped.
    # If `train_dl` dropped it, then `num_cols` returned in state DOES NOT contain it.
    # So `len(num_cols)` is 51 (presumably).
    # BUT `train_df` passed here HAS `customer_id`.
    # `encoder.transform(train_df[cat_cols])` uses `cat_cols`.
    
    # We just need to be sure `input_dim` is 51.
    input_dim = len(num_cols) + encoder.transform(train_df[cat_cols]).shape[1]
    model = BankNet(input_dim).to(device)
    
    # Load CURRENT Persistent Model
    try:
        model.load_state_dict(torch.load("models/dl_model_current.pth"))
        print("Loaded persistent DL model for unlearning.")
    except FileNotFoundError:
        print("Current model not found, falling back to baseline.")
        model.load_state_dict(torch.load("models/dl_model_baseline.pth"))

    criterion = nn.CrossEntropyLoss()

    # -------- Deletion Request Logic --------
    # Use actual indices passed from orchestrator
    forget_indices = np.array(forget_indices_list)
    
    # Identify retained indices (all others)
    all_indices = np.arange(len(train_df))
    retain_indices = np.setdiff1d(all_indices, forget_indices)

    forget_df = train_df.iloc[forget_indices]
    retain_df = train_df.iloc[retain_indices]

    X_forget_ts, y_forget_ts = process(forget_df)
    X_retain_ts, y_retain_ts = process(retain_df)

    forget_loader = DataLoader(TensorDataset(X_forget_ts, y_forget_ts), batch_size=32, shuffle=True)
    retain_loader = DataLoader(TensorDataset(X_retain_ts, y_retain_ts), batch_size=64, shuffle=True)

    unlearn_model = copy.deepcopy(model)

    # ================= SSD Dampening =================
    importance = {n: torch.zeros_like(p) for n, p in unlearn_model.named_parameters()}

    unlearn_model.eval()
    for x, y in forget_loader:
        unlearn_model.zero_grad()
        loss = criterion(unlearn_model(x.to(device)), y.to(device))
        loss.backward()

        for n, p in unlearn_model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.detach() ** 2

    for n in importance:
        if len(forget_loader) > 0:
            importance[n] /= len(forget_loader)

    with torch.no_grad():
        for n, p in unlearn_model.named_parameters():
            damp_factor = 1.0 / (1.0 + 0.05 * importance[n])
            p.mul_(damp_factor)

    # ================= Gradient Ascent =================
    optimizer_forget = optim.SGD(unlearn_model.parameters(), lr=0.0001)

    for _ in range(3):
        for x, y in forget_loader:
            optimizer_forget.zero_grad()
            loss = criterion(unlearn_model(x.to(device)), y.to(device))
            (-loss).backward()
            optimizer_forget.step()

    # ================= Recovery =================
    optimizer_recover = optim.Adam(unlearn_model.parameters(), lr=0.0002)

    for _ in range(2):
        for x, y in retain_loader:
            optimizer_recover.zero_grad()
            loss = criterion(unlearn_model(x.to(device)), y.to(device))
            loss.backward()
            optimizer_recover.step()

    # SAVE STATEFUL UPDATE
    torch.save(unlearn_model.state_dict(), "models/dl_model_current.pth")
    print("Updated DL persistent model.")

    return unlearn_model

def unlearn_sisa(X_train, y_train, forget_indices_list):

    try:
        sisa = joblib.load("models/sisa_model_current.pkl")
        print("Loaded persistent SISA model.")
    except FileNotFoundError:
        print("Current SISA model not found, loading baseline.")
        sisa = joblib.load("models/sisa_model_baseline.pkl")

    # Use exact indices passed from orchestrator
    forget_indices = np.array(forget_indices_list)

    # Efficient Unlearning (Only retrains affected shards)
    sisa.unlearn(X_train, y_train, forget_indices)

    # SAVE STATEFUL UPDATE
    joblib.dump(sisa, "models/sisa_model_current.pkl")
    print("Updated SISA persistent model.")

    return sisa

def run_unlearning(training_state, model_type, forget_indices):
    if model_type == "DL":
        print(f"Running DL Unlearning (SSD) for {len(forget_indices)} records...")
        dl_state = training_state["dl"]
        unlearned_model = unlearn_dl(
            dl_state["train_df"],
            dl_state["process"],
            dl_state["encoder"],
            dl_state["scaler"],
            dl_state["num_cols"],
            dl_state["cat_cols"],
            forget_indices
        )
        return unlearned_model
    else:
        print(f"Running ML Unlearning (SISA) for {len(forget_indices)} records...")
        sisa_state = training_state["sisa"]
        unlearned_model = unlearn_sisa(
            sisa_state["X_train"],
            sisa_state["y_train"],
            forget_indices
        )
        return unlearned_model
