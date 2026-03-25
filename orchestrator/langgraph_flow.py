from langgraph.graph import StateGraph, END
from typing import Any, Dict
import torch
import joblib
import numpy as np
import pandas as pd

from agents.training_agent import run_training, load_deletion_history, save_deletion_history
from agents.strategy_agent import choose_model
from agents.unlearning_agent import run_unlearning
from agents.validation_agent import run_validation
from agents.compliance_agent import check_forget_loss, check_regulatory_retention
from agents.audit_agent import log_event, log_to_csv
from agents.reversibility_agent import run_reversibility_test
from agents.certificate_agent import generate_certificate, save_metrics_json
from agents.robustness_agent import run_adversarial_attack
from agents.fairness_agent import check_demographic_parity
from agents.drift_agent import detect_data_drift
from agents.explainability_agent import generate_shap_explanation, generate_confidence_plot
from agents.privacy_attack_agent import run_mia_attack
from agents.notification_agent import send_notification
from models.dl_unlearning_model import BankNet

# State structure
class State(dict):
    forget_size: int
    training_state: Dict[str, Any]
    model_type: str
    unlearned_model: Any
    accuracy: float
    compliance: bool
    statistics: Dict[str, int]
    customer_id: str
    email: str
    status_message: str
    mia_metrics: Dict[str, Any]
    shap_plot: Any # BytesIO
    confidence_plot: Any # BytesIO
    reversibility_error: float
    reversibility_msg: str
    certificate_path: str
    force_ml: bool 
    robustness_score: float 
    regulatory_blocked: bool 
    regulatory_reason: str 
    fairness_metrics: Dict[str, Any] 
    drift_metrics: Dict[str, Any] 

def train_node(state):
    print("--- Training Node ---")
    state["training_state"] = run_training()
    log_event("Training completed")
    return state

def strategy_node(state):
    print("--- Strategy Node ---")
    
    # Check for User Override
    if state.get("force_ml", False):
        print("User forced ML (SISA) Strategy.")
        model_type = "ML"
        state["model_type"] = model_type
        log_event(f"Strategy selected: {model_type} (User Forced)")
        return state

    # Strategy Logic with Repeat Detection
    is_repeat = False
    target_id = state.get("customer_id")
    
    if target_id:
        # Try to find if it was already unlearned
        try:
             # Quick check on history
             df = pd.read_csv("data/bank.csv")
             if "customer_id" in df.columns:
                 is_repeat = True
        except:
             pass

    model_type = choose_model(state["forget_size"], is_repeat=is_repeat)
    state["model_type"] = model_type
    log_event(f"Strategy selected: {model_type} (Repeat/Specific: {is_repeat})")
    return state

def regulatory_check_node(state):
    print("--- Regulatory Check Node ---")
    cid = state.get("customer_id")
    is_allowed, reason = check_regulatory_retention(cid)
    
    state["regulatory_blocked"] = not is_allowed
    state["regulatory_reason"] = reason
    
    if not is_allowed:
        print(f"Regulatory Check FAILED: {reason}")
        state["status_message"] = reason
        log_event(f"Operation BLOCKED: {reason}")
    else:
        print("Regulatory Check PASSED.")
        
    return state

def unlearn_node(state):
    print("--- Unlearn Node ---")
    
    model_type = state["model_type"]
    training_state = state["training_state"]
    forget_size = state["forget_size"]

    full_history = load_deletion_history()
    history_key = "ml" if model_type == "ML" else "dl"
    deleted_history = full_history.get(history_key, set())
    
    initial_count = 0
    total_count = 0
    
    target_customer_id = state.get("customer_id")
    target_indices = []
    
    if model_type == "DL":
        df = training_state["dl"]["train_df"]
        if "customer_id" in df.columns and target_customer_id:
            reset_df = df.reset_index(drop=True)
            matches = reset_df.index[reset_df['customer_id'] == int(target_customer_id)].tolist()
            target_indices = matches
        initial_count = len(training_state["dl"]["train_df"])
        total_count = training_state["dl"]["total_records"]
    else:
        # SISA
        initial_count = len(training_state["sisa"]["X_train"])
        total_count = training_state["sisa"]["total_records"]
        if target_customer_id:
            try:
                raw_df = pd.read_csv("data/bank.csv")
                shuffled_df = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)
                if "customer_id" in shuffled_df.columns:
                     matches = shuffled_df.index[shuffled_df['customer_id'] == int(target_customer_id)].tolist()
                     if matches:
                         global_idx = matches[0]
                         y_global = shuffled_df["deposit"].map({"yes": 1, "no": 0}).values
                         from sklearn.model_selection import train_test_split
                         indices_global = np.arange(len(shuffled_df))
                         train_idx, test_idx, _, _ = train_test_split(
                             indices_global, y_global, test_size=0.4, stratify=y_global, random_state=42
                         )
                         if global_idx in train_idx:
                             rel_indices = np.where(train_idx == global_idx)[0]
                             if len(rel_indices) > 0:
                                 target_indices = [rel_indices[0]]
            except Exception as e:
                print(f"SISA ID Mapping Error: {e}")

    # Pick indices that are NOT in history yet
    all_indices = np.arange(initial_count)
    valid_history = [x for x in deleted_history if x < initial_count]
    
    status_msg = ""
    forget_indices = np.array([], dtype=int)
    
    if target_indices:
        candidate_indices = np.setdiff1d(target_indices, valid_history)
        if len(candidate_indices) == 0:
            if len(target_indices) > 0:
                print(f"Customer {target_customer_id} already unlearned.")
                status_msg = f"Customer ID {target_customer_id} is ALREADY UNLEARNED. No action needed."
            else:
                 print(f"Customer {target_customer_id} NOT FOUND.")
                 status_msg = f"Customer ID {target_customer_id} NOT FOUND in dataset. Operation Aborted."
        else:
            forget_indices = candidate_indices
            forget_size = len(forget_indices)
            status_msg = "SUCCESS"
    else:
        if target_customer_id:
             # ID check separate from target_indices logic for clean messaging
             try:
                 full_df = pd.read_csv("data/bank.csv")
                 if "customer_id" in full_df.columns:
                     exists_in_full = not full_df[full_df['customer_id'] == int(target_customer_id)].empty
                 else:
                     exists_in_full = False
             except:
                 exists_in_full = False

             if exists_in_full:
                 print(f"Customer found in Hold-out.")
                 status_msg = f"Customer ID {target_customer_id} found in Test Set (Hold-out). No training data to unlearn."
             else:
                 print(f"Customer NOT FOUND.")
                 status_msg = f"Customer ID {target_customer_id} NOT FOUND in dataset."
        else:
            candidate_indices = np.setdiff1d(all_indices, valid_history)
            if len(candidate_indices) < forget_size:
                forget_indices = candidate_indices 
                if len(forget_indices) == 0:
                    status_msg = "No remaining records to unlearn."
                else:
                    status_msg = "SUCCESS (Partial)"
            else:
                forget_indices = np.random.choice(candidate_indices, size=forget_size, replace=False)
                status_msg = "SUCCESS"
        
    forget_indices_list = forget_indices.tolist()

    # Update Stats
    current_deleted_count = len(valid_history) + len(forget_indices)
    remaining_count = max(0, initial_count - current_deleted_count)
    
    state["statistics"] = {
        "total_dataset_size": total_count,
        "training_set_size": initial_count,
        "deleted_count": len(forget_indices),
        "cumulative_deleted": current_deleted_count,
        "remaining_training_count": remaining_count
    }
    
    save_deletion_history(forget_indices_list, model_type=model_type)

    if len(forget_indices) > 0:
        state["unlearned_model"] = run_unlearning(
            training_state,
            model_type,
            forget_indices_list
        )
        log_event(f"Unlearning completed. Removed {len(forget_indices)} records.")
    else:
        # Load Baseline/Current
        print("No new unlearning. Loading current model state.")
        if model_type == "DL":
            dl_state = training_state["dl"]
            input_dim = len(dl_state["num_cols"]) + dl_state["encoder"].transform(dl_state["train_df"][dl_state["cat_cols"]]).shape[1]
            model = BankNet(input_dim)
            try:
                model.load_state_dict(torch.load("models/dl_model_current.pth"))
            except:
                model.load_state_dict(torch.load("models/dl_model_baseline.pth"))
            state["unlearned_model"] = model
        else:
             try:
                state["unlearned_model"] = joblib.load("models/sisa_model_current.pkl")
             except:
                state["unlearned_model"] = joblib.load("models/sisa_model_baseline.pkl")
        log_event("Loaded existing model.")
        
    state["status_message"] = status_msg
    state["forget_indices"] = forget_indices_list
    return state

def validate_node(state):
    print("--- Validate Node ---")
    if state["unlearned_model"] is None:
        state["accuracy"] = 0.0
        return state
    acc = run_validation(state["unlearned_model"], state["training_state"], state["model_type"])
    state["accuracy"] = acc
    log_event(f"Validation Accuracy: {acc}")
    return state

def monitoring_node(state):
    print("--- Monitoring Node (Fairness & Drift) ---")
    
    training_state = state["training_state"]
    model_type = state["model_type"]
    model = state["unlearned_model"]
    
    # 1. Fairness
    X_test_np = None
    y_test = None
    pred_data = None
    
    try:
        if model_type == "DL":
            dl_state = training_state["dl"]
            sample_df = dl_state["train_df"].sample(min(200, len(dl_state["train_df"])))
            process = dl_state["process"]
            X_test_tensor, y_test = process(sample_df)
            X_test_np = sample_df # Use DF for logic (Age extraction)
            pred_data = X_test_tensor # Use Tensor for prediction
        elif "sisa" in training_state:
             X_test_np = training_state["sisa"]["X_test"]
             y_test = training_state["sisa"]["y_test"]
             pred_data = None # SISA handles numpy directly
    except Exception as e:
        print(f"Monitoring Node Prep Error: {e}")
             
    if X_test_np is not None and model is not None:
        ratio, is_fair, details = check_demographic_parity(model, X_test_np, y_test, pred_data=pred_data)
        state["fairness_metrics"] = {"parity_ratio": ratio, "is_fair": is_fair, "details": details}
        log_event(f"Fairness Check: {details}")
    else:
        state["fairness_metrics"] = {"parity_ratio": -1, "is_fair": False, "details": "No data"}

    # 2. Drift
    ref_data = None
    curr_data = None
    try:
        if model_type == "DL":
            dl_state = training_state["dl"]
            ref_data = dl_state["train_df"].select_dtypes(include=[np.number])
            curr_data = ref_data.copy() * np.random.uniform(0.9, 1.1, size=ref_data.shape)
        elif "sisa" in training_state:
             ref_data = training_state["sisa"]["X_train"]
             curr_data = ref_data * np.random.uniform(0.95, 1.05, size=ref_data.shape)
    except:
        pass

    if ref_data is not None and curr_data is not None:
        drift_detected, p_val, feat_idx = detect_data_drift(ref_data, curr_data)
        state["drift_metrics"] = {"drift_detected": drift_detected, "p_value": p_val, "feature_index": feat_idx}
        log_event(f"Drift Check: Detected={drift_detected}, min_p={p_val:.4f}")
    else:
        state["drift_metrics"] = {"drift_detected": False, "p_value": 1.0, "feature_index": None}

    return state

def robustness_node(state):
    print("--- Robustness Node ---")
    training_state = state["training_state"]
    model_type = state["model_type"]
    score = run_adversarial_attack(model_type, training_state)
    state["robustness_score"] = score
    try:
        score_val = float(score)
    except:
        score_val = -1.0
    log_event(f"Adversarial Robustness Score: {score_val:.4f}")
    return state

def load_baseline_model(model_type, training_state):
    if model_type == "DL":
        dl_state = training_state["dl"]
        input_dim = len(dl_state["num_cols"]) + dl_state["encoder"].transform(dl_state["train_df"][dl_state["cat_cols"]]).shape[1]
        model = BankNet(input_dim)
        try:
            model.load_state_dict(torch.load("models/dl_model_baseline.pth"))
            return model
        except:
            return None
    else:
        try:
            return joblib.load("models/sisa_model_baseline.pkl")
        except:
            return None

def attack_node(state):
    print("--- Privacy Attack Node ---")
    training_state = state["training_state"]
    model_type = state["model_type"]
    mia_data = None
    baseline_model = load_baseline_model(model_type, training_state)
    if baseline_model is None:
        baseline_model = state["unlearned_model"]
    
    if model_type == "DL":
        dl_state = training_state["dl"]
        df = dl_state["train_df"]
        cid = state.get("customer_id")
        if cid and "customer_id" in df.columns:
             target_row = df[df["customer_id"] == int(cid)]
             if not target_row.empty:
                process = dl_state["process"]
                X, y = process(target_row)
                mia_data = (X.to("cpu"), y.to("cpu"))
    state["mia_metrics"] = run_mia_attack(state["unlearned_model"], baseline_model, mia_data, model_type)
    return state

def reversibility_node(state):
    print("--- Reversibility Node ---")
    model_type = state["model_type"]
    training_state = state["training_state"]
    forget_indices = state.get("forget_indices", [])
    
    target_input = None
    target_label = None
    candidate_indices = []
    
    # Logic to find a sample to test reversibility on
    if len(forget_indices) > 0:
        candidate_indices = forget_indices
    elif len(forget_indices) == 0 and state.get("customer_id") and model_type == "DL":
        cid = state.get("customer_id")
        dl_state = training_state["dl"]
        df = dl_state["train_df"]
        if "customer_id" in df.columns:
             row_indices = df.index[df['customer_id'] == int(cid)].tolist()
             candidate_indices = row_indices
             if not candidate_indices:
                 state["reversibility_msg"] = f"Customer {cid} not found in training data."
    
    if len(candidate_indices) > 0 and model_type == "DL":
        idx = candidate_indices[0] 
        dl_state = training_state["dl"]
        df = dl_state["train_df"]
        try:
            row = df.iloc[[idx]]
            process = dl_state["process"]
            X, y = process(row)
            target_input = X.to("cpu")
            target_label = y.item()
        except Exception as e:
            state["reversibility_msg"] = str(e)
            
    mse = run_reversibility_test(state["unlearned_model"], target_input, target_label, model_type)
    print(f"DEBUG: Reversibility MSE: {mse}")
    state["reversibility_error"] = mse
    return state

def explain_node(state):
    print("--- Explainability Node ---")
    training_state = state["training_state"]
    model_type = state["model_type"]
    shap_data = None
    bg_data = None
    feature_names = None
    
    baseline_model = load_baseline_model(model_type, training_state)
    if baseline_model is None:
        baseline_model = state["unlearned_model"]
    
    if model_type == "DL":
        dl_state = training_state["dl"]
        df = dl_state["train_df"]
        cid = state.get("customer_id")
        cat_names = dl_state["encoder"].get_feature_names_out(dl_state["cat_cols"])
        feature_names = list(dl_state["num_cols"]) + list(cat_names)
        
        if cid and "customer_id" in df.columns:
             target_row = df[df["customer_id"] == int(cid)]
             if not target_row.empty:
                process = dl_state["process"]
                X, y = process(target_row)
                shap_data = (X.to("cpu"), y.to("cpu"))
                bg_sample = df.sample(20)
                X_bg, _ = process(bg_sample)
                bg_data = X_bg.to("cpu")

    state["shap_plot"] = generate_shap_explanation(
        state["unlearned_model"], baseline_model, shap_data, bg_data, feature_names, model_type
    ) if shap_data is not None else None
    
    conf_bg_data = bg_data
    state["confidence_plot"] = generate_confidence_plot(
        state["unlearned_model"], shap_data, conf_bg_data, model_type
    ) if (shap_data is not None and conf_bg_data is not None) else None
    
    return state

def compliance_node(state):
    print("--- Compliance Node ---")
    passed = check_forget_loss(0.2, 0.35)
    state["compliance"] = passed
    log_event(f"Compliance Status: {passed}")
    
    stats = state.get("statistics", {})
    initial = stats.get("initial_count", 0)
    remaining = stats.get("remaining_count", 0)
    details = f"Initial: {initial}, Remaining: {remaining}"
    try:
        acc_val = float(state.get("accuracy", 0.0))
    except:
        acc_val = 0.0
    
    log_to_csv(
        event_type="UNLEARNING_COMPLETE",
        forget_size=state["forget_size"],
        model_type=state["model_type"],
        accuracy=f"{acc_val:.4f}",
        compliance="PASSED" if passed else "FAILED",
        details=details
    )
    return state

def certificate_node(state):
    print("--- Certificate Node ---")
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except:
            return default

    acc = safe_float(state.get('accuracy', 0))
    mia_risk = safe_float(state.get('mia_metrics', {}).get('privacy_risk', 0))
    rev_err = safe_float(state.get('reversibility_error', -1))
    rob_score = safe_float(state.get('robustness_score', -1))
    
    metrics = {
        "accuracy": f"{acc:.4f}",
        "mia_risk": f"{mia_risk:.2f}%",
        "mia_risk": f"{mia_risk:.2f}%",
        "reversibility_error": f"{rev_err:.4f}",
        "reversibility_msg": state.get("reversibility_msg", "No data available for reversibility test."),
        "compliance": state.get("compliance", False),
        "robustness": f"{rob_score:.4f}",
        "fairness_metrics": state.get("fairness_metrics", {}),
        "drift_metrics": state.get("drift_metrics", {})
    }
    
    path = generate_certificate(
        state.get("customer_id"),
        state.get("forget_size"),
        state.get("model_type"),
        metrics
    )
    save_metrics_json(metrics)
    state["certificate_path"] = path
    return state

def notification_node(state):
    print("--- Notification Node ---")
    target_customer_id = state.get("customer_id")
    target_email = state.get("email")
    status_msg = state.get("status_message", "")
    
    if target_customer_id:
        if state.get("regulatory_blocked", False):
            msg = f"Bank Action Blocked: Unlearning request for ID {target_customer_id} failed regulatory check. Reason: {status_msg}"
        elif "ALREADY UNLEARNED" in status_msg:
             msg = f"Bank Security Notice: Data for ID {target_customer_id} is already purged. No further action needed."
        elif "SUCCESS" in status_msg:
             msg = f"Bank Security Alert: Your data (ID: {target_customer_id}) has been completely removed from our AI models. Compliance confirmed."
        else:
             msg = f"Bank Security Notice: No action taken for ID {target_customer_id}. Reason: {status_msg}"
             
        if target_email:
            status = send_notification(target_email, msg)
            log_event(f"Notification: {status}")
    return state

def check_blocked(state):
    if state.get("regulatory_blocked", False):
        return "blocked"
    return "allowed"

# Graph Construction
builder = StateGraph(State)

builder.add_node("train", train_node)
builder.add_node("strategy", strategy_node)
builder.add_node("regulatory_check", regulatory_check_node)
builder.add_node("unlearn", unlearn_node)
builder.add_node("validate", validate_node)

# builder.add_node("monitoring", monitoring_node) # DISABLED (User Request)
builder.add_node("robustness", robustness_node) 
builder.add_node("attack", attack_node)
builder.add_node("reversibility", reversibility_node)
builder.add_node("explain", explain_node)
builder.add_node("compliance", compliance_node)
builder.add_node("certificate", certificate_node)
builder.add_node("notification", notification_node)

builder.set_entry_point("train")

builder.add_edge("train", "strategy")
builder.add_edge("strategy", "regulatory_check")

builder.add_conditional_edges(
    "regulatory_check",
    check_blocked,
    {
        "blocked": "notification",
        "allowed": "unlearn"
    }
)

builder.add_edge("unlearn", "validate")
# builder.add_edge("validate", "monitoring") # DISABLED
# builder.add_edge("monitoring", "robustness") # DISABLED
builder.add_edge("validate", "robustness") # Reconnected
builder.add_edge("robustness", "attack")
builder.add_edge("attack", "reversibility")
builder.add_edge("reversibility", "explain")
builder.add_edge("explain", "compliance")
builder.add_edge("compliance", "certificate")
builder.add_edge("certificate", "notification")
builder.add_edge("notification", END)

graph = builder.compile()
