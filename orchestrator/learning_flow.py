from langgraph.graph import StateGraph, END
from typing import Any, Dict
from agents.learning_agent import incremental_train_sisa, incremental_train_dl
from agents.validation_agent import run_validation_simple
from agents.audit_agent import log_event

# State
class LearningState(dict):
    new_data: Dict[str, Any]
    model_type: str
    status_message: str
    success: bool
    accuracy: float

def determine_strategy_node(state):
    print("--- Learning Strategy Node ---")
    # Default to DL if not specified, or robust selection?
    # For now, we learn on BOTH or let user choose? 
    # Usually you add data to the system, so both models should learn it to stay synced.
    # The plan says "SISA Append / DL Fine-tune".
    # Let's try to update the PRIMARY model type or both.
    # For simplicity in this flow, we'll respect `model_type` passed from UI.
    
    m_type = state.get("model_type", "DL")
    log_event(f"Learning Strategy: Incremental update for {m_type}")
    return state

def learn_node(state):
    print("--- Incremental Learning Node ---")
    new_data = state["new_data"]
    m_type = state["model_type"]
    
    success = False
    msg = ""
    
    if m_type == "DL":
        success, msg = incremental_train_dl(new_data)
    else:
        success, msg = incremental_train_sisa(new_data)
        
    state["success"] = success
    state["status_message"] = msg
    log_event(f"Learning Result: {msg}")
    return state

def validate_learning_node(state):
    print("--- Validate Learning Node ---")
    # We should check if the model now predicts the NEW data correctly?
    # Or general accuracy?
    # Let's do a quick general validation if possible.
    # But run_validation requires `training_state` which is heavy to load here.
    # Let's just trust the training loop loss for now, or successful execution.
    # We can add a "Prediction Check" on the new data point.
    
    
    if state["success"]:
        state["accuracy"] = run_validation_simple(state["model_type"])
    else:
        state["accuracy"] = 0.0
        
    return state

def notify_node(state):
    print("--- Learning Notification Node ---")
    print(f"Final Status: {state['status_message']}")
    return state

# Graph
builder = StateGraph(LearningState)

builder.add_node("strategy", determine_strategy_node)
builder.add_node("learn", learn_node)
builder.add_node("validate", validate_learning_node)
builder.add_node("notify", notify_node)

builder.set_entry_point("strategy")

builder.add_edge("strategy", "learn")
builder.add_edge("learn", "validate")
builder.add_edge("validate", "notify")
builder.add_edge("notify", END)

learning_graph = builder.compile()
