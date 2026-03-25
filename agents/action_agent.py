from agents.strategy_agent import choose_model
from rag.faiss_store import store_event
from orchestrator.langgraph_flow import graph

def perform_unlearning(forget_size, customer_id=None, email=None, force_ml=False):
    print(f"Initiating Action Agent for {forget_size} samples...")
    
    # Store trigger event
    store_event(f"Action Agent Triggered: Forget Size {forget_size}, Target ID: {customer_id}, Force ML: {force_ml}")
    
    # Run the full agentic graph
    try:
        result = graph.invoke({
            "forget_size": forget_size,
            "customer_id": customer_id,
            "email": email,
            "force_ml": force_ml
        })
        
        # Extract results from state
        acc = result.get("accuracy", "N/A")
        compliance = result.get("compliance", "Unknown")
        model_type = result.get("model_type", "Unknown")
        stats = result.get("statistics", {})
        
        total = stats.get("total_dataset_size", "Unknown")
        training = stats.get("training_set_size", "Unknown")
        deleted = stats.get("deleted_count", forget_size)
        cumulative = stats.get("cumulative_deleted", "Unknown")
        remaining = stats.get("remaining_training_count", "Unknown")
        
        status_msg = result.get("status_message", "Unknown")
        mia = result.get("mia_metrics", {})
        shap_buf = result.get("shap_plot", None)
        
        # Save SHAP plot
        if shap_buf:
             with open("dashboard/shap_latest.png", "wb") as f:
                 f.write(shap_buf.getbuffer())

        # Save Confidence plot
        conf_buf = result.get("confidence_plot", None)
        if conf_buf:
             with open("dashboard/confidence_plot.png", "wb") as f:
                 f.write(conf_buf.getbuffer())
        
        reversibility = result.get("reversibility_error", -1)
        reversibility_msg = result.get("reversibility_msg", "No details")
        certificate = result.get("certificate_path", None)
        
        mia_text = f"- Privacy Risk: {mia.get('privacy_risk', 0):.1f}% ({mia.get('attack_status', 'N/A')})"
        if reversibility == -1:
             rev_text = f"- Reversibility Error: N/A ({reversibility_msg})"
        else:
             rev_text = f"- Reversibility Error: {reversibility:.4f} (Higher is better)"
        
        # NOTE: Metrics are already saved by certificate_node in langgraph_flow.py
        # We do not need to overwrite them here.
        
        # Log detailed stats to RAG to prevent hallucinations
        if "ALREADY UNLEARNED" in status_msg:
            # Short report
            log_message = (
                f"Unlearning Status: **{status_msg}**\n\n"
                f"No changes were made to the model or dataset because this ID is already in the deletion history.\n"
                f"- Cumulative Deleted Records: {cumulative}\n"
                f"- Compliance Status: {compliance} (Pre-verified)"
            )
        else:
            # Full report
            # Safe formatting helper
            def safe_format(val, decimals=4):
                try:
                    return f"{float(val):.{decimals}f}"
                except:
                    return str(val)

            log_message = (
                f"Unlearning Operation Report:\n"
                f"**Status: {status_msg}**\n"
                f"- Model Type: {model_type}\n"
                f"- Total Dataset Size: {total}\n"
                f"- Training Set Size: {training} (Subset used for unlearning)\n"
                f"- Deleted This Run: {deleted}\n"
                f"- Cumulative Deleted Records: {cumulative}\n"
                f"- Remaining Training Records: {remaining}\n"
                f"- New Accuracy: {safe_format(acc)}\n"
                f"{mia_text}\n"
                f"{rev_text}\n"
                f"- Compliance Status: {compliance}\n"
                f"- **Certificate Generated**: {certificate if certificate else 'Failed'}"
            )
        
        store_event(log_message)
        
        return log_message
        
    except Exception as e:
        import traceback
        return f"Error during agent execution: {str(e)}"
