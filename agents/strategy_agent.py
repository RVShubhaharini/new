from rag.llm_client import ask_llm

def choose_strategy(model_type):
    prompt = f"Choose best unlearning method for {model_type} between SISA or SSD."
    decision = ask_llm(prompt)
    return decision

def choose_model(unlearning_size, is_repeat=False):
    print(f"--- Strategy Agent: Deciding for {unlearning_size} records (Repeat: {is_repeat}) ---")
    
    prompt = f"""
    You are an AI Architect for a Banking System.
    We need to perform machine unlearning for {unlearning_size} customer records.
    Context: The user has requested unlearning for data that might have been processed before (Repeat Request: {is_repeat}).
    
    Available Strategies:
    1. 'DL' (Deep Learning Unlearning): Best for small batches (< 100). Uses gradient updates. Fast but approximate.
    2. 'ML' (SISA - Sharded/Sliced): Best for large batches (>= 100) OR for exact unlearning guarantees. Retrains specific shards. Slower but distinct from DL.
    
    Guidance:
    - If Repeat Request is True, prefer 'ML' (SISA) to ensure the data is cleared from the alternative model (Deep Cleaning).
    - Otherwise, follow standard logic (DL for small, ML for large).
    
    Task: Decide the best strategy.
    Output: Return ONLY the string 'DL' or 'ML'. do not add any other text.
    """
    
    try:
        decision = ask_llm(prompt).strip()
    except Exception as e:
        print(f"LLM Strategy error: {e}. Fallback to rule-based.")
        decision = "DL" if unlearning_size < 100 else "ML"
        
    # Validation
    if "DL" in decision:
        clean_decision = "DL"
    elif "ML" in decision:
        clean_decision = "ML"
    else:
        # Fallback
        clean_decision = "DL" if unlearning_size < 100 else "ML"
        
    print(f"Strategy Agent Selected: {clean_decision} (Reasoning: {decision})")
    return clean_decision
