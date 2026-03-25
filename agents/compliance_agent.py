def detect_deletion_request(customer_id):
    print(f"Deletion request received for ID: {customer_id}")
    return True

def check_forget_loss(loss_before, loss_after):
    gain = loss_after - loss_before
    if gain > 0:
        print("Compliance PASSED: Data influence removed")
        return True
    else:
        print("Compliance FAILED")
        return False

def check_regulatory_retention(customer_id):
    """
    Simulates a Regulatory Check (AML/KYC).
    Returns: (is_allowed: bool, reason: str)
    """
    if not customer_id:
        return True, "No ID provided"
        
    cid = int(customer_id)
    
    # 1. Mock Suspicious List (AML)
    suspicious_ids = [10099, 10100, 10555] # Example restricted IDs
    if cid in suspicious_ids:
        return False, f"BLOCKED: Customer {cid} is flagged for AML Investigation. Retention Mandatory."
        
    # 2. Mock Account Age (Data Retention Laws - e.g., < 5 years cannot delete)
    # We simulate this by checking if ID is effectively "new" (high number?)
    # Let's say IDs > 90000 are new accounts created this year.
    if cid > 90000:
        return False, f"BLOCKED: Account {cid} is less than 5 years old. Retention required by Law."
        
    return True, "Regulatory Check Passed."
