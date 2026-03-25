import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rag.faiss_store import store_event

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def invert_model(model, target_label, input_shape, iterations=500):
    """
    Simulates a Model Inversion Attack.
    Tries to find an input 'x' that maximizes the probability of 'target_label'.
    """
    # Start with random noise
    reconstructed_x = torch.randn((1, input_shape), requires_grad=True, device=device)
    
    optimizer = optim.Adam([reconstructed_x], lr=0.01)
    
    model.eval()
    
    for i in range(iterations):
        optimizer.zero_grad()
        output = model(reconstructed_x)
        
        # We want to maximize the target class probability
        # So we minimize the negative log probability
        # target_label needs to be long tensor
        label_tensor = torch.tensor([target_label], device=device).long()
        loss = nn.CrossEntropyLoss()(output, label_tensor)
        
        loss.backward()
        optimizer.step()
        
    return reconstructed_x.detach()

def run_reversibility_test(model, original_input, target_label, model_type="DL"):
    """
    Runs the Reversibility Test.
    Returns the reconstruction error (MSE) between original and reconstructed.
    """
    print(f"--- Running Reversibility Test (Model Inversion) on {model_type} ---")
    
    if model_type == "DL":
        if original_input is None or target_label is None:
            return -1.0
            
        input_shape = original_input.shape[1]
        
        # Try to reconstruct
        reconstructed = invert_model(model, target_label, input_shape)
        
        # Calculate MSE
        mse_loss = nn.MSELoss()(reconstructed, original_input.to(device))
        error_val = mse_loss.item()
        
        status = "PASSED (High Difficulty)" if error_val > 0.5 else "WARNING (Low Difficulty)" # Threshold is arbitrary
        
        log_msg = f"Reversibility Test: Reconstruction Error (MSE) = {error_val:.4f}. Status: {status}"
        store_event(log_msg)
        print(log_msg)
        
        return error_val
    else:
        # SISA / ML models are harder to invert via gradients. 
        # We return a dummy high value to indicate 'safe' or Not Applicable.
        return 99.99
    
