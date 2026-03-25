import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fgsm_attack(model, data, target, epsilon):
    """
    Generates adversarial examples using Fast Gradient Sign Method (FGSM).
    """
    data.requires_grad = True
    output = model(data)
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Collect data gradients
    data_grad = data.grad.data
    
    # Create adversarial image
    sign_data_grad = data_grad.sign()
    perturbed_data = data + epsilon * sign_data_grad
    
    return perturbed_data

def run_adversarial_attack(model_type, training_state, epsilon=0.1):
    """
    Runs an adversarial attack on the model to test robustness.
    Returns a robustness score (Accuracy on Adversarial Data).
    """
    print(f"Running Adversarial Attack (FGSM) on {model_type} model...")
    
    if model_type == "DL":
        return run_dl_attack(training_state, epsilon)
    else:
        # For ML (SISA/Sklearn), gradient-based attacks are harder (black box).
        # We simulate a "Boundary Attack" or noise injection for robustness proxy.
        return run_ml_noise_attack(training_state, epsilon)

def run_dl_attack(training_state, epsilon):
    try:
        model = training_state["dl"]["model"]
        test_loader = training_state["dl"]["test_loader"]
        
        model.eval()
        correct = 0
        total = 0
        
        for data, target in test_loader:
            if isinstance(data, list): data = data[0] # Handle list if needed
            
            # Move to device if needed (assuming CPU for now based on project context)
            # data, target = data.to(device), target.to(device)
            
            # Generate Adversarial Example
            perturbed_data = fgsm_attack(model, data, target, epsilon)
            
            # Re-classify
            output = model(perturbed_data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        adv_accuracy = correct / total
        print(f"DL Adversarial Accuracy (Epsilon={epsilon}): {adv_accuracy:.4f}")
        return adv_accuracy
        
    except Exception as e:
        print(f"DL Attack Error: {e}")
        return -1.0

def run_ml_noise_attack(training_state, noise_level):
    """
    Simulates robustness test for non-differentiable models (SISA/RF/LR)
    by adding Gaussian noise to the test set and measuring stability.
    """
    try:
        model = training_state["sisa"]["model"]
        X_test = training_state["sisa"]["X_test"]
        y_test = training_state["sisa"]["y_test"]
        
        # Add Noise
        noise = np.random.normal(0, noise_level, X_test.shape)
        X_test_adv = X_test + noise
        
        # Evaluate
        # SISA model might have a predict method or we need to aggregate shards
        # The SISA wrapper implementation usually has .predict()
        y_pred = model.predict(X_test_adv)
        
        accuracy = (y_pred == y_test).mean()
        print(f"ML (SISA) Robustness (Noise={noise_level}): {accuracy:.4f}")
        return accuracy
        
    except Exception as e:
        print(f"ML Attack Error: {e}")
        return -1.0
