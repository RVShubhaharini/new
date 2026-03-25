import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import torch.nn.functional as F
from rag.faiss_store import store_event

def get_feature_importance(model, inputs, feature_names=None, model_type="DL"):
    """
    Calculates feature importance via simple perturbation (Sensitivity Analysis).
    Importance = | Confidence(X) - Confidence(X_masked) |
    """
    importances = []
    
    if model_type == "DL":
        inputs = inputs.to(next(model.parameters()).device)
        model.eval()
        
        with torch.no_grad():
            # Base Confidence
            outputs = model(inputs)
            base_prob = F.softmax(outputs, dim=1).max().item()
            
            # Perturb each feature
            # inputs shape: (1, num_features)
            num_features = inputs.shape[1]
            
            for i in range(num_features):
                masked = inputs.clone()
                # Mask with 0 (assuming scaled data, 0 is mean)
                masked[0][i] = 0.0 
                
                out_masked = model(masked)
                prob_masked = F.softmax(out_masked, dim=1).max().item()
                
                # Impact is the absolute change in max probability
                impact = base_prob - prob_masked
                # We care about "Positive Contribution" to the class?
                # Actually, simpler: How much does it DROP if we remove it?
                # If impact is positive, feature helped. If negative, feature hurt.
                importances.append(impact)
                
    else:
        # SISA
        base_probs = model.predict_proba(inputs)
        base_prob = np.max(base_probs)
        
        num_features = inputs.shape[1]
        for i in range(num_features):
            masked = inputs.copy()
            masked[0][i] = 0.0
            
            prob_masked = np.max(model.predict_proba(masked))
            importances.append(base_prob - prob_masked)

    return np.array(importances)

def generate_shap_explanation(model, baseline_model, target_data, background_data, feature_names=None, model_type="DL"):
    """
    REPLACED SHAP WITH CUSTOM PERTURBATION PLOT.
    Function name kept for compatibility with Orchestrator, but logic is custom.
    """
    print(f"--- Generating Feature Importance (Perturbation) for {model_type} ---")
    
    try:
        if model_type == "DL":
            inputs = target_data[0][:1] # Just the target
        else:
            inputs = target_data[0][:1]
            
        # 1. Calculate Importance for Baseline
        imp_base = get_feature_importance(baseline_model, inputs, feature_names, model_type)
        
        # 2. Calculate Importance for Current
        imp_curr = get_feature_importance(model, inputs, feature_names, model_type)
        
        # 3. Select Top Features (based on Baseline importance)
        # We want to see how the "Important features" changed
        top_indices = np.argsort(np.abs(imp_base))[-10:] # Top 10
        
        top_imp_base = imp_base[top_indices]
        top_imp_curr = imp_curr[top_indices]
        
        if feature_names:
            labels = [feature_names[i] for i in top_indices]
        else:
            labels = [f"Feature {i}" for i in top_indices]
            
        # 4. Plot Side-by-Side
        plt.figure(figsize=(12, 6))
        
        y_pos = np.arange(len(labels))
        width = 0.35
        
        plt.barh(y_pos + width/2, top_imp_base, width, label='Before (Original)', color='skyblue')
        plt.barh(y_pos - width/2, top_imp_curr, width, label='After (Unlearned)', color='orange')
        
        plt.yticks(y_pos, labels)
        plt.xlabel('Feature Contribution (Drop in Confidence if Removed)')
        plt.title('Feature Importance Comparison: Which features drive the prediction?')
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        store_event("Custom Feature Importance Plot generated.")
        return buf

    except Exception as e:
        print(f"Feature Importance Gen Failed: {e}")
        return None

def generate_confidence_plot(model, target_data, background_data, model_type="DL"):
    """
    Generates a generic Confidence Distribution Plot.
    """
    print(f"--- Generating Confidence Plot for {model_type} ---")
    try:
        if model_type == "DL":
            # 1. Get Background Confidences
            model.eval()
            bg_inputs = background_data.to(target_data[0].device)
            with torch.no_grad():
                bg_outputs = model(bg_inputs)
                bg_probs = torch.nn.functional.softmax(bg_outputs, dim=1)
                bg_confidences, _ = torch.max(bg_probs, dim=1)
                bg_confidences = bg_confidences.cpu().numpy()
            
            # 2. Get Target Confidence
            target_input = target_data[0][:1].to(bg_inputs.device)
            with torch.no_grad():
                target_output = model(target_input)
                target_prob = torch.nn.functional.softmax(target_output, dim=1)
                target_conf, _ = torch.max(target_prob, dim=1)
                target_conf = target_conf.item()
                
        else:
            # SISA
            bg_inputs = background_data[0] # assuming tuple
            bg_probs = model.predict_proba(bg_inputs)
            bg_confidences = np.max(bg_probs, axis=1)
            
            target_input = target_data[0][:1]
            target_probs = model.predict_proba(target_input)
            target_conf = np.max(target_probs)

        # 3. Plot
        plt.figure(figsize=(8, 5))
        
        # Hist of retained
        plt.hist(bg_confidences, bins=10, color='skyblue', alpha=0.7, label='Retained Population')
        
        # Line for target
        plt.axvline(target_conf, color='red', linestyle='dashed', linewidth=2, label=f'Unlearned Target ({target_conf:.2f})')
        
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency (Count)')
        plt.title('Unlearning Efficacy: Outlier Analysis')
        plt.legend()
        plt.xlim(0, 1.0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        store_event("Confidence Plot generated.")
        return buf

    except Exception as e:
        print(f"Confidence Plot Failed: {e}")
        return None
