import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
from rag.faiss_store import store_event

def calculate_confidence(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        return torch.mean(max_probs).item()

def run_mia_attack(model, baseline_model, target_data, model_type="DL"):
    """
    Simulates a Membership Inference Attack (MIA) on both Current and Baseline models.
    Generates a comparison plot.
    """
    print(f"--- Running Comparative Privacy Attack (MIA) on {model_type} Model ---")
    
    if target_data is None or len(target_data) == 0:
        return {"privacy_risk": 0, "attack_status": "No Target Data"}

    # Prepare Data
    if model_type == "DL":
        inputs = target_data[0] # X
        
        # 1. Evaluate Current Model (Unlearned)
        model.eval()
        conf_current = calculate_confidence(model, inputs)
        
        # 2. Evaluate Baseline Model (Original)
        baseline_model.eval()
        conf_baseline = calculate_confidence(baseline_model, inputs)
        
        # 3. Calculate Risks
        # Risk = (Confidence - 0.5) * 2 * 100
        risk_current = max(0.0, (conf_current - 0.5) * 2) * 100
        risk_baseline = max(0.0, (conf_baseline - 0.5) * 2) * 100
        
        # 4. Generate Plot
        plt.figure(figsize=(6, 4))
        labels = ['Before (Original)', 'After (Unlearned)']
        risks = [risk_baseline, risk_current]
        colors = ['red', 'green']
        
        bars = plt.bar(labels, risks, color=colors)
        plt.ylabel('Privacy Risk Score (%)')
        plt.title('Membership Inference Attack: Risk Reduction')
        plt.ylim(0, 100)
        
        # Add text labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
                    
        # Add threshold line
        plt.axhline(y=50, color='orange', linestyle='--', label='Risk Threshold')
        plt.legend()
        
        # Save Plot
        plt.savefig("dashboard/mia_plot.png", bbox_inches='tight')
        plt.close()
            
        status = "Safe / Unlearned" if risk_current < 50 else "High Risk"
        
        log_msg = f"MIA Simulation: Risk dropped from {risk_baseline:.1f}% to {risk_current:.1f}%"
        store_event(log_msg)
        print(log_msg)
        
        return {
            "is_member": risk_current > 50,
            "confidence_score": conf_current,
            "privacy_risk": risk_current,
            "attack_status": status,
            "risk_baseline": risk_baseline
        }
            
    else:
        # SISA Support
        try:
            # Current
            probs = model.predict_proba(target_data[0])
            conf_current = np.mean(np.max(probs, axis=1))
            
            # Baseline
            probs_b = baseline_model.predict_proba(target_data[0])
            conf_baseline = np.mean(np.max(probs_b, axis=1))
            
            risk_current = max(0.0, (conf_current - 0.5) * 2) * 100
            risk_baseline = max(0.0, (conf_baseline - 0.5) * 2) * 100
            
            # Plot
            plt.figure(figsize=(6, 4))
            plt.bar(['Before', 'After'], [risk_baseline, risk_current], color=['red', 'green'])
            plt.ylabel('Privacy Risk (%)')
            plt.title('MIA Risk Reduction (SISA)')
            plt.ylim(0, 100)
            plt.savefig("dashboard/mia_plot.png", bbox_inches='tight')
            plt.close()
            
            return {
                "privacy_risk": risk_current,
                "attack_status": "Safe" if risk_current < 50 else "Risk"
            }
        except Exception as e:
            print(f"SISA MIA Error: {e}")
            return {"attack_status": "Error"}
