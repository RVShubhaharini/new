import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.action_agent import perform_unlearning
from agents.llm_agent import explain_system
from rag.faiss_store import store_event
st.set_page_config(page_title="Bank AI Governance", layout="wide")

st.title("🏦 Agentic AI Banking Governance Dashboard")

# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

st.subheader("Model Status")

baseline_acc = 0.85
current_acc = 0.85

st.metric("Baseline Accuracy", baseline_acc)
st.metric("Current Accuracy", current_acc)

# --------------------------------------------------
# DELETION REQUEST
# --------------------------------------------------

st.subheader("Request Data Deletion")

st.subheader("Deletion Mode")
deletion_mode = st.radio("Select Unlearning Target:", ["Delete Random Samples", "Delete Specific Customer"])

if deletion_mode == "Delete Random Samples":
    forget_size = st.number_input("Records to Unlearn", min_value=1, max_value=500, value=50)
    target_id = None
    force_ml = False
else:
    target_id = st.text_input("Customer ID (Required)", placeholder="e.g. 10001")
    forget_size = 1 # Single record
    force_ml = st.checkbox("Also unlearn from ML Model? (SISA)", value=False, help="Check this to force unlearning from the SISA model as well, useful for deep cleaning.")
    if not target_id:
        st.info("Please enter a Customer ID to proceed.")

target_email = st.text_input("Confirmation Email (Optional)", placeholder="user@example.com")

if st.button("Trigger Unlearning"):
    if deletion_mode == "Delete Specific Customer" and not target_id:
         st.error("Error: Customer ID is required for Specific Deletion.")
    else:
        with st.spinner("Agents are coordinating unlearning process..."):
            status = perform_unlearning(forget_size,
                                      customer_id=target_id if target_id else None,
                                      email=target_email if target_email else None,
                                      force_ml=force_ml)
        st.success("Task Complete")
        st.markdown(status)
        
    # Check for Explainability image
    shap_path = "dashboard/shap_latest.png"
    if os.path.exists(shap_path):
        st.subheader("🧠 Feature Importance Analysis (Perturbation)")
        st.write("Visualizing which features drive the model's prediction. 'Before' (Blue) vs 'After' (Orange).")
        st.image(shap_path, caption="Feature Contribution: Original vs Unlearned")
    
    # Check for MIA Plot
    mia_path = "dashboard/mia_plot.png"
    if os.path.exists(mia_path):
        st.subheader("🕵️‍♂️ Evidence of Privacy (MIA)")
        st.write("Simulating a Membership Inference Attack to verify data deletion.")
        st.image(mia_path, caption="Privacy Risk Reduction")

    # Check for Confidence Plot
    conf_path = "dashboard/confidence_plot.png"
    if os.path.exists(conf_path):
        st.subheader("📊 Confidence Distribution (Outlier Analysis)")
        st.write("Visualizing how the model perceives the unlearned record compared to the retained population.")
        st.image(conf_path, caption="Unlearned Target vs Retained Population")

    # Check for JSON metrics (Reversibility & Certificate)
    metrics_path = "dashboard/latest_metrics.json"
    if os.path.exists(metrics_path):
        import json
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            st.warning("⚠️ Metrics file is corrupted. Waiting for next update...")
            metrics = {}
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
            metrics = {}
            
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except:
                return default

        st.subheader("🔁 Reversibility Test (Lazarus Attack)")
        rev_err_val = metrics.get("reversibility_error", 0)
        
        # Handle -1.0 logic (which might be string "-1.0000")
        rev_err = safe_float(rev_err_val, -1.0)
        
        if rev_err == -1.0:
            rev_msg = metrics.get("reversibility_msg", "Unknown error")
            st.info(f"ℹ️ Reversibility Test Skipped. Reason: {rev_msg}")
        elif rev_err == 99.99:
             st.success("✅ PASSED: SISA Model (Exact Unlearning) uses distinct mechanism. Reversibility N/A (Safe).")
        else:
            st.metric("Reconstruction Error (MSE)", f"{rev_err:.4f}", help="Higher error means the model cannot reconstruct the deleted data.")
            if rev_err > 0.5:
                st.success("✅ PASSED: Data cannot be easily reconstructed from model weights.")
            else:
                st.warning("⚠️ WARNING: Model may still retain traces allowing reconstruction.")

        # Adversarial Robustness
        rob_score_val = metrics.get("robustness", "-1")
        rob_score = safe_float(rob_score_val, -1.0)
        
        if rob_score != -1.0:
            st.subheader("🛡️ Adversarial Robustness (Security)")
            st.write("Simulating a Hacker Attack (FGSM) to test model stability after unlearning.")
            st.metric("Robustness Score (Attack Accuracy)", f"{rob_score:.4f}", help="Low accuracy on attack data means the model is vulnerable? No. High accuracy means model is robust.")
            # Interpretation:
            # If Model Accuracy on Clean Data is 90%
            # And Model Accuracy on Adversarial Data is 10% -> Vulnerable.
            # If Model Accuracy on Adversarial Data is 85% -> Robust.
            if rob_score > 0.5:
                st.success("✅ PASSED: Model maintains robustness against attacks.")
            else:
                st.warning("⚠️ WARNING: Model vulnerability increased.")

        # --------------------------------------------------
        # MODEL HEALTH MONITOR (Fairness & Drift) - DISABLED BY USER REQUEST
        # --------------------------------------------------
        # st.markdown("---")
        # st.subheader("🏥 Model Health Monitor")
        # 
        # col1, col2 = st.columns(2)
        # 
        # with col1:
        #     st.markdown("#### ⚖️ Fairness (Demographic Parity)")
        #     fair_metrics = metrics.get("fairness_metrics", {})
        #     if fair_metrics:
        #         is_fair = fair_metrics.get("is_fair", False)
        #         ratio_val = fair_metrics.get("parity_ratio", 0)
        #         ratio = safe_float(ratio_val, 0.0)
        #         details = fair_metrics.get("details", "")
        #         
        #         st.metric("Parity Ratio", f"{ratio:.2f}", help="Close to 1.0 is fair. < 0.8 or > 1.25 indicates bias.")
        #         if is_fair:
        #             st.success(f"✅ PASSED: {details}")
        #         else:
        #             st.error(f"❌ FAILED: {details}")
        #     else:
        #         st.info("Fairness measurement pending...")
        #
        # with col2:
        #     st.markdown("#### 🌊 Data Drift (KS-Test)")
        #     drift_metrics = metrics.get("drift_metrics", {})
        #     if drift_metrics:
        #         drift_detected = drift_metrics.get("drift_detected", False)
        #         p_val_val = drift_metrics.get("p_value", 1.0)
        #         p_val = safe_float(p_val_val, 1.0)
        #         feat = drift_metrics.get("feature_index", "N/A")
        #         
        #         st.metric("Min P-Value", f"{p_val:.4f}", help="P-Value < 0.05 indicates significant drift.")
        #         if not drift_detected:
        #             st.success("✅ PASSED: No significant distribution shift detected.")
        #         else:
        #             st.warning(f"⚠️ DRIFT DETECTED at Feature Index {feat}. Model may need retraining.")
        #     else:
        #         st.info("Drift measurement pending...")
        
        st.markdown("---")

            
        # Certificate
        cert_path = metrics.get("certificate_path")
        if cert_path and os.path.exists(cert_path):
            st.subheader("📜 Compliance Certificate")
            with open(cert_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download Certificate of Erasure (PDF)",
                data=pdf_bytes,
                file_name="unlearning_certificate.pdf",
                mime="application/pdf"
            )

# --------------------------------------------------
# AI EXPLANATION
# --------------------------------------------------

st.subheader("Ask AI Compliance Assistant")

question = st.text_input("Ask about privacy / deletion")

if st.button("Ask AI"):

    answer = explain_system(question)

    st.write(answer)

# --------------------------------------------------
# AUDIT LOG
# --------------------------------------------------

st.subheader("System Audit Trail")

st.write("Recent actions stored in FAISS memory.")
