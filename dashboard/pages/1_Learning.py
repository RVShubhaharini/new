import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add parent dir to path to import agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator.learning_flow import learning_graph

st.set_page_config(page_title="Incremental Learning", page_icon="🎓", layout="wide")

st.title("🎓 Teach the AI (Incremental Learning)")
st.markdown("Use this interface to add **new knowledge** to the system. The model will learn from this new data point without needing a full retrain.")

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
st.subheader("New Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["admin.", "technician", "services", "management", "retired", "blue-collar", "unemployed", "entrepreneur", "housemaid", "unknown", "self-employed", "student"])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["secondary", "tertiary", "primary", "unknown"])
    balance = st.number_input("Balance (EUR)", value=1000)

with col2:
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    default = st.selectbox("Has Credit Default?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    day = st.number_input("Day of Month", 1, 31, 15)

with col3:
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Duration (sec)", 0, 5000, 200)
    campaign = st.number_input("Campaign Contacts", 1, 50, 1)
    pdays = st.number_input("Pdays (-1 if new)", -1, 1000, -1)
    previous = st.number_input("Previous Contacts", 0, 50, 0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "failure", "other", "success"])

st.markdown("---")
st.subheader("Target Label")
deposit = st.radio("Did the client subscribe to a Term Deposit?", ["yes", "no"], horizontal=True)

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------
new_data_dict = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "deposit": deposit
}

# --------------------------------------------------
# ACTION
# --------------------------------------------------
st.markdown("---")
st.subheader("Training Control")

model_choice = st.radio("Select Model to Teach:", ["Deep Learning (Fine-Tune)", "Machine Learning (SISA Append)"])

if st.button("🧠 Teach Model"):
    model_type_code = "DL" if "Deep Learning" in model_choice else "SISA"
    
    with st.spinner(f"Teaching {model_type_code} model new knowledge..."):
        # Invoke Graph
        inputs = {
            "new_data": new_data_dict,
            "model_type": model_type_code
        }
        
        final_state = learning_graph.invoke(inputs)
        
        if final_state.get("success"):
            st.success(f"✅ Learning Complete! {final_state.get('status_message')}")
            st.session_state['last_input'] = new_data_dict
            st.session_state['last_model'] = model_type_code
        else:
            st.error(f"❌ Learning Failed: {final_state.get('status_message')}")

# --------------------------------------------------
# VERIFICATION
# --------------------------------------------------
if 'last_input' in st.session_state:
    st.markdown("---")
    st.subheader("🧪 Verify Learning")
    st.info("Test if the model actually remembers what you just taught it.")
    
    if st.button("Predict on this New Data"):
        
        from agents.learning_agent import verify_prediction
        m_code = st.session_state['last_model']
        data = st.session_state['last_input']
        
        pred, conf = verify_prediction(data, m_code)
        
        st.write(f"**Model Used:** {m_code}")
        st.write(f"**Target Label:** {data['deposit']}")
        st.write(f"**Model Prediction:** {pred}")
        st.write(f"**Confidence:** {conf:.4f}")
        
        target_bool = 1 if data['deposit'] == "yes" else 0
        
        # Handle logic for display
        pred_label = "yes" if pred == 1 else "no"
        
        if pred == target_bool:
             st.success(f"✅ SUCCESS: The model correctly predicted '{pred_label}'.")
        else:
             st.warning(f"⚠️ WARNING: The model predicted '{pred_label}' but you taught it '{data['deposit']}'. It might need more training examples.")

# --------------------------------------------------
# QUICK TEST
# --------------------------------------------------
with st.expander("Developer Options"):
    if st.button("Auto-Fill Random Data"):
        st.info("To implement auto-fill, we'd need session state management. For now, just use defaults.")
