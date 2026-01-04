import streamlit as st
import pandas as pd
import os
import sys

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Add src to path to ensure modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.predict import load_model, make_prediction

# Enable LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Page config
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Credit Risk Assessment")
st.markdown("Enter the customer details below to assess credit risk.")

# Load model
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "credit_risk_model.joblib")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

model = get_model()

# Tabs for navigation
tab1, tab2 = st.tabs(["🚀 Prediction", "📊 Monitoring Dashboard"])

with tab1:
    if model is None:
        st.error("⚠️ Model not found. Please train the model first by running `python src/models/train.py`.")
    else:
        with st.form("risk_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                sex = st.selectbox("Sex", ["male", "female"])
                job = st.selectbox("Job", [0, 1, 2, 3], format_func=lambda x: {
                    0: "0 - Unskilled and non-resident",
                    1: "1 - Unskilled and resident",
                    2: "2 - Skilled",
                    3: "3 - Highly skilled"
                }[x])
                housing = st.selectbox("Housing", ["own", "rent", "free"])
                
            with col2:
                saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "quite rich", "rich"])
                checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
                credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
                duration = st.number_input("Duration (months)", min_value=1, value=12)
                purpose = st.selectbox("Purpose", [
                    "radio/TV", "education", "furniture/equipment", "car", 
                    "business", "domestic appliances", "repairs", "vacation/others"
                ])
                
            submit = st.form_submit_button("Assess Risk", use_container_width=True)
            
            if submit:
                input_data = {
                    "Age": age, "Sex": sex, "Job": job, "Housing": housing,
                    "Saving accounts": saving_accounts, "Checking account": checking_account,
                    "Credit amount": credit_amount, "Duration": duration, "Purpose": purpose
                }
                
                try:
                    from langsmith import traceable
                    from models.enrich_inference import generate_inference_description
                    from monitoring.logger import log_inference
                    
                    @traceable(name="Streamlit Assessment")
                    def predict_flow(data):
                        desc = generate_inference_description(data)
                        pred, prob = make_prediction(model, data)
                        log_inference(data, pred)
                        return pred, prob, desc

                    with st.spinner("Analyzing profile..."):
                        prediction, probability, description = predict_flow(input_data)
                    
                    st.divider()
                    st.subheader("Assessment Result")
                    st.info(f"**Customer Profile Analysis:**\n\n{description}")
                    
                    if prediction == "Good Risk":
                        st.success(f"✅ **Good Risk**")
                    else:
                        st.error(f"⚠️ **Bad Risk**")
                    if probability is not None:
                        st.caption(f"Probability of Bad Risk: {probability:.2%}")
                            
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.subheader("📡 Real-time Model Monitoring")
    st.markdown("Monitor de **Data Drift** (PSI) y **Prediction Drift** basado en logs de producción.")
    
    if st.button("Refresh Monitoring Report"):
        from monitoring.run_monitoring import run_monitoring_suite
        report = run_monitoring_suite()
        
        if report.get("status") == "success":
            # Summary Metrics
            cols = st.columns(2)
            drift_count = sum(1 for f in report["data_drift"].values() if f["drift"])
            cols[0].metric("Features with Drift", drift_count)
            
            p_drift = report.get("prediction_drift", {})
            cols[1].metric("Bad Risk Rate", f"{p_drift.get('current_rate', 0):.1%}")

            # Data Drift Table
            st.write("### Feature Stability (PSI)")
            psi_data = []
            for feat, res in report["data_drift"].items():
                psi_data.append({
                    "Feature": feat, 
                    "PSI": round(res["psi"], 4), 
                    "Status": "🚨 DRIFT" if res["drift"] else "✅ OK"
                })
            st.table(pd.DataFrame(psi_data))
        else:
            st.warning(f"Info: {report.get('message', 'Waiting for more data...')}")
