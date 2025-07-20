import streamlit as st
import requests
import numpy as np
import pandas as pd

# Page config
st.set_page_config(page_title="RiskLens AI", layout="wide")
st.title("📊 RiskLens AI - Fraud Risk Score Dashboard")
st.markdown("Enter the 30 feature values from the dataset (V1–V28, Amount, Time).")

# Feature names and default values
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
default_values = [round(0.1 * (i % 10), 2) for i in range(30)]

# Form for input
with st.form("prediction_form"):
    user_input = []
    st.subheader("🔢 Input Features")

    cols = st.columns(3)
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(f"{feature}", value=default_values[i], step=0.01, format="%.2f")
            user_input.append(val)

    submitted = st.form_submit_button("🔍 Predict Risk")

# After form submission
if submitted:
    st.info("📡 Sending data to prediction API...")

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"data": user_input}
        )

        if response.status_code == 200:
            result = response.json()
            risk_score = result["risk_score"]
            shap_vals = result.get("shap_values", {})

            st.success(f"✅ Predicted Risk Score: **{risk_score:.4f}**")

            if risk_score > 0.7:
                st.error("🚨 High Risk of Fraud!")
            elif risk_score > 0.3:
                st.warning("⚠️ Moderate Risk")
            else:
                st.success("🟢 Low Risk")

            # SHAP Feature Importance
            if shap_vals:
                st.subheader("🔍 Feature Importance (SHAP Values)")

                # Convert values to float if nested list
                shap_df = pd.DataFrame([
                    {"Feature": k, "Impact": float(v[0]) if isinstance(v, list) else float(v)}
                    for k, v in shap_vals.items()
                ])
                shap_df["|Impact|"] = shap_df["Impact"].abs()
                shap_df_sorted = shap_df.sort_values("|Impact|", ascending=False).head(10)

                st.bar_chart(shap_df_sorted.set_index("Feature")["|Impact|"])

        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.exception(f"🔥 Error connecting to API: {e}")

    st.code(f"Input Sent: {user_input}", language="python")
