from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import shap

# Load model
print("🚀 app.py is executing")
model = joblib.load("risk_model.pkl")

# Define consistent feature names
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# Setup SHAP explainer once
explainer = shap.Explainer(model)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json["data"]

        # Convert input to DataFrame with correct feature names
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Prediction
        risk_score = model.predict_proba(input_df)[0][1]

        # SHAP values
        shap_values = explainer(input_df)
        shap_dict = dict(zip(input_df.columns, shap_values.values[0].tolist()))

        return jsonify({
            "risk_score": round(risk_score, 4),
            "shap_values": shap_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("✅ Flask app is starting...")
    app.run(debug=True)
