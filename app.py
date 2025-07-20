from flask import Flask, request, jsonify
import joblib
import numpy as np
import shap
import pandas as pd

print("🚀 app.py is executing")

# Load model and SHAP explainer
model = joblib.load("risk_model.pkl")
explainer = shap.Explainer(model)

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json["data"]
        input_array = np.array(input_data).reshape(1, -1)

        # Get prediction
        risk_score = model.predict_proba(input_array)[0][1]

        # SHAP values (with column names matching the model)
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        input_df = pd.DataFrame(input_array, columns=feature_names)
        shap_values = explainer(input_df)
        shap_dict = dict(zip(input_df.columns, shap_values.values[0]))

        return jsonify({
            "risk_score": round(risk_score, 4),
            "shap_values": shap_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("✅ Flask app is starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)
