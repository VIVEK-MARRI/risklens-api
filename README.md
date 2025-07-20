# 📊 RiskLens AI - Fraud Detection Dashboard

RiskLens AI is a full-stack machine learning project that predicts the risk of fraudulent credit card transactions using a trained machine learning model (Random Forest). It features:

- 🔍 A Flask API backend for model prediction and SHAP explanation
- 📈 A Streamlit dashboard frontend for user interaction
- ✅ Real-time prediction and explainability with SHAP values

---

## 🚀 Project Structure

```
risklens-api/
│
├── app.py                # Flask API backend
├── dashboard.py          # Streamlit frontend dashboard
├── risk_model.pkl        # Trained ML model (Random Forest)
├── requirements.txt      # List of Python dependencies
├── README.md             # Project overview (this file)
├── .gitignore            # Ignored files for Git
├── data/                 # (Optional) Dataset
└── notebooks/            # (Optional) Jupyter Notebooks for training
```

---

## 💡 Features

- Credit card fraud detection using a Random Forest classifier
- Easy-to-use web dashboard with Streamlit
- SHAP explainability for feature impact visualization
- RESTful API via Flask backend

---

## 🧪 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/VIVEK-MARRI/risklens-api.git
cd risklens-api
```

### 2. Create a Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
# OR
source myenv/bin/activate  # On Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask API

```bash
python app.py
```

It runs at: `http://127.0.0.1:5000`

### 5. Start the Streamlit Dashboard

In another terminal:

```bash
streamlit run dashboard.py
```

---

## 🔍 Sample API Usage (cURL)

```bash
curl -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d "{"data": [0.1, 0.2, ..., 0.3]}"
```

---

## 📊 Example Use Case

You can input 30 features (`V1–V28`, `Amount`, `Time`) on the dashboard and instantly get:

- A **fraud risk score**
- A **visual SHAP bar chart** showing feature contributions

---

## 📦 Requirements

- Python 3.9
- Flask
- Streamlit
- scikit-learn
- shap
- pandas
- numpy

All dependencies are listed in `requirements.txt`.

---

## 🙌 Acknowledgements

- Dataset: [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- SHAP: SHapley Additive exPlanations for model interpretability

---

## 🔐 License

MIT License - feel free to use, share, and improve!
