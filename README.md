# RiskLens AI — Fraud Risk Scoring API + Dashboard

<!-- Badges -->
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-5B2C83)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

A compact, production-minded reference implementation for **credit-card fraud risk scoring** with:

- **Flask API** for online inference
- **Streamlit dashboard** for interactive scoring
- **SHAP explanations** for model interpretability

> The included model (`risk_model.pkl`) is a trained **Random Forest** classifier.

---

## Table of contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [API](#api)
  - [Health](#health)
  - [Predict](#predict)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Architecture

```text
┌──────────────┐        HTTP         ┌───────────────┐
│ Streamlit UI │  ───────────────▶   │   Flask API   │
│ dashboard.py │                    │    app.py     │
└──────────────┘                    └───────┬───────┘
                                            │
                                            │ loads
                                            ▼
                                     ┌───────────────┐
                                     │ risk_model.pkl │
                                     └───────────���───┘
```

---

## Project structure

```text
risklens-api/
├── app.py                # Flask API backend
├── dashboard.py          # Streamlit frontend dashboard
├── risk_model.pkl        # Trained ML model (Random Forest)
├── requirements.txt      # Python dependencies
├── README.md             # You are here
├── .gitignore
├── data/                 # (optional) dataset
└── notebooks/            # (optional) model training / exploration
```

---

## Quickstart

### 1) Clone

```bash
git clone https://github.com/VIVEK-MARRI/risklens-api.git
cd risklens-api
```

### 2) Create & activate a virtual environment

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash/zsh):**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the API

```bash
python app.py
```

API (default): `http://127.0.0.1:5000`

### 5) Run the dashboard

In a second terminal (same venv):

```bash
streamlit run dashboard.py
```

---

## API

### Health

If you have a health endpoint, prefer to document it here. If not, consider adding one (recommended for deployments).

Example (typical):

```bash
curl -s http://127.0.0.1:5000/health
```

### Predict

The API accepts **30 features** (commonly: `Time`, `Amount`, and `V1–V28`).

> Note: Feature order matters. Ensure the order matches the model training pipeline.

**cURL**

```bash
curl -X POST "http://127.0.0.1:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [0.1, 0.2, 0.3]}'
```

Expected response (example):

```json
{
  "prediction": 0,
  "probability": 0.013
}
```

If SHAP explanations are returned by your API, document the response shape here as well (e.g., top features, raw shap values).

---

## Dashboard

The Streamlit UI provides:

- Manual entry of transaction features
- Fraud risk scoring
- SHAP-based feature contribution visualization

Run:

```bash
streamlit run dashboard.py
```

---

## Configuration

This repository currently runs with sensible local defaults. For production deployments, consider parameterizing:

- `MODEL_PATH` (default: `risk_model.pkl`)
- `FLASK_HOST` / `FLASK_PORT`
- CORS policy (if hosting UI separately)
- Logging level & structured logging

If you add environment variables, document them in a `.env.example` and reference it here.

---

## Development

### Recommended quality checks

```bash
python -m compileall .
```

Optional (if you add them):

- Linting: `ruff` / `flake8`
- Formatting: `black`
- Type checking: `mypy`
- Tests: `pytest`

---

## Troubleshooting

- **`ModuleNotFoundError`**: confirm you activated the venv and installed `requirements.txt`.
- **SHAP issues on Windows**: ensure compatible versions of `numpy`, `numba`, and `shap`.
- **Model input mismatch**: verify the feature vector length and ordering match training.

---

## Acknowledgements

- Dataset: Credit Card Fraud Detection dataset on Kaggle
- SHAP: SHapley Additive exPlanations

---

## License

MIT — see `LICENSE`.