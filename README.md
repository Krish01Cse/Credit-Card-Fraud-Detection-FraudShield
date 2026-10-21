# 🛡️ FraudShield — Credit Card Fraud Detection System

> **Detect Suspicious Transactions. Protect Every Swipe.**

FraudShield is an academic machine learning project that analyses credit card transaction data, trains and compares four classification models, and predicts whether individual transactions may be fraudulent.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Type** | Academic Machine Learning Project & Fraud Sentinel |
| **Frontend** | React 19 + Vite (InsightHub Rounded Aesthetics) |
| **Backend API** | FastAPI + Uvicorn (REST API on port 8000) |
| **Machine Learning** | Scikit-learn · Pandas · NumPy · Joblib |
| **Visualisation** | Interactive SVG ROC Curves · Precision-Recall Curves · Confusion Matrix Heatmaps |
| **Classifiers** | Random Forest · K-Nearest Neighbors · Decision Tree · Logistic Regression |

---

## Key Features

1. **Enterprise Fraud Detection Dashboard**:
   - Monitored swipe counts, fraud anomaly counts, live transaction stream with status badges (`BLOCKED`, `APPROVED`).
   - Monthly fraud volume trends and prevented financial losses.

2. **Custom Dataset Upload & Dynamic Model Retraining**:
   - Upload any custom European cardholders CSV dataset (`creditcard.csv`).
   - Automated schema validation (`Time`, `Amount`, `V1–V28`, `Class`), standard scaling, and one-click retraining across all 4 classifiers.
   - Built-in one-click demo benchmark dataset generator (2,500 rows).

3. **Model Evaluation & Graph Visualizer**:
   - Dedicated tab with model selector pills for **Random Forest**, **KNN**, **Decision Tree**, and **Logistic Regression**.
   - Interactive SVG **Receiver Operating Characteristic (ROC)** curve with Area Under Curve (AUC) scoring.
   - Interactive SVG **Precision-Recall (PR)** curve with Average Precision metrics.
   - 2x2 **Confusion Matrix Heatmap** (True Negatives, False Positives, False Negatives, True Positives).
   - Class-level Classification Report metrics.

4. **Real-time Transaction Fraud Inspector & Authentic Catalog**:
   - Single-transaction fraud probability gauge with instant visual verdict and dynamic risk badge.
   - Authentic Kaggle preset scenario buttons: `[ 📋 Fill Legit Sample ]` and `[ ⚠️ Fill Fraud Sample ]`.
   - **Interactive Authentic Transaction Repository**: Browse curated lists of legitimate (`✓ Verified Clean Swipes`) and fraudulent (`⚠️ High-Risk Fraud Samples`) transactions extracted from real Kaggle benchmarks.
   - Multi-category filtering (`Crypto & Cash`, `ATM Withdrawal`, `Groceries`, `Fuel`, `Electronics`, `High-Ticket Retail`) and instant text search.
   - Click-to-load feature: select any card in the list to populate its complete 30-feature vector (`Amount`, `Time`, `V1–V28`) into the inspector and trigger real-time model re-evaluation.
   - Collapsible advanced PCA vector editor (V1 to V28).

---

## Quick Start & Running Locally

Run both the FastAPI backend and React frontend with a single command:

```bash
bash run.sh
```

Or start the services individually:

```bash
# Terminal 1 — Start FastAPI Backend (Port 8000)
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Start React Frontend (Port 5173)
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173) in your browser.


---

## Usage Workflow

1. **Dashboard** — Monitor swipe volume, anomaly statistics, and real-time transaction feeds.
2. **Dataset & Upload** — Upload custom `creditcard.csv` or load the Kaggle benchmark baseline with one click.
3. **Model Graphs & ROC** — Inspect ROC curves, Precision-Recall curves, Confusion Matrices, and classification reports across Random Forest, KNN, Decision Tree, and Logistic Regression.
4. **Live Fraud Predictor** — Test single transactions with instant presets (`[ 📋 Fill Legit Sample ]` and `[ ⚠️ Fill Fraud Sample ]`) and live probability gauge.
5. **Model Benchmarks** — Train single or all classifiers with side-by-side performance comparison.

---

## Architecture Notes

- **`frontend/`** — React 19 + Vite modern client with InsightHub rounded glassmorphism aesthetic.
- **`main.py`** — FastAPI REST API endpoints (`/api/dashboard`, `/api/dataset/upload`, `/api/models/graphs`, `/api/train`, `/api/predict`).
- **`backend.py`** — Core machine learning pipelines: preprocessing, data balancing, training, evaluation, and prediction.
- Trained models are cached in-memory and serialized via Joblib for instant inference.

---

## Academic Disclaimer

FraudShield is built purely for educational purposes. It does **not** guarantee fraud prevention or transaction safety. Results should not be used for real financial decisions.

---

## License

This project is for academic use only.
