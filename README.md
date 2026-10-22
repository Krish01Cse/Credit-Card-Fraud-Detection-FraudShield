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

3. **Data Cleaning & Preprocessing Studio**:
   - Comprehensive automated data health audit covering missing values, duplicate entries, Tukey's 1.5×IQR statistical outliers, and class imbalance metrics.
   - Customizable imputation strategies (Median, Mean, Zero-fill), outlier handling (IQR Winsorization/Clipping, None), and scaling algorithms (StandardScaler, RobustScaler).
   - Stratified 80/20 train/test split visualizer and SMOTE synthetic oversampling balance projection.
   - Interactive side-by-side Raw vs Cleaned transaction record comparator.
   - One-click automated dataset sanitization and real-time model retraining.

4. **Universal Schema Adaptor for Any Fraud Dataset**:
   - Automated intelligent column resolution supporting diverse schemas (e.g., `is_fraud`, `amt`, `trans_date_trans_time`, `unix_time`).
   - Automated PCA vector synthesis: if external datasets lack PCA vectors `V1`–`V28`, FraudShield automatically standardizes numeric features and projects them into 28 orthogonal principal components.

5. **Sleek Dark / Light Theme**:
   - Toggle button in the navigation bar with animated sun/moon icon transitions.
   - Complete HSL design token system with dark mode glassmorphism, high-contrast tables, and chart harmony for ROC, PR, and Confusion Matrix SVGs.
   - Persistent theme preferences saved in `localStorage`.

6. **Model Evaluation & Graph Visualizer**:
   - Dedicated tab with model selector pills for **Random Forest**, **KNN**, **Decision Tree**, and **Logistic Regression**.
   - Interactive SVG **Receiver Operating Characteristic (ROC)** curve with Area Under Curve (AUC) scoring.
   - Interactive SVG **Precision-Recall (PR)** curve with Average Precision metrics.
   - 2x2 **Confusion Matrix Heatmap** (True Negatives, False Positives, False Negatives, True Positives).
   - Class-level Classification Report metrics.

7. **Real-time Transaction Fraud Inspector & Authentic Catalog**:
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
2. **Dataset & Upload** — Upload custom transaction datasets (any schema: `creditcard.csv`, `fraudTest.csv`, Kaggle datasets) or load the Kaggle benchmark baseline with one click.
3. **Data Cleaning Studio** — Audit missing values, deduplicate records, inspect Tukey IQR statistical outliers, view Stratified 80/20 train/test split & SMOTE balance projections, and execute custom preprocessing.
4. **Model Graphs & ROC** — Inspect ROC curves, Precision-Recall curves, Confusion Matrices, and classification reports across Random Forest, KNN, Decision Tree, and Logistic Regression.
5. **Live Fraud Predictor** — Test single transactions with authentic presets, catalog selector, and real-time fraud probability gauge.
6. **Model Benchmarks** — Train single or all classifiers with side-by-side performance comparison.
7. **Dark / Light Theme** — Click the top-right toggle button to switch between sleek dark mode and vibrant light mode seamlessly.

---

## Architecture Notes

- **`frontend/`** — React 19 + Vite modern client with InsightHub rounded glassmorphism aesthetic and full dark/light theme engine.
- **`main.py`** — FastAPI REST API endpoints (`/api/dashboard`, `/api/dataset/upload`, `/api/dataset/cleaning-report`, `/api/dataset/clean`, `/api/models/graphs`, `/api/train`, `/api/predict`).
- **`backend.py`** — Core machine learning pipelines: universal schema adaptor, Tukey IQR anomaly detection, missing imputation, stratified splitting, standard & robust scaling, SMOTE class balancing, and model evaluation.
- Trained models are cached in-memory and serialized via Joblib for instant inference.

---

## Academic Disclaimer

FraudShield is built purely for educational purposes. It does **not** guarantee fraud prevention or transaction safety. Results should not be used for real financial decisions.

---

## License

This project is for academic use only.
