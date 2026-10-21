# 🛡️ FraudShield — Credit Card Fraud Detection System

> **Detect Suspicious Transactions. Protect Every Swipe.**

FraudShield is an academic machine learning project that analyses credit card transaction data, trains and compares four classification models, and predicts whether individual transactions may be fraudulent.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Type** | Academic mini-project (third-year engineering) |
| **Frontend** | Streamlit |
| **Backend / ML** | Python · Scikit-learn · Pandas · NumPy |
| **Visualisation** | Plotly · Matplotlib |
| **Model persistence** | Joblib |

### Models trained and compared

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### Evaluation metrics

Accuracy · Precision · Recall · F1-Score · ROC-AUC · Average Precision · Confusion Matrix · ROC Curve · Precision-Recall Curve

---

## Project Structure

```
Credit-Card-Fraud-Detection-FraudShield/
├── app.py              ← Entry point — run this with streamlit
├── backend.py          ← All ML: load, preprocess, train, evaluate, predict
├── pages/
│   ├── home.py         ← Home / Dashboard page
│   ├── dataset.py      ← Dataset & Preprocessing page
│   ├── train.py        ← Model Training & Comparison page
│   ├── predict.py      ← Fraud Prediction page
│   └── about.py        ← Results & About page
├── utils/
│   └── ui.py           ← Shared CSS, sidebar, and reusable UI components
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── models/             ← Created automatically when models are trained
    ├── scaler.joblib
    ├── Logistic_Regression.joblib
    ├── Decision_Tree.joblib
    ├── Random_Forest.joblib
    └── K-Nearest_Neighbors.joblib
```

---

## Dataset

FraudShield is designed to work with the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (`creditcard.csv`).

- **284,807** transactions (September 2013, European cardholders)
- **492** fraud cases (0.172 % — highly imbalanced)
- Features: `V1`–`V28` (PCA-transformed), `Time`, `Amount`, `Class`
- `Class`: 0 = Legitimate, 1 = Fraud

> Download `creditcard.csv` from Kaggle before running the application.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection-FraudShield.git
cd Credit-Card-Fraud-Detection-FraudShield
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
# Option A — if streamlit is on your PATH
streamlit run app.py

# Option B — macOS (Python 3.9 user install)
/Users/krish_01/Library/Python/3.9/bin/streamlit run app.py
```

The application opens at **http://localhost:8501** in your default browser.

---

## Usage Workflow

1. **Dataset page** — Upload `creditcard.csv` and inspect summary statistics.
2. **Train Models page** — Preprocess data and train all four classifiers.
3. **Model Comparison page** — Compare models using all evaluation metrics.
4. **Predict page** — Select a model and enter transaction features to get a fraud prediction.

---

## Architecture Notes

- **`frontend.py`** — All Streamlit pages, navigation, charts, tables, and forms. Calls functions from `backend.py` to retrieve results.
- **`backend.py`** — All data processing and ML logic. No Streamlit imports; can be tested independently.
- Trained models are saved to the `models/` directory using Joblib and reloaded automatically.
- Session state is used to pass data between pages without re-running computations.

---

## Academic Disclaimer

FraudShield is built purely for educational purposes. It does **not** guarantee fraud prevention or transaction safety. Results should not be used for real financial decisions.

---

## License

This project is for academic use only.
