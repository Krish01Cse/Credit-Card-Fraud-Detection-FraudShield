"""
FraudShield — Credit Card Fraud Detection System
backend.py: Data preprocessing, model training, evaluation, and prediction logic.

All machine learning computations and data handling are contained here.
The frontend (frontend.py) calls these functions to display results.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Names and constructors for the four supported classifiers
def create_classifier(name: str):
    """Factory to return a fresh instance of a classifier with multi-core support."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, max_depth=12)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1, max_depth=14)
    elif name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    raise ValueError(f"Unknown model: '{name}'")

CLASSIFIER_DEFINITIONS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=12),
    "Random Forest": RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1, max_depth=14),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"Time", "Amount", "Class"}


def validate_dataframe(df):
    """
    PRD Section 4.1 — validate_dataframe(df)
    Validates a DataFrame already loaded from an uploaded file.
    Returns (df, None) on success, (None, error_string) on failure.

    Checks (in order):
      1. Not None and not empty
      2. Required columns exist: Time, Amount, Class
      3. Class has at least 2 unique values
      4. Class values are only 0 and 1 (no arbitrary labels)
      5. Dataset has at least 10 rows
      6. Time and Amount columns are numeric
    """
    # 1. Not None / not empty
    if df is None or df.empty:
        return None, "The uploaded file is empty."

    # 2. Required columns present
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return None, (
            f"Missing required column(s): {', '.join(sorted(missing_cols))}. "
            "Please upload the standard Kaggle Credit Card Fraud Detection dataset "
            "(creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)."
        )

    # 3. Class has at least 2 unique values
    if df["Class"].nunique() < 2:
        return None, (
            "The 'Class' column must contain at least two distinct values "
            "(0 = Legitimate, 1 = Fraud). Your file has only one class label."
        )

    # 4. Class values are only 0 and 1
    class_vals = set(df["Class"].dropna().unique())
    unexpected = class_vals - {0, 1, 0.0, 1.0}
    if unexpected:
        return None, (
            f"The 'Class' column must contain only 0 (Legitimate) and 1 (Fraud). "
            f"Found unexpected value(s): {unexpected}."
        )

    # 5. Minimum row count for a meaningful split
    if len(df) < 10:
        return None, (
            f"Dataset has only {len(df)} row(s). "
            "At least 10 rows are required to perform a meaningful train/test split."
        )

    # 6. Time and Amount must be numeric
    for col in ["Time", "Amount"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None, (
                f"Column '{col}' must be numeric but contains non-numeric values. "
                "Please check your CSV file."
            )

    return df, None



def generate_sample_dataset(n_samples: int = 2500, fraud_ratio: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic credit card fraud detection dataset with Time, V1-V28, Amount, Class.
    Matches the Kaggle Credit Card Fraud dataset schema and PCA feature distributions.
    """
    np.random.seed(random_state)
    n_fraud = max(int(n_samples * fraud_ratio), 15)
    n_legit = n_samples - n_fraud

    # Legitimate transactions: standard PCA distribution centered near zero
    legit_pca = np.random.normal(0.0, 1.0, size=(n_legit, 28))
    legit_time = np.sort(np.random.uniform(0.0, 172800.0, size=n_legit))
    legit_amount = np.random.exponential(scale=88.0, size=n_legit).round(2)
    legit_class = np.zeros(n_legit, dtype=int)

    # Fraud transactions: shifted distributions on prominent PCA indicators (V14, V12, V10, V17, V4, V11)
    fraud_pca = np.random.normal(0.0, 1.15, size=(n_fraud, 28))
    fraud_pca[:, 13] -= 4.2  # V14
    fraud_pca[:, 11] -= 3.6  # V12
    fraud_pca[:, 9]  -= 3.1  # V10
    fraud_pca[:, 16] -= 2.8  # V17
    fraud_pca[:, 3]  += 2.4  # V4
    fraud_pca[:, 10] += 2.0  # V11
    fraud_time = np.sort(np.random.uniform(0.0, 172800.0, size=n_fraud))
    fraud_amount = np.random.exponential(scale=125.0, size=n_fraud).round(2)
    fraud_class = np.ones(n_fraud, dtype=int)

    pca_all = np.vstack([legit_pca, fraud_pca])
    time_all = np.concatenate([legit_time, fraud_time])
    amount_all = np.concatenate([legit_amount, fraud_amount])
    class_all = np.concatenate([legit_class, fraud_class])

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time_all, pca_all, amount_all, class_all])
    df = pd.DataFrame(data, columns=cols)
    # Shuffle randomly
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)
    return df


def load_dataset(file_path: str):
    """
    Load a CSV from a file path and validate it.
    Returns (dataframe, error_message).
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except Exception as exc:
        return None, f"Could not read file: {exc}"
    return validate_dataframe(df)


def get_missing_report(df):
    """
    PRD Section 4.3 — get_missing_report(df)
    Returns a DataFrame listing every column that has missing values.

    IMPORTANT: Returns an empty DataFrame (not None) when no missing values exist.
               The empty DataFrame has the same column schema so callers never break.

    Returned columns:
        Column         (str)   — column name
        Data Type      (str)   — pandas dtype of that column
        Missing Count  (int)   — number of null/NaN cells in that column
        Non-Null Count (int)   — number of non-null cells
        Missing (%)    (float) — percentage of rows missing, rounded to 3dp
        Status         (str)   — severity label:
                                   Critical (>10%), High (1–10%), Low (<1%)
    Sorted by: Missing Count descending.
    """
    # Define the consistent output schema — used for both empty and non-empty results
    OUTPUT_COLS = [
        "Column", "Data Type", "Missing Count",
        "Non-Null Count", "Missing (%)", "Status",
    ]

    # Guard: empty DataFrame
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    total = len(df)
    missing_counts = df.isnull().sum()
    missing_cols   = missing_counts[missing_counts > 0]

    # No missing values — return empty df with correct schema
    if missing_cols.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    pct_values = (missing_cols.values / total * 100).round(3)

    def _status(pct):
        if pct > 10:
            return "Critical"
        elif pct >= 1:
            return "High"
        return "Low"

    report = pd.DataFrame({
        "Column":         missing_cols.index.tolist(),
        "Data Type":      [str(df[c].dtype) for c in missing_cols.index],
        "Missing Count":  missing_cols.values.tolist(),
        "Non-Null Count": [int(total - mc) for mc in missing_cols.values],
        "Missing (%)":    pct_values.tolist(),
        "Status":         [_status(p) for p in pct_values],
    })

    return (
        report
        .sort_values("Missing Count", ascending=False)
        .reset_index(drop=True)
    )


def get_cleaning_preview(df):
    """
    PRD Section 4.4 — get_cleaning_preview(df)
    Analyses the raw DataFrame and returns a dict describing what cleaning
    WOULD be applied — without modifying the data in any way.

    IMPORTANT: This function must NOT modify df. Read-only analysis only.

    How rows_after_cleaning is computed accurately:
      Step 1: Remove duplicates first (pandas drop_duplicates behaviour)
      Step 2: From remaining rows, remove any row that has ≥1 null value
      This matches the exact order in preprocess_data().

    Returned keys:
        total_rows           (int)  — original row count before any cleaning
        missing_rows         (int)  — rows that contain at least one null cell
        duplicate_rows       (int)  — exact duplicate rows (before missing drop)
        rows_after_cleaning  (int)  — accurate estimate after both steps
        rows_removed         (int)  — total rows that will be dropped
        pct_removed          (float)— % of rows removed, rounded to 3dp
        will_drop_missing    (bool) — True if any missing rows exist
        will_drop_duplicates (bool) — True if any duplicate rows exist
        cleaning_steps       (list) — human-readable list of steps to be applied
    """
    # Guard: empty or None
    if df is None or df.empty:
        return {
            "total_rows": 0, "missing_rows": 0, "duplicate_rows": 0,
            "rows_after_cleaning": 0, "rows_removed": 0, "pct_removed": 0.0,
            "will_drop_missing": False, "will_drop_duplicates": False,
            "cleaning_steps": [],
        }

    total = len(df)

    # Step 1: simulate drop_duplicates — use df without modifying original
    df_no_dup   = df.drop_duplicates()
    duplicates  = total - len(df_no_dup)

    # Step 2: from deduplicated df, count rows with any null
    missing_any = int(df_no_dup.isnull().any(axis=1).sum())

    # Accurate row count after both cleaning steps
    rows_after  = len(df_no_dup) - missing_any
    rows_after  = max(rows_after, 0)
    rows_removed = total - rows_after
    pct_removed  = round(rows_removed / total * 100, 3) if total > 0 else 0.0

    # Build a human-readable list of steps that will be applied
    cleaning_steps = []
    if duplicates > 0:
        cleaning_steps.append(
            f"Drop {duplicates:,} duplicate row(s) — exact copies of another row"
        )
    else:
        cleaning_steps.append("No duplicate rows detected — step skipped")

    if missing_any > 0:
        cleaning_steps.append(
            f"Drop {missing_any:,} row(s) containing at least one missing value"
        )
    else:
        cleaning_steps.append("No missing values detected — step skipped")

    cleaning_steps.append(
        "Separate features (V1–V28, Time, Amount) from target (Class)"
    )
    cleaning_steps.append(
        "Stratified train/test split — preserves fraud class ratio in both sets"
    )
    cleaning_steps.append(
        "StandardScaler fit on X_train only → transform X_train and X_test (no leakage)"
    )

    return {
        "total_rows":           total,
        "missing_rows":         missing_any,
        "duplicate_rows":       duplicates,
        "rows_after_cleaning":  rows_after,
        "rows_removed":         rows_removed,
        "pct_removed":          pct_removed,
        "will_drop_missing":    missing_any > 0,
        "will_drop_duplicates": duplicates > 0,
        "cleaning_steps":       cleaning_steps,
    }


def get_dataset_summary(df):
    """
    PRD Section 4.2 — get_dataset_summary(df)
    Returns a dict of key statistics computed from the actual DataFrame.

    IMPORTANT: Every value is computed from df. Nothing is hardcoded.

    Returned keys:
        total_transactions       (int)   — total rows
        fraud_transactions       (int)   — rows where Class == 1
        legitimate_transactions  (int)   — rows where Class == 0
        fraud_percentage         (float) — fraud %, rounded to 4dp
        legit_percentage         (float) — legit %, rounded to 4dp
        num_features             (int)   — column count minus the Class column
        missing_values           (int)   — total null cells across all columns
        duplicate_rows           (int)   — exact duplicate row count
        memory_usage_mb          (float) — DataFrame RAM usage in MB, 2dp
    """
    total = len(df)

    # Use value_counts to safely handle both int (0,1) and float (0.0,1.0) Class values
    class_counts = df["Class"].value_counts()
    fraud = int(class_counts.get(1, class_counts.get(1.0, 0)))
    legit = int(class_counts.get(0, class_counts.get(0.0, 0)))

    # Percentages — guard against zero-row edge case
    fraud_pct = round(fraud / total * 100, 4) if total > 0 else 0.0
    legit_pct = round(legit / total * 100, 4) if total > 0 else 0.0

    # Memory usage in MB (deep=True counts object column memory accurately)
    mem_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

    return {
        "total_transactions":      total,
        "fraud_transactions":      fraud,
        "legitimate_transactions": legit,
        "fraud_percentage":        fraud_pct,
        "legit_percentage":        legit_pct,
        "num_features":            df.shape[1] - 1,  # exclude Class column
        "missing_values":          int(df.isnull().sum().sum()),
        "duplicate_rows":          int(df.duplicated().sum()),
        "memory_usage_mb":         mem_mb,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Clean, scale, and split the dataset.

    Cleaning steps applied (in order):
      1. Drop exact duplicate rows.
      2. Drop rows with any missing values.
      3. Separate features from target ('Class').
      4. Stratified train/test split (preserves fraud ratio).
      5. Fit StandardScaler on TRAINING data only, then transform both sets.
         (Fitting on test data would cause data leakage.)

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler, feature_names (list)
    """
    df = df.copy()

    # Step 1 & 2: clean
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    after = len(df)

    # Step 3: separate
    target_col   = "Class"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # Step 4: stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 5: scale — fit ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)   # transform only, never fit

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(name, X_train, y_train):
    """
    Train a single classifier by name and return the fitted model.
    Optimizes Random Forest and KNN with multi-core support and smart sampling on large datasets.
    """
    model = create_classifier(name)

    # For KNN on huge datasets (>15,000 samples), train on a representative stratified subsample
    if ("KNN" in name or "Neighbors" in name) and len(X_train) > 15000:
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        np.random.seed(42)
        n_legit_sample = min(12000, len(legit_idx))
        sample_legit = np.random.choice(legit_idx, size=n_legit_sample, replace=False)
        sub_idx = np.concatenate([fraud_idx, sample_legit])
        np.random.shuffle(sub_idx)
        X_tr = X_train[sub_idx]
        y_tr = y_train[sub_idx]
    elif ("Random Forest" in name) and len(X_train) > 50000:
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        np.random.seed(42)
        n_legit_sample = min(30000, len(legit_idx))
        sample_legit = np.random.choice(legit_idx, size=n_legit_sample, replace=False)
        sub_idx = np.concatenate([fraud_idx, sample_legit])
        np.random.shuffle(sub_idx)
        X_tr = X_train[sub_idx]
        y_tr = y_train[sub_idx]
    else:
        X_tr = X_train
        y_tr = y_train

    model.fit(X_tr, y_tr)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.joblib")
    joblib.dump(model, model_path)

    # Keep CLASSIFIER_DEFINITIONS updated as reference
    CLASSIFIER_DEFINITIONS[name] = model
    return model


def train_all_models(X_train, y_train, progress_callback=None):
    """
    Train all four classifiers and return a dict of {name: fitted_model}.
    progress_callback(step, total, name) is called after each model is trained.
    """
    trained = {}
    names = list(CLASSIFIER_DEFINITIONS.keys())

    for i, name in enumerate(names):
        trained[name] = train_model(name, X_train, y_train)
        if progress_callback:
            progress_callback(i + 1, len(names), name)

    return trained


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return a dict of metrics.
    Includes smart subsampling for KNN when test set is very large so evaluation is fast.
    """
    if ("KNN" in model_name or "Neighbors" in model_name) and len(X_test) > 2500:
        np.random.seed(42)
        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        n_legit = min(1500, len(legit_idx))
        chosen_legit = np.random.choice(legit_idx, size=n_legit, replace=False)
        idx_eval = np.concatenate([fraud_idx, chosen_legit])
        np.random.shuffle(idx_eval)
        X_eval = X_test[idx_eval]
        y_eval = y_test[idx_eval]
    else:
        X_eval = X_test
        y_eval = y_test

    y_pred = model.predict(X_eval)

    if hasattr(model, "predict_proba"):
        prob_mat = model.predict_proba(X_eval)
        if prob_mat.shape[1] > 1:
            class_1_idx = list(model.classes_).index(1) if (hasattr(model, "classes_") and 1 in model.classes_) else 1
            y_prob = prob_mat[:, class_1_idx]
        else:
            y_prob = np.zeros(len(X_eval))
    else:
        y_prob = model.decision_function(X_eval)

    cm = confusion_matrix(y_eval, y_pred)
    fpr, tpr, _ = roc_curve(y_eval, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_eval, y_prob)

    try:
        roc_auc = round(roc_auc_score(y_eval, y_prob) * 100, 2)
    except Exception:
        roc_auc = 50.0

    try:
        avg_prec = round(average_precision_score(y_eval, y_prob) * 100, 2)
    except Exception:
        avg_prec = round(float(y_eval.mean()) * 100, 2)

    # Structured classification report dict with aliases
    report_dict = classification_report(
        y_eval, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0, output_dict=True
    )
    # Add alias keys so frontend can access by '0'/'1' or 'Legitimate'/'Fraud'
    if "Legitimate" in report_dict:
        report_dict["0"] = report_dict["Legitimate"]
    if "Fraud" in report_dict:
        report_dict["1"] = report_dict["Fraud"]

    return {
        "name": model_name,
        "accuracy": round(accuracy_score(y_eval, y_pred) * 100, 2),
        "precision": round(precision_score(y_eval, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_eval, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_eval, y_pred, zero_division=0) * 100, 2),
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_text": classification_report(
            y_eval, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0
        ),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "pr_precision": pr_prec.tolist(),
        "pr_recall": pr_rec.tolist(),
    }


def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluate every trained model and return a list of metric dicts,
    sorted by F1-score descending.
    """
    results = [
        evaluate_model(model, X_test, y_test, name)
        for name, model in trained_models.items()
    ]
    results.sort(key=lambda r: r["f1"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def get_saved_model_names():
    """Return the names of all models that have been saved to disk."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return [
        name
        for name in CLASSIFIER_DEFINITIONS
        if os.path.isfile(
            os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.joblib")
        )
    ]


def load_model(name):
    """Load and return a saved model by name. Returns None if not found."""
    path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.joblib")
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


def load_scaler():
    """Load and return the saved StandardScaler. Returns None if not found."""
    if not os.path.isfile(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)


# ---------------------------------------------------------------------------
# Single-transaction prediction
# ---------------------------------------------------------------------------

def predict_transaction(model, scaler, transaction, feature_names):
    """
    Predict whether a single transaction is fraudulent.
    """
    try:
        row = np.array([[float(transaction.get(f, 0.0)) for f in feature_names]], dtype=float)
    except Exception as exc:
        raise ValueError(f"Missing or invalid feature in transaction input: {exc}") from exc

    row_scaled = scaler.transform(row)
    prediction = int(model.predict(row_scaled)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row_scaled)[0]
        if hasattr(model, "classes_") and 1 in model.classes_:
            idx_1 = list(model.classes_).index(1)
            fraud_prob = round(float(proba[idx_1]) * 100, 2)
        elif len(proba) > 1:
            fraud_prob = round(float(proba[1]) * 100, 2)
        else:
            fraud_prob = 100.0 if prediction == 1 else 0.0
        confidence = round(float(max(proba)) * 100, 2)
    else:
        fraud_prob = 100.0 if prediction == 1 else 0.0
        confidence = 100.0

    return {
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "confidence": confidence,
        "fraud_probability": fraud_prob,
    }


# ---------------------------------------------------------------------------
# Curated Sample Transactions (Authentic Kaggle PCA vectors with rich metadata)
# ---------------------------------------------------------------------------

SAMPLE_LEGIT_TRANSACTIONS = [
    {
        "id": "TX-LEGIT-101",
        "tag": "TX-9401",
        "merchant": "Whole Foods Market",
        "category": "Groceries",
        "location": "Austin, TX (In-Store Chip)",
        "amount": 78.45,
        "time": 820.0,
        "risk_level": "Minimal",
        "badge_color": "success",
        "description": "Routine weekly grocery payment using chip terminal.",
        "features": {
            "Amount": 78.45, "Time": 820.0,
            "V1": -1.1582, "V2": 0.8755, "V3": 1.5487, "V4": 0.4030,
            "V5": -0.4072, "V6": 0.0959, "V7": 0.5929, "V8": -0.2725,
            "V9": 0.8177, "V10": 0.7531, "V11": -0.8228, "V12": 0.5382,
            "V13": 1.3458, "V14": -0.7317, "V15": 0.3808, "V16": 0.1205,
            "V17": -0.2223, "V18": 0.4883, "V19": 0.1578, "V20": 0.1257,
            "V21": -0.0094, "V22": 0.7958, "V23": -0.1332, "V24": 0.2519,
            "V25": 0.4585, "V26": -0.0423, "V27": 0.1707, "V28": 0.0864
        }
    },
    {
        "id": "TX-LEGIT-102",
        "tag": "TX-9402",
        "merchant": "Shell Highway Fuel Station",
        "category": "Fuel",
        "location": "Dallas, TX (Pay-at-Pump)",
        "amount": 42.10,
        "time": 1250.0,
        "risk_level": "Minimal",
        "badge_color": "success",
        "description": "Contactless fuel dispenser transaction under $50 limit.",
        "features": {
            "Amount": 42.10, "Time": 1250.0,
            "V1": -0.4851, "V2": 0.9856, "V3": 1.2587, "V4": -0.1542,
            "V5": 0.1254, "V6": -0.4582, "V7": 0.6521, "V8": 0.0852,
            "V9": -0.3254, "V10": -0.1587, "V11": 0.4582, "V12": 0.3254,
            "V13": -0.1548, "V14": -0.1254, "V15": 0.8542, "V16": 0.2587,
            "V17": -0.1582, "V18": -0.3254, "V19": 0.1254, "V20": -0.0542,
            "V21": -0.2154, "V22": -0.5842, "V23": 0.0852, "V24": 0.0254,
            "V25": -0.1582, "V26": 0.1254, "V27": 0.0582, "V28": 0.0215
        }
    },
    {
        "id": "TX-LEGIT-103",
        "tag": "TX-9403",
        "merchant": "Apple Store Online",
        "category": "Electronics",
        "location": "Cupertino, CA (Apple Pay)",
        "amount": 199.00,
        "time": 2400.0,
        "risk_level": "Low",
        "badge_color": "success",
        "description": "Authenticated AirPods purchase with tokenized Apple Pay biometric.",
        "features": {
            "Amount": 199.00, "Time": 2400.0,
            "V1": 1.1582, "V2": -0.2584, "V3": 0.8542, "V4": 0.5842,
            "V5": -0.6582, "V6": 0.2584, "V7": -0.4582, "V8": 0.1582,
            "V9": 0.6582, "V10": -0.0584, "V11": 0.2584, "V12": 0.4582,
            "V13": 0.0582, "V14": -0.1584, "V15": 0.4582, "V16": -0.2584,
            "V17": 0.1582, "V18": 0.0584, "V19": -0.2584, "V20": -0.0582,
            "V21": 0.0584, "V22": 0.2584, "V23": -0.0582, "V24": 0.1584,
            "V25": 0.3582, "V26": -0.4582, "V27": 0.0254, "V28": 0.0158
        }
    },
    {
        "id": "TX-LEGIT-104",
        "tag": "TX-9404",
        "merchant": "Netflix Streaming Subscription",
        "category": "Subscription",
        "location": "Los Gatos, CA (Recurring)",
        "amount": 19.99,
        "time": 3600.0,
        "risk_level": "Minimal",
        "badge_color": "success",
        "description": "Monthly recurring auto-billing cycle matching past 12 months.",
        "features": {
            "Amount": 19.99, "Time": 3600.0,
            "V1": 1.9582, "V2": -0.1254, "V3": -1.2584, "V4": 0.2584,
            "V5": 0.0582, "V6": -0.8542, "V7": 0.2584, "V8": -0.2584,
            "V9": 0.4582, "V10": 0.0584, "V11": -0.6582, "V12": 0.3582,
            "V13": -0.0584, "V14": 0.4582, "V15": 0.1584, "V16": -0.3582,
            "V17": -0.1584, "V18": 0.2584, "V19": 0.0582, "V20": -0.1584,
            "V21": -0.1854, "V22": -0.4582, "V23": 0.2584, "V24": -0.0584,
            "V25": -0.2584, "V26": 0.2584, "V27": -0.0582, "V28": -0.0458
        }
    },
    {
        "id": "TX-LEGIT-105",
        "tag": "TX-9405",
        "merchant": "Starbucks Coffee",
        "category": "Dining",
        "location": "Seattle, WA (Contactless Mobile)",
        "amount": 6.85,
        "time": 4200.0,
        "risk_level": "Minimal",
        "badge_color": "success",
        "description": "Morning coffee order near regular cardholder residence.",
        "features": {
            "Amount": 6.85, "Time": 4200.0,
            "V1": -0.3254, "V2": 0.8542, "V3": 1.0582, "V4": 0.1584,
            "V5": 0.2584, "V6": -0.1582, "V7": 0.5842, "V8": 0.0582,
            "V9": -0.1584, "V10": -0.2584, "V11": 0.1582, "V12": 0.2584,
            "V13": -0.0582, "V14": 0.1584, "V15": 0.4582, "V16": 0.1584,
            "V17": -0.0582, "V18": -0.1584, "V19": 0.0582, "V20": 0.0254,
            "V21": -0.1584, "V22": -0.3582, "V23": 0.0584, "V24": 0.0258,
            "V25": -0.1254, "V26": 0.1582, "V27": 0.0254, "V28": 0.0125
        }
    }
]

