# FraudShield — Backend Product Requirements Document (PRD)

**Project:** Credit Card Fraud Detection System  
**File:** `backend.py`  
**Version:** 1.0  
**Author:** Engineering Student — Third Year Mini Project  

---

## 1. Purpose

`backend.py` is the **sole ML engine** of FraudShield. It contains every data operation, model training step, evaluation computation, and prediction function. The frontend (`pages/`) imports and calls these functions — it never performs any computation itself.

> **Rule:** If it involves data or math → it belongs in `backend.py`.  
> **Rule:** If it involves `st.` (Streamlit) → it does NOT belong in `backend.py`.

---

## 2. Current Status — What Already Exists

| Function | Status | Notes |
|----------|--------|-------|
| `validate_dataframe(df)` | ✅ Done | Checks required columns, non-empty, Class has 2 values |
| `load_dataset(file_path)` | ✅ Done | Loads CSV from path, calls validate_dataframe |
| `get_dataset_summary(df)` | ✅ Done | Returns dict: total, fraud, legit counts, % |
| `get_missing_report(df)` | ✅ Done | DataFrame of columns with missing values |
| `get_cleaning_preview(df)` | ✅ Done | Preview what cleaning will do without modifying data |
| `preprocess_data(df, test_size, random_state)` | ✅ Done | Clean → split → scale (no leakage) |
| `train_model(name, X_train, y_train)` | ✅ Done | Train single model by name, save to disk |
| `train_all_models(X_train, y_train, callback)` | ✅ Done | Train all 4, optional progress callback |
| `evaluate_model(model, X_test, y_test, name)` | ✅ Done | Full metrics dict |
| `evaluate_all_models(trained_models, X_test, y_test)` | ✅ Done | Evaluate all, sort by F1 |
| `get_saved_model_names()` | ✅ Done | List models saved on disk |
| `load_model(name)` | ✅ Done | Load a saved model from disk |
| `load_scaler()` | ✅ Done | Load saved StandardScaler from disk |
| `predict_transaction(model, scaler, transaction, feature_names)` | ✅ Done | Single-row prediction |

---

## 3. Data Flow

```
creditcard.csv (uploaded by user)
        │
        ▼
validate_dataframe(df)
        │
        ▼
get_dataset_summary(df)   ←── used by Home page cards & Dataset page
get_missing_report(df)    ←── used by Dataset quality section
get_cleaning_preview(df)  ←── used by Dataset preprocessing plan
        │
        ▼
preprocess_data(df, test_size=0.2)
        │
        ├──► X_train, y_train  ──► train_model() / train_all_models()
        │                                   │
        │                                   ▼
        │                          models saved to /models/*.joblib
        │                                   │
        └──► X_test, y_test   ──► evaluate_model() / evaluate_all_models()
                                            │
                                            ▼
                                    metrics dict (accuracy, precision,
                                    recall, F1, ROC-AUC, avg_precision,
                                    confusion_matrix, fpr/tpr, pr curves)
                                            │
                                            ▼
                              predict_transaction(model, scaler, inputs)
                                            │
                                            ▼
                                  {label, confidence, fraud_probability}
```

---

## 4. Function Specifications

### 4.1 `validate_dataframe(df)`

```python
def validate_dataframe(df) -> tuple[pd.DataFrame | None, str | None]:
```

**Input:** A pandas DataFrame loaded from user upload  
**Output:** `(df, None)` on success | `(None, error_string)` on failure  

**Validation checks (in order):**
1. `df` is not None and not empty
2. Required columns exist: `Time`, `Amount`, `Class`
3. `Class` column has exactly 2 unique values (0 and 1)

**Error messages must be:**
- Human-readable (not Python tracebacks)
- Specific about what is missing
- Include a fix hint (e.g. "download from Kaggle")

---

### 4.2 `get_dataset_summary(df)`

```python
def get_dataset_summary(df) -> dict:
```

**Returns dict with keys:**

| Key | Type | Description |
|-----|------|-------------|
| `total_transactions` | int | Total rows |
| `fraud_transactions` | int | Rows where Class == 1 |
| `legitimate_transactions` | int | Rows where Class == 0 |
| `fraud_percentage` | float | Fraud % rounded to 4 decimal places |
| `num_features` | int | Number of columns minus Class column |
| `missing_values` | int | Total null cells across entire DataFrame |
| `duplicate_rows` | int | Exact duplicate row count |

> **Never invent or hardcode any of these values. Compute from the actual df.**

---

### 4.3 `get_missing_report(df)`

```python
def get_missing_report(df) -> pd.DataFrame:
```

**Returns** a DataFrame with columns: `Column`, `Missing Count`, `Missing (%)`  
**Sorted** by Missing Count descending  
**Returns empty DataFrame** (not None) when no missing values exist  

---

### 4.4 `get_cleaning_preview(df)`

```python
def get_cleaning_preview(df) -> dict:
```

**Returns dict with keys:**

| Key | Type | Description |
|-----|------|-------------|
| `total_rows` | int | Before cleaning |
| `missing_rows` | int | Rows with any null value |
| `duplicate_rows` | int | Exact duplicate rows |
| `rows_after_cleaning` | int | Estimated rows remaining |
| `will_drop_missing` | bool | True if any missing rows exist |
| `will_drop_duplicates` | bool | True if any duplicates exist |

> This function **must not modify** the DataFrame. Analysis only.

---

### 4.5 `preprocess_data(df, test_size=0.2, random_state=42)`

```python
def preprocess_data(df, test_size=0.2, random_state=42) -> tuple:
```

**Steps (in strict order — do not change order):**

| Step | Operation | Reason |
|------|-----------|--------|
| 1 | `df.drop_duplicates()` | Remove identical rows |
| 2 | `df.dropna()` | Remove rows with any null |
| 3 | Separate `X` (features) and `y` (Class column) | Target isolation |
| 4 | `train_test_split(X, y, stratify=y)` | Stratified split preserves fraud ratio |
| 5 | `scaler.fit_transform(X_train)` | Fit scaler ONLY on training data |
| 6 | `scaler.transform(X_test)` | Transform test with training scaler |
| 7 | Save scaler to `models/scaler.joblib` | Reuse during prediction |

**Returns:** `(X_train, X_test, y_train, y_test, scaler, feature_names)`

> ⚠️ **Critical:** The scaler must NEVER be fit on test data. Fitting on test data causes **data leakage** — the model gets unfair knowledge of the test distribution. Always fit on X_train only.

> ⚠️ **Stratified split** is required because the dataset is highly imbalanced (0.17% fraud). Without stratify, the test set may have no fraud samples at all.

---

### 4.6 `train_model(name, X_train, y_train)`

```python
def train_model(name, X_train, y_train) -> sklearn_model:
```

**Supported model names (exact strings):**
- `"Logistic Regression"` → `LogisticRegression(max_iter=1000, random_state=42)`
- `"Decision Tree"` → `DecisionTreeClassifier(random_state=42)`
- `"Random Forest"` → `RandomForestClassifier(n_estimators=100, random_state=42)`
- `"K-Nearest Neighbors"` → `KNeighborsClassifier(n_neighbors=5)`

**After training:**
- Save model to `models/{name_with_underscores}.joblib`
- Create `models/` directory if it doesn't exist

**Raises** `ValueError` with clear message if name not recognised.

---

### 4.7 `evaluate_model(model, X_test, y_test, model_name)`

```python
def evaluate_model(model, X_test, y_test, model_name="Model") -> dict:
```

**Returns dict with keys:**

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Model name |
| `accuracy` | float | Percentage (0–100), rounded to 2dp |
| `precision` | float | Percentage (0–100), rounded to 2dp |
| `recall` | float | Percentage (0–100), rounded to 2dp |
| `f1` | float | Percentage (0–100), rounded to 2dp |
| `roc_auc` | float | Percentage (0–100), rounded to 2dp |
| `avg_precision` | float | Area under PR curve × 100, 2dp |
| `confusion_matrix` | list | 2×2 list of ints [[TN,FP],[FN,TP]] |
| `classification_report` | str | Full sklearn text report |
| `fpr` | list | False positive rates for ROC curve |
| `tpr` | list | True positive rates for ROC curve |
| `pr_precision` | list | Precision values for PR curve |
| `pr_recall` | list | Recall values for PR curve |

> All metrics computed from the **actual test set predictions** — never fabricated.  
> Use `zero_division=0` on precision/recall/F1 to handle edge cases gracefully.

---

### 4.8 `predict_transaction(model, scaler, transaction, feature_names)`

```python
def predict_transaction(model, scaler, transaction, feature_names) -> dict:
```

**Input:**
- `model` — a fitted sklearn classifier
- `scaler` — the same StandardScaler used during training
- `transaction` — `dict` of `{feature_name: float_value}`
- `feature_names` — ordered list of feature names

**Returns dict:**

| Key | Type | Description |
|-----|------|-------------|
| `prediction` | int | 0 = Legitimate, 1 = Fraud |
| `label` | str | `"Legitimate"` or `"Fraud"` |
| `confidence` | float | max(class probabilities) × 100 |
| `fraud_probability` | float | P(class=1) × 100 |

**Error handling:**
- Raise `ValueError` if any feature is missing from `transaction` dict
- Handle models without `predict_proba` (use decision_function fallback)

---

## 5. Model Parameters — Fixed for Reproducibility

All models use `random_state=42` where applicable. Parameters must not change between runs to ensure reproducible results for the viva demo.

| Model | Key Parameters |
|-------|---------------|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| KNN | `n_neighbors=5` |

---

## 6. File Storage

```
models/
├── scaler.joblib                  ← StandardScaler (fitted on X_train)
├── Logistic_Regression.joblib     ← Trained LR model
├── Decision_Tree.joblib           ← Trained DT model
├── Random_Forest.joblib           ← Trained RF model
└── K-Nearest_Neighbors.joblib     ← Trained KNN model
```

- Directory is created automatically if it does not exist
- Models are saved with `joblib.dump()` and loaded with `joblib.load()`
- Filenames use the model name with spaces replaced by underscores

---

## 7. Error Handling Rules

| Situation | Required Behaviour |
|-----------|-------------------|
| File not found | Return `(None, "File not found: path")` |
| Unreadable CSV | Return `(None, "Could not read file: <reason>")` |
| Missing required column | Return `(None, "Missing column(s): X, Y. Upload creditcard.csv from Kaggle.")` |
| Empty DataFrame | Return `(None, "The uploaded file is empty.")` |
| Unknown model name | Raise `ValueError` with list of valid names |
| Missing feature in prediction | Raise `ValueError` naming the missing feature |
| Model not trained yet | Return `None` from `load_model()` — never crash |

---

## 8. Session State Contract

The backend functions return values. The frontend is responsible for storing them in `st.session_state`. The agreed keys are:

| Key | Set by | Value type |
|-----|--------|-----------|
| `df` | Dataset page | `pd.DataFrame` |
| `dataset_ready` | Dataset page | `bool` |
| `feature_names` | Dataset page | `list[str]` |
| `X_train` | Dataset page | `np.ndarray` |
| `X_test` | Dataset page | `np.ndarray` |
| `y_train` | Dataset page | `np.ndarray` |
| `y_test` | Dataset page | `np.ndarray` |
| `scaler` | Dataset page | `StandardScaler` |
| `trained_models` | Train page | `dict[str, model]` |
| `models_ready` | Train page | `bool` |
| `eval_results` | Train page | `list[dict]` |

---

## 9. What the Backend Must NEVER Do

- ❌ Import or use `streamlit`
- ❌ Call `st.write()`, `st.error()`, or any `st.*` function
- ❌ Print to stdout (use return values / raise exceptions)
- ❌ Fabricate or hardcode any metric, statistic, or prediction
- ❌ Fit the scaler on test data
- ❌ Modify the original DataFrame in-place (always use `.copy()`)
- ❌ Delete the `models/` directory or existing saved models

---

## 10. Testing Checklist (Before Viva)

- [ ] `validate_dataframe()` rejects a CSV missing the `Class` column
- [ ] `validate_dataframe()` rejects an empty CSV
- [ ] `preprocess_data()` produces stratified splits (fraud % similar in train and test)
- [ ] Scaler is fit only on X_train (verify by checking `scaler.mean_` matches X_train stats)
- [ ] All 4 models train without error on the Kaggle dataset
- [ ] `evaluate_model()` returns non-zero recall for fraud class
- [ ] `predict_transaction()` returns `"Fraud"` for a known fraud sample
- [ ] `predict_transaction()` returns `"Legitimate"` for a known clean sample
- [ ] Saved models load correctly after app restart
- [ ] `get_missing_report()` returns empty DataFrame for creditcard.csv (it has no nulls)

---

## 11. Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

No other libraries are needed or permitted in `backend.py`.
