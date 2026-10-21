"""
main.py — FastAPI Backend for FraudShield & InsightHub
Exposes clean REST APIs for the React frontend:
- Dashboard KPI summaries and monthly trends
- Dataset status and demo data initialization
- Individual and bulk ML model training (Random Forest, KNN, Decision Tree, Logistic Regression)
- Real-time transaction fraud prediction with confidence and probability scoring
"""

import os
import io
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import backend

app = FastAPI(title="FraudShield & InsightHub API", version="2.0.0")

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State Storage ──────────────────────────────────────────────────────
STATE: Dict[str, Any] = {
    "df": None,
    "dataset_ready": False,
    "feature_names": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "scaler": None,
    "trained_models": {},
    "eval_results": [],
}


def ensure_data_and_models():
    """Ensure sample data and models are active on startup."""
    if not STATE["dataset_ready"] or STATE["df"] is None:
        try:
            demo_df = backend.generate_sample_dataset()
            STATE["df"] = demo_df
            STATE["dataset_ready"] = True
            X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(demo_df)
            STATE["X_train"] = X_tr
            STATE["X_test"] = X_te
            STATE["y_train"] = y_tr
            STATE["y_test"] = y_te
            STATE["scaler"] = scl
            STATE["feature_names"] = f_names

            # Load or train all models
            loaded = {}
            for name in backend.CLASSIFIER_DEFINITIONS:
                m = backend.load_model(name)
                if m is None:
                    m = backend.train_model(name, X_tr, y_tr)
                loaded[name] = m

            STATE["trained_models"] = loaded
            STATE["eval_results"] = backend.evaluate_all_models(loaded, X_te, y_te)
        except Exception as exc:
            print("Auto-initialization notice:", exc)


@app.on_event("startup")
def startup_event():
    ensure_data_and_models()


# ── Pydantic Request Models ───────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_name: str = "all"  # "all", "Random Forest", "K-Nearest Neighbors", etc.


class PredictRequest(BaseModel):
    model_name: Optional[str] = "Random Forest"
    transaction: Dict[str, float]


# ── Presets ───────────────────────────────────────────────────────────────────

LEGIT_PRESET = {
    "Amount": 149.62,
    "Time": 406.0,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
}

FRAUD_PRESET = {
    "Amount": 529.00,
    "Time": 406.0,
    "V1": -4.3979, "V2": 3.8864, "V3": -4.6871, "V4": 4.5676,
    "V5": -2.4568, "V6": -1.8492, "V7": -4.5544, "V8": 2.8564,
    "V9": -3.8021, "V10": -5.6044, "V11": 4.1098, "V12": -6.2231,
    "V13": -0.8597, "V14": -7.1245, "V15": 0.4851, "V16": -3.8965,
    "V17": -5.8456, "V18": -1.2589, "V19": 1.1564, "V20": 0.8465,
    "V21": 1.1082, "V22": -0.4371, "V23": -0.5621, "V24": 0.2645,
    "V25": 0.1856, "V26": 0.4012, "V27": 0.8462, "V28": -0.3214,
}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    ensure_data_and_models()
    return {
        "status": "online",
        "dataset_ready": STATE["dataset_ready"],
        "total_rows": len(STATE["df"]) if STATE["df"] is not None else 0,
        "models_count": len(STATE["trained_models"]),
        "trained_models": list(STATE["trained_models"].keys()),
    }


@app.get("/api/dashboard")
def get_dashboard_data():
    ensure_data_and_models()
    df = STATE["df"]
    eval_results = STATE["eval_results"]

    # Top 4 Metric Cards (Matching screenshot: 547, 339, 147, 89.75%)
    top_metrics = {
        "active_employees": {
            "label": "Active Employees",
            "value": "547",
            "icon": "user",
        },
        "number_projects": {
            "label": "Number of Projects",
            "value": "339",
            "icon": "folder",
        },
        "number_tasks": {
            "label": "Number of Task",
            "value": "147",
            "icon": "paperclip",
        },
        "target_percentage": {
            "label": "Target Percentage Completed",
            "value": "89.75%",
            "icon": "percent",
        },
    }

    # Monthly trends matching screenshot chart
    monthly_trends = [
        {"month": "Jan", "value": 45},
        {"month": "Feb", "value": 52},
        {"month": "Mar", "value": 48},
        {"month": "Apr", "value": 62},
        {
            "month": "Mei",
            "value": 137,
            "highlight": True,
            "tooltip": {
                "project_done": 137,
                "project_new": 123,
                "project_cost": 84,
            },
        },
        {"month": "Jun", "value": 38},
        {"month": "Jul", "value": 70},
        {"month": "Aug", "value": 65},
        {"month": "Sep", "value": 72},
    ]

    # Tasks / Projects (Journey Scarves, Edfier, and Active ML Model)
    tasks = [
        {
            "id": 1,
            "title": "Journey Scarves",
            "subtitle": "Rebranding and Website Design",
            "tag": "JS",
            "avatars": ["👩‍💻", "👨‍🎨", "👩‍💼"],
        },
        {
            "id": 2,
            "title": "Edfier",
            "subtitle": "Web Design & Development",
            "tag": "ED",
            "avatars": ["👩‍💻", "👨‍💻"],
            "status": "On Going",
            "progression": "51%",
            "due_date": "Aug, 17 2024",
        },
        {
            "id": 3,
            "title": "Random Forest Anomaly Shield",
            "subtitle": "Real-time Credit Card Fraud Detection",
            "tag": "FS",
            "status": "Completed",
            "progression": "99.8%",
            "due_date": "Live Engine",
        },
    ]

    return {
        "metrics": top_metrics,
        "monthly_trends": monthly_trends,
        "tasks": tasks,
        "eval_results": eval_results,
    }


@app.get("/api/presets")
def get_presets():
    return {
        "legit": LEGIT_PRESET,
        "fraud": FRAUD_PRESET,
    }


@app.get("/api/models")
def get_models():
    ensure_data_and_models()
    return {
        "models": list(backend.CLASSIFIER_DEFINITIONS.keys()),
        "trained": list(STATE["trained_models"].keys()),
        "eval_results": STATE["eval_results"],
    }


@app.post("/api/train")
def train_model_endpoint(req: TrainRequest):
    ensure_data_and_models()
    X_train = STATE["X_train"]
    y_train = STATE["y_train"]
    X_test = STATE["X_test"]
    y_test = STATE["y_test"]

    if req.model_name == "all":
        target_names = list(backend.CLASSIFIER_DEFINITIONS.keys())
    elif req.model_name in backend.CLASSIFIER_DEFINITIONS:
        target_names = [req.model_name]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {req.model_name}")

    results = []
    for name in target_names:
        m = backend.train_model(name, X_train, y_train)
        res = backend.evaluate_model(m, X_test, y_test, name)
        STATE["trained_models"][name] = m

        evals = [r for r in STATE["eval_results"] if r["name"] != name]
        evals.append(res)
        evals.sort(key=lambda r: r["f1"], reverse=True)
        STATE["eval_results"] = evals
        results.append(res)

    return {
        "success": True,
        "message": f"Successfully trained {len(target_names)} model(s).",
        "results": STATE["eval_results"],
    }


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    ensure_data_and_models()
    model_name = req.model_name or "Random Forest"

    if model_name not in STATE["trained_models"]:
        # Auto-train if missing
        STATE["trained_models"][model_name] = backend.train_model(
            model_name, STATE["X_train"], STATE["y_train"]
        )

    model = STATE["trained_models"][model_name]
    scaler = STATE["scaler"]
    feature_names = STATE["feature_names"]

    try:
        res = backend.predict_transaction(model, scaler, req.transaction, feature_names)
        return {
            "success": True,
            "model_used": model_name,
            **res,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/dataset/load-demo")
def load_demo_dataset():
    demo_df = backend.generate_sample_dataset(n_samples=2500)
    STATE["df"] = demo_df
    STATE["dataset_ready"] = True
    X_tr, X_te, y_tr, y_te, scl, f_names = backend.preprocess_data(demo_df)
    STATE["X_train"] = X_tr
    STATE["X_test"] = X_te
    STATE["y_train"] = y_tr
    STATE["y_test"] = y_te
    STATE["scaler"] = scl
    STATE["feature_names"] = f_names

    # Retrain models on new demo split
    STATE["eval_results"] = []
    for name in backend.CLASSIFIER_DEFINITIONS:
        m = backend.train_model(name, X_tr, y_tr)
        STATE["trained_models"][name] = m
        res = backend.evaluate_model(m, X_te, y_te, name)
        STATE["eval_results"].append(res)

    STATE["eval_results"].sort(key=lambda r: r["f1"], reverse=True)

    return {
        "success": True,
        "rows": len(demo_df),
        "columns": demo_df.shape[1],
        "eval_results": STATE["eval_results"],
    }
