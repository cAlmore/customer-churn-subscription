from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from ml_pipeline import (
    load_model, train_model, predict_customer,
    get_at_risk_customers, get_full_analysis, load_and_preprocess
)

app = FastAPI(title="Churn Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
_model = None

@app.on_event("startup")
def startup():
    global _model
    print("Loading model...")
    _model = load_model()
    print(f"Model ready. Accuracy: {_model['accuracy']:.4f}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/api/model-metrics")
def model_metrics():
    m = _model
    report = m["report"]
    return {
        "accuracy": round(m["accuracy"] * 100, 2),
        "roc_auc": round(m["roc_auc"] * 100, 2),
        "cv_accuracy": round(m["cv_mean"] * 100, 2),
        "cv_std": round(m["cv_std"] * 100, 2),
        "precision_churn": round(report["1"]["precision"] * 100, 2),
        "recall_churn": round(report["1"]["recall"] * 100, 2),
        "f1_churn": round(report["1"]["f1-score"] * 100, 2),
        "precision_retain": round(report["0"]["precision"] * 100, 2),
        "recall_retain": round(report["0"]["recall"] * 100, 2),
        "f1_retain": round(report["0"]["f1-score"] * 100, 2),
        "confusion_matrix": m["confusion_matrix"],
        "roc_curve": m["roc_curve"],
        "feature_importance": {
            k: round(float(v), 4) for k, v in m["feature_importance"].items()
        },
    }


@app.get("/api/stats")
def dashboard_stats():
    return get_full_analysis(_model)


@app.get("/api/at-risk")
def at_risk_customers(threshold: float = 0.65, limit: int = 100):
    df = get_at_risk_customers(_model, threshold)
    df = df.head(limit)
    df["churn_probability"] = df["churn_probability"].round(4)
    records = df.to_dict(orient="records")
    for r in records:
        r["risk_level"] = str(r["risk_level"])
        r["churn_probability"] = float(r["churn_probability"])
    return {"count": len(records), "customers": records}


@app.get("/api/dataset-preview")
def dataset_preview(limit: int = 20):
    df = load_and_preprocess()
    cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "Contract", "MonthlyCharges", "TotalCharges",
            "InternetService", "TechSupport", "Churn"]
    preview = df[cols].head(limit).to_dict(orient="records")
    return {"rows": preview, "total": len(df)}


class CustomerInput(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0
    customerID: Optional[str] = "PREDICT-001"
    Churn: Optional[str] = "No"


@app.post("/api/predict")
def predict(customer: CustomerInput):
    result = predict_customer(_model, customer.dict())
    risk_factors = _get_risk_factors(customer.dict(), result["churn_probability"])
    return {**result, "risk_factors": risk_factors}


@app.post("/api/retrain")
def retrain():
    global _model
    _model = train_model()
    return {"status": "retrained", "accuracy": round(_model["accuracy"] * 100, 2)}


def _get_risk_factors(c: dict, prob: float):
    factors = []
    if c["Contract"] == "Month-to-month":
        factors.append({"factor": "Month-to-month contract", "impact": "High"})
    if c["tenure"] <= 6:
        factors.append({"factor": "New customer (tenure ≤ 6 months)", "impact": "High"})
    if c["InternetService"] == "Fiber optic" and c["TechSupport"] == "No":
        factors.append({"factor": "Fiber optic without tech support", "impact": "High"})
    if c["PaymentMethod"] == "Electronic check":
        factors.append({"factor": "Electronic check payment", "impact": "Medium"})
    if c["MonthlyCharges"] > 85:
        factors.append({"factor": f"High monthly charges (${c['MonthlyCharges']})", "impact": "Medium"})
    if c["OnlineSecurity"] == "No" and c["DeviceProtection"] == "No":
        factors.append({"factor": "No security services", "impact": "Medium"})
    if c["PaperlessBilling"] == "Yes":
        factors.append({"factor": "Paperless billing", "impact": "Low"})
    if not factors:
        factors.append({"factor": "No significant risk factors detected", "impact": "Low"})
    return factors
