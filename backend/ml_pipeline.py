import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

def load_and_preprocess(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargePerService"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["HasSecurity"] = ((df["OnlineSecurity"] == "Yes") | (df["DeviceProtection"] == "Yes")).astype(int)
    df["HasSupport"] = (df["TechSupport"] == "Yes").astype(int)
    df["NumServices"] = (
        (df["PhoneService"] == "Yes").astype(int) +
        (df["MultipleLines"] == "Yes").astype(int) +
        (df["OnlineSecurity"] == "Yes").astype(int) +
        (df["OnlineBackup"] == "Yes").astype(int) +
        (df["DeviceProtection"] == "Yes").astype(int) +
        (df["TechSupport"] == "Yes").astype(int) +
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] >= 48).astype(int)
    df["MonthlyTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    contract_risk = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["ContractRisk"] = df["Contract"].map(contract_risk)
    return df

def encode_features(df):
    df = df.copy()
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    multi_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                  "Contract", "PaymentMethod"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    return df

def get_feature_columns(df):
    return [c for c in df.columns if c not in ["customerID", "Churn"]]

def train_model():
    df_raw = load_and_preprocess()
    df = encode_features(df_raw)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
    lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    lr.fit(X_train_s, y_train)

    rf_probs = rf.predict_proba(X_test)[:, 1]
    gbm_probs = gbm.predict_proba(X_test)[:, 1]
    lr_probs = lr.predict_proba(X_test_s)[:, 1]
    ensemble_probs = 0.45 * gbm_probs + 0.40 * rf_probs + 0.15 * lr_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, ensemble_preds)
    roc = roc_auc_score(y_test, ensemble_probs)
    report = classification_report(y_test, ensemble_preds, output_dict=True)
    cm = confusion_matrix(y_test, ensemble_preds).tolist()
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gbm, X, y, cv=cv, scoring="accuracy")

    fi_gbm = pd.Series(gbm.feature_importances_, index=feature_cols)
    fi_rf = pd.Series(rf.feature_importances_, index=feature_cols)
    fi_avg = ((fi_gbm + fi_rf) / 2).sort_values(ascending=False)

    model_bundle = {
        "rf": rf, "gbm": gbm, "lr": lr, "scaler": scaler,
        "feature_cols": feature_cols,
        "accuracy": float(accuracy),
        "roc_auc": float(roc),
        "report": report,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "feature_importance": fi_avg.head(15).to_dict(),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Model trained. Accuracy: {accuracy:.4f} | ROC-AUC: {roc:.4f}")
    return model_bundle

def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def _predict_probs(model_bundle, X):
    X = X.fillna(0)
    X_s = model_bundle["scaler"].transform(X)
    p_rf = model_bundle["rf"].predict_proba(X)[:, 1]
    p_gbm = model_bundle["gbm"].predict_proba(X)[:, 1]
    p_lr = model_bundle["lr"].predict_proba(X_s)[:, 1]
    return 0.45 * p_gbm + 0.40 * p_rf + 0.15 * p_lr

def predict_customer(model_bundle, customer_dict):
    df_raw = load_and_preprocess()
    row = pd.DataFrame([customer_dict])
    combined = pd.concat([df_raw, row], ignore_index=True)
    combined_enc = encode_features(combined)
    feature_cols = model_bundle["feature_cols"]
    for col in feature_cols:
        if col not in combined_enc.columns:
            combined_enc[col] = 0
    X_new = combined_enc[feature_cols].iloc[[-1]]
    prob = float(_predict_probs(model_bundle, X_new)[0])
    risk_label = "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")
    return {"churn_probability": round(prob, 4), "risk_level": risk_label}

def get_at_risk_customers(model_bundle, threshold=0.65):
    df_raw = load_and_preprocess()
    df_enc = encode_features(df_raw)
    feature_cols = model_bundle["feature_cols"]
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    X = df_enc[feature_cols]
    probs = _predict_probs(model_bundle, X)
    df_raw["churn_probability"] = probs
    df_raw["risk_level"] = pd.cut(probs, bins=[0, 0.35, 0.65, 1.0], labels=["Low", "Medium", "High"])
    df_raw["actual_churn"] = df_raw["Churn"]
    at_risk = df_raw[probs >= threshold].sort_values("churn_probability", ascending=False)
    return at_risk[["customerID", "tenure", "Contract", "MonthlyCharges",
                    "InternetService", "TechSupport", "NumServices",
                    "churn_probability", "risk_level", "actual_churn"]].head(200)

def get_full_analysis(model_bundle):
    df_raw = load_and_preprocess()
    df_enc = encode_features(df_raw)
    feature_cols = model_bundle["feature_cols"]
    for col in feature_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    X = df_enc[feature_cols]
    probs = _predict_probs(model_bundle, X)
    df = df_raw.copy()
    df["churn_probability"] = probs

    contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: round((x == "Yes").mean() * 100, 2)).to_dict()
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 48, 72], labels=["0-6m", "7-12m", "13-24m", "25-48m", "48-72m"])
    tenure_churn = df.groupby("tenure_bucket", observed=True)["Churn"].apply(lambda x: round((x == "Yes").mean() * 100, 2)).to_dict()
    internet_churn = df.groupby("InternetService")["Churn"].apply(lambda x: round((x == "Yes").mean() * 100, 2)).to_dict()
    payment_churn = df.groupby("PaymentMethod")["Churn"].apply(lambda x: round((x == "Yes").mean() * 100, 2)).to_dict()
    churned_charges = df[df["Churn"] == "Yes"]["MonthlyCharges"].describe().round(2).to_dict()
    retained_charges = df[df["Churn"] == "No"]["MonthlyCharges"].describe().round(2).to_dict()
    risk_dist = {
        "High": int((probs >= 0.65).sum()),
        "Medium": int(((probs >= 0.35) & (probs < 0.65)).sum()),
        "Low": int((probs < 0.35).sum()),
    }
    prob_by_contract = df.groupby("Contract")["churn_probability"].mean().round(4).to_dict()

    return {
        "total_customers": len(df),
        "churned_customers": int((df["Churn"] == "Yes").sum()),
        "churn_rate": round((df["Churn"] == "Yes").mean() * 100, 2),
        "at_risk_count": risk_dist["High"],
        "contract_churn": contract_churn,
        "tenure_churn": {str(k): v for k, v in tenure_churn.items()},
        "internet_churn": internet_churn,
        "payment_churn": payment_churn,
        "monthly_charges": {"churned": churned_charges, "retained": retained_charges},
        "risk_distribution": risk_dist,
        "prob_by_contract": prob_by_contract,
    }
