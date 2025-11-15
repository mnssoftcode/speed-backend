from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# -----------------------------------------
# FASTAPI INITIALIZATION
# -----------------------------------------
app = FastAPI()

# -----------------------------------------
# LOAD ONLY SMALL MODELS GLOBALLY
# (Large models will be lazy-loaded)
# -----------------------------------------
speed_model = joblib.load("speed_model.pkl")        # very small
crash_model = joblib.load("crash_model.pkl")        # small
risk_model = joblib.load("risk_model.pkl")          # small (Decision Tree)

# -----------------------------------------
# REQUEST BODY SCHEMAS
# -----------------------------------------
class SpeedInput(BaseModel):
    speeds: list

class CrashInput(BaseModel):
    speed: float
    accel: float
    gyro: float
    jerk: float

class RiskInput(BaseModel):
    speed: float
    accel: float
    brake: float
    gyro: float
    jerk: float


# -----------------------------------------
# LABELS USED IN MULTIPLE MODELS
# -----------------------------------------
risk_labels = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

# -----------------------------------------
# 1️⃣ SPEED PREDICTION
# -----------------------------------------
@app.post("/predict-speed")
def predict_speed(data: SpeedInput):
    speeds = np.array(data.speeds).reshape(1, -1)
    next_speed = speed_model.predict(speeds)[0]

    return {
        "next_speed": float(next_speed)
    }


# -----------------------------------------
# 2️⃣ CRASH DETECTION
# -----------------------------------------
@app.post("/predict-crash")
def predict_crash(input: CrashInput):
    X = np.array([[input.speed, input.accel, input.gyro, input.jerk]])
    pred = crash_model.predict(X)[0]

    result = "CRASH DETECTED" if pred == 1 else "NORMAL DRIVING"

    return {
        "crash_raw": int(pred),
        "status": result
    }


# -----------------------------------------
# 3️⃣ RISK PREDICTION (Decision Tree - lightweight)
# -----------------------------------------
@app.post("/predict-risk")
def predict_risk(data: RiskInput):
    X = np.array([[data.speed, data.accel, data.brake, data.gyro, data.jerk]])
    pred = risk_model.predict(X)[0]

    return {
        "risk_level": int(pred),
        "status": risk_labels[int(pred)]
    }


# -----------------------------------------
# 4️⃣ RISK (Random Forest - LAZY LOAD)
# -----------------------------------------
@app.post("/predict-risk-rf")
def predict_risk_rf(data: RiskInput):
    # Lazy load heavy model
    rf_model = joblib.load("risk_rf_model.pkl")

    X = np.array([[data.speed, data.accel, data.brake, data.gyro, data.jerk]])
    pred = rf_model.predict(X)[0]

    return {
        "risk_level": int(pred),
        "status": risk_labels[int(pred)]
    }


# -----------------------------------------
# 5️⃣ RISK (XGBoost - LAZY LOAD, special handling)
# -----------------------------------------
@app.post("/predict-risk-xgb")
def predict_risk_xgb(data: RiskInput):
    # Lazy load XGBoost model (very heavy)
    xgb_model = joblib.load("risk_xgb_model.pkl")

    X = np.array([[data.speed, data.accel, data.brake, data.gyro, data.jerk]])
    pred = xgb_model.predict(X)[0]

    return {
        "risk_level": int(pred),
        "status": risk_labels[int(pred)]
    }
    
# -----------------------------------------
# 6️⃣ RISK PREDICTION WITH SVM (Lazy Load)
# -----------------------------------------
@app.post("/predict-risk-svm")
def predict_risk_svm(data: RiskInput):
    # Lazy load SVM + Scaler (heavy)
    svm_model = joblib.load("risk_svm_model.pkl")
    scaler = joblib.load("risk_svm_scaler.pkl")

    # Prepare features
    X = np.array([[data.speed, data.accel, data.brake, data.gyro, data.jerk]])

    # Scale the input
    X_scaled = scaler.transform(X)

    # Predict
    pred = svm_model.predict(X_scaled)[0]

    risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}

    return {
        "risk_level": int(pred),
        "status": risk_labels[int(pred)]
    }
