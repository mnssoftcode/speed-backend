from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost as xgb
import pandas as pd

app = FastAPI()

# Load all models
speed_model = joblib.load("speed_model.pkl")
crash_model = joblib.load("crash_model.pkl")
risk_model = joblib.load("risk_model.pkl")
risk_rf_model = joblib.load("risk_rf_model.pkl")
risk_xgb_model = joblib.load("risk_xgb_model.pkl")

# Body schema for speed prediction
class SpeedInput(BaseModel):
    speeds: list

# Body schema for crash prediction
class CrashInput(BaseModel):
    speed: float
    accel: float
    gyro: float
    jerk: float


# Body schema for risk prediction
class RiskInput(BaseModel):
    speed: float
    accel: float
    brake: float
    gyro: float
    jerk: float


# -----------------------------------------
# 1️⃣ SPEED PREDICTION ENDPOINT
# -----------------------------------------
@app.post("/predict-speed")
def predict_speed(data: SpeedInput):
    speeds = np.array(data.speeds).reshape(1, -1)
    next_speed = speed_model.predict(speeds)[0]
    return {"next_speed": float(next_speed)}


# -----------------------------------------
# 2️⃣ CRASH PREDICTION ENDPOINT
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
# 3️⃣ RISK PREDICTION ENDPOINT
# -----------------------------------------
@app.post("/predict-risk")
def predict_risk(data: RiskInput):
    X = np.array([[data.speed, data.accel, data.brake, data.gyro, data.jerk]])
    pred = risk_model.predict(X)[0]

    # Convert model output to text label
    risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}

    return {
        "risk_level": int(pred),
        "status": risk_labels[pred]
    }


# -----------------------------------------
# 4️⃣ RISK PREDICTION WITH RANDOM FOREST
# -----------------------------------------
@app.post("/predict-risk-rf")
def predict_risk_rf(data: RiskInput):
    X = np.array([[data.speed, data.accel, data.brake, data.gyro, data.jerk]])
    pred = risk_rf_model.predict(X)[0]

    risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}

    return {
        "risk_level": int(pred),
        "status": risk_labels[pred]
    }


# -----------------------------------------
# 5️⃣ RISK PREDICTION WITH XGBOOST
# -----------------------------------------
@app.post("/predict-risk-xgb")
def predict_risk_xgb(data: RiskInput):
    # Create dataframe with correct feature names

    df_input = pd.DataFrame([{
        "speed": data.speed,
        "accel": data.accel,
        "brake": data.brake,
        "gyro": data.gyro,
        "jerk": data.jerk
    }])

    dinput = xgb.DMatrix(df_input)

    pred = int(risk_xgb_model.predict(dinput)[0])

    risk_labels = {0: "LOW RISK", 1: "MEDIUM RISK", 2: "HIGH RISK"}

    return {
        "risk_level": pred,
        "status": risk_labels[pred]
    }
