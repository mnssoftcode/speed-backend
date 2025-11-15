from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load both models
speed_model = joblib.load("speed_model.pkl")
crash_model = joblib.load("crash_model.pkl")

# Body schema for speed prediction
class SpeedInput(BaseModel):
    speeds: list

# Body schema for crash prediction
class CrashInput(BaseModel):
    speed: float
    accel: float
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
