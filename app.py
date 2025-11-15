from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("speed_model.pkl")

# Input format
class SpeedInput(BaseModel):
    speeds: list

@app.post("/predict-speed")
def predict_speed(data: SpeedInput):
    speeds = np.array(data.speeds).reshape(1, -1)
    next_speed = model.predict(speeds)[0]
    return {"next_speed": float(next_speed)}
