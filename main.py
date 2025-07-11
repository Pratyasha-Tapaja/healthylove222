from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("ensemble_calorie_model.joblib")

class InputData(BaseModel):
    bmi: float
    age: int
    duration: int
    heart_rate: int

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.bmi, data.age, data.duration, data.heart_rate]])
    prediction = model.predict(input_array)
    return {"calories_burnt": round(float(prediction[0]), 2)}
