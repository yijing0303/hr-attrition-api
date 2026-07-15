# ============================================================
# Task 2d: Model Deployment - FastAPI Application
# HR Employee Attrition Prediction API
# ============================================================

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Dict, Any
import uvicorn
import os

# ============================================================
# Load the trained model and scaler
# ============================================================
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.pkl')

print("✅ Model loaded successfully!")


# ============================================================
# Define input data schema (30 features)
# ============================================================
class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: int
    DailyRate: int
    Department: int
    DistanceFromHome: int
    Education: int
    EducationField: int
    EnvironmentSatisfaction: int
    Gender: int
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: int
    JobSatisfaction: int
    MaritalStatus: int
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: int
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


class PredictionResponse(BaseModel):
    employee_data: Dict[str, Any]
    attrition_probability: float
    attrition_prediction: int
    prediction_label: str


# ============================================================
# Create FastAPI app
# ============================================================
app = FastAPI(
    title="HR Employee Attrition Prediction API",
    description="Predicts employee attrition risk using XGBoost model",
    version="1.0.0"
)

EXPECTED_COLUMNS = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "message": "HR Employee Attrition Prediction API",
        "status": "running",
        "model": "XGBoost Optimized",
        "features": len(EXPECTED_COLUMNS),
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/predict": "POST - Submit employee data for prediction"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict")
def predict(employee: EmployeeData):
    try:
        input_dict = employee.dict()
        input_data = pd.DataFrame([input_dict])
        input_data = input_data[EXPECTED_COLUMNS]

        probability = model.predict_proba(input_data)[0][1]
        threshold = 0.37
        prediction_binary = 1 if probability >= threshold else 0

        return PredictionResponse(
            employee_data=input_dict,
            attrition_probability=round(float(probability), 4),
            attrition_prediction=prediction_binary,
            prediction_label="At Risk" if prediction_binary == 1 else "Stable"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)