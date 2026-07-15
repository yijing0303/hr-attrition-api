import requests
import json

url = "http://localhost:8000/predict"
data = {
    "Age": 34,
    "BusinessTravel": 0,
    "DailyRate": 1102,
    "Department": 0,
    "DistanceFromHome": 1,
    "Education": 3,
    "EducationField": 0,
    "EnvironmentSatisfaction": 2,
    "Gender": 0,
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 3,
    "JobRole": 0,
    "JobSatisfaction": 2,
    "MaritalStatus": 0,
    "MonthlyIncome": 6000,
    "MonthlyRate": 11002,
    "NumCompaniesWorked": 1,
    "OverTime": 1,
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 6,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 4
}

response = requests.post(url, json=data)
print(response.json())
