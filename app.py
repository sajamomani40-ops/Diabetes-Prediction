from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

models = joblib.load("trained_models_all_data.pkl")

FEATURES = ['gender', 'age', 'hypertension', 'heart_disease',
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
    gender: float = Form(...),
    age: float = Form(...),
    hypertension: float = Form(...),
    heart_disease: float = Form(...),
    smoking_history: float = Form(...),
    bmi: float = Form(...),
    HbA1c_level: float = Form(...),
    blood_glucose_level: float = Form(...)
):
    try:
        input_data = pd.DataFrame([[gender, age, hypertension, heart_disease,
                                    smoking_history, bmi, HbA1c_level, blood_glucose_level]],
                                  columns=FEATURES)
        
        predictions = {}
        votes = 0

        for name, model in models.items():
            pred = model.predict(input_data)[0]
            label = "Diabetes" if pred == 1 else "No Diabetes"
            predictions[name] = label
            if pred == 1:
                votes += 1

        majority = "Diabetes ✅" if votes > len(models) / 2 else "No Diabetes 🟢"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "predictions": predictions,
            "majority": majority
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "predictions": {"Error": str(e)},
            "majority": "❌ Prediction Failed"
        })
