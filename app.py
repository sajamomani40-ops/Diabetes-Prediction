from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import gzip
import joblib
import pandas as pd
from typing import Union, Dict, Any

app = FastAPI()
# تأكد أن مجلد templates يحتوي على index.html
templates = Jinja2Templates(directory="templates")

FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

def load_model():
    base = "trained_models_all_data.pkl"
    gz = base + ".gz"
    if os.path.exists(base):
        return joblib.load(base)
    elif os.path.exists(gz):
        import gzip
        with gzip.open(gz, "rb") as f:
            return joblib.load(f)
    else:
        raise FileNotFoundError("Model file not found: trained_models_all_data.pkl[.gz]")

models = load_model()

_GENDER_MAP = {
    "male": 1, "m": 1, "ذكر": 1,
    "female": 0, "f": 0, "أنثى": 0, "انثى": 0,
}

def maybe_cast_gender(x: str) -> Union[int, str]:
    if x is None:
        return x
    key = str(x).strip().lower()
    return _GENDER_MAP.get(key, x)

def maybe_int(x: Union[str, int, float]) -> int:
    return int(float(x))

def maybe_float(x: Union[str, int, float]) -> float:
    return float(x)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # أول تحميل للصفحة: لا يوجد نتائج
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "majority": None, "predictions": None, "error_msg": None},
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: str = Form(...),
    age: str = Form(...),
    hypertension: str = Form(...),
    heart_disease: str = Form(...),
    smoking_history: str = Form(...),
    bmi: str = Form(...),
    HbA1c_level: str = Form(...),
    blood_glucose_level: str = Form(...),
):
    try:
        # بناء صف الإدخال
        row: Dict[str, Any] = {
            "gender": maybe_cast_gender(gender),
            "age": maybe_float(age),
            "hypertension": maybe_int(hypertension),
            "heart_disease": maybe_int(heart_disease),
            "smoking_history": str(smoking_history).strip(),
            "bmi": maybe_float(bmi),
            "HbA1c_level": maybe_float(HbA1c_level),
            "blood_glucose_level": maybe_float(blood_glucose_level),
        }
        x = pd.DataFrame([row], columns=FEATURES)

        def predict_one(model) -> int:
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(x)[0][1])
                return int(proba >= 0.5)
            return int(model.predict(x)[0])

        # تجميع التصويتات
        votes = []
        if isinstance(models, dict):
            for mdl in models.values():
                votes.append(predict_one(mdl))
        else:
            votes.append(predict_one(models))

        ones = sum(votes)
        zeros = len(votes) - ones
        is_positive = ones >= zeros

        # يجب أن تكون majority إنجليزية لتطابق شرط القالب (== 'Diabetes')
        majority = "Diabetes" if is_positive else "No Diabetes"

        # مرر predictions كقيمة Truthy ليُظهر قسم النتائج
        predictions = {"votes_1": ones, "votes_0": zeros, "n_models": len(votes)}

        # Debug (اختياري)
        print("INPUT ROW:", row)
        print("VOTES:", votes, "MAJORITY:", majority)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "majority": majority,
                "predictions": predictions,
                "error_msg": None,
            },
        )

    except Exception as e:
        # في حال الخطأ، نعرض رسالة ودية
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "majority": None,
                "predictions": None,
                "error_msg": f"{type(e).__name__}: {str(e)}",
            },
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
