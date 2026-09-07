import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Churn Classifier API",
    description="Microservice d'inférence en production pour prédire le désabonnement client (Telco Churn).",
    version="1.0.0"
)

# Configuration MLflow
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        try:
            # Tente de charger le modèle depuis le Model Registry
            MODEL = mlflow.sklearn.load_model("models:/ChurnClassifier/latest")
        except Exception:  # noqa: BLE001
            # Fallback vers le dernier run si disponible
            if os.path.exists(".last_run_id"):
                with open(".last_run_id", "r", encoding="utf-8") as f:
                    run_id = f.read().strip()
                MODEL = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            else:
                raise RuntimeError("Aucun modèle trouvé dans MLflow.")
    return MODEL


class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float | None = 0.0
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API ChurnClassifier. Rendez-vous sur /docs pour tester."}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ChurnClassifier"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        model = get_model()
        data_df = pd.DataFrame([customer.model_dump()])
        prediction = int(model.predict(data_df)[0])
        probability = float(model.predict_proba(data_df)[0, 1])

        return {
            "churn_prediction": prediction,
            "churn_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4)
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
