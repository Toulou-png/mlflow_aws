# api_pipeline.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from aynid_pipeline import AynidCartAbandonmentAWS
import uvicorn
import pandas as pd
import joblib
import os

app = FastAPI(title="Aynid ML Pipeline API", version="1.1")

# Initialisation du pipeline ML
pipeline = AynidCartAbandonmentAWS()

# === Modèles de données ===
class TrainRequest(BaseModel):
    n_samples: int = 5000

class PredictRequest(BaseModel):
    session_duration: float
    pages_visited: int
    cart_value: float
    time_of_day: int
    device_mobile: int
    user_returning: int
    items_in_cart: int


# === Endpoints ===
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API FastAPI du pipeline ML Aynid 🚀"}


@app.post("/train")
def train_model(request: TrainRequest):
    """Lance un entraînement complet"""
    try:
        df = pipeline.prepare_data(n_samples=request.n_samples)
        model, metrics = pipeline.train_model(df)

        # Séparation des données train/test
        train_size = int(0.8 * len(df))
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        # Sauvegarde du modèle localement
        model_path = "model_latest.pkl"
        joblib.dump(model, model_path)

        return {
            "status": "OK",
            "metrics": metrics,
            "model_path": model_path,
            "raw_data": df.head(100).to_dict(orient="records"),
            "train_data": df_train.head(50).to_dict(orient="records"),
            "test_data": df_test.head(50).to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'entraînement : {str(e)}")


@app.post("/predict")
def predict(data: PredictRequest):
    """Retourne la prédiction d’abandon ou d’achat"""
    try:
        model_path = "model_latest.pkl"

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Aucun modèle entraîné trouvé. Lancez /train d'abord.")

        # Charger le modèle
        model = joblib.load(model_path)

        # Convertir les données d’entrée
        input_df = pd.DataFrame([data.dict()])

        # Faire la prédiction
        pred = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0][1]

        result = "🛒 Abandon du panier" if pred == 1 else "✅ Achat effectué"

        return {
            "prediction": int(pred),
            "probability": round(float(pred_proba), 3),
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")


@app.get("/metrics")
def get_metrics():
    """Retourne un message de monitoring Prometheus"""
    return {"metrics": "Voir /train pour en générer ou consultez Prometheus sur /metrics Prometheus"}


# === Lancement ===
if __name__ == "__main__":
    uvicorn.run("api_pipeline:app", host="0.0.0.0", port=8001, reload=True)
