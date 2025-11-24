import os
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sentinelhub import SHConfig
import mlflow.pyfunc

from urban_watch.ml_logic.data import get_data
### Load model
from urban_watch.ml_logic.preprocessing import preprocess_image

# ============================================================
# INIT (comme dans TaxiFare)
# ============================================================

load_dotenv()

app = FastAPI()

# CORS (pour pouvoir appeler l'API depuis Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SentinelHub config (credentials dans .env)
### config = SHConfig()
### config.sh_client_id = os.getenv("SH_CLIENT_ID")
### config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

app = FastAPI()

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
    lat: float,   
    lon: float,   
    km_size: int = 5  
):

    # Télécharger l'image Sentinel-2 
    try:
        images, metadata = get_data([(lat, lon)], config)
    except Exception as e:
        return {"error": f"SentinelHub download failed: {str(e)}"}

    if len(images) == 0:
        return {"error": "No image returned from SentinelHub"}

    img = images[0]

    # Préprocessing 
    X_processed, mask_valid = preprocess_image(img)

    # Prédiction du modèle MLflow
    model = app.state.model
    assert model is not None

    y_pred = model.predict(X_processed)
    y_pred = np.array(y_pred).squeeze()

    # Score global d'urbanisation = moyenne des prédictions
    urbanization_score = float(y_pred.mean())

    # Réponse JSON simple
    return {
        "lat": lat,
        "lon": lon,
        "km_size": km_size,
        "urbanization_score": urbanization_score 
    }

@app.get("/")
def root():
    return {
        "greeting": "Hello from UrbanWatch API"
    }
