import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sentinelhub import SHConfig

from urban_watch.ml_logic.data import download_sentinel_image
from urban_watch.interface.main import pred

# INIT

load_dotenv()

## SentinelHub credentials
config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

## FastAPI app
app = FastAPI()

## CORS (pour Streamlit plus tard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROOT

@app.get("/")
def root():
    return {"status": "UrbanWatch API running ✅"}

# PREDICT ENDPOINT

@app.get("/predict")
def predict(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    date: str
):
    """
    Example:
    /predict?x_min=5.1&y_min=43.2&x_max=5.2&y_max=43.3&date=2021-06-15

    coords format required:
    (x_min, y_min, x_max, y_max)
    """

    ## validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "error": "Invalid date format. Expected YYYY-MM-DD",
            "example": "2021-06-15"
        }

    ## correct bbox format for download_sentinel_image()
    bbox = (x_min, y_min, x_max, y_max)

    ## download Sentinel-2 image
    try:
        image = download_sentinel_image(date, bbox, config)
    except Exception as e:
        return {"error": f"SentinelHub download failed: {str(e)}"}

    ## call the existing prediction function
    y_pred_full, mean_score = pred(image)

    ## API response
    return {
        "bbox": bbox,
        "date": date,
        "urbanization_score": float(mean_score),
        "prediction_shape": y_pred_full.shape
    }
