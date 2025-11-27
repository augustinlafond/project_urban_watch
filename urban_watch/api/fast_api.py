"""
# UrbanWatch API
This module exposes a FastAPI-based HTTP API used to:
- Download Sentinel‑2 satellite imagery from SentinelHub.
- Generate an RGB visualization from raw satellite data.
- Run an urbanization prediction model (Random Forest).
- Return predictions, RGB image, and aggregated urbanization scores.


The API provides one main endpoint:
- **/predict**: Fetches a satellite tile for coordinates and date, processes it, predicts urbanization, and returns results.
"""


# Standard library
from datetime import datetime

# Third-party libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentinelhub import SHConfig

# Project-specific imports
from urban_watch.params import SH_CLIENT_ID, SH_CLIENT_SECRET
from urban_watch.ml_logic.data import download_sentinel_image, image_rgb
from urban_watch.ml_logic.registry import load_model
from urban_watch.interface.main import pred


# Configure access to the SentinelHub API
config = SHConfig()
config.sh_client_id = SH_CLIENT_ID
config.sh_client_secret = SH_CLIENT_SECRET

## FastAPI app
app = FastAPI()

## CORS (For Streamlit)
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
    return {"status": "UrbanWatch API running :coche_blanche:"}


app.state.model = load_model(
    model_name="xgb_model", model_type="XGB", stage="Production"
)


# PREDICT ENDPOINT
@app.get("/predict")
def predict(date: str, lon: float, lat: float, size_km: float):
    """
    lon and lat are the coordinates in WGS84 format of the center of the bbox
    size_km is the size of the window, eg., 3 correspond to a window of 3kmx3km centered on lon/lat coordinates
    """
    ## validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "error": "Invalid date format. Expected YYYY-MM-DD",
            "example": "2021-06-15",
        }

    ## download Sentinel-2 image
    try:
        image_sat = download_sentinel_image(date, lon, lat, size_km, config)
    except Exception as e:
        return {"error": f"SentinelHub download failed: {str(e)}"}

    ## Return RGB image
    rgb_image = image_rgb(image_sat)

    ## call the existing prediction function
    y_pred_full, mean_urban_score = pred(X_pred=image_sat, model=app.state.model)

    ## API response
    return {
        "urbanization_score": float(round(mean_urban_score, 2)),
        "prediction": y_pred_full.tolist(),
        "rgb": rgb_image.tolist(),
    }
