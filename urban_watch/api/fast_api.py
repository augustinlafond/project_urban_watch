from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentinelhub import SHConfig
from urban_watch.ml_logic.data import download_sentinel_image, image_rgb
from urban_watch.interface.main import pred
from urban_watch.params import *
# Configure access to the SentinelHub API
config = SHConfig()
config.sh_client_id = SH_CLIENT_ID
config.sh_client_secret = SH_CLIENT_SECRET
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
    return {"status": "UrbanWatch API running :coche_blanche:"}
# PREDICT ENDPOINT
@app.get("/predict")
def predict(
    date: str,
    lon: float,
    lat: float,
    size_km : float
):
    """
    Example:
    /predict?lon_lat=(0.10432014043169194, 43.23224498737862)&date=2021-06-15&size_km=3
    lon_lat is a tuple corresponding to the coordinates (longitude, latiude) in WGS84 format of the center of the bbox
    size_km is the size of the window, eg., 3 correspond to a window of 3kmx3km centered on lon_lat coordinates
    """
    ## validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "error": "Invalid date format. Expected YYYY-MM-DD",
            "example": "2021-06-15"
        }
    ## download Sentinel-2 image
    try:
        image_sat = download_sentinel_image(date, lon, lat, size_km, config)
    except Exception as e:
        return {"error": f"SentinelHub download failed: {str(e)}"}
    ## Return RGB image
    rgb_image = image_rgb(image_sat)
    ## call the existing prediction function
    y_pred_full, mean_urban_score = pred(image_sat, model_name="random_forest_model", model_type="RandomForest", stage="Production")
    ## API response
    breakpoint()
    return {
        "urbanization_score": float(round(mean_urban_score,2)),
        "prediction": y_pred_full.tolist(),
        "rgb" : rgb_image.tolist()
    }
