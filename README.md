🛰️ UrbanWatch – Detecting urban artificialization using Sentinel-2 images

UrbanWatch is a data science and AI project that automatically detects urban artificialization using Sentinel-2 satellite images (10 m resolution). It relies on: a robust pipeline for downloading and preprocessing SentinelHub data, the ESA WorldCover 2021 layer as ground truth, an XGBoost model trained on more than 9 million pixels, a FastAPI API that can be used to request an artificialization prediction for any area in the world and returns an average urbanization score for that area as well as the predicted image composed of urban pixels (value 1) and non-urban pixels (value 0).

🎥 **Project pitch & demo (Le Wagon Demo Day)**  
[Watch the pitch and live demo](https://www.youtube.com/watch?v=FckypTvmKao)

📌 Objective

The objective of the project is to produce a model capable of predicting an average urbanization score and an urbanization map from a Sentinel-2 tile centered around a GPS point. The final pipeline allows: automatic downloading of Sentinel-2 satellite images, application of an s2cloudless cloud mask, calculation of spectral indices (NDVI, NDBI, MNDWI, etc.), standardize and flatten each pixel into a vector, associate each pixel with its urban vs. non-urban class using the model, generate a complete map of the model's predictions and an average urbanization score for the area, and expose everything in an API that allows the model to be queried.


📦 Project architecture

```text
urban_watch/
├── ml_logic/
│   ├── data.py               # SentinelHub download, loading, metadata
│   ├── preprocessing.py      # Cloud mask, NDVI / NDBI / MNDWI indices, normalization
│   ├── labels.py             # WorldCover conversion, reprojection, cropping
│   ├── model.py              # Training, prediction, evaluation
│   ├── registry.py           # MLflow tracking
│
├── interface/
│   ├── main.py               # Orchestration: full pipeline, training, prediction
│
├── api/
│   ├── api.py                # FastAPI server for real-time predictions
│
├── data/
│   ├── features_x/           # Downloaded Sentinel-2 images
│   ├── labels_y/             # Reprojected WorldCover tiles
│
├── requirements.txt
└── README.md
```

🚀 Methodology

1. 🛰️ Sentinel-2 download (SentinelHub)

For each GPS coordinate, a 5 km × 5 km bbox is generated. Then the satellite image is downloaded via SentinelHub API with the following parameters :

- SentinelHubRequest (SENTINEL2_L2A)
- Resolution: 10 m
- MosaickingOrder: LEAST_CLOUD_COVERAGE
- The 10 Sentinel-2 bands are retrieved (B01, B02, B03, … B12).

The raw data is saved in .npy format.

2. ☁️ Cloud masking (s2cloudless)
   
A cloud mask is generated and cloudy pixels are removed.

3. 🧮 Calculation of spectral indices

Three essential indices are added:
- NDVI – vegetation
- NDBI – urban areas
- MNDWI – water and wetlands
→ The image goes from 10 to 13 bands.

4. ⚙️ Normalization & flattening

Each image:
- is normalized band-by-band (min-max/standardization),
- is converted into a 2D table with the following format: N_valid_pixels × 13 bands

5. 🏷️ Label construction (WorldCover 2021)

Each bbox is converted:
- from Sentinel-2 CRS → WGS84,
- cut out from the corresponding ESA tile,
- reprojected into the Sentinel-2 CRS of tile X (hence the natural appearance of a few -1 = NoData values, which are in turn removed from y and X).
- The WorldCover values are then converted:
Urban → 1
Non-urban → 0

6. 🤖 Modeling: XGBoost

- More than 9 million pixels used for training
- Features: 13 values per pixel
- Target: urban vs. non-urban (binary)

The final model is saved in MLflow, then deployed in the API.

🔮 FastAPI API

The API exposes an endpoint that allows you to:
- download a Sentinel-2 image around a GPS point,
- launch the preprocessing pipeline,
- produce a prediction,
- return an average artificialization score.

Main endpoint:
```text
GET /predict?lon=5.4389&lat=43.5306&date=2021-06-15&size_km=3
```
Réponse :
```text
{
  "urbanization_score": 0.27,
  "prediction": [...300x300...],
  "image_rgb": [...encoded RGB...]
}
```

🧪 Reproducibility

Installation

```text
pip install -r requirements.txt
```
Environment variables

Create a .env file:

```text
SH_CLIENT_ID=xxx       #SentinelHub client ID
SH_CLIENT_SECRET=xxx   #SentinelHub client secret
BUCKET_NAME=...
GCP_PROJECT=...
```

Run the complete pipeline

```text
from urban_watch.interface.main import full_preproc_pipeline
X, y = full_preproc_pipeline()
```

Train the model
```text
from urban_watch.interface.main import train
train(X,y, model_type="xgb_tuned")
```

Launch the API
```text
uvicorn api.api:app --reload
```

📈 Results

The XGBoost model achieves:
```text
Précision : 0.84
Recall : 0.83
F1-score : 0.84
Accuracy : 0.91
```
