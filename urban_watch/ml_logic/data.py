"""
# Sentinel-2 Data Processing Utilities


This module provides helper functions to generate bounding boxes, download Sentinel-2
imagery from SentinelHub, preprocess bands into RGB composite images, and load previously
saved tiles from disk. It includes:


- Automatic generation of geographic bounding boxes (using UTM when possible)
- Batch download of multi-band Sentinel-2 tiles
- Image normalization and RGB extraction
- Local storage of tiles and metadata
- Reloading of previously downloaded datasets


The core purpose of this module is to simplify large-scale extraction of Sentinel-2 imagery
for geospatial machine learning workflows.
"""


from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig, MosaickingOrder, bbox_to_dimensions
from pyproj import Transformer
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pyproj import CRS as pyCRS
import math
from pathlib import Path
import time
from datetime import datetime, timedelta

def make_bbox_global(lat, lon, km_size=5):
    """
    Create a bounding box centered on (lat, lon) covering a km_size × km_size area.
    Automatically uses UTM when applicable, otherwise falls back to WGS84.

    Returns: SentinelHub BBox + CRS used.
    """

    # 1) Tente d'utiliser l'UTM correspondant
    try:
        # Automatically find UTM zone
        utm_crs = pyCRS.from_epsg(f"326{int((lon + 180) / 6) + 1}")
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        # Convert center to UTM coordinates (meters)
        x_center, y_center = transformer.transform(lon, lat)

        half = (km_size * 1000) / 2

        # UTM bbox in meters
        bbox_utm = [x_center - half, y_center - half,
                    x_center + half, y_center + half]

        # Return bbox and UTM CRS
        return BBox(bbox_utm, CRS(utm_crs.to_epsg()))

    except Exception:
        # 2) If no UTM available, fallback to approximate WGS84
        delta_lat = (km_size / 111) / 2
        delta_lon = (km_size / (111 * math.cos(math.radians(lat)))) / 2

        bbox_wgs = [
            lon - delta_lon,
            lat - delta_lat,
            lon + delta_lon,
            lat + delta_lat,
        ]

        return BBox(bbox_wgs, CRS.WGS84)



RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "features_x"

def get_data(list_bbox, config):

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    images = []
    metadata_list = []

    for i, (lat, lon) in enumerate(list_bbox):

        print(f"📡 Downloading tile {i} at {lat},{lon} ...")

        # 1) Generate bbox automatically
        bbox = make_bbox_global(lat, lon, km_size=5)

        # 2) SentinelHub request
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B01","B02","B03","B04","B05","B06","B08","B8A","B11","B12"],
                output: { bands: 10, sampleType: "FLOAT32"}
            };
        }
        function evaluatePixel(sample) {
            return [
            sample.B01,
            sample.B02,
            sample.B03,
            sample.B04,
            sample.B05,
            sample.B06,
            sample.B08,
            sample.B8A,
            sample.B11,
            sample.B12
        ];
        }
        """

        try:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(
                    DataCollection.SENTINEL2_L2A,
                    time_interval=("2021-06-01", "2021-07-31"),
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                )],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                resolution=(10, 10),
                config=config
            )
            image = request.get_data()[0]  # (H, W, 10)
            images.append(image)

            # 3) Create tile folder
            tile_dir = os.path.join(RAW_DATA_DIR, f"tile_{i}")
            os.makedirs(tile_dir, exist_ok=True)

            # 4) Save numpy array
            np.save(os.path.join(tile_dir, "X.npy"), image)

            # 5) Save metadata
            meta = {
                "lat": lat,
                "lon": lon,
                "bbox": list(bbox),
                "bbox_crs": str(bbox.crs),
                "bands": ["B01","B02","B03","B04","B05","B06","B08","B8A","B11","B12"],
                "resolution": 10
            }

            with open(os.path.join(tile_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)

            metadata_list.append(meta)
            print(f"✔ Saved tile {i} in {tile_dir}.\nImage shape: {image.shape}")

            # PAUSE to avoid SentinelHub throttling
            time.sleep(5)

        except Exception as e:
            print(f"  ❌ Error downloading tile {i}: {str(e)}")

    print(f"\n✅ Downloaded {len(images)} images")
    return images, metadata_list




def download_sentinel_image(date, lon, lat, size_km, config):
    """
    Download a Sentinel-2 tile centered on a WGS84 point (lon, lat)
    inside a square window of size_km × size_km kilometers.

    Args:
        date (str): "YYYY-MM-DD"
        lon, lat: WGS84 coordinates of the center of the area of interest
        size_km (float): size of the window in kilometers
        config: SentinelHub configuration

    Returns:
        numpy array (H, W, 10)
    """

    # --- Date range handling ---
    date = datetime.strptime(date, "%Y-%m-%d")
    date_minus_15 = date - timedelta(days=15)
    date_plus_15 = date + timedelta(days=15)

    # --- Approximate km → degree conversion ---

    dlat = (size_km / 2) / 111 # 1° latitude ≈ 111 km
    dlon = (size_km / 2) / (111 * np.cos(np.radians(lat))) # 1° longitude depend latitude

    # --- Build WGS84 bbox ---
    xmin = lon - dlon
    xmax = lon + dlon
    ymin = lat - dlat
    ymax = lat + dlat

    bbox = BBox(bbox=[xmin, ymin, xmax, ymax], crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=10)

    # --- Script Sentinel-2 ---
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B01","B02","B03","B04","B05","B06","B08","B8A","B11","B12"],
            output: { bands: 10, sampleType: "FLOAT32"}
        };
    }
    function evaluatePixel(sample) {
        return [
            sample.B01, sample.B02, sample.B03, sample.B04, sample.B05,
            sample.B06, sample.B08, sample.B8A, sample.B11, sample.B12
        ];
    }
    """

    # --- SentinelHub request ---
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            DataCollection.SENTINEL2_L2A,
            time_interval=(
                date_minus_15.strftime("%Y-%m-%d"),
                date_plus_15.strftime("%Y-%m-%d")
            ),
            mosaicking_order=MosaickingOrder.LEAST_CC,
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config
    )

    image = request.get_data()[0]
    return image


def image_rgb(image_sat):
    # Extract correct RGB bands
    B2 = image_sat[:, :, 0]   # Blue
    B3 = image_sat[:, :, 1]   # Green
    B4 = image_sat[:, :, 2]   # Red

    # Stack RGB
    RGB = np.dstack([B4, B3, B2])

    """Percentile normalization + gamma correction"""
    RGB = RGB.astype(float)
    RGB = (RGB - RGB.min()) / (RGB.max() - RGB.min() + 1e-6)

    p2 = np.percentile(RGB, 2)
    p98 = np.percentile(RGB, 98)
    RGB_stretched = np.clip((RGB - p2) / (p98 - p2), 0, 1)

    gamma = 0.5
    return np.clip(RGB_stretched ** gamma, 0, 1)


def load_data(raw_data_dir=RAW_DATA_DIR):
    """
    Load all tiles stored in raw_data/.

    Returns:
      - X : array shape (n_tiles, H, W, bands)
      - meta : list of dict (lat, lon, bbox, bbox_crs, ...)
    """

    X_list = []
    meta_list = []

    # # Iterate through raw_data/tile_XX folders
    for tile_name in sorted(os.listdir(RAW_DATA_DIR)):
        tile_dir = os.path.join(RAW_DATA_DIR, tile_name)

        if not os.path.isdir(tile_dir):
            continue

        # Load X.npy
        x_path = os.path.join(tile_dir, "X.npy")
        if not os.path.exists(x_path):
            continue

        X = np.load(x_path)
        X_list.append(X)

        # Load meta.json
        meta_path = os.path.join(tile_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            meta = {}

        meta_list.append(meta)

    # Convert list to numpy array
    X_array = np.stack(X_list, axis=0)   # shape = (n_tiles, H, W, 10)

    return X_array, meta_list
