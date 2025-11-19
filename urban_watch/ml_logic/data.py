from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig
from dotenv import load_dotenv
from pyproj import Transformer
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import rasterio
import folium
from pyproj import CRS as pyCRS
import math


def make_bbox_global(lat, lon, km_size=3):
    """
    Crée une bbox centrée sur (lat, lon) couvrant km_size x km_size km.
    Utilise UTM automatiquement si applicable, sinon WGS84 ajusté.

    Retourne : BBox SentinelHub + CRS utilisé
    """

    # 1) Tente d'utiliser l'UTM correspondant
    try:
        # Trouve automatiquement la zone UTM
        utm_crs = pyCRS.from_epsg(f"326{int((lon + 180) / 6) + 1}")
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        # Convertit le centre en UTM (en mètres)
        x_center, y_center = transformer.transform(lon, lat)

        half = (km_size * 1000) / 2

        # bbox UTM en mètres
        bbox_utm = [x_center - half, y_center - half,
                    x_center + half, y_center + half]

        # Retourne la bbox et le CRS UTM correct
        return BBox(bbox_utm, CRS(utm_crs.to_epsg()))

    except Exception:
        # 2) Si UTM pas disponible, fallback WGS84 approximatif
        delta_lat = (km_size / 111) / 2
        delta_lon = (km_size / (111 * math.cos(math.radians(lat)))) / 2

        bbox_wgs = [
            lon - delta_lon,
            lat - delta_lat,
            lon + delta_lon,
            lat + delta_lat,
        ]

        return BBox(bbox_wgs, CRS.WGS84)



RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "raw_data")

def get_data(list_bbox, config):

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    images = []
    metadata_list = []

    for i, (lat, lon) in enumerate(list_bbox):

        print(f"📡 Downloading tile {i} at {lat},{lon} ...")

        # 1) Generate bbox automatically
        bbox = make_bbox_global(lat, lon, km_size=3)

        # 2) SentinelHub request
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "B11"],
                output: { bands: 5 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                DataCollection.SENTINEL2_L2A,
                time_interval=("2025-08-01", "2025-08-30")
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            resolution=(10, 10),
            config=config
        )

        image = request.get_data()[0]  # (H, W, 5)
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
            "bands": ["B02", "B03", "B04", "B08", "B11"],
            "resolution": 10
        }

        with open(os.path.join(tile_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        metadata_list.append(meta)

        print(f"✔ Saved tile {i} in {tile_dir}")

    return images, metadata_list




def load_data(raw_data_dir=RAW_DATA_DIR):
    """
    Charge toutes les tuiles stockées dans raw_data/

    Retourne :
      - X : array shape (n_tiles, H, W, bands)
      - meta : liste de dict (lat, lon, bbox, bbox_crs, ...)
    """

    X_list = []
    meta_list = []

    # liste des dossiers raw_data/tile_XX
    for tile_name in sorted(os.listdir(RAW_DATA_DIR)):
        tile_dir = os.path.join(RAW_DATA_DIR, tile_name)

        if not os.path.isdir(tile_dir):
            continue

        # charge X.npy
        x_path = os.path.join(tile_dir, "X.npy")
        if not os.path.exists(x_path):
            continue

        X = np.load(x_path)
        X_list.append(X)

        # charge meta.json
        meta_path = os.path.join(tile_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            meta = {}

        meta_list.append(meta)

    # convertit la liste en numpy array
    X_array = np.stack(X_list, axis=0)   # shape = (n_tiles, 512, 512, 5)

    return X_array, meta_list
