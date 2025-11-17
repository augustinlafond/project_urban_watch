from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig
from dotenv import load_dotenv
from pyproj import Transformer
import os
import numpy as np
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



def get_data(list_bbox):

    config = SHConfig()
    config.sh_client_id = os.environ.get("SH_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SH_CLIENT_SECRET")

    images = []  # liste pour stocker les X

    for i, (lat, lon) in enumerate(list_bbox):

        bbox = make_bbox_global(lat, lon, km_size=3)

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
            resolution = (10,10),
            config=config
        )


        image = request.get_data()[0]
        images.append(image)

    return images
