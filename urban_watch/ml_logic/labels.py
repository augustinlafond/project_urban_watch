import numpy as np
import json
from dotenv import load_dotenv
import os
from pyproj import Transformer
from google.cloud import storage
import rasterio
from io import BytesIO
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from urban_watch.ml_logic.data import RAW_DATA_DIR
import math



def get_bbox_from_features(raw_data_dir=RAW_DATA_DIR):
    """
    Récupère les bbox et le CRS des features. Retourne une liste de bbox et une liste de CRS.
    """
    list_bbox = []
    list_crs = []

    for tile_name in sorted(os.listdir(RAW_DATA_DIR)):
            meta_path = os.path.join(RAW_DATA_DIR, tile_name, "meta.json")
            with open(meta_path, "r") as f:
                data = json.load(f)
                list_bbox.append(data['bbox'])
                list_crs.append(data['bbox_crs'])

    return list_bbox, list_crs




def bbox_to_wgs84(list_bbox, list_crs):
    """
    Convertit une bounding box d'un CRS projeté vers WGS84.
    list_bbox_wgs84 = (minx, miny, maxx, maxy)
    """
    list_bbox_wgs84 = []

    for bbox, crs in zip(list_bbox, list_crs):
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        xmin, ymin = transformer.transform(bbox[0], bbox[1])
        xmax, ymax = transformer.transform(bbox[2], bbox[3])

        list_bbox_wgs84.append([xmin, ymin, xmax, ymax])

    return list_bbox_wgs84



def tile_name_from_bbox_wgs84(list_bbox_wgs84):
    """
    Créé le nom du fichier tiff des labels y à aller récupérer dans GCP, à partir des bbox des features X.
    tile_names = [tiff_name_1, tif_name_2, ...]
    """
    tile_names = []

    for bbox_wgs84 in list_bbox_wgs84:
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84

        # Tuile WorldCover = grille régulière de 3° x 3°
        lat_tile = 3 * math.floor(min_lat / 3)
        lon_tile = 3 * math.floor(min_lon / 3)

        # Préfixes formatés
        lat_prefix = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
        lon_prefix = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"

        tile_names.append(f"ESA_WorldCover_10m_2021_V200_{lat_prefix}{lon_prefix}_Map.tif")

    return tile_names



def get_label_array(tile_names, list_bbox_wgs84, list_bbox_x, list_crs_x):
    """
    - tile_names : liste des noms des TIFF dans GCP
    - list_bbox_wgs84 : bbox en WGS84 pour découper WorldCover
    - list_bbox_x : bbox en CRS des tuiles X (pour construire transform_x)
    - list_crs_x : CRS EPSG:326xx des tuiles X
    """

    bucket_name = os.getenv("BUCKET_NAME")
    project = os.getenv("GCP_PROJECT")

    labels = []

    for tile_name, bbox_wgs84, bbox_x, crs_x in zip(tile_names, list_bbox_wgs84, list_bbox_x, list_crs_x):

        # --- 1. Télécharger la tuile depuis GCP ---
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"y-labels/{tile_name}")
        content = blob.download_as_bytes()

        # --- 2. Lire WorldCover ---
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as src:

                # Découpe en WGS84
                min_lon, min_lat, max_lon, max_lat = bbox_wgs84
                window = rasterio.windows.from_bounds(
                    min_lon, min_lat, max_lon, max_lat,
                    transform=src.transform
                )

                wc_cut = src.read(1, window=window)
                wc_transform = src.window_transform(window)
                wc_crs = src.crs  # EPSG:4326

        # --- 3. Reprojeter sur la grille X (300×300) ---

        xmin, ymin, xmax, ymax = bbox_x  # bbox en EPSG:326xx
        dst_transform = from_bounds(xmin, ymin, xmax, ymax, 300, 300)

        dst = np.empty((300, 300), dtype=np.uint8)

        reproject(
            source=wc_cut,
            destination=dst,
            src_transform=wc_transform,
            src_crs=wc_crs,
            dst_transform=dst_transform,
            dst_crs=crs_x,
            resampling=Resampling.nearest   # très important pour classification
        )

        labels.append(dst)

    return labels
