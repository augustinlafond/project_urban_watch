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
from pathlib import Path


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
    - tile_names : noms des fichiers TIFF dans GCP (WorldCover tiles)
    - list_bbox_wgs84 : bbox en WGS84 servant à découper WorldCover
    - list_bbox_x : bbox des X en CRS projeté (EPSG:326xx)
    - list_crs_x : CRS du X (EPSG:326xx)

    Sauvegarde chaque Y dans :
        data/labels_y/tile_{i}/label.tif

    Renvoie : liste des labels (numpy arrays)
    """

    bucket_name = os.getenv("BUCKET_NAME")
    project = os.getenv("GCP_PROJECT")

    # --- dossier de sortie ---
    project_root = Path(__file__).resolve().parents[2]  # remonte depuis urban_watch/ml_logic/labels.py
    labels_root = project_root / "data" / "labels_y"
    labels_root.mkdir(parents=True, exist_ok=True)

    base_folder = labels_root

    labels = []

    for i, (tile_name, bbox_wgs84, bbox_x, crs_x) in enumerate(
            zip(tile_names, list_bbox_wgs84, list_bbox_x, list_crs_x)
    ):
        # Dossier local pour cette tile
        tile_folder = base_folder / f"tile_{i}"
        tile_folder.mkdir(parents=True, exist_ok=True)

        # Chemin du label local
        local_label_path = tile_folder / "label.tif"

        print(f"⬇️ Téléchargement GCP + reprojection pour tile {i}")

        # --- 1. Télécharger la tuile depuis GCP ---
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"y-labels/{tile_name}")
        content = blob.download_as_bytes()

        # --- 2. Lire & découper en WGS84 ---
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as src:

                min_lon, min_lat, max_lon, max_lat = bbox_wgs84

                window = rasterio.windows.from_bounds(
                    min_lon, min_lat, max_lon, max_lat,
                    transform=src.transform
                )

                wc_cut = src.read(1, window=window)
                wc_transform = src.window_transform(window)
                wc_crs = src.crs  # EPSG:4326

        # --- 3. Reprojection sur la grille X (500×500) ---
        xmin, ymin, xmax, ymax = bbox_x
        dst_transform = from_bounds(xmin, ymin, xmax, ymax, 500, 500)

        dst = np.empty((500, 500), dtype=np.uint8)

        reproject(
            source=wc_cut,
            destination=dst,
            src_transform=wc_transform,
            src_crs=wc_crs,
            dst_transform=dst_transform,
            dst_crs=crs_x,
            resampling=Resampling.nearest
        )

        labels.append(dst)

        # --- 4. Sauvegarde locale du label ---
        with rasterio.open(
            local_label_path,
            "w",
            driver="GTiff",
            height=500,
            width=500,
            count=1,
            dtype=dst.dtype,
            crs=crs_x,
            transform=dst_transform,
        ) as dst_file:
            dst_file.write(dst, 1)

        print(f"💾 Sauvegardé dans {local_label_path}")

    return labels



LABELS_DIR = Path(__file__).resolve().parents[2] / "data" / "labels_y"

def load_labels_y(labels_dir=LABELS_DIR):
    """
    Charge tous les labels Y enregistrés dans data/labels_y/tile_i/label.tif.

    Retourne :
        labels : list[np.ndarray]   (chaque array a une shape (500, 500))
    """

    labels = []

    # --- Parcourir les sous-dossiers tile_i triés par index ---
    tile_dirs = sorted(labels_dir.glob("tile_*"),
                       key=lambda p: int(p.name.split("_")[1]))

    for tile_dir in tile_dirs:
        tif_path = tile_dir / "label.tif"

        # Lecture avec rasterio
        with rasterio.open(tif_path) as src:
            arr = src.read(1)  # lire bande unique
            labels.append(arr)

    print(f"📥 Chargement terminé : {len(labels)} labels trouvés.")
    print(f" - Shape des labels : {labels[0].shape if len(labels) else 'N/A'}")

    return labels
