"""
# Label Extraction and Reprojection Pipeline


This module performs the following operations:


1. **Bounding Box Extraction**: Reads metadata from raw feature directories to collect projected-coordinate bounding boxes and their CRS.
2. **WGS84 Conversion**: Converts each bounding box from its projected CRS to WGS84.
3. **WorldCover Tile Mapping**: Determines the corresponding ESA WorldCover tile name for each WGS84 bounding box.
4. **Label Extraction from GCP**:
- Downloads the relevant WorldCover tile from Google Cloud Storage.
- Crops the tile according to the WGS84 bounding box of the feature.
- Reprojects the cropped raster to the CRS and spatial extent of the feature grid (500×500).
5. **Local Storage**: Saves each reprojected label into `data/labels_y/tile_i/label.tif`.
6. **Label Loading**: Provides a loader to read all generated labels for downstream ML workflows.


The code does **not** modify any feature arrays; it focuses solely on generating and loading label rasters aligned with input feature tiles.
"""


# Standard library
import os
import math
import json
from pathlib import Path
from io import BytesIO

# Third-party libraries
import numpy as np
from pyproj import Transformer
from google.cloud import storage
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

# Project-specific imports
from urban_watch.ml_logic.data import RAW_DATA_DIR


def get_bbox_from_features(raw_data_dir=RAW_DATA_DIR):
    """
    Retrieves bounding boxes and CRS information from feature metadata.

    Returns:
    list_bbox: list of bounding boxes
    list_crs: list of associated CRS
    """
    list_bbox = []
    list_crs = []

    for tile_name in sorted(os.listdir(RAW_DATA_DIR)):
        meta_path = os.path.join(RAW_DATA_DIR, tile_name, "meta.json")
        with open(meta_path, "r") as f:
            data = json.load(f)
            list_bbox.append(data["bbox"])
            list_crs.append(data["bbox_crs"])

    return list_bbox, list_crs


def bbox_to_wgs84(list_bbox, list_crs):
    """
    Converts bounding boxes from their projected CRS to WGS84.

    Returns:
    list_bbox_wgs84: list of bounding boxes in WGS84 format (minx,
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
    Create the name of the tiff file containing the labels (y) to be retrieved from GCP, based on the feature boxes (X).
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

        tile_names.append(
            f"ESA_WorldCover_10m_2021_V200_{lat_prefix}{lon_prefix}_Map.tif"
        )

    return tile_names


def get_label_array(tile_names, list_bbox_wgs84, list_bbox_x, list_crs_x):
    """
    - tile_names: names of the TIFF files in GCP (WorldCover tiles)
    - list_bbox_wgs84: WGS84 bbox used to slice WorldCover
    - list_bbox_x: bbox of the X tiles in projected CRS (EPSG:326xx)
    - list_crs_x: CRS of the X tile (EPSG:326xx)

    Saves each Y tile in:
        data/labels_y/tile_{i}/label.tif

    Returns: list of labels (numpy arrays)
    """

    bucket_name = os.getenv("BUCKET_NAME")
    project = os.getenv("GCP_PROJECT")

    # --- # Output directory ---
    project_root = Path(__file__).resolve().parents[2]
    labels_root = project_root / "data" / "labels_y"
    labels_root.mkdir(parents=True, exist_ok=True)

    base_folder = labels_root

    labels = []

    for i, (tile_name, bbox_wgs84, bbox_x, crs_x) in enumerate(
        zip(tile_names, list_bbox_wgs84, list_bbox_x, list_crs_x)
    ):
        # Local folder for this tile
        tile_folder = base_folder / f"tile_{i}"
        tile_folder.mkdir(parents=True, exist_ok=True)

        # Local Label Path
        local_label_path = tile_folder / "label.tif"

        print(f" Download GCP + reprojection for tile {i}")

        # --- 1. Download tile from GCP ---
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"y-labels/{tile_name}")
        content = blob.download_as_bytes()

        # --- 2. Read and crop WorldCover tile ---
        with rasterio.MemoryFile(content) as memfile:
            with memfile.open() as src:

                min_lon, min_lat, max_lon, max_lat = bbox_wgs84

                window = rasterio.windows.from_bounds(
                    min_lon, min_lat, max_lon, max_lat, transform=src.transform
                )

                wc_cut = src.read(1, window=window)
                wc_transform = src.window_transform(window)
                wc_crs = src.crs  # EPSG:4326

        # --- 3. Reprojection onto grid X (500×500) ---
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
            resampling=Resampling.nearest,
        )

        labels.append(dst)

        # --- 4. Local preservation of the label ---
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

        print(f"Saved in {local_label_path}")

    return labels


LABELS_DIR = Path(__file__).resolve().parents[2] / "data" / "labels_y"


def load_labels_y(labels_dir=LABELS_DIR):
    """
    Loads all the Y labels stored in data/labels_y/tile_i/label.tif.

    Retourns:
        labels: list[np.ndarray] (each array has a shape (500, 500))
    """

    labels = []

    # --- Browse the tile_i subfolders sorted by index ---
    tile_dirs = sorted(
        labels_dir.glob("tile_*"), key=lambda p: int(p.name.split("_")[1])
    )

    for tile_dir in tile_dirs:
        tif_path = tile_dir / "label.tif"

        # Reading with rasterio
        with rasterio.open(tif_path) as src:
            arr = src.read(1)  # read single strip
            labels.append(arr)

    print(f"📥 Loading complete : {len(labels)} labels found.")
    print(f" - Shape of the labels : {labels[0].shape if len(labels) else 'N/A'}")

    return labels
