from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig
from pyproj import Transformer
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

class IndexCalculator:#calcul le spectre d'indice

    @staticmethod
    def ndvi(B4: np.ndarray, B8: np.ndarray) -> np.ndarray:
        return (B8 - B4) / (B8 + B4 + 1e-6)

    @staticmethod
    def ndbi(B11: np.ndarray, B8: np.ndarray) -> np.ndarray:
        return (B11 - B8) / (B11 + B8 + 1e-6)

    @staticmethod
    def ndwi(B8: np.ndarray, B11: np.ndarray) -> np.ndarray:
        return (B8 - B11) / (B8 + B11 + 1e-6)


class ImageNormalizer:
    def normalize_minmax(image : np.ndarray) -> np.ndarray:
        #normalization to [0,1]
        return (image - image.min()) / (image.max() - image.min() + 1e-6)

    def normalize_percentile(image: np.ndarray, p_low: int = 2, p_high: int = 98) -> np.ndarray:
        #percentile based normalization avoids outliers
        #args : image = imput image / p_low =Lower
        p2= np.percentile(image, p_low)
        p98 = np.percentile(image, p_high)
        return np.clip((image-p2)/ (p98 -p2 +1e-6), 0, 1)
