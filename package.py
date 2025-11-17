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

    def nvdi(B4:np.ndarray, B8: np.ndarray) ->np.ndarray:
        return (B8 - B4) / (B8 + B4 + 1e-6)
