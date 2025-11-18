from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig
from pyproj import Transformer
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

#____________
#class 1, IndexCalculator /calculate spectral (NDVI, NDBI and others)
#Use : Convert raw bands into useful indicators (vegetation, water and others)
#____________
class IndexCalculator:#calcul le spectre d'indice

    @staticmethod
    def ndvi(B4: np.ndarray, B8: np.ndarray) -> np.ndarray:
        return (B8 - B4) / (B8 + B4 + 1e-6)
    #NDVI = Normalized Difference Vegetation Index
    '''
    Values :
        - Close to +1 : Dense vegetation (forests, grass)
        - Close to 0  : Mixed zones
        - Close to -1 : Areas without vegetation (concrete, water, bare soil)

        Args:
            B4: Red Band
            B8: Near Infrared Band (NIR)
    '''

    @staticmethod
    def ndbi(B11: np.ndarray, B8: np.ndarray) -> np.ndarray:
        return (B11 - B8) / (B11 + B8 + 1e-6)
    """
        NDBI = Normalized Difference Built-up Index
        Detects built-up areas (buildings, roads, concrete).

        Values :
        - Positive : Urban areas (buildings, roads)
        - Negative : Vegetation, water, natural soil

        Args:
            B11: Short Wave Infrared Band (SWIR)
            B8: Near Infrared Band (NIR)
    """

    @staticmethod
    def ndwi(B8: np.ndarray, B11: np.ndarray) -> np.ndarray:
        return (B8 - B11) / (B8 + B11 + 1e-6)
    """
        NDWI = Normalized Difference Water Index
        Detects water and soil moisture.

        Values :
        - Positive : Water surfaces, wet areas
        - Negative : Dry zones

        Args:
            B8: Near Infrared Band (NIR)
            B11: Short Wave Infrared Band (SWIR)
    """

    @staticmethod
    def mndwi(B3: np.ndarray, B11: np.ndarray) -> np.ndarray:
        return (B3 - B11) / (B3 + B11 + 1e-6)
    """
        MNDWI = Modified Normalized Difference Water Index
        Improved version of NDWI, better water detection.

        Values :
        - Positive : Water (lakes, rivers, seas)
        - Negative : Other (land, vegetation)

        Args:
            B3: Green Band
            B11: Short Wave Infrared Band (SWIR)
    """


#____________
#class 2 : NormalizationConfig /Store normalization parameters
#Use :reusable configuration to normalize images
#____________
class NormalizationConfig:

    #reusable normalization configuration
    def __init__(self, gamma: float = 0.8, p_low: int = 2, p_high: int = 98):
        self.gamma = gamma
        self.p_low = p_low
        self.p_high = p_high
    """
        Args:
            gamma (float): Gamma correction for brightness
                - 0.5 = brighter image
                - 0.8 = default
                - 1.2 = darker image

            p_low (int): Lower percentile (default: 2%)
                - Pixels below this percentile are ignored
                - Eliminates extreme shadows

            p_high (int): Upper percentile (default: 98%)
                - Pixels above this percentile are ignored
                - Eliminates extreme highlights
    """

    def __repr__(self):
        return f"NormalizationConfig(gamma={self.gamma}, p_low={self.p_low}, p_high={self.p_high})"
    """
    Displays the configuration in a readable format when printed
    """


#____________
#class 3 : ImageNormalizer / Normalize images for better quality
#Use : Prepare images for visualization or machine learning
#____________
class ImageNormalizer:
    @staticmethod
    def normalize_minmax(image : np.ndarray) -> np.ndarray:
        #normalization to [0,1]
        return (image - image.min()) / (image.max() - image.min() + 1e-6)
    """
        Min-Max Normalization (classical approach).

        Transforms values to [0, 1] using minimum and maximum.

        Problem : Sensitive to outliers (extremely bright/dark pixels)

        Formula: (pixel - min) / (max - min)
    """

    @staticmethod
    def normalize_percentile(image: np.ndarray, p_low: int = 2, p_high: int = 98) -> np.ndarray:
        #percentile based normalization avoids outliers
        #args : image = imput image / p_low =Lower
        p2= np.percentile(image, p_low)
        p98 = np.percentile(image, p_high)
        return np.clip((image-p2)/ (p98 -p2 +1e-6), 0, 1)
    """
        Percentile-Based Normalization.

        Transforms values to [0, 1] using percentiles.

        Advantage : Ignores extreme values (outliers)
        - The 2% darkest pixels are ignored
        - The 2% brightest pixels are ignored
        - Better contrast than normalize_minmax

        Args:
            image: The image to normalize
            p_low: Lower percentile (default: 2%)
            p_high: Upper percentile (default: 98%)
    """

    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
        return np.clip(image ** gamma, 0, 1)
    """
        Gamma Correction to adjust brightness.

        Makes the image brighter or darker without loss of detail.

        - gamma < 1.0 : Brighter image (ex: 0.8)
        - gamma = 1.0 : No change
        - gamma > 1.0 : Darker image (ex: 1.2)

        Formula: pixel ^ gamma
    """


    @staticmethod
    def normalize_full (image: np.ndarray, config: Optional[NormalizationConfig] = None ) -> np.ndarray:
        if config is None:
            config = NormalizationConfig() #create config by default

        normalized = ImageNormalizer.normalize_percentile(image, config.p_low, config.p_high)
        return ImageNormalizer.gamma_correction(normalized, config.gamma)
    """
        COMPLETE normalization pipeline : Percentile + Gamma.

        Combines both methods to get an optimal image :
        1. Percentile normalization (improves contrast)
        2. Gamma correction (adjusts brightness)

        Args:
            image: The image to normalize
            config: Configuration (if None, uses default values)

        Usage example:
            config = NormalizationConfig(gamma=0.5, p_low=3, p_high=97)
            normalized_image = ImageNormalizer.normalize_full(image, config)
    """
