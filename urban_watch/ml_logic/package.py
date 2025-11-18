from sentinelhub import SentinelHubRequest, DataCollection, MimeType, CRS, BBox, SHConfig
from pyproj import Transformer
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
from s2cloudless import S2PixelCloudDetector

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

#___
#cloud masking
#__________________________________

class CloudMasker:


    BAND_IDX = {
        "B01": 0,   # Coastal aerosol
        "B02": 1,   # Blue
        "B03": 2,   # Green
        "B04": 3,   # Red
        "B05": 4,   # Vegetation Red Edge
        "B06": 5,   # Vegetation Red Edge
        "B08": 6,   # NIR
        "B8A": 7,   # Vegetation Red Edge
        "B11": 8,   # SWIR
        "B12": 9    # SWIR
    }

    def __init__(self, threshold: float = 0.5, average_over: int = 4, dilation_size: int = 1):
        self.detector = S2PixelCloudDetector(threshold=threshold,
                                             average_over =average_over,
                                             dilation_size=dilation_size,
                                             all_bands=False)

    def detect_clouds(self, image: np.ndarray) -> np.ndarray:
            """
            image : np.ndarray shape (H,W,10) = B01,B02,B03,B04,B05,B06,B08,B8A,B11,B12
            retourne mask booléen
            """

            if image.ndim == 3:
                image_batch = image[np.newaxis, :, :, :]
            else:
                image_batch = image

            cloud_probs_batch = self.detector.get_cloud_probability_maps(image_batch)  # shape (1,H,W)
            mask = cloud_probs_batch[0] > self.detector.threshold
            return mask

    @staticmethod
    def apply_mask(image: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        image_masked = image.copy().astype(float)
        image_masked[mask] = fill_value
        return image_masked

    @staticmethod
    def get_cloud_percentage(mask: np.ndarray) -> float:
        return (mask.sum() / mask.size) * 100

#_____________
#data cleaning
#_____________
class DataCleaner: #normalise les bandes de 0 a 1

    def normalize_bands(self, image: np.ndarray) -> np.ndarray:
        img = image.astype("float32")
        min_val= np.nanmin(img, axis=(0,1), keepdims=True )
        max_val= np.nanmax(img, axis=(0,1), keepdims=True )
        return (img - min_val) / (max_val - min_val  + 1e-6)

    def standardize(self, image: np.ndarray) -> np.ndarray:
        #je voulais importer StandardScaler ici de scikitlearn, surtout pas !!!
        #IMAGE 2d (pixels x features) + de toute facon impossible avec 3 dimensions
        mean = np.nanmean(image, axis=(0,1), keepdims=True)
        std= np.nanstd(image, axis=(0,1), keepdims=True) + 1e-6
        return (image-mean)/std


#_____________
#full pipeline
#_____________

def preprocess_image(img):
    cleaner = DataCleaner()
    cloud_detector = CloudMasker()
    img_norm = cleaner.normalize_bands(img) #normalisation bands 0-1
    cloud_mask = cloud_detector.detect_clouds(img_norm) #s2cloudless
    img_masked = cloud_detector.apply_mask(img_norm, cloud_mask, fill_value=np.nan)
    '''masquage des pixel nuageux'''

    B3  = img_masked[:, :, CloudMasker.BAND_IDX["B03"]]
    B4  = img_masked[:, :, CloudMasker.BAND_IDX["B04"]]
    B8  = img_masked[:, :, CloudMasker.BAND_IDX["B08"]]
    B11 = img_masked[:, :, CloudMasker.BAND_IDX["B11"]]
    '''ici les bands principales'''

    ndvi  = IndexCalculator.ndvi(B4,  B8)
    ndbi  = IndexCalculator.ndbi(B11, B8)
    mndwi = IndexCalculator.mndwi(B3, B11)

    ndvi = ndvi[..., np.newaxis]
    ndbi = ndbi[..., np.newaxis]
    mndwi = mndwi[..., np.newaxis]
    #ici cest pour concaténer 13 bandes

    img_13 = np.concatenate([img_masked, ndvi, ndbi,mndwi], axis=-1)

    img_std = cleaner.standardize(img_13)
    return img_std
