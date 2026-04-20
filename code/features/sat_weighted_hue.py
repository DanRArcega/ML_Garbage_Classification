"""
file: features/sat_weighted_hue.py
description: An implementation of saturation-weighted hue histogram feature extraction.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm



HISTOGRAM_BINS = 64
HUE_RANGE = (0, 180)



def extract_sat_weighted_hue(image: Image.Image) -> np.ndarray:
    """
    Extracts a saturation-weighted hue histogram for a single image.
    :param image: The raw image to draw from.
    :type image: Image.Image
    :return: A 1d numpy array representing the saturation-weighted hue histogram.
    :rtype: np.ndarray
    """
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].flatten().astype(np.float32)
    saturation = hsv[:, :, 1].flatten().astype(np.float32) / 255.0

    histogram, _ = np.histogram(
        hue,
        bins = HISTOGRAM_BINS,
        range = HUE_RANGE,
        weights = saturation
    )
    total_weight = saturation.sum()
    if total_weight > 0:
        histogram = histogram / total_weight
    else:
        histogram = np.zeros(HISTOGRAM_BINS, dtype = np.float32)
    return histogram.astype(np.float32)



def extract_weighted_features(manifest_path: Path, split: str) -> pd.DataFrame:
    """
    Extracts the saturation weighted hue histograms for all the images of a given split.
    :param manifest_path: The path to the manifest csv.
    :type manifest_path: Path
    :param split: The split to draw from.
    :type split: str
    :return: A dataframe containing each image's associated path, label, and descriptor.
    :rtype: pd.DataFrame
    """
    assert split in ("train", "validation", "test"), \
        f"split must be 'train', 'validation', or 'test', got '{split}'"

    manifest = pd.read_csv(manifest_path)
    split_dataframe = manifest[manifest["split"] == split].reset_index(drop = True)
    if split_dataframe.empty:
        raise ValueError(f"No records found in split '{split}' in manifest {manifest_path}")

    rows = []
    for _, row in tqdm(
        split_dataframe.iterrows(),
        total = len(split_dataframe),
        desc = f"Extraction Saturation-Weighted Hue Histograms",
        unit = "img"
    ):
        try:
            image = Image.open(row["processed_path"]).convert("RGB")
            descriptor = extract_sat_weighted_hue(image)
            rows.append({
                "path": row["processed_path"],
                "label": row["label"],
                "descriptor": descriptor
            })
        except Exception as e:
            print(f"Could not process image {row['processed_path']}: {e}")
    return pd.DataFrame(rows)



def descriptor_length() -> int:
    """
    Calculates the length of a descriptor vector for a single image.
    :return: Length of the descriptor vector.
    :rtype: int
    """
    return HISTOGRAM_BINS



if __name__ == "__main__":
    from preprocessing.config import DATA_CONFIG

    manifest_path = DATA_CONFIG.processed_data_path / "manifest.csv"
    print(f"Descriptor Length: {descriptor_length()}")

    dataframe = extract_weighted_features(manifest_path, split = "train")
    print(f"\nExtracted {len(dataframe)} descriptor vectors.")
    print(f"Descriptor Shape: {dataframe["descriptor"].iloc[0].shape}")
    print(dataframe[["path", "label"]].head())
