"""
file: features/color_histogram.py
description: Extracts HSV color histograms from preprocessed images.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""



import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


HISTOGRAM_BINS = 64


def extract_color_histogram(image: Image.Image) -> np.ndarray:
    """
    Computes a normalized HSV histogram of the given image.
    :param image: The image to compute from.
    :type image: Image.Image
    :return: A 1d numpy array of 32 bits * 3 channels.
    :rtype: np.ndarray
    """
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    channel_ranges = [(0, 180), (0, 256), (0, 256)]
    histograms = []
    for channel_index, (channel_min, channel_max) in enumerate(channel_ranges):
        histogram = cv2.calcHist(
            [hsv],
            [channel_index],
            None,
            [HISTOGRAM_BINS],
            [channel_min, channel_max]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()
        histograms.append(histogram)

    return np.concatenate(histograms)



def extract_histogram_features(manifest_path: Path, split: str) -> pd.DataFrame:
    """
    Extracts HSV color descriptors from all images in the given split.
    :param manifest_path: The path to the manifest file.
    :type manifest_path: Path
    :param split: The split to extract features for.
    :type split: str
    :return: A dataframe containing the descriptors.
    :rtype: pd.DataFrame
    """
    assert split in ("train", "validation", "test"), \
        f"split must be 'train', 'validation', or 'test', got '{split}'"

    manifest = pd.read_csv(manifest_path)
    split_dataframe = manifest[manifest["split"] == split].reset_index(drop = True)
    if split_dataframe.empty:
        raise ValueError(f"No records found in split '{split}' in {manifest_path}.")

    rows = []
    for _, row in tqdm(
        split_dataframe.iterrows(),
        total = len(split_dataframe),
        desc = f"Extracting color histograms for {split}",
        unit = "img"
    ):
        try:
            image = Image.open(row["processed_path"]).convert("RGB")
            descriptor = extract_color_histogram(image)
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
    Calculates the length of a color histogram descriptor.
    :return: The length of the descriptor vector (#Bins * 3 channels).
    :rtype: int
    """
    return HISTOGRAM_BINS * 3



if __name__ == "__main__":
    from preprocessing.config import DATA_CONFIG
    manifest_path = DATA_CONFIG.processed_data_path / "manifest.csv"
    print(f"Color histogram descriptor length: {descriptor_length()}")

    dataframe = extract_histogram_features(manifest_path, split = "train")
    print(f"Extracted {len(dataframe)} color histogram descriptors.")
    print(f"Descriptor shape: {dataframe["descriptor"].iloc[0].shape}")
    print(dataframe[["path", "label"]].head())