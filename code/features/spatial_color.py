"""
file: features/spatial_color.py
description: An implementation of the spatial color histogram for the feature pipeline for svm training.
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



CHANNEL_RANGES = [(0, 180), (0, 256), (0, 256)]



def extract_spatial_histogram(image: Image.Image, grid_size: tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Computes a single HSV spatial-color histogram for a PIL image.
    :param image: The image to compute from.
    :type image: Image.Image
    :param grid_size: The grid layout.
    :type grid_size: tuple[int, int]
    :return: A 1d numpy array of length HISTOGRAM_BINS * 3 * grid rows * grid columns.
    :rtype: np.ndarray
    """
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)

    height, width = hsv.shape[:2]
    grid_rows, grid_cols = grid_size

    cell_height = height // grid_rows
    cell_width = width // grid_cols

    regional_histograms = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            row_start = row * cell_height
            if row < grid_rows - 1:
                row_end = row_start + cell_height
            else:
                row_end = height
            col_start = col * cell_width
            if col < grid_cols - 1:
                col_end = col_start + cell_width
            else:
                col_end = width

            region = hsv[row_start:row_end, col_start:col_end]

            for channel_index, (range_min, range_max) in enumerate(CHANNEL_RANGES):
                histogram = cv2.calcHist(
                    [region],
                    [channel_index],
                    None,
                    [HISTOGRAM_BINS],
                    [range_min, range_max]
                )
                histogram = cv2.normalize(histogram, histogram).flatten()
                regional_histograms.append(histogram)
    return np.concatenate(regional_histograms)



def extract_color_spatial_features(
        manifest_path: Path,
        split: str,
        grid_size: tuple[int, int] = (2, 2)
) -> pd.DataFrame:
    """
    Extracts the color-spatial histograms for all images on the manifest for a given split.
    :param manifest_path: The path to the manifest csv file.
    :type manifest_path: Path
    :param split: The split to draw from.
    :type split: str
    :param grid_size: The grid layout.
    :type grid_size: tuple[int, int]
    :return: A dataframe containing each histogram's path, label and descriptor.
    :rtype: pd.DataFrame
    """
    assert split in ("train", "validation", "test"), \
        f"split must be 'train', 'validation', or 'test', got '{split}'"

    manifest = pd.read_csv(manifest_path)
    split_dataframe = manifest[manifest["split"] == split].reset_index(drop = True)
    if split_dataframe.empty:
        raise ValueError(f"Split '{split}' not found in {manifest_path}")

    rows = []
    for _, row in tqdm(
        split_dataframe.iterrows(),
        total = len(split_dataframe),
        desc = f"Extracting spatial histogram for {split}",
        unit = "img"
    ):
        try:
            image = Image.open(row["processed_path"]).convert("RGB")
            descriptor = extract_spatial_histogram(image, grid_size = grid_size)
            rows.append({
                "path": row["processed_path"],
                "label": row["label"],
                "descriptor": descriptor
            })
        except Exception as e:
            print(f"Could not process image {row['processed_path']}: {e}")
    return pd.DataFrame(rows)



def descriptor_length(grid_size: tuple[int, int] = (2, 2)) -> int:
    """
    Calculates the length of the histogram vector.
    :param grid_size: The grid layout.
    :type grid_size: tuple[int, int]
    :return: Length of the histogram vector.
    :rtype: int
    """
    grid_rows, grid_cols = grid_size
    return HISTOGRAM_BINS * grid_rows * grid_cols * 3



if __name__ == "__main__":
    from preprocessing.config import DATA_CONFIG

    manifest_path = DATA_CONFIG.processed_data_path / "manifest.csv"

    for grid_size in ((2, 2), (3, 3)):
        print(f"\nGrid size: {grid_size}")
        print(f"Descriptor length: {descriptor_length(grid_size)}")
        dataframe = extract_color_spatial_features(manifest_path, split = "train", grid_size = grid_size)