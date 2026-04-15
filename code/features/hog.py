"""
file: features/hog.py
description: An implementation of HOG (histogram of oriented gradients) for an SVM pipeline.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from tqdm import tqdm


HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


def extract_hog_descriptor(image: Image.Image) -> np.ndarray:
    """
    Extracts the HOG descriptor from an image
    :param image: The PIL image to work with.
    :type image: Image.Image
    :return: An array representing the HOG descriptor.
    :rtype: np.ndarray
    """
    grayscale = rgb2gray(np.array(image))
    descriptor = hog(
        grayscale,
        orientations = HOG_ORIENTATIONS,
        pixels_per_cell = HOG_PIXELS_PER_CELL,
        cells_per_block = HOG_CELLS_PER_BLOCK,
        block_norm = HOG_BLOCK_NORM,
    )
    return descriptor


def extract_hog_features(manifest_path: Path, split: str) -> pd.DataFrame:
    """
    Extracts all hog descriptors for all images in the given split of the manifest.
    :param manifest_path: The path to the manifest file.
    :type manifest_path: Path
    :param split: The name of the split.
    :type split: str
    :return: A dataframe containing all the descriptors for each image.
    :rtype: pd.DataFrame
    """
    assert split in ("train", "validation", "test"), \
        f"split must be 'train', 'validation', or 'test', got '{split}'"

    manifest = pd.read_csv(manifest_path)
    split_dataframe = manifest[manifest["split"] == split].reset_index(drop = True)
    if split_dataframe.empty:
        raise ValueError(f"No records found for split '{split}' in {manifest_path}")

    rows = []
    for _, row in tqdm(
        split_dataframe.iterrows(),
        total = len(split_dataframe),
        desc = f"Extracting HOG descriptors for {split}",
        unit = "img"
    ):
        try:
            image = Image.open(row["processed_path"]).convert("RGB")
            descriptor = extract_hog_descriptor(image)
            rows.append({
                "path": row["processed_path"],
                "label": row["label"],
                "descriptor": descriptor,
            })
        except Exception as e:
            print(f"Could not process image {row['processed']}: {e}")

    return pd.DataFrame(rows)



def descriptor_length(image_size: tuple[int, int] = (224, 224)) -> int:
    """
    Computes the length of the HOG descriptor for an image.
    :param image_size: The dimensions of the image to work with (defaults to 224x224).
    :type image_size: tuple[int, int]
    :return: Length of the HOG descriptor vector for an image.
    :rtype: int
    """
    dummy_array = np.zeros(image_size)
    descriptor = hog(
        dummy_array,
        orientations = HOG_ORIENTATIONS,
        pixels_per_cell = HOG_PIXELS_PER_CELL,
        cells_per_block = HOG_CELLS_PER_BLOCK,
        block_norm = HOG_BLOCK_NORM
    )
    return len(descriptor)



if __name__ == "__main__":
    from preprocessing.config import DATA_CONFIG
    manifest_path = DATA_CONFIG.processed_data_path / "manifest.csv"
    print(f"HOG descriptor length for {DATA_CONFIG.target_size}: {descriptor_length(DATA_CONFIG.target_size)}")

    dataframe = extract_hog_features(manifest_path, split = "train")
    print(f"\nExtracted HOG features for {len(dataframe)} training images.")
    print(f"Descriptor shape: {dataframe['descriptor'].iloc[0].shape}")
    print(dataframe[["path", "label"]].head())
