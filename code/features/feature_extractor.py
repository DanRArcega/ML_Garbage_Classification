"""
file: features/feature_extractor.py
description:
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from features.hog import extract_hog_features
from features.color_histogram import extract_histogram_features
from features.spatial_color import extract_color_spatial_features


GRID_SIZES = {
    "spatial_2x2": (2, 2),
    "spatial_3x3": (3, 3)
}

FEATURE_MODE = Literal["hog", "color", "spatial_2x2", "spatial_3x3", "both"]


def descriptor_to_matrix(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Converts the DataFrame column to a 2d numpy matrix.
    :param dataframe: The DataFrame containing the descriptor column.
    :type dataframe: pd.DataFrame
    :return: 2d numpy matrix of shape samples x descriptor length.
    :rtype: np.ndarray
    """
    return np.stack(dataframe["descriptor"].values)



def extract_hog_matrix(
        manifest_path: Path,
        split: str,
        scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extracts and normalizes HOG features for a given split.
    :param manifest_path: The path to the manifest csv.
    :type manifest_path: Path
    :param split: The split to extract HOG features for.
    :type split: str
    :param scaler: The fitted scaler, or None to fit a new one.
    :type scaler: StandardScaler | None
    :return: The feature matrix, the labels, and the scaler.
    :rtype: tuple[np.ndarray, np.ndarray, StandardScaler]
    """
    hog_dataframe = extract_hog_features(manifest_path, split)
    matrix = descriptor_to_matrix(hog_dataframe)
    if scaler is None:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
    else:
        matrix = scaler.transform(matrix)
    return matrix, hog_dataframe["label"].values, scaler



def extract_color_matrix(
        manifest_path: Path,
        split: str,
        scaler: StandardScaler | None = None
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extracts and normalizes color histogram features for a given split.
    :param manifest_path: The path to the manifest csv.
    :type manifest_path: Path
    :param split: The split to extract color histogram features for.
    :type split: str
    :param scaler: The fitted scaler, or None to fit a new one.
    :type scaler: StandardScaler | None
    :return: The feature matrix, the labels, and the scaler.
    :rtype: tuple[np.ndarray, np.ndarray, StandardScaler]
    """
    color_dataframe = extract_histogram_features(manifest_path, split)
    matrix = descriptor_to_matrix(color_dataframe)
    if scaler is None:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
    else:
        matrix = scaler.transform(matrix)
    return matrix, color_dataframe["label"].values, scaler



def extract_spatial_matrix(
        manifest_path: Path,
        split: str,
        grid_size: tuple[int, int],
        scaler: StandardScaler | None = None
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Extracts and normalizes spatial color histogram features for a given split.
    :param manifest_path: The path to the manifest csv.
    :type manifest_path: Path
    :param split: The split to read from.
    :type split: str
    :param grid_size: The given grid layout.
    :type grid_size: tuple[int, int]
    :param scaler: The fitted scaler, or None to fit a new one.
    :type scaler: StandardScaler | None
    :return: The feature matrix, the labels and the fitted scaler.
    :rtype: tuple[np.ndarray, np.ndarray, StandardScaler]
    """
    spatial_dataframe = extract_color_spatial_features(manifest_path, split, grid_size)
    matrix = descriptor_to_matrix(spatial_dataframe)
    if scaler is None:
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
    else:
        matrix = scaler.transform(matrix)
    return matrix, spatial_dataframe["label"].values, scaler



def encode_labels(labels: np.ndarray, encoder: LabelEncoder | None = None) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encodes class string labels as integers. Fits to a new LabelEncoder if one isn't provided.
    :param labels: The class labels to encode.
    :type labels: np.ndarray
    :param encoder: The fitted encoder, or none.
    :type encoder: LabelEncoder | None
    :return: The encoded labels and the fitted encoder.
    :rtype: tuple[np.ndarray, LabelEncoder]
    """
    if encoder is None:
        encoder = LabelEncoder()
        return encoder.fit_transform(labels), encoder
    else:
        return encoder.transform(labels), encoder



def extract_features(
        manifest_path: Path,
        split: str,
        mode: FEATURE_MODE = "both",
        scalers: dict[str, StandardScaler] | None = None,
        label_encoder: LabelEncoder | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, StandardScaler], LabelEncoder]:
    """
    Extracts, normalizes and concatenates features for a given split.
    :param manifest_path: The path to the manifest csv.
    :type manifest_path: Path
    :param split: The split to extract features for.
    :type split: str
    :param mode: The feature mode, using HOG, color histogram or both.
    :type mode: str
    :param scalers: The dictionary of StandardScalers fitted to each feature type.
    :type scalers: dict[str, StandardScaler] | None
    :param label_encoder: The fitted label encoder, or none.
    :type label_encoder: LabelEncoder | None
    :return: X, y, hog scaler, color scaler, label encoder.
    :rtype: tuple[np.ndarray, np.ndarray, dict[str, StandardScaler], LabelEncoder]
    """
    assert mode in ("hog", "color", "spatial_2x2", "spatial_3x3", "both"), \
        f"mode must be 'hog', 'color', 'spatial_2x2', 'spatial_3x3' or 'both', got '{mode}'"

    feature_groups = []
    labels = None
    if scalers is None:
        scalers = {}

    if mode in ("hog", "both"):
        hog_matrix, labels, scalers["hog"] = extract_hog_matrix(manifest_path, split, scalers.get("hog"))
        feature_groups.append(hog_matrix)
    if mode in ("color", "both"):
        color_matrix, labels, scalers["color"] = extract_color_matrix(manifest_path, split, scalers.get("color"))
        feature_groups.append(color_matrix)
    if mode in GRID_SIZES:
        grid_size = GRID_SIZES[mode]
        matrix, labels, scalers["spatial"] = extract_spatial_matrix(manifest_path, split, grid_size, scalers.get("spatial"))
        feature_groups.append(matrix)


    X = np.concatenate(feature_groups, axis = 1)
    y, label_encoder = encode_labels(labels, label_encoder)

    print(f"{split} feature matrix shape : {X.shape}")
    print(f"{split} label vector shape : {y.shape}")
    print(f"Classes = {list(label_encoder.classes_)}")
    return X, y, scalers, label_encoder



if __name__ == "__main__":
    from preprocessing.config import DATA_CONFIG
    manifest_path = DATA_CONFIG.processed_data_path / "manifest.csv"

    for mode in ("hog", "color", "spatial_2x2", "spatial_3x3", "both"):
        print("\n" + ("=" * 50))
        print(f"Mode: {mode}")
        X_train, y_train, scalers, label_encoder = extract_features(
            manifest_path,
            split = "train",
            mode = mode
        )
        X_test, y_test, _, _ = extract_features(
            manifest_path,
            split = "test",
            mode = mode,
            scalers = scalers,
            label_encoder = label_encoder
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
