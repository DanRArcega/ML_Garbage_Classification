import torch

from download_data import get_dataframe
from typing import Tuple, List
import pandas as pd
from PIL import Image
from pathlib import Path
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


PROCESSED_IMAGES = "../data/Garbage classification/processed"


def load_split(split_path: Path) -> pd.DataFrame:
    """

    :param split_path:
    :return:
    """
    records = []
    for label_dir in split_path.iterdir():
        if not label_dir.is_dir():
            continue
        for image_path in label_dir.glob("*.jpg"):
            records.append({"path": image_path, "label": label_dir.name})
    return pd.DataFrame(records)


def new_load_train_test_imgs(dir: str = PROCESSED_IMAGES) -> Tuple[List[Image.Image], List[Image.Image], List[str], List[str]]:
    """
    Loads the testing and training splits from the processed image directories.
    :param dir: The root directory of the processed images.
    :type dir: str
    :return: A list of both sets of images and their labels.
    :rtype: Tuple[List[Image.Image], List[Image.Image], List[str], List[str]]
    """
    full_train = load_split(Path(dir) / "train")
    full_test = load_split(Path(dir) / "test")

    train_images = [Image.open(p) for p in full_train["path"]]
    test_images = [Image.open(p) for p in full_test["path"]]
    return train_images, full_train["label"].to_list(), train_images, full_test["label"].to_list()


def load_train_test_imgs(
    dir: str = '../data/Garbage classification'
) -> Tuple[List[Image.Image], List[Image.Image], List[str], List[str]]:
    
    # Load all data from the 'train' folder
    full_path = os.path.join(dir, 'train')
    full_data = get_dataframe(full_path)

    # Split the dataframe into train and test (80% train, 20% test)
    # By stratifying on the label column, we ensure balanced class distribution
    train_data, test_data = train_test_split(
        full_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=full_data['label']
    )
    
    # Open images using the 'path' column (matching your original working code)
    train_imgs = [Image.open(path) for path in train_data['path']]
    test_imgs = [Image.open(path) for path in test_data['path']]
    
    # Extract the labels directly from the split dataframes
    train_labels = train_data['label'].to_list()
    test_labels = test_data['label'].to_list()
    
    return train_imgs, test_imgs, train_labels, test_labels


def convert_to_grayscale(imgs: List[Image.Image]) -> List[Image.Image]:
    return [img.convert('L') for img in imgs]


def sample_per_label(X, y, n_per_label=10):
    label_dict = defaultdict(list)

    for xi, yi in zip(X, y):
        label_dict[yi].append(xi)

    X_sample = []
    y_sample = []

    for label, items in label_dict.items():
        selected = items[:n_per_label]
        X_sample.extend(selected)
        y_sample.extend([label] * len(selected))

    return X_sample, y_sample


def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)

    precision = []
    recall = []
    f1 = []

    for i in range(len(cm)):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

    return {
        'accuracy': accuracy,
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1_score': np.mean(f1)
    }
# Extract all data from a data loader and return appropiate np.arrays
# Can be dangerous if the total dataset is too big
def extract_dataloader_data(dataloader):
    scaled_pixels = []
    encoded_labels = []
    for images, labels in dataloader:
        image = images.cpu().numpy()
        scaled_pixels.append(image.reshape(image.shape[0], -1))
        encoded_labels.append(labels.cpu().numpy())
    scaled_pixels = np.concatenate(scaled_pixels)
    encoded_labels = np.concatenate(encoded_labels)
    return scaled_pixels, encoded_labels
