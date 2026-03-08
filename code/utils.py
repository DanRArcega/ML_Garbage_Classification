from download_data import get_dataframe
from typing import Tuple, List
import pandas as pd
from PIL import Image
import os
from collections import defaultdict


def load_train_test_imgs(
    dir: str = '../data/Garbage classification'
) -> Tuple[List[Image.Image], List[Image.Image], List[str], List[str]]:
    train_path = os.path.join(dir, 'train')
    test_path = os.path.join(dir, 'test')

    train_data = get_dataframe(train_path)
    test_data = get_dataframe(test_path)

    train_imgs = [Image.open(path) for path in train_data['path']]
    test_imgs = [Image.open(path) for path in test_data['path']]

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
