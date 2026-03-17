"""
file: preprocessing/transforms.py
description: Handles a number of image transforms for preprocessing stages.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""

import cv2
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms

from config import DataConfig


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Transforms a PIL RGB image into a BGR array for OpenCV.
    :param image: The image to be transformed.
    :type image: PIL.Image
    :return: The transformed image, as a BGR array.
    :rtype: np.ndarray
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """
    Transforms a BGR array in OpenCV format to a PIL image object.
    :param image: The OpenCV image to be transformed.
    :type image: np.ndarray
    :return: The transformed image.
    :rtype: PIL.Image
    """
    return Image.fromarray(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB))


def resize_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resizes a PIL image into a target size (uses Lanczos resampling).
    :param image: The PIL image to be resized.
    :type image: PIL.Image
    :param target_size: The size to be resized to.
    :type target_size: tuple
    :return: A resized image.
    :rtype: PIL.Image
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def apply_clahe(image: Image.Image, clip_limit: float, tile_size: tuple[int, int]) -> Image.Image:
    """
    Applies a CLAHE transformation to a PIL image.
    :param image: The image to be transformed.
    :type image: PIL.Image
    :param clip_limit: The threshold value for contrasting (higher being more contrast).
    :type clip_limit: float
    :param tile_size: The grid size for the histogram.
    :type tile_size: tuple
    :return: A transformed image based on the CLAHE algorithm.
    :rtype: PIL.Image
    """
    bgr = pil_to_cv2(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_size)
    l_equalization = clahe.apply(l)
    lab_equation = cv2.merge([l_equalization, a, b])
    bgr_equalization = cv2.cvtColor(lab_equation, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(bgr_equalization)


def build_training_transforms(cfg: DataConfig) -> torchvision.transforms.Compose:
    """
    Builds the transform pipline for training with torchvision.
    :param cfg: The configuration to use for the transformation.
    :type cfg: DataConfig
    :return: A list of callable transforms for training.
    :rtype: torchvision.transforms.Compose
    """
    transforms_list = []

    transforms_list.append(transforms.Resize(cfg.target_size))
    if cfg.apply_random_flip:
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomVerticalFlip())
    if cfg.apply_random_rotation:
        transforms_list.append(transforms.RandomRotation(cfg.random_rotation_degrees))
    if cfg.apply_color_jitter:
        transforms_list.append(transforms.ColorJitter(
            brightness = cfg.color_jitter_brightness,
            contrast = cfg.color_jitter_contrast,
            saturation = cfg.color_jitter_saturation,
            hue = cfg.color_jitter_hue
        ))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(
        mean = cfg.normalization_mean,
        std = cfg.normalization_std
    ))
    return transforms.Compose(transforms_list)



def build_evaluation_transforms(cfg: DataConfig) -> torchvision.transforms.Compose:
    """
    Builds a torchvision transform pipeline for the validation or testing step.
    :param cfg: The configuration to use for the transformation set.
    :type cfg: DataConfig
    :return: A callable list of transformations.
    :rtype: torchvision.transforms.Compose
    """
    transforms_list = []
    transforms_list.append(transforms.Resize(cfg.target_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(
        mean = cfg.normalization_mean,
        std = cfg.normalization_std
    ))
    return transforms.Compose(transforms_list)