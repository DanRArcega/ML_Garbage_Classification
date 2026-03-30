"""
file: preprocessing/preprocess.py
description: Handles offline preprocessing steps for image training.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


import argparse
import shutil
from pathlib import Path
import pandas as pd
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

from config import DATA_CONFIG, DataConfig
from transforms import resize_image, apply_clahe


def gather_images(data_directory: Path) -> pd.DataFrame:
    """
    Walks through the data directory and collects all the images.
    :param data_directory: The path to the data directory.
    :type data_directory: Path
    :return: A dataframe containing all the images and their corresponding labels.
    :rtype: pd.DataFrame
    """
    records = []
    base_directory = data_directory / "Garbage classification" / "raw"
    for label_directory in sorted(base_directory.iterdir()):
        if not label_directory.is_dir():
            continue
        for image_path in label_directory.glob("*.jpg"):
            records.append({
                "path": str(image_path),
                "label": label_directory.name,
            })
    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        raise FileNotFoundError(f"No images found in {data_directory}.")
    return dataframe



def split_images(
        dataframe: pd.DataFrame,
        training_ratio: float,
        test_ratio: float,
        validation_ratio: float,
        random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the image dataframe into a training/testing/validation split.
    :param dataframe: The unified dataframe containing all the images and their corresponding labels.
    :type dataframe: pd.DataFrame
    :param training_ratio: The portion of the dataframe to go to training.
    :type training_ratio: float
    :param test_ratio: The portion of the dataframe to go to testing.
    :type test_ratio: float
    :param validation_ratio: The portion of the dataframe to go to validation.
    :type validation_ratio: float
    :param random_seed: The seed for random number generation.
    :type random_seed: int
    :return: Three dataframes split by the given ratios.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    assert abs(training_ratio + test_ratio + validation_ratio - 1.0) < 1e-6
    training_dataframe, alternate_dataframe = train_test_split(
        dataframe,
        test_size = (1.0 - training_ratio),
        stratify = dataframe["label"],
        random_state = random_seed
    )
    relative_value = validation_ratio / (validation_ratio + test_ratio)
    validation_dataframe, test_dataframe = train_test_split(
        alternate_dataframe,
        test_size = (1.0 - relative_value),
        stratify = alternate_dataframe["label"],
        random_state = random_seed
    )
    return training_dataframe, validation_dataframe, test_dataframe



def split_summary(train_dataframe: pd.DataFrame, validation_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame) -> None:
    """
    Reports on pre-augmentation counts for training, testing and validation splits.
    :param train_dataframe: The training split.
    :type train_dataframe: pd.DataFrame
    :param validation_dataframe: The validation split.
    :type validation_dataframe: pd.DataFrame
    :param test_dataframe: The testing split.
    :type test_dataframe: pd.DataFrame
    :return: None.
    :rtype: None
    """
    train_count = len(train_dataframe)
    test_count = len(test_dataframe)
    validation_count = len(validation_dataframe)
    total = train_count + test_count + validation_count
    print(f"\nSplit summary - total: {total} images")
    print(f"    Train: {train_count:5d} ({train_count / total * 100:.1f}%)")
    print(f"    Test: {test_count:5d} ({test_count / total * 100:.1f}%)")
    print(f"    Validation: {validation_count:5d} ({validation_count / total * 100:.1f}%)")



def augment_image(image: Image.Image, seed: int, config: DataConfig) -> Image.Image:
    """
    Randomly applies a number of transformations to an image.
    :param image: The PIL image to augment.
    :type image: Image.Image
    :param seed: The random seed for reproduction.
    :type seed: int
    :param config: The representation of config options for types of transformations to apply.
    :type config: DataConfig
    :return: The transformed image.
    :rtype: Image.Image
    """
    rng = random.Random(seed)
    if config.apply_random_flip:
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        if rng.random() < 0.5:
            image = ImageOps.flip(image)
    if config.apply_random_rotation:
        if rng.random() < 0.5:
            angle = rng.uniform(-config.random_rotation_degrees, config.random_rotation_degrees)
            image = image.rotate(angle, resample = Image.Resampling.BILINEAR, expand = False)
    if config.apply_color_jitter:
        if rng.random() < 0.5:
            factor = rng.uniform(1 - config.color_jitter_brightness, 1 + config.color_jitter_brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if rng.random() < 0.5:
            factor = rng.uniform(1 - config.color_jitter_contrast, 1 + config.color_jitter_contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if rng.random() < 0.5:
            factor = rng.uniform(1 - config.color_jitter_saturation, 1 + config.color_jitter_saturation)
            image = ImageEnhance.Color(image).enhance(factor)
        if rng.random() < 0.5:
            shift = rng.uniform(-config.color_jitter_hue, config.color_jitter_hue)
            bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(float)
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + shift * 180) % 180
            hsv_image = hsv_image.clip(0, 255).astype(np.uint8)
            bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    return image





def process_and_save(
        dataframe: pd.DataFrame,
        split_name: str,
        output_directory: Path,
        cfg: DataConfig,
        target_count: int | None = None
) -> list[dict]:
    """
    Gathers all images, runs all offline preprocessing steps, and deposits the changed images into a directory defined
    in the config file. Also produces an image manifest for referencing.
    :param dataframe: The dataframe storing the images to transform.
    :type dataframe: pd.DataFrame
    :param split_name: The particular split part of the image directory.
    :type split_name: str
    :param output_directory: The name of the directory the processed images will be saved to.
    :type output_directory: Path
    :param cfg: The config file to be passed in to handle how images are processed in the offline stage.
    :type cfg: DataConfig
    :param target_count: The target number of training images per class.
    :type target_count: int
    :return: The manifest of images saved after processing.
    :rtype: list[dict]
    """
    manifest_rows = []
    for _, row in tqdm(dataframe.iterrows(), total = len(dataframe), desc = f"Processing {split_name} originals", unit = "image"):
        source_path = Path(row["path"])
        label = row["label"]
        destination_directory = output_directory / split_name / label
        destination_directory.mkdir(parents = True, exist_ok = True)
        destination_path = destination_directory / source_path.name

        try:
            image = Image.open(source_path).convert("RGB")
            image = resize_image(image, cfg.target_size)
            if cfg.apply_clahe:
                image = apply_clahe(image, cfg.clahe_clip_limit, cfg.clahe_tile_size)
            image.save(destination_path, format = "JPEG", quality = 100)
            manifest_rows.append({
                "split": split_name,
                "label": label,
                "processed_path": str(destination_path),
                "original_path": str(source_path),
                "augmented": False
            })
        except Exception as e:
            print(f"Error processing image: {source_path.name} - {e}")

    if split_name == "train" and target_count is not None:
        for label in dataframe["label"].unique():
            label_rows = dataframe[dataframe["label"] == label]
            current_count = len(label_rows)
            needed = target_count - current_count

            if needed <= 0:
                print(f"    {label} already at {current_count} images. Skipping augmentation.")
                continue
            print(f"    {label}: {current_count} originals. Generating {needed} augmented images...")
            source_paths = label_rows["path"].tolist()
            destination_directory = output_directory / split_name / label
            augmentation_index = 0
            with tqdm(total = needed, desc = f"Augmenting {label}", unit = "image") as progress_bar:
                while augmentation_index < needed:
                    source_path = Path(source_paths[augmentation_index % len(source_paths)])
                    seed = augmentation_index * 31 + hash(label) % 1000
                    try:
                        image = Image.open(source_path).convert("RGB")
                        image = resize_image(image, cfg.target_size)
                        if cfg.apply_clahe:
                            image = apply_clahe(image, cfg.clahe_clip_limit, cfg.clahe_tile_size)
                        image = augment_image(image, seed, cfg)
                        augmented_filename = f"{source_path.stem}_aug{augmentation_index:04d}.jpg"
                        destination_path = destination_directory / augmented_filename
                        image.save(destination_path, format = "JPEG", quality = 100)
                        manifest_rows.append({
                            "split": split_name,
                            "label": label,
                            "processed_path": str(destination_path),
                            "original_path": str(source_path),
                            "augmented": True
                        })
                        augmentation_index += 1
                        progress_bar.update(1)
                    except Exception as e:
                        print(f"Error processing image: {source_path.name} - {e}")
                        augmentation_index += 1
    return manifest_rows




def main(cfg: DataConfig = DATA_CONFIG) -> None:
    parser = argparse.ArgumentParser(description = "Offline image preprocessing.")
    parser.add_argument("--raw_data_dir", type = Path, default = cfg.raw_data_path)
    parser.add_argument("--processed_data_dir", type = Path, default = cfg.processed_data_path)
    parser.add_argument("--target_count", type = int, default = 500,
                        help = "Target number of training images per class.")
    args = parser.parse_args()

    cfg.raw_data_path = args.raw_data_dir
    cfg.processed_data_path = args.processed_data_dir

    print(f"Loading image data from {cfg.raw_data_path}...")
    print(f"Output directory: {cfg.processed_data_path}")
    print(f"Target size: {cfg.target_size}")
    print(f"CLAHE: {"on" if cfg.apply_clahe else "off"}")
    print(f"Augmentation target/class: {args.target_count}")

    if cfg.processed_data_path.exists():
        print(f"Removing existing processed data at {cfg.processed_data_path}...")
        shutil.rmtree(cfg.processed_data_path)
    cfg.processed_data_path.mkdir(parents = True)

    dataframe = gather_images(cfg.raw_data_path)
    print(f"Found {len(dataframe)} images across {dataframe["label"].nunique()} classes.")

    train_dataframe, validation_dataframe, test_dataframe = split_images(
        dataframe,
        cfg.training_ratio,
        cfg.test_ratio,
        cfg.validation_ratio,
        cfg.split_random_seed
    )
    split_summary(train_dataframe, validation_dataframe, test_dataframe)

    manifest = []
    manifest.extend(process_and_save(train_dataframe, "train", cfg.processed_data_path, cfg, target_count = args.target_count))
    manifest.extend(process_and_save(validation_dataframe, "validation", cfg.processed_data_path, cfg))
    manifest.extend(process_and_save(test_dataframe, "test", cfg.processed_data_path, cfg))
    manifest_dataframe = pd.DataFrame(manifest)
    manifest_path = cfg.processed_data_path / "manifest.csv"
    manifest_dataframe.to_csv(manifest_path, index = False)

    print(f"\nManifest saved to {manifest_path}")
    print(f"Final training class counts (originals + augmented):")
    train_manifest = manifest_dataframe[manifest_dataframe["split"] == "train"]
    print(train_manifest["label"].value_counts().to_string())
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

