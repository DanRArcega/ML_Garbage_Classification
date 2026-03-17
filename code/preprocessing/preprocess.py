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
from PIL import Image
from tqdm import tqdm

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
    base_directory = data_directory / "Garbage classification"
    print("Found splits:", [d.name for d in base_directory.iterdir() if d.is_dir()])
    for split in ("test", "train", "val"):
        split_directory = base_directory / split
        for label_directory in split_directory.iterdir():
            if not label_directory.is_dir():
                continue
            for image_path in label_directory.glob("*.jpg"):
                records.append({
                    "path": str(image_path),
                    "label": label_directory.name,
                    "split": split
                })
    return pd.DataFrame(records)



def process_and_save(
        dataframe: pd.DataFrame,
        split_name: str,
        output_directory: Path,
        cfg: DataConfig
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
    :return: The manifest of images saved after processing.
    :rtype: list[dict]
    """
    manifest_rows = []
    for _, row in tqdm(dataframe.iterrows(), total = len(dataframe), desc = f"processing {split_name}...", unit = "img"):
        src_path = Path(row["path"])
        label = row["label"]
        destination_directory = output_directory / split_name / label
        destination_directory.mkdir(parents = True, exist_ok = True)
        destination_path = destination_directory / src_path.name

        try:
            image = Image.open(src_path).convert("RGB")
            image = resize_image(image, cfg.target_size)
            if cfg.apply_clahe:
                image = apply_clahe(image, cfg.clahe_clip_limit, cfg.clahe_tile_size)
            image.save(destination_path, format = "JPEG", quality = 100)
            manifest_rows.append({
                "split": split_name,
                "label": label,
                "processed_path": str(destination_path),
                "original_path": str(src_path)
            })
        except Exception as e:
            print(f"Error in processing image {src_path.name}: {e}")
    return manifest_rows




def main(cfg: DataConfig = DATA_CONFIG) -> None:
    parser = argparse.ArgumentParser(description = "Offline image preprocessing.")
    parser.add_argument("--raw_data_dir", type = Path, default = cfg.raw_data_path)
    parser.add_argument("--processed_data_dir", type = Path, default = cfg.processed_data_path)
    args = parser.parse_args()

    cfg.raw_data_path = args.raw_data_dir
    cfg.processed_data_path = args.processed_data_dir

    print(f"Loading image data from {cfg.raw_data_path}...")
    print(f"Output directory: {cfg.processed_data_path}")
    print(f"Target size = {cfg.target_size}")
    print(f"CLAHE: {"on" if cfg.apply_clahe else "off"}")

    if cfg.processed_data_path.exists():
        print(f"Removing existing processed data at {cfg.processed_data_path}...")
        shutil.rmtree(cfg.processed_data_path)
    cfg.processed_data_path.mkdir(parents = True)

    dataframe = gather_images(cfg.raw_data_path)
    print(f"Found {len(dataframe)} images across {dataframe["label"].nunique()} classes.")

    manifest = []
    for split_name in ("train", "val", "test"):
        split_dataframe = dataframe[dataframe["split"] == split_name]
        manifest.extend(process_and_save(split_dataframe, split_name, cfg.processed_data_path, cfg))
    manifest_path = cfg.processed_data_path / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index = False)
    print(f"Manifest saved to {manifest_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()

