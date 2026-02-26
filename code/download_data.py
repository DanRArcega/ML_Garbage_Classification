"""
file: download_data.py
description: downloads all necessary data to the data directory.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""

import pandas as pd
import mlcroissant as mlc
import os
import zipfile
import requests
from tqdm import tqdm


GARBAGE_DATA_URL = "https://www.kaggle.com/datasets/hassnainzaidi/garbage-classification/croissant/download"
SAVE_DIR = "../data"



def get_kaggle_credentials() -> dict:
    """
    Retrieves the Username and API key for Kaggle from the system user's environment variables.
    :return: The username and API key for Kaggle.
    :rtype: dict
    """
    return {
        "username": os.environ.get("KAGGLE_USERNAME"),
        "key": os.environ.get("KAGGLE_KEY")
    }


def download_pull(url: str, directory: str) -> None:
    """
    Downloads and unzips the dataset from the url, to the given directory (should be the data directory by default).
    :param url: The url of the dataset to be downloaded.
    :type url: str
    :param directory: The directory to download the dataset to.
    :type directory: str
    :return: None.
    :rtype: None
    """
    os.makedirs(directory, exist_ok = True)
    credentials = get_kaggle_credentials()
    print(f"Downloading Archive...")
    response = requests.get(url,
                            auth = (credentials["username"], credentials["key"]),
                            stream = True
                            )
    response.raise_for_status()

    content_count = int(response.headers.get("Content-Length", 0))
    zip_path = os.path.join(directory, "garbage_archive.zip")
    with open(zip_path, "wb") as f, tqdm(total = content_count, unit = "B", unit_scale = True, desc = "garbage_archive") as bar:
        for chunk in response.iter_content(chunk_size = 1024):
            f.write(chunk)
            bar.update(len(chunk))

    print("Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()
        for member in tqdm(members, desc = "extracting", unit = "file"):
            zip_ref.extract(member, directory)
    os.remove(zip_path)


def get_dataframe(directory: str) -> pd.DataFrame:
    """
    Produces a dataframe of all the records downloaded to the data directory and all the sub directories.
    :param directory: The directory to search through.
    :type directory: str
    :return: A dataframe of all the found records.
    :rtype: pd.DataFrame
    """
    records = []
    for root, directories, files in os.walk(directory):
        for file_name in tqdm(files, desc = os.path.basename(root), unit = "file", leave = False):
            if file_name.lower().endswith(".jpg"):
                label = os.path.basename(root)
                path = os.path.join(root, file_name)
                records.append({"path": path, "label": label})
    return pd.DataFrame(records)



def main():
    garbage_dataset = mlc.Dataset(GARBAGE_DATA_URL)
    dataset_url = garbage_dataset.metadata.file_objects[0].content_url
    download_pull(dataset_url, SAVE_DIR)
    dataframe = get_dataframe(SAVE_DIR)
    print(dataframe.head())
    print(f"\n{len(dataframe)} images downloaded across {dataframe['label'].nunique()} classes:")
    print(dataframe["label"].value_counts())



if __name__ == '__main__':
    main()