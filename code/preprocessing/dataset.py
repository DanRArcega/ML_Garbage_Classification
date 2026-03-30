"""
file: preprocessing/dataset.py
description: A dataset implementation for pytorch, for applying online image transforms.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .config import CLASSES, DATA_CONFIG, Classes, DataConfig
from .transforms import build_evaluation_transforms, build_training_transforms


class GarbageDataset(Dataset):

    def __init__(self,
                 manifest_path: Path,
                 split: str,
                 label_to_index: dict[str, int],
                 transform = None) -> None:
        """
        Initializes the Dataset object.
        :param manifest_path: The path to the data manifest.
        :type manifest_path: Path
        :param split: The split to use(train, test, val).
        :type split: str
        :param label_to_index: Mapping from label to index.
        :type label_to_index: dict[str, int]
        :param transform: The transformation to use (with torchvision), defaults to None.
        :type transform: torch.nn.Module, optional
        """
        assert split in ["train", "validation", "test"], \
        f"split must be \"train\", \"validation\", or \"test\"."

        self.label_to_index = label_to_index
        self.transform = transform
        manifest = pd.read_csv(str(manifest_path))
        self.dataframe = manifest[manifest["split"] == split].reset_index(drop = True)

        if self.dataframe.empty:
            raise ValueError(f"No records found for split '{split}' in {manifest_path}")

        unknown = set(self.dataframe["label"].unique()) - set(self.label_to_index.keys())
        if unknown:
            raise ValueError(f"Unknown labels found in {manifest_path}: {unknown}")



    def __len__(self) -> int:
        """
        Returns the length of the dataset, reflected by the length of the internal dataframe.
        :return: The length of the dataframe.
        :rtype: int
        """
        return len(self.dataframe)


    def __getitem__(self, idx: int) -> tuple:
        """
        Gets the image and its label for a particular index.
        :param idx: The index to fetch.
        :type idx: int
        :return: The image and its label.
        :rtype: tuple
        """
        row = self.dataframe.iloc[idx]
        image = Image.open(row["processed_path"]).convert("RGB")
        label = self.label_to_index[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


    def class_counts(self) -> dict[str, int]:
        """
        Gets the count of records for each class.
        :return: A dictionary where the key is the class and the value is the number of records.
        :rtype: dict[str, int]
        """
        return self.dataframe["label"].value_counts().to_dict()


def build_dataloaders(
        cfg: DataConfig = DATA_CONFIG,
        classes: Classes = CLASSES
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds the PyTorch DataLoaders for the dataset.
    :param cfg: The config file as defined in config.py
    :param classes: The class set for the types of garbage, defined in config.py
    :type classes: Classes
    :return: A tuple of Dataloaders.
    :rtype: tuple[DataLoader, DataLoader, DataLoader]
    """
    manifest_path = cfg.processed_data_path / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest file found at {manifest_path}. Run preprocess.py first.")
    train_transform = build_training_transforms(cfg)
    eval_transform = build_evaluation_transforms(cfg)

    train_dataset = GarbageDataset(manifest_path, "train", classes.label_to_index, train_transform)
    val_dataset = GarbageDataset(manifest_path, "validation", classes.label_to_index, eval_transform)
    test_dataset = GarbageDataset(manifest_path, "test", classes.label_to_index, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size = cfg.batch_size,
        shuffle = True,
        num_workers = cfg.num_workers,
        pin_memory = True
    )
    print(f"Train: {len(train_dataset)} images")
    val_loader = DataLoader(
        val_dataset,
        batch_size = cfg.batch_size,
        shuffle = False,
        num_workers = cfg.num_workers,
        pin_memory = True
    )
    print(f"Validation: {len(val_dataset)} images")
    test_loader = DataLoader(
        test_dataset,
        batch_size = cfg.batch_size,
        shuffle = False,
        num_workers = cfg.num_workers,
        pin_memory = True
    )
    print(f"Test: {len(test_dataset)} images")
    return train_loader, val_loader, test_loader



def check_process(cfg: DataConfig = DATA_CONFIG, classes: Classes = CLASSES) -> None:
    """
    Loads a batch from the train loader and prints a distribution check. For debugging/testing.
    :param cfg: The config file as defined in config.py
    :type cfg: DataConfig
    :param classes: The class set for the types of garbage, defined in config.py
    :type classes: Classes
    :return: None.
    :rtype: None
    """
    import torch
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    train_loader, _, _ = build_dataloaders(cfg, classes)
    images, labels = next(iter(train_loader))
    print(f"\n Batch image tensor shape: {images.shape}")
    print(f"Batch label tensor shape: {labels.shape}")
    print(f"Label indices in batch: {labels.tolist()}")
    print(f"Label names in batch: {[classes.index_to_label[l.item()] for l in labels]}")

    mean = torch.tensor(cfg.normalization_mean).view(3, 1, 1)
    std = torch.tensor(cfg.normalization_std).view(3, 1, 1)
    images_display = images * std + mean
    images_display = images_display.clamp(0, 1)

    grid = make_grid(images_display[:16], nrow = 4)
    plt.figure(figsize = (10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Sample training batch (first 16 images)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_process(cfg = DATA_CONFIG, classes = CLASSES)








