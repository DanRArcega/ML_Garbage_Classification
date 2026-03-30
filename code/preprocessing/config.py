"""
file: preprocessing/config.py
description: Config file for use in preprocessing of images.
language: python3
author: Sam Whitney, shw9601@rit.edu
"""


from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class DataConfig:

    # Data paths
    raw_data_path: Path = PROJECT_ROOT / "data"
    processed_data_path: Path = PROJECT_ROOT / "data" / "Garbage classification" / "processed"

    # Image size
    target_size: tuple[int, int] = (224, 224)

    # Training splits
    training_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    split_random_seed: int = 42

    # Preprocessing options
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple[int, int] = (8, 8)

    # Online augmentation
    apply_random_flip: bool = True
    apply_random_rotation: bool = True
    random_rotation_degrees: float = 30.0
    apply_color_jitter: bool = True
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # Normalization
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Dataloader
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class Classes:
    """
    List of garbage classes to be used for training the model.
    """
    names: list[str] = field(default_factory = lambda: [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ])

    @property
    def num_classes(self) -> int:
        return len(self.names)

    @property
    def label_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.names)}

    @property
    def index_to_label(self) -> dict[int, str]:
        return {index: name for index, name in enumerate(self.names)}


# Module singletons
DATA_CONFIG = DataConfig()
CLASSES = Classes()