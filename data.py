"""Derm7Pt CSV splits, labels, and PyTorch dataset."""

import os
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


CONCEPT_COLUMNS = [
    "pigment_network",
    "blue_whitish_veil",
    "vascular_structures",
    "pigmentation",
    "streaks",
    "dots_and_globules",
    "regression_structures",
]

CONCEPT_VALUES = {
    "pigment_network": ["absent", "typical", "atypical"],
    "blue_whitish_veil": ["absent", "present"],
    "vascular_structures": ["absent", "regular", "irregular"],
    "pigmentation": ["absent", "regular", "irregular"],
    "streaks": ["absent", "regular", "irregular"],
    "dots_and_globules": ["absent", "regular", "irregular"],
    "regression_structures": ["absent", "present"],
}

DIAGNOSIS_VALUES = [
    "basal cell carcinoma",
    "nevus",
    "melanoma",
    "DF/LT/MLS/MISC",
    "seborrheic keratosis",
]


def _normalise_label(value: str) -> str:
    return str(value).strip().lower()


def _grouped_concept_value(column: str, value: str) -> str:
    value = _normalise_label(value)
    if column == "vascular_structures":
        if value in {"arborizing", "comma", "hairpin", "within regression", "wreath"}:
            return "regular"
        if value in {"dotted", "linear irregular", "dotted/irregular"}:
            return "irregular"
    elif column == "pigmentation":
        if value in {"diffuse regular", "localized regular"}:
            return "regular"
        if value in {"diffuse irregular", "localized irregular"}:
            return "irregular"
    elif column == "regression_structures":
        if value in {"blue areas", "white areas", "combinations"}:
            return "present"
    return value


def _grouped_diagnosis(value: str) -> str:
    diagnosis = _normalise_label(value)
    if diagnosis in {
        "blue nevus", "clark nevus", "combined nevus", "congenital nevus",
        "dermal nevus", "recurrent nevus", "reed or spitz nevus",
    }:
        return "nevus"
    if diagnosis in {
        "dermatofibroma", "lentigo", "melanosis", "miscellaneous",
        "vascular lesion",
    }:
        return "DF/LT/MLS/MISC"
    if diagnosis.startswith("melanoma"):
        return "melanoma"
    return diagnosis


class Derm7PtConceptDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        image_root: str,
        indexes: Sequence[int],
        image_size: int = 224,
    ) -> None:
        self.metadata = metadata.iloc[list(indexes)].reset_index(drop=True)
        self.image_root = image_root
        self.image_size = image_size
        self.concept_to_index = {
            column: {value: index for index, value in enumerate(CONCEPT_VALUES[column])}
            for column in CONCEPT_COLUMNS
        }
        self.diagnosis_to_index = {
            value: index for index, value in enumerate(DIAGNOSIS_VALUES)
        }

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        image_path = os.path.join(self.image_root, str(row["derm"]))
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        image_tensor = (image_tensor - 0.5) / 0.5

        concept_targets = []
        for column in CONCEPT_COLUMNS:
            label = _grouped_concept_value(column, row[column])
            if label not in self.concept_to_index[column]:
                raise ValueError(f"Unknown {column} label {row[column]!r}.")
            concept_targets.append(self.concept_to_index[column][label])

        diagnosis = _grouped_diagnosis(row["diagnosis"])
        if diagnosis not in self.diagnosis_to_index:
            raise ValueError(f"Unknown diagnosis label {row['diagnosis']!r}.")

        return (
            image_tensor,
            torch.tensor(concept_targets, dtype=torch.long),
            torch.tensor(self.diagnosis_to_index[diagnosis], dtype=torch.long),
        )


def build_datasets(data_root: str, image_size: int) -> Dict[str, Dataset]:
    metadata_root = os.path.join(data_root, "meta")
    metadata = pd.read_csv(os.path.join(metadata_root, "meta.csv"))
    datasets = {}
    for split in ("train", "valid", "test"):
        indexes = pd.read_csv(
            os.path.join(metadata_root, f"{split}_indexes.csv")
        )["indexes"].tolist()
        datasets[split] = Derm7PtConceptDataset(
            metadata, os.path.join(data_root, "images"), indexes, image_size
        )
    return datasets
