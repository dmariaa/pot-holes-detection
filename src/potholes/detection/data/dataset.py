import json
import os
from enum import IntEnum
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms as tr
from tqdm import tqdm
from pprint import pprint

from potholes.tools.generator import generate_samples


class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).float()


class PotholesDataset(Dataset):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    def __init__(self, config: dict, transforms: list = None ):
        self.config = config

        self.__read_data__()

        if transforms is None:
            transforms = [
                ToTensor(),
                tr.Resize((224, 224)),
                tr.Normalize(mean=PotholesDataset.imagenet_mean * 2, std = PotholesDataset.imagenet_std * 2)
            ]

        self.transforms = tr.Compose(transforms)

    def __read_data__(self):
        self.data_folder = self.config.get('data_folder')

        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Data folder {self.data_folder} not found")

        if self.config.get('generate', False):
            self.__generate_data__()
        else:
            self.__load_data__()

    def __load_data__(self):
        data_files = sorted(Path(self.data_folder).glob('**/*.npz'))
        if len(data_files) == 0:
            raise FileNotFoundError(f"No data files found in {self.data_folder}, maybe generate parameter should be on?")

        self._data_ = []

        with tqdm(total=len(data_files), disable=not self.config.get('verbose', True)) as pbar:
            pbar.set_description(desc="Loading data samples")
            for data_file in data_files:
                data = np.load(data_file, allow_pickle=True)
                sample = data['sample']
                metadata = json.loads(str(data['meta']))
                metadata.setdefault("session_id", data_file.parent.name)
                self._data_.append((sample, metadata))
                pbar.update(1)

    def __generate_data__(self):
        session_files = sorted(Path(self.data_folder).glob('**/session_*.csv'))
        if len(session_files) == 0:
            raise FileNotFoundError(f"No session files found in {self.data_folder}")

        self._data_ = []
        for session_file in session_files:
            df = pd.read_csv(session_file, parse_dates=['timestamp'])
            total_samples, iterator, spec_params = generate_samples(df,
                                                                    window_size=self.config.get('window_size'),
                                                                    step=self.config.get('step'))

            with tqdm(total=total_samples, disable=not self.config.get('verbose', True)) as pbar:
                pbar.set_description(desc=f"Generating data samples for session {session_file.name}")
                for sample, meta in iterator:
                    meta.setdefault("session_id", session_file.parent.name)
                    self._data_.append((sample, meta))
                    pbar.update(1)

    def __len__(self):
        return len(self._data_)

    def __getitem__(self, item):
        data_file = self._data_[item]
        sample = data_file[0]
        metadata = data_file[1]
        return self.transforms(sample), metadata


class RoadLabel(IntEnum):
    NORMAL = 0, "normal", "Normal road"
    MANHOLE = 1, "manhole", "Manhole cover"
    SPEED_BUMP = 2, "speed_bump", "Speed bump"
    OTHER = 3, "other", "Other"
    POTHOLE = 4, "pothole", "Pothole"
    MULTIPLE = 5, "multiple", "Multiple labels"

    def __new__(cls, value, name, text):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj._label = name
        obj._text = text
        return obj

    def __str__(self):
        return self._label

    @property
    def label(self):
        return self._label

    @property
    def text(self):
        return self._text

    @classmethod
    def from_label(cls, label: str):
        for member in cls:
            if member.label == label:
                return member
        raise ValueError(f"{label} is not a valid {cls.__name__}")



def labels_to_id(labels: list) -> RoadLabel:
    if len(labels) == 0:
        return RoadLabel.NORMAL
    elif len(labels) > 1:
        return RoadLabel.MULTIPLE
    else:
        return RoadLabel.from_label(labels[0]['label'])


def random_split_dataset(dataset: Dataset, val_ratio: float, test_ratio: float) -> List[Subset[Any]]:
    assert val_ratio + test_ratio < 1.0, "Validation + test ratio should be smaller than 1.0"

    train_ratio = 1.0 - val_ratio - test_ratio
    split = torch.utils.data.random_split(dataset, [train_ratio, val_ratio, test_ratio])
    return split


def stratified_split_indices(dataset: Dataset, val_ratio: float, test_ratio: float, shuffle: bool = True) -> list[list[int]]:
    assert val_ratio + test_ratio < 1.0, "Validation + test ratio should be smaller than 1.0"

    labels_indices = {}

    for i in range(len(dataset)):
        _, metadata = dataset[i]
        labels = metadata.get('labels', [])
        label = labels_to_id(labels)

        if label not in labels_indices:
            labels_indices[label] = []

        labels_indices[label].append(i)

    rng = np.random.default_rng()

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for indices in labels_indices.values():
        indices = np.array(indices, dtype=int)
        rng.shuffle(indices)

        n_total = len(indices)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val - n_test

        train_indices.extend(indices[:n_train].tolist())
        val_indices.extend(indices[n_train:n_train + n_val].tolist())
        test_indices.extend(indices[n_train + n_val:].tolist())

    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

    return [train_indices, val_indices, test_indices]


def _get_sample_metadata(dataset: Dataset, index: int) -> dict:
    if hasattr(dataset, "_data_"):
        return dataset._data_[index][1]

    _, metadata = dataset[index]
    return metadata


def session_split_indices(
        dataset: Dataset,
        *,
        val_sessions: list[str],
        test_sessions: list[str],
        shuffle: bool = True,
        seed: int | None = None,
) -> list[list[int]]:
    val_session_set = set(val_sessions)
    test_session_set = set(test_sessions)
    overlap = val_session_set & test_session_set
    if overlap:
        raise ValueError(f"Sessions cannot be in both validation and test splits: {sorted(overlap)}")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    seen_sessions: set[str] = set()

    for i in range(len(dataset)):
        metadata = _get_sample_metadata(dataset, i)
        session_id = metadata.get("session_id")
        if session_id is None:
            raise ValueError("Dataset sample metadata is missing 'session_id'")

        seen_sessions.add(session_id)
        if session_id in val_session_set:
            val_indices.append(i)
        elif session_id in test_session_set:
            test_indices.append(i)
        else:
            train_indices.append(i)

    unknown_val_sessions = val_session_set - seen_sessions
    unknown_test_sessions = test_session_set - seen_sessions
    if unknown_val_sessions or unknown_test_sessions:
        raise ValueError(
            "Unknown split sessions. "
            f"Validation: {sorted(unknown_val_sessions)}. "
            f"Test: {sorted(unknown_test_sessions)}."
        )

    if len(val_indices) == 0:
        raise ValueError("Validation split is empty")
    if len(test_indices) == 0:
        raise ValueError("Test split is empty")
    if len(train_indices) == 0:
        raise ValueError("Train split is empty")

    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

    return [train_indices, val_indices, test_indices]


def print_labels_distribution(dataset: Dataset):
    labels = {
        'normal': 0,
        'manhole': 0,
        'speed_bump': 0,
        'other': 0,
        'pothole': 0,
        'multiple': 0
    }

    for idx in range(len(dataset)):
        _, metadata = dataset[idx]
        label = labels_to_id(metadata.get('labels'))
        labels[label.label] = labels[label.label] + 1 if label.label in labels else 1

    pprint(labels, sort_dicts=False)

if __name__ == '__main__':
    from potholes.detection.data import get_dataset

    config = {
        'version': 2,
        'data_folder': 'dataset',
        'generate': False,
        'window_size': 10,
        'step': 1,
        'verbose': True,
        'split_ratio': (0.2, 0.2)   # (validation, test)
    }

    dataset = get_dataset(config=config)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0][0].shape}")

    train_idx, val_idx, test_idx = stratified_split_indices(dataset, val_ratio=0.2, test_ratio=0.2)
    print(f"Train/val/test sizes: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print("Train set labels distribution:")
    print_labels_distribution(train_dataset)
    print("Val set labels distribution:")
    print_labels_distribution(val_dataset)
    print("Test set labels distribution:")
    print_labels_distribution(test_dataset)

