from collections import Counter

import numpy as np

from potholes.detection.data.dataset import labels_to_id, session_split_indices, stratified_split_indices


def build_split_indices(dataset, config: dict) -> tuple[list[int], list[int], list[int]]:
    split_config = config.get("split") or {}
    split_strategy = split_config.get("strategy", "stratified")
    if split_strategy == "session":
        return session_split_indices(
            dataset,
            train_sessions=split_config.get("train_sessions"),
            val_sessions=split_config.get("val_sessions", []),
            test_sessions=split_config.get("test_sessions", []),
            shuffle=split_config.get("shuffle", False),
            seed=split_config.get("seed"),
        )
    if split_strategy == "stratified":
        return stratified_split_indices(
            dataset,
            val_ratio=split_config.get("val_ratio", 0.2),
            test_ratio=split_config.get("test_ratio", 0.2),
            shuffle=split_config.get("shuffle", False),
        )

    raise ValueError(f"Unknown split strategy: {split_strategy}")


def metadata_at(dataset, idx: int) -> dict:
    if hasattr(dataset, "_data_"):
        return dataset._data_[idx][1]

    _, metadata = dataset[idx]
    return metadata


def split_distribution(dataset, indices: list[int] | np.ndarray) -> dict:
    labels = Counter()
    sessions = Counter()

    for idx in indices:
        metadata = metadata_at(dataset, int(idx))
        labels[labels_to_id(metadata.get("labels", [])).label] += 1
        sessions[metadata.get("session_id", "unknown")] += 1

    return {
        "size": len(indices),
        "labels": dict(sorted(labels.items())),
        "sessions": dict(sorted(sessions.items())),
    }


def build_split_summary(dataset, train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> dict:
    return {
        "dataset_size": len(dataset),
        "splits": {
            "train": split_distribution(dataset, train_idx),
            "val": split_distribution(dataset, val_idx),
            "test": split_distribution(dataset, test_idx),
        },
    }
