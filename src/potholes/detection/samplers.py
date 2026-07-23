from collections import defaultdict
from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BalancedLabelBatchSampler(Sampler[list[int]]):
    """
    Yields batches containing N classes and K samples per class.

    Indices are local to the dataset passed to DataLoader. Sampling is with
    replacement when a class has fewer than K examples.
    """
    def __init__(
            self,
            labels: list[int],
            *,
            classes_per_batch: int,
            samples_per_class: int,
            batches_per_epoch: int,
            seed: int | None = None,
    ):
        if classes_per_batch < 1:
            raise ValueError("classes_per_batch must be >= 1")
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be >= 2 for supervised contrastive learning")
        if batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be >= 1")

        self.labels = labels
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batches_per_epoch = batches_per_epoch
        self.rng = np.random.default_rng(seed)

        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[int(label)].append(idx)

        self.label_to_indices = {label: np.array(indices, dtype=int) for label, indices in label_to_indices.items()}
        self.labels_with_enough_samples = sorted(
            label for label, indices in self.label_to_indices.items() if len(indices) > 0
        )

        if len(self.labels_with_enough_samples) < classes_per_batch:
            raise ValueError(
                f"Need at least {classes_per_batch} labels with samples, "
                f"found {len(self.labels_with_enough_samples)}"
            )

    def __iter__(self) -> Iterator[list[int]]:
        labels = np.array(self.labels_with_enough_samples, dtype=int)
        for _ in range(self.batches_per_epoch):
            selected_labels = self.rng.choice(labels, size=self.classes_per_batch, replace=False)
            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[int(label)]
                replace = len(indices) < self.samples_per_class
                batch.extend(self.rng.choice(indices, size=self.samples_per_class, replace=replace).tolist())

            self.rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch
