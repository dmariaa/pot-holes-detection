from pathlib import Path

import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

from potholes.detection.data.dataset import PotholesDataset, stratified_split_indices, print_labels_distribution


class OldPotholeDataset(PotholesDataset):
    def __init__(self, config: dict, transforms: list = None):
        super().__init__(config, transforms)

    @staticmethod
    def __extract_file_labels__(file_name: str):
        file_path = Path(file_name)
        stem = file_path.stem
        parts = stem.split("_", 4)
        labels_part = parts[-1]
        labels = [ { 'label': s } for s in labels_part.split("+") ]
        return labels

    def __read_data__(self):
        self.data_folder = Path(self.config.get('data_folder'))
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder {self.data_folder} not found")

        data_files = list(self.data_folder.glob("*.npy"))
        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.data_folder}")

        labels_file = self.data_folder.joinpath("data_labels.csv")
        if not labels_file.exists():
            raise FileNotFoundError(f"No labels file found in {self.data_folder}")

        self._data_ = []
        with tqdm(total=len(data_files), disable=not self.config.get('verbose', True)) as pbar:
            pbar.set_description(desc="Loading data samples")
            for data_file in data_files:
                data = np.load(data_file)
                labels = self.__extract_file_labels__(data_file.name)
                meta = {
                    'labels': labels
                }
                self._data_.append((data, meta))
                pbar.update(1)


if __name__ == "__main__":
    from torchvision import transforms as tr
    from potholes.detection.data.dataset import ToTensor
    from potholes.detection.data import get_dataset

    config = {
        'version': 1,
        'data_folder': 'data_old',
        'verbose': True,
    }

    transforms = tr.Compose([
        ToTensor(),
        tr.Resize((224, 224))
    ])

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
