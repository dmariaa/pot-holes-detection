import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file (with columns 'path','label')
            transform (callable, optional): Optional transform to apply to the data
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label"])

        # Load numpy array
        data = np.load(path)

        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data, label


if __name__ == "__main__":
    from torchvision import transforms as tr


    class ToTensor(object):
        def __call__(self, sample):
            return torch.from_numpy(sample).float()


    transforms = tr.Compose([
        tr.Resize((224, 224))
    ])

    dataset = CustomCSVDataset("data_old/train_unbalanced.csv", transform=transforms)
    row = dataset[0]
    pass