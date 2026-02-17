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
