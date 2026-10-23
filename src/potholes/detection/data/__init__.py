from potholes.detection.data.dataset import PotholesDataset
from potholes.detection.data.old_dataset import OldPotholeDataset


def get_dataset(*, config: dict):
    data_version = config["version"]

    if data_version == 1:
        dataset = OldPotholeDataset(config=config)
    else:
        dataset = PotholesDataset(config=config)

    return dataset
