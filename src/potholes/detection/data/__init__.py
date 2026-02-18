from potholes.detection.data.dataset import PotholesDataset
from potholes.detection.data.old_dataset import OldPotholeDataset


def get_dataset(*, config: dict):
    data_version = config.get('version', 2) # if no version, the version is 2

    if data_version == 1:
        dataset = OldPotholeDataset(config=config)
    else:
        dataset = PotholesDataset(config=config)

    return dataset