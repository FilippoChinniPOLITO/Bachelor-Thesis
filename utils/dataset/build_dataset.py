from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from experiments.weed_mapping_experiment.backend.dataset.dataset_interface import WeedMapDatasetInterface


def load_MNIST_data(root_folder):
    dataset = MNIST(root=root_folder, download=True, transform=ToTensor(), train=True)
    test_dataset = MNIST(root=root_folder, download=True, transform=ToTensor(), train=False)

    train_dataset, val_dataset = data_split(dataset=dataset, train_split=0.8)

    return train_dataset, val_dataset, test_dataset


def data_split(dataset, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def load_weedmap_data():
    return WeedMapDatasetInterface(dict(
        root="../../../../0_rotations_processed_003_test/RedEdge",
        # root="../data/Weed Map Dataset Processed/RedEdge",
        channels=['R', 'G', 'B', 'NIR', 'RE'],
        train_folders=["000", "001", "002", "004"],
        test_folders=["003"],
        batch_size=6,
        val_batch_size=12,
        test_batch_size=12,
        hor_flip=True,
        ver_flip=True,
        return_path=True,
        num_classes=3,
    ))
