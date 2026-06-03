"""MNIST dataset module for federated learning."""

from fl_module.mnist.dataset import MNISTDataset
from fl_module.mnist.utils import (
    setup_federated_data,
    load_client_data,
    load_test_data,
    create_client_dataloaders,
    create_test_dataloader
)
