"""CIFAR-10 dataset utilities and adapter."""

from fl_module.cifar10.utils import (
    setup_federated_data as setup_cifar10_federated_data,
    load_client_data as load_cifar10_client_data,
    load_test_data as load_cifar10_test_data,
    create_client_dataloaders as create_cifar10_client_dataloaders,
    create_test_dataloader as create_cifar10_test_dataloader,
)
from fl_module.cifar10.adapter import CIFAR10Adapter
from fl_module.cifar10.dataset import CIFAR10Dataset
