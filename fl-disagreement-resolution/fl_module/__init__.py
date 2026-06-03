"""Datasets, models and utilities for the federated learning experiments."""

# Import main classes and functions from submodules
from fl_module.base import BaseDataset
from fl_module.n_cmapss.dataset import NCMAPSSDataset
from fl_module.mnist.dataset import MNISTDataset
from fl_module.tabular.dataset import TabularDataset

# Import model classes and functions
from fl_module.models import (
    BaseModel,
    RULPredictor,
    MNISTClassifier,
    CIFAR10Classifier,
    TabularClassifier,
    create_model
)

#Import utility functions
from fl_module.n_cmapss.utils import (
    load_client_data as load_ncmapss_client_data,
    load_test_data as load_ncmapss_test_data,
    preprocess_data as preprocess_ncmapss_data,
    create_client_dataloaders as create_ncmapss_client_dataloaders,
    create_test_dataloader as create_ncmapss_test_dataloader
)

from fl_module.mnist.utils import (
    setup_federated_data as setup_mnist_federated_data,
    load_client_data as load_mnist_client_data,
    load_test_data as load_mnist_test_data,
    create_client_dataloaders as create_mnist_client_dataloaders,
    create_test_dataloader as create_mnist_test_dataloader
)

from fl_module.cifar10.utils import (
    setup_federated_data as setup_cifar10_federated_data,
    load_client_data as load_cifar10_client_data,
    load_test_data as load_cifar10_test_data,
    create_client_dataloaders as create_cifar10_client_dataloaders,
    create_test_dataloader as create_cifar10_test_dataloader,
)

from fl_module.tabular.utils import (
    setup_federated_data as setup_tabular_federated_data,
    load_client_data as load_tabular_client_data,
    load_test_data as load_tabular_test_data,
    create_client_dataloaders as create_tabular_client_dataloaders,
    create_test_dataloader as create_tabular_test_dataloader
)

#Custom dataset utilities (optional, only if machine_unlearning_tool is available)
try:
    from fl_module.custom.utils import (
        load_custom_test_data,
        create_custom_test_dataloader
    )
    _CUSTOM_DATASET_AVAILABLE = True
except ImportError:
    _CUSTOM_DATASET_AVAILABLE = False

# Import adapters and registry
from fl_module.base import DatasetAdapter
from fl_module.registry import DatasetAdapterRegistry
from fl_module.n_cmapss.adapter import NCMAPSSAdapter
from fl_module.mnist.adapter import MNISTAdapter
from fl_module.cifar10.adapter import CIFAR10Adapter
from fl_module.tabular.adapter import TabularAdapter
from fl_module.adult.adapter import AdultAdapter

# Auto-register adapters
DatasetAdapterRegistry.register("n_cmapss", NCMAPSSAdapter())
DatasetAdapterRegistry.register("mnist", MNISTAdapter())
DatasetAdapterRegistry.register("cifar10", CIFAR10Adapter())
DatasetAdapterRegistry.register("tabular", TabularAdapter())
DatasetAdapterRegistry.register("adult", AdultAdapter())

# Register Adult model factory (same MLP architecture as tabular)
from fl_module.model_registry import ModelRegistry
from fl_module.models import TabularClassifier

def _factory_adult(**kwargs):
    input_dim = kwargs.get("input_dim", 105)  #105 = actual Adult Income feature count after preprocessing
    output_dim = kwargs.get("output_dim", 2)
    hidden_dims = kwargs.get("hidden_dims", [128, 64])
    dropout = kwargs.get("dropout", 0.2)
    return TabularClassifier(input_dim=input_dim, hidden_dims=hidden_dims,
                             output_dim=output_dim, dropout=dropout)

ModelRegistry.register("adult", _factory_adult)
