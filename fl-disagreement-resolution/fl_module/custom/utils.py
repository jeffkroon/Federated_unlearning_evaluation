"""Utility functions for custom dataset loading."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from fl_module.custom.adapter import CustomDatasetAdapter
from fl_module.registry import DatasetAdapterRegistry
from machine_unlearning_tool.schemas import DatasetSchema


def register_custom_dataset(
    dataset_path: str,
    schema: DatasetSchema,
    experiment_type: str = "custom",
    num_clients: int = 6,
    iid: bool = True,
    client_column: Optional[str] = None,
    train_frac: float = 0.8,
    test_frac: float = 0.2,
    seed: int = 42
) -> CustomDatasetAdapter:
    """Build a CustomDatasetAdapter for the given file/schema and register it under experiment_type."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    adapter = CustomDatasetAdapter(
        dataset_path=dataset_path,
        schema=schema,
        num_clients=num_clients,
        iid=iid,
        client_column=client_column,
        train_frac=train_frac,
        test_frac=test_frac,
        seed=seed
    )

    DatasetAdapterRegistry.register(experiment_type, adapter)

    print(f"Registered custom dataset adapter for experiment_type='{experiment_type}'")
    print(f"  Dataset: {dataset_path}")
    print(f"  Clients: {num_clients}")
    print(f"  Partitioning: {'IID' if iid else 'Non-IID'}")
    print(f"  Training samples: {sum(len(df) for df in adapter.client_data.values())}")
    print(f"  Test samples: {len(adapter.test_df)}")

    return adapter


def load_custom_test_data(experiment_type: str = "custom", sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load test data from the custom adapter registered under experiment_type."""
    adapter = DatasetAdapterRegistry.get_adapter(experiment_type)
    if adapter is None:
        raise ValueError(f"No adapter registered for experiment_type='{experiment_type}'")

    if not isinstance(adapter, CustomDatasetAdapter):
        raise ValueError(f"Adapter for '{experiment_type}' is not a CustomDatasetAdapter")

    return adapter.load_test_data(sample_size=sample_size)


def create_custom_test_dataloader(
    samples: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    is_classification: bool = True
) -> DataLoader:
    """Wrap custom test arrays in a DataLoader (label dtype depends on the task)."""
    X_tensor = torch.from_numpy(samples).float()

    if is_classification:
        y_tensor = torch.from_numpy(labels).long()
    else:
        y_tensor = torch.from_numpy(labels).float()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader
