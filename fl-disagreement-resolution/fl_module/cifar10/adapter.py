"""Adapter for CIFAR-10 dataset."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from fl_module.base import DatasetAdapter
from fl_module.cifar10.utils import load_client_data as _load_cifar10_client_data
import fl_module


class CIFAR10Adapter(DatasetAdapter):

    def load_client_data(self, client_id: int, data_dir: str, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        return _load_cifar10_client_data(client_id, data_dir, sample_size)

    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, Any]:
        #flatten images to vectors
        n_pixels = np.prod(X.shape[1:])
        X_flat = X.reshape(X.shape[0], n_pixels)
        pixel_cols = [f"pixel_{i}" for i in range(n_pixels)]

        df = pd.DataFrame(X_flat, columns=pixel_cols)
        df["client_id"] = client_ids
        df["target"] = y

        return {
            "df": df,
            "X": X_flat,
            "y": y,
            "input_cols": pixel_cols,
            "target_col": "target",
            "id_column": "client_id",
            "seq_len": 1  # images are not sequences
        }

    def get_sequence_length(self) -> int:
        return 1

    def is_classification(self) -> bool:
        return True

    def get_input_dim(self) -> int:
        return 3072  # 3 * 32 * 32

    def get_output_dim(self) -> int:
        return 10

    def setup_data(self, data_config: dict, client_ids: List[int]) -> None:
        if not data_config.get("setup_data", False):
            return
        num_clients = len(client_ids) if client_ids else 1
        train_exists = all(
            os.path.exists(os.path.join("data/cifar10", "train", f"client_{i}", "cifar10_data.npz"))
            for i in range(num_clients)
        )
        test_exists = os.path.exists(os.path.join("data/cifar10", "test", "cifar10_test.npz"))
        if (train_exists and test_exists) and not data_config.get("force_setup_data", False):
            print("CIFAR-10 data already exists. Skipping setup.")
            return
        print("Setting up CIFAR-10 federated data...")
        fl_module.setup_cifar10_federated_data(
            num_clients=num_clients,
            samples_per_client=data_config.get("client_sample_size", 5000),
            iid=data_config.get("iid", True),
        )
        print("CIFAR-10 data setup complete.")
