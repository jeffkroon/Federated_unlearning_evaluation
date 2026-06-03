"""Adapter for MNIST dataset."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from fl_module.base import DatasetAdapter
from fl_module.mnist.utils import load_client_data as _load_mnist_client_data
import fl_module


class MNISTAdapter(DatasetAdapter):

    def load_client_data(self, client_id: int, data_dir: str, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        return _load_mnist_client_data(client_id, data_dir, sample_size)

    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, Any]:
        # X is already flattened: (n_samples, n_pixels)
        n_pixels = X.shape[1]
        pixel_cols = [f"pixel_{i}" for i in range(n_pixels)]
        
        df = pd.DataFrame(X, columns=pixel_cols)
        df["client_id"] = client_ids
        df["target"] = y
        
        return {
            "df": df,
            "X": X,
            "y": y,
            "input_cols": pixel_cols,
            "target_col": "target",
            "id_column": "client_id",
            "seq_len": 1  #MNIST doesn't use sequences
        }
    
    def get_sequence_length(self) -> int:
        return 1

    def is_classification(self) -> bool:
        return True

    def get_input_dim(self) -> int:
        return 784  #28 * 28

    def get_output_dim(self) -> int:
        return 10

    def setup_data(self, data_config: dict, client_ids: List[int]) -> None:
        if not data_config.get("setup_data", False):
            return
        num_clients = len(client_ids) if client_ids else 1
        train_exists = all(
            os.path.exists(os.path.join("data/mnist", "train", f"client_{i}", "mnist_data.npz"))
            for i in range(num_clients)
        )
        test_exists = os.path.exists(os.path.join("data/mnist", "test", "mnist_test.npz"))
        if (train_exists and test_exists) and not data_config.get("force_setup_data", False):
            print("MNIST data already exists. Skipping setup.")
            return
        print("Setting up MNIST federated data...")
        fl_module.setup_mnist_federated_data(
            num_clients=num_clients,
            samples_per_client=data_config.get("client_sample_size", 1000),
            iid=data_config.get("iid", True),
        )
        print("MNIST data setup complete.")

