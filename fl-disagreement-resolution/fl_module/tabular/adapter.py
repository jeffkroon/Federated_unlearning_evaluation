"""Adapter for tabular dataset."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from fl_module.base import DatasetAdapter
from fl_module.tabular.utils import load_client_data as _load_tabular_client_data
import fl_module


class TabularAdapter(DatasetAdapter):

    def load_client_data(self, client_id: int, data_dir: str, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        return _load_tabular_client_data(client_id, data_dir, sample_size)

    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, Any]:
        n_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(n_features)]

        # one-hot labels -> class indices
        if y.ndim > 1:
            y = np.argmax(y, axis=1).astype(np.float32)
        elif y.ndim == 1:
            y = y.astype(np.float32)

        df = pd.DataFrame(X, columns=feature_cols)
        df["client_id"] = client_ids
        df["target"] = y
        
        return {
            "df": df,
            "X": X,
            "y": y,
            "input_cols": feature_cols,
            "target_col": "target",
            "id_column": "client_id",
            "seq_len": 1  # Tabular data doesn't use sequences
        }
    
    def get_sequence_length(self) -> int:
        return 1

    def is_classification(self) -> bool:
        return True

    def get_input_dim(self) -> Optional[int]:
        return None  #consumers use config model_params

    def get_output_dim(self) -> Optional[int]:
        return None  #consumers use config model_params

    def setup_data(self, data_config: dict, client_ids: List[int]) -> None:
        if not data_config.get("setup_data", False):
            return
        num_clients = len(client_ids) if client_ids else 1
        train_exists = all(
            os.path.exists(os.path.join("data/tabular", "train", f"client_{i}", "tabular_data.npz"))
            for i in range(num_clients)
        )
        test_exists = os.path.exists(os.path.join("data/tabular", "test", "tabular_test.npz"))
        if (train_exists and test_exists) and not data_config.get("force_setup_data", False):
            print("Tabular data already exists. Skipping setup.")
            return
        print("Setting up tabular federated data...")
        fl_module.setup_tabular_federated_data(
            num_clients=num_clients,
            samples_per_client=data_config.get("client_sample_size", 1000),
            n_features=20,
            n_classes=2,
            task="classification",
            iid=data_config.get("iid", True),
        )
        print("Tabular data setup complete.")

