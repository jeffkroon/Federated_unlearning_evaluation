"""Dataset adapter for the Adult Income (UCI) dataset."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List

from fl_module.base import DatasetAdapter
from fl_module.adult.utils import load_client_data, load_test_data, setup_federated_data


def _get_input_dim() -> Optional[int]:
    """Return input dim by reading from saved data files (no cached fallback)."""
    for path in [
        "data/adult/test/adult_test.npz",
        "data/adult/train/client_0/adult_data.npz",
    ]:
        if os.path.exists(path):
            return int(np.load(path)["features"].shape[1])
    return None  #Not set up yet - caller should use config fallback


class AdultAdapter(DatasetAdapter):
    """Adapter for the Adult Income dataset (UCI / OpenML)."""

    def load_client_data(
        self, client_id: int, data_dir: str, sample_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        return load_client_data(client_id, data_dir, sample_size)

    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray,
    ) -> Dict[str, Any]:
        n_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(n_features)]

        y_flat = y.astype(np.float32)
        if y_flat.ndim > 1:
            y_flat = np.argmax(y_flat, axis=1).astype(np.float32)

        df = pd.DataFrame(X, columns=feature_cols)
        df["client_id"] = client_ids
        df["target"] = y_flat

        return {
            "df": df,
            "X": X,
            "y": y_flat,
            "input_cols": feature_cols,
            "target_col": "target",
            "id_column": "client_id",
            "seq_len": 1,
        }

    def get_sequence_length(self) -> int:
        return 1

    def is_classification(self) -> bool:
        return True

    def get_input_dim(self) -> Optional[int]:
        """Return input dim from saved data, or None if data not set up yet."""
        return _get_input_dim()

    def get_output_dim(self) -> Optional[int]:
        return 2  # binary: <=50K (0) or >50K (1)

    def setup_data(self, data_config: dict, client_ids: List[int]) -> None:
        if not data_config.get("setup_data", False):
            return
        num_clients = len(client_ids) if client_ids else 5
        setup_federated_data(
            num_clients=num_clients,
            samples_per_client=data_config.get("client_sample_size", 1000),
            data_dir="data/adult",
            iid=data_config.get("iid", True),
            force=data_config.get("force_setup_data", False),
        )
