"""Adapter for N-CMAPSS dataset."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from fl_module.base import DatasetAdapter
from fl_module.n_cmapss.utils import load_client_data as _load_ncmapss_client_data


class NCMAPSSAdapter(DatasetAdapter):

    def load_client_data(self, client_id: int, data_dir: str, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        return _load_ncmapss_client_data(client_id, data_dir, sample_size)

    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, Any]:
        #X is already flattened: (n_samples, n_features)
        n_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        
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
            "seq_len": 50  # N-CMAPSS sequence length
        }
    
    def get_sequence_length(self) -> int:
        return 50

    def is_classification(self) -> bool:
        return False

    def get_input_dim(self) -> int:
        return 50 * 20  # seq_len * n_features

    def get_output_dim(self) -> int:
        return 1

