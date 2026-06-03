"""Adapter for custom datasets (CSV/Parquet uploads)."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from fl_module.base import DatasetAdapter

#pull in the machine_unlearning_tool loaders/schema
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
machine_unlearning_path = os.path.join(base_dir, "machine_unlearning_tool")
if machine_unlearning_path not in sys.path:
    sys.path.insert(0, machine_unlearning_path)

try:
    from machine_unlearning_tool.loaders import CsvAdapter, ParquetAdapter, BaseDatasetAdapter
    from machine_unlearning_tool.schemas import DatasetSchema
    from machine_unlearning_tool.data_utils import extract_features_targets
    _MACHINE_UNLEARNING_AVAILABLE = True
except ImportError:
    _MACHINE_UNLEARNING_AVAILABLE = False
    #dummy so the type hints below still resolve
    class DatasetSchema:
        pass
    print("Warning: machine_unlearning_tool not available. Custom dataset adapter will not work.")


class CustomDatasetAdapter(DatasetAdapter):
    """Adapter for a single CSV/Parquet file: loads it, splits train/test, and
    partitions training data into clients (IID or non-IID).

    Usage:
        adapter = CustomDatasetAdapter(dataset_path="data/my_dataset.csv",
                                       schema=DatasetSchema(...), num_clients=6, iid=True)
        DatasetAdapterRegistry.register("custom", adapter)
    """

    def __init__(
        self,
        dataset_path: str,
        schema: DatasetSchema,
        num_clients: int = 6,
        iid: bool = True,
        client_column: Optional[str] = None,
        train_frac: float = 0.8,
        test_frac: float = 0.2,
        seed: int = 42
    ):
        if not _MACHINE_UNLEARNING_AVAILABLE:
            raise ImportError("machine_unlearning_tool is required for CustomDatasetAdapter")

        self.dataset_path = dataset_path
        self.schema = schema
        self.num_clients = num_clients
        self.iid = iid
        self.client_column = client_column or schema.client_column
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.seed = seed

        # pick file reader from extension
        file_ext = os.path.splitext(dataset_path)[1].lower()
        if file_ext == '.csv':
            self.file_adapter = CsvAdapter()
        elif file_ext in ['.parquet', '.pq']:
            self.file_adapter = ParquetAdapter()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .parquet")

        self._load_and_partition()

    def _load_and_partition(self):
        """Load the file, split train/test, and partition training data into clients."""
        df = self.file_adapter.load(self.dataset_path)

        if self.schema.rename_map:
            df = df.rename(columns=self.schema.rename_map)

        required_cols = set(self.schema.input_cols) | {self.schema.target_col, self.schema.id_column}
        if self.client_column:
            required_cols.add(self.client_column)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        np.random.seed(self.seed)
        n_total = len(df)
        n_train = int(n_total * self.train_frac)
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        self.train_df = df.iloc[train_indices].reset_index(drop=True)
        self.test_df = df.iloc[test_indices].reset_index(drop=True)

        if self.client_column and self.client_column in self.train_df.columns:
            print(f"Using existing client column: {self.client_column}")
            self.client_data = {}
            for client_id in range(self.num_clients):
                client_df = self.train_df[self.train_df[self.client_column] == client_id]
                if len(client_df) > 0:
                    self.client_data[client_id] = client_df
        else:
            print(f"Partitioning data into {self.num_clients} clients (IID={self.iid})")
            self.client_data = self._partition_data(self.train_df, self.num_clients, self.iid)

        # virtual dir paths; data lives in memory, not on disk
        self.client_dirs = {}
        for client_id in self.client_data.keys():
            self.client_dirs[client_id] = f"custom_client_{client_id}"

    def _partition_data(self, df: pd.DataFrame, num_clients: int, iid: bool) -> Dict[int, pd.DataFrame]:
        """Split a dataframe into num_clients parts (random if iid, else sorted by label)."""
        client_data = {}

        if iid:
            df_shuffled = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            samples_per_client = len(df_shuffled) // num_clients

            for client_id in range(num_clients):
                start_idx = client_id * samples_per_client
                end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else len(df_shuffled)
                client_data[client_id] = df_shuffled.iloc[start_idx:end_idx].reset_index(drop=True)
        else:
            # non-IID: sort by label so clients see skewed distributions
            df_sorted = df.sort_values(by=self.schema.target_col).reset_index(drop=True)
            samples_per_client = len(df_sorted) // num_clients

            for client_id in range(num_clients):
                start_idx = client_id * samples_per_client
                end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else len(df_sorted)
                client_data[client_id] = df_sorted.iloc[start_idx:end_idx].reset_index(drop=True)

        return client_data

    def load_client_data(self, client_id: int, data_dir: str, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        #data_dir is unused (data is in memory); kept for interface compatibility
        if client_id not in self.client_data:
            raise ValueError(f"Client {client_id} not found in partitioned data")

        client_df = self.client_data[client_id]

        if len(client_df) > sample_size:
            client_df = client_df.sample(n=sample_size, random_state=self.seed).reset_index(drop=True)

        X, y = extract_features_targets(
            client_df,
            self.schema.input_cols,
            self.schema.target_col
        )

        return X, y

    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, Any]:
        df = pd.DataFrame(X, columns=self.schema.input_cols)
        df[self.schema.id_column] = client_ids
        df[self.schema.target_col] = y

        return {
            "df": df,
            "X": X,
            "y": y,
            "input_cols": self.schema.input_cols,
            "target_col": self.schema.target_col,
            "id_column": self.schema.id_column,
            "seq_len": 1 if not self.schema.timestamp_column else 24  #24 = default window for time series
        }

    def get_sequence_length(self) -> int:
        return 1 if not self.schema.timestamp_column else 24

    def is_classification(self) -> bool:
        """Infer from target: few unique integer-like values -> classification."""
        if not hasattr(self, "test_df") or self.test_df is None or len(self.test_df) == 0:
            return True  # default
        y = self.test_df[self.schema.target_col].values
        n_unique = len(np.unique(y))
        if n_unique <= 2 or (n_unique <= 100 and np.issubdtype(y.dtype, np.integer)):
            return True
        return False

    def get_input_dim(self) -> int:
        return len(self.schema.input_cols)

    def get_output_dim(self) -> Optional[int]:
        return None  # consumers use config model_params

    def load_test_data(self, sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load the held-out test set (optionally subsampled)."""
        test_df = self.test_df
        if sample_size and len(test_df) > sample_size:
            test_df = test_df.sample(n=sample_size, random_state=self.seed).reset_index(drop=True)

        X, y = extract_features_targets(
            test_df,
            self.schema.input_cols,
            self.schema.target_col
        )

        return X, y
