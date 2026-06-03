"""Base dataset classes and the adapter interface."""

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseDataset(Dataset):
    """Base class for experiment datasets; subclasses implement __len__ and __getitem__."""

    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError("Dataset class must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset class must implement __getitem__")


class DatasetAdapter(ABC):
    """Common interface for loading client data and converting it to the unlearning format."""

    @abstractmethod
    def load_client_data(self, client_id: int, data_dir: str, sample_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Load (samples, labels) for one client."""
        pass

    @abstractmethod
    def to_unlearning_format(
        self,
        X: np.ndarray,
        y: np.ndarray,
        client_ids: np.ndarray
    ) -> Dict[str, Any]:
        """Convert arrays to the unlearning dict: df, X, y, input_cols, target_col, id_column, seq_len."""
        pass

    @abstractmethod
    def get_sequence_length(self) -> int:
        """Sequence length (1 for non-sequential data)."""
        pass

    @abstractmethod
    def is_classification(self) -> bool:
        """True for classification, False for regression. Drives metrics, loss and plots."""
        pass

    def get_input_dim(self) -> Optional[int]:
        # None = let the consumer fall back to config (model_params.input_dim)
        return None

    def get_output_dim(self) -> Optional[int]:
        # None = let the consumer fall back to config (model_params.output_dim)
        return None

    def setup_data(self, data_config: dict, client_ids: list) -> None:
        """Optional hook to download/partition data. No-op by default."""
        pass

    def flatten_samples(self, samples: np.ndarray) -> np.ndarray:
        """Flatten samples to 2D (n_samples, n_features) for the unlearning tool."""
        if samples.ndim == 2:
            return samples
        elif samples.ndim == 3:
            # time series: (N, seq_len, feat) -> (N*seq_len, feat)
            n_samples, seq_len, n_features = samples.shape
            return samples.reshape(-1, n_features)
        elif samples.ndim == 4:
            #images: (N, C, H, W) -> (N, C*H*W)
            return samples.reshape(len(samples), -1)
        else:
            return samples.reshape(len(samples), -1)
