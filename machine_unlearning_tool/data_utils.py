from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def drop_rows(df: pd.DataFrame, id_column: str, forget_ids: Iterable) -> pd.DataFrame:
    forget_ids_set = set(forget_ids)
    return df[~df[id_column].isin(forget_ids_set)].reset_index(drop=True)


def filter_by_id(df: pd.DataFrame, id_column: str, ids: Iterable) -> pd.DataFrame:
    ids_set = set(ids)
    return df[df[id_column].isin(ids_set)].reset_index(drop=True)


def exclude_ids(df: pd.DataFrame, id_column: str, ids: Iterable) -> pd.DataFrame:
    return drop_rows(df, id_column, ids)


def extract_features_targets(
    df: pd.DataFrame, input_cols: Sequence[str], target_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[list(input_cols)].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    # one-hot labels -> class indices
    if y.ndim > 1:
        y = np.argmax(y, axis=1).astype(np.float32)
    elif y.ndim == 1:
        y = y.astype(np.float32)

    return X, y


class ArraySequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        if X.ndim != 2:
            raise ValueError("X must be 2D: [N, input_dim]")
        if y.ndim != 1:
            raise ValueError("y must be 1D: [N]")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]  # window of seq_len steps
        y_next = self.y[idx + self.seq_len]  # value right after the window
        x_tensor = torch.from_numpy(x_seq)
        y_tensor = torch.tensor(y_next, dtype=torch.float32)
        return x_tensor, y_tensor


def split_into_slices(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    id_column: str,
    num_shards: int,
    num_slices: int,
    random_state: int = 42,
    client_aware: bool = False,
):
    if num_shards < 1 or num_slices < 1:
        raise ValueError("num_shards and num_slices must be >= 1")
    rng = np.random.default_rng(random_state)
    result = []

    if client_aware:
        #keep each client in one shard, spread clients round-robin
        grouped = df.groupby(id_column).indices
        client_ids = sorted(grouped.keys())

        shard_clients = [[] for _ in range(num_shards)]
        for idx, cid in enumerate(client_ids):
            shard_idx = idx % num_shards
            shard_clients[shard_idx].append(cid)

        for shard_idx, clients in enumerate(shard_clients):
            if not clients:
                continue

            #each slice gets whole clients, so forget-ids stay together
            slice_client_groups = np.array_split(clients, min(num_slices, len(clients)))
            for slice_idx, client_group in enumerate(slice_client_groups):
                if len(client_group) == 0:
                    continue
                slice_indices = []
                for cid in client_group:
                    slice_indices.extend(grouped[cid])
                slice_indices = np.array(sorted(slice_indices))

                slice_df = df.iloc[slice_indices]
                slice_X = X[slice_indices]
                slice_y = y[slice_indices]
                slice_ids = slice_df[id_column].tolist()
                result.append(
                    {
                        "shard": shard_idx,
                        "slice": slice_idx,
                        "indices": slice_indices,
                        "X": slice_X,
                        "y": slice_y,
                        "ids": slice_ids,
                    }
                )
    else:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        shards = np.array_split(indices, num_shards)
        for shard_idx, shard_indices in enumerate(shards):
            if len(shard_indices) == 0:
                continue
            shard_slices = np.array_split(shard_indices, num_slices)
            for slice_idx, slice_indices in enumerate(shard_slices):
                if len(slice_indices) == 0:
                    continue
                slice_df = df.iloc[slice_indices]
                slice_X = X[slice_indices]
                slice_y = y[slice_indices]
                slice_ids = slice_df[id_column].tolist()
                result.append(
                    {
                        "shard": shard_idx,
                        "slice": slice_idx,
                        "indices": slice_indices,
                        "X": slice_X,
                        "y": slice_y,
                        "ids": slice_ids,
                    }
                )
    return result

