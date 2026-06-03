from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def train_val_test_split_indices(
    n: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def apply_forget_filter(df: pd.DataFrame, id_column: str, forget_ids: Iterable) -> pd.DataFrame:
    forget_set = set(forget_ids)
    return df[~df[id_column].isin(forget_set)].reset_index(drop=True)
