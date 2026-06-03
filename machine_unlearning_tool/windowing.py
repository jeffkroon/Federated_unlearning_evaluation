from typing import Tuple

import numpy as np
import pandas as pd

from .data_utils import ArraySequenceDataset, extract_features_targets
from .schemas import DatasetSchema


def build_sequences_from_df(
    df: pd.DataFrame,
    schema: DatasetSchema,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = extract_features_targets(df, schema.input_cols, schema.target_col)
    return X, y


def make_sequence_dataset(
    df: pd.DataFrame,
    schema: DatasetSchema,
    seq_len: int,
):
    X, y = build_sequences_from_df(df, schema, seq_len)
    return ArraySequenceDataset(X, y, seq_len=seq_len)
