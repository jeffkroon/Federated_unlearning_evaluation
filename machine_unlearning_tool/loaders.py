from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from .partition import apply_forget_filter, train_val_test_split_indices
from .scalers import StandardScalerDF
from .schemas import DatasetSchema


class BaseDatasetAdapter(ABC):
    @abstractmethod
    def load(self, source: str) -> pd.DataFrame:
        ...


class CsvAdapter(BaseDatasetAdapter):
    def __init__(self, **read_csv_kwargs):
        self.read_csv_kwargs = read_csv_kwargs

    def load(self, source: str) -> pd.DataFrame:
        return pd.read_csv(source, **self.read_csv_kwargs)


class ParquetAdapter(BaseDatasetAdapter):
    def __init__(self, **read_parquet_kwargs):
        self.read_parquet_kwargs = read_parquet_kwargs

    def load(self, source: str) -> pd.DataFrame:
        return pd.read_parquet(source, **self.read_parquet_kwargs)


def normalize_schema(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    out = df.copy()
    if schema.rename_map:
        out = out.rename(columns=schema.rename_map)
    needed = set(schema.input_cols) | {schema.target_col, schema.id_column}
    if schema.client_column:
        needed.add(schema.client_column)
    if schema.timestamp_column:
        needed.add(schema.timestamp_column)
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return out


def split_scale_prepare(
    df: pd.DataFrame,
    schema: DatasetSchema,
    forget_ids=None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    scale_inputs: bool = True,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScalerDF]:
    df_norm = normalize_schema(df, schema)
    if forget_ids:
        df_norm = apply_forget_filter(df_norm, schema.id_column, forget_ids)
    idx_train, idx_val, idx_test = train_val_test_split_indices(len(df_norm), train_frac, val_frac, seed)
    train_df = df_norm.iloc[idx_train].reset_index(drop=True)
    val_df = df_norm.iloc[idx_val].reset_index(drop=True)
    test_df = df_norm.iloc[idx_test].reset_index(drop=True)
    scaler = StandardScalerDF()
    if scale_inputs:
        train_df = scaler.fit_transform(train_df, schema.input_cols)
        val_df = scaler.transform(val_df, schema.input_cols)
        test_df = scaler.transform(test_df, schema.input_cols)
    return train_df, val_df, test_df, scaler


