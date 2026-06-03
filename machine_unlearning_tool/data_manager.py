from typing import Dict

import pandas as pd
from torch.utils.data import DataLoader

from .loaders import BaseDatasetAdapter, split_scale_prepare
from .schemas import DatasetSchema
from .training import create_loader
from .windowing import make_sequence_dataset


class DataManager:
    def __init__(
        self,
        adapter: BaseDatasetAdapter,
        schema: DatasetSchema,
        seq_len: int = 24,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.adapter = adapter
        self.schema = schema
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self._scaler = None

    def load_dataframe(self, source: str) -> pd.DataFrame:
        df = self.adapter.load(source)
        return df

    def prepare_loaders(
        self,
        df: pd.DataFrame,
        forget_ids=None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        scale_inputs: bool = True,
    ) -> Dict[str, DataLoader]:
        train_df, val_df, test_df, scaler = split_scale_prepare(
            df,
            self.schema,
            forget_ids=forget_ids,
            train_frac=train_frac,
            val_frac=val_frac,
            scale_inputs=scale_inputs,
            seed=self.seed,
        )
        self._scaler = scaler
        ds_train = make_sequence_dataset(train_df, self.schema, self.seq_len)
        ds_val = make_sequence_dataset(val_df, self.schema, self.seq_len)
        ds_test = make_sequence_dataset(test_df, self.schema, self.seq_len)
        return {
            "train": create_loader(ds_train, batch_size=self.batch_size, shuffle=True),
            "val": create_loader(ds_val, batch_size=self.batch_size, shuffle=False),
            "test": create_loader(ds_test, batch_size=self.batch_size, shuffle=False),
        }

    @property
    def scaler(self):
        return self._scaler


