"""Registry for dataset adapters."""

from typing import Dict, Optional
from fl_module.base import DatasetAdapter


class DatasetAdapterRegistry:
    """Maps a dataset type (e.g. 'mnist') to its adapter."""

    _adapters: Dict[str, DatasetAdapter] = {}

    @classmethod
    def register(cls, dataset_type: str, adapter: DatasetAdapter):
        cls._adapters[dataset_type] = adapter
        print(f"Registered adapter for dataset type: {dataset_type}")

    @classmethod
    def get_adapter(cls, dataset_type: str) -> Optional[DatasetAdapter]:
        return cls._adapters.get(dataset_type)

    @classmethod
    def is_registered(cls, dataset_type: str) -> bool:
        return dataset_type in cls._adapters

    @classmethod
    def list_registered(cls) -> list:
        return list(cls._adapters.keys())
