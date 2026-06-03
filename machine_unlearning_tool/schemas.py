from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass
class DatasetSchema:
    id_column: str
    timestamp_column: Optional[str]
    input_cols: Sequence[str]
    target_col: str
    client_column: Optional[str] = None
    rename_map: Optional[Dict[str, str]] = None  #rename cols to our names
