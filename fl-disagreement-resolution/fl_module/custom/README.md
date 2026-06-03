# Custom Dataset Adapter

Unified data loading voor custom datasets (CSV/Parquet files) met automatische client partitioning.

## Quick Start

```python
from machine_unlearning_tool.schemas import DatasetSchema
from fl_module.custom.utils import register_custom_dataset

# 1. Define schema
schema = DatasetSchema(
    id_column="user_id",  # Voor unlearning (identificeert samples)
    input_cols=["feat1", "feat2", "feat3"],  # Feature columns
    target_col="target",  # Target/label column
    timestamp_column=None  # Optional: voor time series
)

# 2. Register dataset
adapter = register_custom_dataset(
    dataset_path="data/my_dataset.csv",
    schema=schema,
    experiment_type="custom",
    num_clients=6,
    iid=True  # Random partitioning
)

# 3. Use in FL config: experiment_type="custom"
```

## Testing

Run het test script:

```bash
cd fl-disagreement-resolution
python examples/demo_custom_dataset.py
```

Dit test:
- Basic adapter functionality
- collect_all_client_data integration
- Test DataLoader creation
- Non-IID partitioning

## Features

- **Unified Loading**: Gebruikt machine_unlearning_tool's CsvAdapter/ParquetAdapter
- **Automatic Partitioning**: IID of non-IID client distribution
- **Full Compatibility**: Werkt met alle unlearning strategieën
- **Optional**: Alleen gebruikt als experiment_type="custom" (breekt niets)

## API

### `register_custom_dataset()`

Registreert een custom dataset adapter.

**Parameters:**
- `dataset_path`: Pad naar CSV/Parquet bestand
- `schema`: DatasetSchema met column mappings
- `experiment_type`: Naam voor registratie (default: "custom")
- `num_clients`: Aantal clients voor partitioning
- `iid`: True voor random, False voor sorted by label
- `client_column`: Optioneel: gebruik bestaande client column
- `train_frac`: Fractie voor training (default: 0.8)

### `load_custom_test_data()`

Laadt test data voor custom dataset.

### `create_custom_test_dataloader()`

Maakt PyTorch DataLoader voor test data.

## Example Dataset Format

CSV bestand moet bevatten:
- Feature columns (bijv. `feature_0`, `feature_1`, ...)
- ID column (bijv. `user_id`) - gebruikt voor unlearning
- Target column (bijv. `target`) - labels

```csv
user_id,feature_0,feature_1,feature_2,target
0,0.5,-0.3,1.2,1
1,0.1,0.8,-0.5,0
...
```

## Integration

De adapter integreert automatisch met:
- `collect_all_client_data()` - gebruikt door unlearning strategies
- `DatasetAdapterRegistry` - automatisch geregistreerd
- Alle unlearning methodes werken zonder wijzigingen
