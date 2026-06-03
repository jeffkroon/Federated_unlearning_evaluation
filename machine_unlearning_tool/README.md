# machine_unlearning_tool (timeseries energy forecasting)

Minimal modular library for machine unlearning on time series (energy consumption).

Modules:
- `data_utils.py`: preprocess/filter, id-based splits, simple sequence dataset.
- `model_utils.py`: LSTM + linear head for one-step forecasting.
- `training.py`: training loops (supervised and soft-label distillation).
- `evaluation.py`: RMSE, MAE, R2 and evaluation helper.
- `unlearning.py`: exact retraining, SISA, knowledge distillation.
- `workflow.py`: high-level API to run unlearning pipelines.
- `__init__.py`: exports public API.

Quick start:
```python
import numpy as np
import pandas as pd
from machine_unlearning_tool import (
    run_exact_retraining,
    run_sisa_unlearning,
    run_knowledge_distillation,
)

# Example inputs (replace with real data)
df = pd.DataFrame({
    "id": np.arange(1000),
    "feat1": np.random.randn(1000),
    "feat2": np.random.randn(1000),
    "target": np.random.randn(1000),
})
input_cols = ["feat1", "feat2"]
target_col = "target"
id_column = "id"
X = df[input_cols].to_numpy(np.float32)
y = df[target_col].to_numpy(np.float32)
forget_ids = [10, 11, 12, 13]

res_retrain = run_exact_retraining(
    X, y, df, input_cols, target_col, id_column, forget_ids,
    device=None,   # auto CUDA if available
    seq_len=24,
    model_params={"hidden_size": 64},
    train_params={"epochs": 5, "batch_size": 64}
)
print(res_retrain["metrics_retain"])
```

Notes:
- SISA and distillation implementations are minimal baselines for research.
- Ensure reproducible seeds and proper validation for experiments.


