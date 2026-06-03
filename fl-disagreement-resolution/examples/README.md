# Examples / integration demos

These are **runnable demonstration scripts**, not part of an automated unit-test
suite. Each one exercises a slice of the framework end-to-end (data loading, an FL
round, a checkpoint round-trip, an unlearning strategy) and prints what it does so
you can follow along. They are kept here for reference and manual reproduction.

| Script | Demonstrates |
|--------|--------------|
| `demo_custom_dataset.py` | Registering a custom `DatasetAdapter` and loading/partitioning its data |
| `demo_custom_fl_round.py` | A full FL round (server + clients + evaluation) on a custom dataset |
| `demo_federated_exact_retraining.py` | `federated_exact_retraining` vs centralized `exact_retraining` |
| `demo_sisa_checkpoint.py` | SISA metadata-based checkpoint save/load (incl. config-mismatch handling) |

## Running

Run them from the framework directory:

```bash
cd fl-disagreement-resolution
python examples/demo_custom_dataset.py
```
