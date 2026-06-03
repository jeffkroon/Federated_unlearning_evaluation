# Resolution Strategies for Client-Level Disagreement Scenarios in Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

> [!Note]
> The entire system, its motivation, and experimental results are described in detail in [the accompanying master's thesis](docs/FL_Disagreement_Resolution_Thesis_DaanRosendal.pdf).


## Table of Contents

- [Citation](#citation)
- [Description](#description)
  - [Machine Unlearning Integration](#machine-unlearning-integration)
  - [Multi-track resolution in action](#multi-track-resolution-in-action)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Usage](#usage)
  - [Running basic experiments](#running-basic-experiments)
  - [Grid Experiments](#grid-experiments-systematic-evaluation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Documentation](#documentation)
- [Dependencies](#dependencies--technologies-used)
- [License](#license)
- [Academic context](#academic-context)
- [Project structure](#project-structure)

## Citation

If you use this work in your research, please cite it as:

```bibtex
@mastersthesis{rosendal2025fldisagreementresolution,
  author  = {Daan E. Rosendal},
  title   = {Resolution Strategies for Client-Level Disagreement Scenarios in Federated Learning},
  school  = {University of Amsterdam},
  year    = {2025},
  type    = {Master's thesis},
  url     = {https://github.com/DaanRosendal/fl-disagreement-resolution}
}
```

## Description

This project addresses a critical limitation in standard Federated Learning (FL): the assumption of unconditional collaboration amongst all clients. In real-world scenarios (e.g., competing companies, regulatory constraints), clients may need to exclude each other's data or model updates due to client-level disagreements.

Our solution introduces a robust multi-track resolution approach that creates and manages multiple, isolated model update paths called "tracks". Each track corresponds to a unique set of client exclusion preferences, guaranteeing strict client exclusion and preventing cross-contamination and unfairness issues.

### Understanding Model Concepts

The framework uses three distinct model concepts that serve different purposes:

1. **Baseline Global Model** (`baseline_global`):
   - **Purpose**: Reference/baseline model for comparison
   - **Aggregation**: Standard FedAvg from ALL clients (regardless of disagreements)
   - **Unlearning**: Never modified by unlearning (always represents full collaboration)
   - **Storage**: Saved to `round_X/global_model_aggregated/model.pt`
   - **Use case**: Baseline comparison to measure the impact of disagreements and unlearning

2. **Global Track** (`tracks/global/`):
   - **Purpose**: A track for clients without specific disagreements
   - **Clients**: All clients except fully excluded ones
   - **Unlearning**: May have unlearning applied if clients are excluded
   - **Storage**: Saved to `round_X/tracks/global/model.pt`
   - **Use case**: Represents the model used by clients without disagreements (may differ from baseline if unlearning was applied)

3. **Server Global Model Object** (`server.global_model`):
   - **Purpose**: Model object used for evaluation (a "working copy")
   - **Function**: Can be loaded with different models (baseline_global, global track, or specific tracks)
   - **Use case**: Used by evaluation functions to test different models without modifying the original

**Key Distinction**: The `baseline_global` model is a **reference** that always aggregates all clients without unlearning, while the `global` track is an actual **track** that may have unlearning applied and represents the model used by clients without disagreements.

### Machine Unlearning Integration

This repository has been extended with **policy-aware machine unlearning** capabilities. When clients are excluded from tracks (due to disagreements) or fully excluded from the federation, their historical data contributions are removed from the model using machine unlearning strategies:

- **Exact Retraining**: Retrain model from scratch on retained data (ground truth, golden standard)
- **SISA**: Sharding, Isolation, Slicing, Aggregation for efficient unlearning
- **Knowledge Distillation**: Teacher-student distillation for unlearning

**Unlearning is automatically applied:**
- **For tracks**: When a track excludes clients (e.g., Client 0 excludes Client 1 -> `track_0_no1` gets unlearning to remove Client 1's influence)
- **For full exclusions**: When clients are fully excluded from the federation (`type: "full"` disagreement)

The system supports both **PyTorch models** (LSTM, MLP, CNN) and **sklearn models** (RandomForest, XGBoost) for comprehensive evaluation across different model architectures.

### Multi-track resolution in action

![Track Contributions Visualisation](results/collected_outputs/s4_mnist_track_contributions.png)

*This visualisation demonstrates temporal disagreement resolution ([Scenario 4](mock_etcd/scenarios_5clients/scenario4.json)) where Client 0 "inbound" excludes Client 1 from rounds 1-3, creating a separate track that automatically becomes inactive once the disagreement period expires.*

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) - Python package and project manager

### Setup instructions

1. **Install uv** (if not already installed):

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/DaanRosendal/fl-disagreement-resolution.git
   cd fl-disagreement-resolution
   ```

3. **Create and activate virtual environment, then install dependencies**:

   ```bash
   uv venv

   # On Unix/macOS
   source .venv/bin/activate

   # On Windows
   .venv\Scripts\activate

   uv sync
   ```

## Data preparation

**MNIST**: Handled automatically by using the `-s` flag on first run.

**N-CMAPSS**: Requires manual preparation using the [N-CMAPSS Data Preparation repository](https://github.com/DaanRosendal/N-CMAPSS_DL). After preparation, organise your `data/n-cmapss/` folder as shown in the [project structure](#-project-structure) below (≤6 clients only).

**Tabular**: Synthetic tabular data is automatically generated on first run. The system uses `sklearn.datasets.make_classification` or `make_regression` to generate synthetic data with configurable features, classes, and task type (classification/regression).

## Usage

**Documentation:**
- [Running Experiments Guide](RUNNING_EXPERIMENTS.md) - Complete guide on how to run experiments, parameter explanations, and troubleshooting
- [Basic Experiments](#running-basic-experiments) - Single scenario runs
- [Grid Experiments](#grid-experiments-systematic-evaluation) - Full experimental matrix

### Running basic experiments

Run a simple federated learning experiment with disagreement resolution:

```bash
# First run with MNIST - automatically sets up data
uv run scripts/run_fl.py -S 1 -e mnist -r 5 -l 1 -s

# Subsequent MNIST runs (data already prepared)
uv run scripts/run_fl.py -S 1 -e mnist -r 5 -l 1

# Run scenario 3 with N-CMAPSS dataset (requires manual data preparation first)
uv run scripts/run_fl.py -S 3 -e n_cmapss -r 5 -l 1

# Run with tabular data (synthetic data, automatically generated)
uv run scripts/run_fl.py -S 1 -e tabular -r 5 -l 1

# Run all scenarios with MNIST dataset
uv run scripts/run_fl.py -S all -e mnist -r 5 -l 1

# Run with custom client configuration
uv run scripts/run_fl.py -S 1 -e mnist -c 4 -r 10 -l 2

# Run with IID data distribution
uv run scripts/run_fl.py -S 1 -e mnist -c 6 -s -i
```

### Command line options

- `-S, --scenario <num>`: Scenario number (0-34) or 'all'
- `-e, --experiment <type>`: Dataset type ('mnist', 'n_cmapss', or 'tabular')
- `-c, --clients <ids>`: Number of clients or specific client IDs
- `-r, --rounds <num>`: Number of FL rounds (default: 3)
- `-l, --local-epochs <num>`: Local training epochs (default: 5)
- `-s, --setup-data`: Set up MNIST or tabular data (first run only, not needed for N-CMAPSS)
- `-i, --iid`: Use IID data distribution (MNIST and tabular)
- `--verbose-plots`: Generate comprehensive visualisations

## Configuration

The system uses a [DYNAMOS](https://github.com/Jorrit05/DYNAMOS)-inspired configuration approach with JSON files in the `mock_etcd/` directory:

### Main configuration (`mock_etcd/configuration.json`)

```json
{
  "experiment": {
    "type": "mnist",
    "fl_rounds": 5,
    "client_ids": [0, 1, 2, 3, 4, 5]
  },
  "disagreement": {
    "initiation_mechanism": "shallow",
    "lifting_mechanism": "shallow",
    "deep_lifting_finetune_rounds": 3
  },
  "training": {
    "batch_size": 64,
    "local_epochs": 10,
    "learning_rate": 0.001
  },
  "unlearning": {
    "enabled": false,
    "model_type": "lstm",
    "strategies": ["exact_retraining", "sisa", "distillation"],
    "use_strategy": "exact_retraining",
    "reuse_existing_branches": true,
    "model_params": {},
    "train_params": {
      "epochs": 10,
      "batch_size": 64,
      "lr": 0.001
    }
  }
}
```

### Scenario definitions (`mock_etcd/scenarios_5clients/`)

Scenarios define specific disagreement patterns:

```json
{
  "name": "Simple Inbound Exclusion",
  "description": "Client 0 excludes client 1 from round 1 onwards",
  "num_clients": 6,
  "disagreements": {
    "client_0": [{
      "type": "inbound",
      "target": "client_1",
      "active_rounds": {"start": 1, "end": null}
    }]
  }
}
```

Available disagreement types:

- **inbound**: Exclude another client's updates from your model
- **outbound**: Prevent your updates from reaching another client
- **bidirectional**: Mutual exclusion between two clients
- **full**: Complete isolation from all other clients

## Testing

### Validation suite

Run the comprehensive test suite to validate disagreement resolution across all scenarios:

```bash
# Test all scenarios with MNIST
uv run scripts/test_disagreement_scenarios.py all -e mnist -r 5 -l 1

# Test specific scenario
uv run scripts/test_disagreement_scenarios.py 1 -e mnist -r 10 -l 2

# Test with N-CMAPSS dataset (limited to ≤6 clients)
uv run scripts/test_disagreement_scenarios.py all -e n_cmapss -r 5 -l 1

# Test with tabular data
uv run scripts/test_disagreement_scenarios.py all -e tabular -r 5 -l 1

# Test with verbose output and comprehensive plots
uv run scripts/test_disagreement_scenarios.py 5 -v --verbose-plots
```

The test suite automatically:

- Validates track creation matches expected patterns
- Verifies client isolation is properly enforced
- Checks temporal disagreement handling

### Scalability testing

Evaluate system performance across multiple scenarios:

```bash
#!/bin/bash

# Run the first set of scalability scenarios (S7-S12) with MNIST dataset
for run in {1..5}; do
  for S in {7..12}; do # or, e.g., "for S in 25 26 29 30 31; do"
    uv run scripts/run_fl.py -S "$S" -e mnist -r 5 -l 1 # or "-e n_cmapss" for S13-S19, or "-e tabular"
  done
done

output_dirs=($(find results -maxdepth 1 -type d ! -name . ! -name results ! -name comparisons ! -name collected_outputs -printf "results/%f\n" | sort))

# assuming the results directory contains only relevant results, i.e., was empty before executing the run_fl.py command
uv run scripts/compare_fl_runs.py "${output_dirs[@]}"
```

### Visualisation generation

Generate comprehensive analysis plots:

```bash
# Create track contribution visualisations (runs automatically at the end of each run)
uv run scripts/visualize_track_contributions.py results/fl_simulation_*

# Gather and compare outputs across scenarios (mostly useful for scalability testing)
uv run scripts/gather_simulation_outputs.py
```

### Grid Experiments: Systematic Evaluation

For comprehensive evaluation across all scenarios, models, and unlearning strategies, use the **grid experiment system**:

```bash
# Quick test: 1 scenario, 1 model, 1 strategy
uv run scripts/grid_run.py -e mnist -s 1 -m lstm -u exact_retraining -o results/test

# All scenarios with one model and strategy (35 runs)
uv run scripts/grid_run.py -e mnist -m lstm -u exact_retraining -o results/grid_lstm

# Full grid: all scenarios × all models × all strategies (525 combinations)
uv run scripts/grid_run.py -e mnist -o results/grid_mnist_full

# Run grid experiments with tabular data
uv run scripts/grid_run.py -e tabular -s 1 -m mlp -u exact_retraining -o results/test_tabular

# Aggregate and compare results
uv run scripts/aggregate_grid_results.py -d results/grid/mnist
```

**Grid Experiment Features:**
- Systematic evaluation of 35 scenarios × 5 models × 3 unlearning strategies
- Automatic result collection and comparison
- Best combination identification per scenario
- Performance metrics (accuracy, RMSE, unlearning time)


### Machine Unlearning Configuration

Enable unlearning in `mock_etcd/configuration.json`:

```json
{
  "unlearning": {
    "enabled": true,
    "model_type": "mlp",
    "strategies": ["exact_retraining", "sisa", "distillation"],
    "use_strategy": "exact_retraining",
    "reuse_existing_branches": true,
    "model_params": {},
    "train_params": {
      "epochs": 5,
      "batch_size": 64,
      "lr": 0.001
    }
  }
}
```

**Note**: `model_type` is auto-detected based on `experiment_type` if not specified:
- `n_cmapss` -> `lstm` (or `mlp`)
- `mnist` -> `mlp` (CNN not supported in unlearning framework)
- `tabular` -> `mlp` (or `random_forest`/`xgboost`)

**Unlearning is automatically triggered in two scenarios:**

1. **Track-based exclusions**: When a track excludes clients (e.g., `track_0_no1` excludes Client 1):
   - Creates a pre-unlearning checkpoint for the track
   - Applies all configured unlearning strategies (exact_retraining, sisa, distillation)
   - Selects the best strategy (default: exact_retraining)
   - Updates the track model with the unlearned version
   - Track continues training without the excluded clients' influence

2. **Full exclusions**: When a client is "fully excluded" (disagreement type: "full"):
   - Creates a pre-unlearning checkpoint for the global model
   - Applies the selected unlearning strategy
   - Creates branch checkpoints for each strategy
   - Continues FL with the unlearned model

**Unlearning strategies are evaluated and compared**, with metrics (RMSE, MAE, R2, unlearning time) saved for analysis. The system automatically selects the best strategy based on the configuration.

## Documentation

### Guides and Tutorials

  - Step-by-step instructions for running grid experiments
  - All combinations: 35 scenarios × 5 models × 3 unlearning strategies
  - Result aggregation and comparison
  - Best practices and troubleshooting

### API Documentation

- **Federated Learning Framework**: See inline documentation in `fl_server/`, `fl_client/`, and `fl_module/`
- **Machine Unlearning Tool**: See [machine_unlearning_tool/README.md](../machine_unlearning_tool/README.md)
- **Unlearning Strategies**: See `fl_server/unlearning_strategies.py` for strategy implementations
- **Branching System**: See `fl_server/branching.py` for checkpoint and version control

### Configuration Files

- **Main Configuration**: `mock_etcd/configuration.json` - System-wide settings
- **Scenario Definitions**: `mock_etcd/scenarios_5clients/` - 35 disagreement scenarios (5 clients)
- **Disagreements**: `mock_etcd/disagreements.json` - Active disagreement rules

### Example Scripts

- `scripts/run_fl.py` - Single experiment runner
- `scripts/grid_run.py` - Grid experiment runner
- `scripts/aggregate_grid_results.py` - Result aggregation
- `scripts/test_disagreement_scenarios.py` - Validation suite

## Dependencies / Technologies Used

**Core Framework:**

- **Python 3.12+**: Main programming language
- **PyTorch 2.7+**: Deep learning framework for model training
- **NumPy 2.2+**: Numerical computing for data handling

**Machine Learning:**

- **scikit-learn 1.6+**: ML utilities and metrics
- **torchvision 0.22+**: Computer vision datasets and transforms

**Visualisation & analysis:**

- **matplotlib 3.10+**: Plotting and visualisation
- **seaborn 0.13+**: Statistical data visualisation
- **brokenaxes 0.6+**: Advanced plot formatting

**Datasets:**

- **MNIST**: Classic handwritten digit recognition (image classification)
- **N-CMAPSS**: NASA Commercial Modular Aero-Propulsion System Simulation for predictive maintenance of aircraft engines (time-series regression)
- **Tabular**: Synthetic tabular data for general classification/regression tasks (generated with sklearn)

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Academic context

This repository contains the complete implementation for the Master's thesis:

**"[Resolution Strategies for Client-Level Disagreement Scenarios in Federated Learning](docs/FL_Disagreement_Resolution_Thesis_DaanRosendal.pdf)"**
*By Daan Eduard Rosendal*
*University of Amsterdam, 2025*

The work serves as a proof-of-concept for handling realistic federated learning scenarios where unconditional client collaboration cannot be assumed.

### Extended with Machine Unlearning

This repository has been extended with **policy-aware machine unlearning** mechanisms for the thesis:

**"Designing Policy-Aware Machine Unlearning Mechanisms for Federated Learning with Adaptive Branching and Version Control"**
*By Jeff Kroon*
*University of Amsterdam, 2025*

The extension integrates machine unlearning strategies into the disagreement-aware federated learning framework:
- **Automatic unlearning for tracks**: When tracks exclude clients, unlearning is automatically applied to remove excluded clients' influence
- **Automatic unlearning for full exclusions**: When clients are fully excluded, unlearning removes their historical contributions
- **Multi-strategy evaluation**: All strategies (exact_retraining, SISA, distillation) are evaluated and compared
- **Systematic evaluation**: Enables comprehensive evaluation of unlearning methods across different models, scenarios, and datasets

## Related projects

- **[DYNAMOS](https://github.com/Jorrit05/DYNAMOS)**: Microservice orchestration middleware that inspired our configuration architecture
- **[N-CMAPSS Data Preparation](https://github.com/DaanRosendal/N-CMAPSS_DL)**: Toolkit for preparing the NASA turbofan engine dataset

## Project structure

```text
fl-disagreement-resolution/
├── data/                            # Dataset storage
│   ├── MNIST/                       # MNIST dataset (auto-downloaded)
│   │   └── raw/                     # Raw MNIST files
│   └── n-cmapss/                    # N-CMAPSS dataset (manual preparation required)
│       ├── test/                    # Test data (.npz files)
│       │   ├── Unit11_win50_str1_smp10.npz
│       │   ├── Unit14_win50_str1_smp10.npz
│       │   └── Unit15_win50_str1_smp10.npz
│       └── train/                   # Training data organised by client
│           ├── client_0/            # Client 0 training data
│           │   └── Unit2_win50_str1_smp10.npz
│           ├── client_1/            # Client 1 training data
│           │   └── Unit5_win50_str1_smp10.npz
│           ├── client_2/            # Client 2 training data
│           │   └── Unit10_win50_str1_smp10.npz
│           ├── client_3/            # Client 3 training data
│           │   └── Unit16_win50_str1_smp10.npz
│           ├── client_4/            # Client 4 training data
│           │   └── Unit18_win50_str1_smp10.npz
│           └── client_5/            # Client 5 training data
│               └── Unit20_win50_str1_smp10.npz
│
├── docs/                            # Documentation and technical diagrams
│   ├── FL_Disagreement_Resolution_Thesis_DaanRosendal.pdf # Master's thesis
│   └── drawio/                      # Technical architecture diagrams
│       ├── data_exchange_archetypes.drawio
│       ├── dynamos-design.drawio
│       ├── fl-disagreement-resolution-design.drawio
│       ├── fl-disagreements-graph.drawio
│       ├── resolution-strategies.drawio
│       ├── system_flow.drawio       # Overall system architecture
│       └── disagreement-scenarios-visualisations/
│           ├── bidirectional-*.drawio # Various bidirectional patterns
│           ├── combination*.drawio  # Combination scenarios
│           ├── full-exclusion.drawio
│           ├── inbound-*.drawio     # Inbound exclusion patterns
│           ├── legend.drawio        # Visualisation legend
│           ├── outbound-*.drawio    # Outbound exclusion patterns
│           ├── partial-data-exclusion.drawio
│           └── template-scenario-*.drawio
│
├── fl_client/                       # Client-side federated learning implementation
│   ├── __init__.py
│   ├── client.py                    # Core FL client logic and communication
│   ├── main.py                      # Client application entry point
│   ├── training.py                  # Local model training procedures
│   └── utils.py                     # Client utility functions
│
├── fl_module/                       # Dataset handlers and ML models
│   ├── __init__.py
│   ├── base.py                      # Base classes for datasets and models
│   ├── models.py                    # Neural network architectures (CNN, MLP, LSTM, TabularClassifier)
│   ├── mnist/                       # MNIST dataset implementation
│   │   ├── __init__.py
│   │   ├── dataset.py               # MNIST data loading and preprocessing
│   │   └── utils.py                 # MNIST-specific utilities
│   ├── n_cmapss/                    # N-CMAPSS dataset implementation
│   │   ├── __init__.py
│   │   ├── dataset.py               # N-CMAPSS data loading and preprocessing
│   │   └── utils.py                 # N-CMAPSS-specific utilities
│   └── tabular/                     # Tabular dataset implementation
│       ├── __init__.py
│       ├── adapter.py               # Tabular data adapter for unlearning
│       ├── dataset.py               # Tabular data loading and preprocessing
│       └── utils.py                 # Tabular data generation and utilities
│
├── fl_server/                       # Server-side coordination and aggregation
│   ├── __init__.py
│   ├── aggregation.py               # Multi-track model aggregation strategies
│   ├── branching.py                 # Branch registry for unlearning checkpoints
│   ├── disagreement.py              # Disagreement detection and resolution logic
│   ├── evaluation.py                # Model evaluation and metrics collection
│   ├── main.py                      # Server application entry point
│   ├── server.py                    # Core FL server orchestration (includes track unlearning)
│   ├── unlearning_strategies.py    # Machine unlearning strategy implementations
│   └── utils.py                     # Server utility functions
│
├── fl_server_sklearn/               # Sklearn-based server implementation
│   ├── __init__.py
│   ├── aggregation.py               # Sklearn model aggregation
│   ├── disagreement.py              # Disagreement resolution for sklearn
│   ├── evaluation.py                # Sklearn model evaluation
│   └── server.py                    # Sklearn FL server orchestration
│
├── fl_client_sklearn/               # Sklearn-based client implementation
│   ├── __init__.py
│   ├── client.py                    # Sklearn FL client
│   └── training.py                  # Sklearn model training
│
├── logs/                            # Runtime logs and debugging output
├── mock_etcd/                       # Configuration and scenario management
│   ├── configuration.json           # Main system configuration
│   ├── disagreements.json           # Disagreement definitions and rules
│   ├── etcd_loader.py              # Configuration loading utilities
│   └── scenarios_5clients/          # Disagreement scenario definitions (35 scenarios, 5 clients)
│       ├── scenario0.json           # Baseline: no disagreements
│       ├── scenario1.json           # Simple inbound exclusion
│       ├── scenario4.json           # Temporal disagreement (featured example)
│       ├── ...                      # Scenarios 2-34 covering various patterns
│       └── archive/                 # Legacy scenario definitions
│
├── results/                         # Experimental outputs and analysis
│   └── collected_outputs/           # Aggregated visualisation outputs
│       ├── s1_mnist_track_contributions.png
│       ├── s4_mnist_track_contributions.png
│       ├── s*_scalability_comparison.png
│       └── ...                      # Generated plots for all scenarios
│
├── scripts/                         # CLI tools and automation scripts
│   ├── aggregate_grid_results.py   # Aggregate and compare grid experiment results
│   ├── compare_fl_runs.py           # Compare results across multiple FL runs
│   ├── gather_simulation_outputs.py # Collect and organize experimental outputs
│   ├── grid_run.py                  # Grid experiment runner (all combinations)
│   ├── run_fl.py                    # Main experiment runner (scenarios 0-34)
│   ├── test_disagreement_scenarios.py # Validation suite for disagreement resolution
│   └── visualize_track_contributions.py # Generate track contribution plots
│
├── fl_orchestrator.py               # High-level orchestration coordinator (PyTorch)
├── fl_orchestrator_sklearn.py      # Sklearn-based orchestration coordinator
├── LICENSE                          # MIT license
├── pyproject.toml                   # Python project configuration and dependencies
├── README.md                        # This comprehensive documentation
└── uv.lock                          # Dependency lock file
```

---

**Ready to explore federated learning with realistic client disagreements? Start with scenario 1:**

```bash
# First run - sets up MNIST data automatically with IID data distribution
uv run scripts/run_fl.py -S 1 -e mnist -r 5 -l 1 -s -i
```
