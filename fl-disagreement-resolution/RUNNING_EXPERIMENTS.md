# Running Experiments: Complete Guide

This guide explains how to run federated learning experiments with machine unlearning and what each parameter does.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Command Structure](#basic-command-structure)
- [Parameters Explained](#parameters-explained)
- [What Happens During a Run](#what-happens-during-a-run)
- [Understanding Results](#understanding-results)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)

## Quick Start

The simplest way to run an experiment:

```bash
cd fl-disagreement-resolution
uv run scripts/run_fl.py -e mnist -S 1 -c 0 -c 1 -c 2 -r 2 -l 3
```

This runs:
- **Dataset**: MNIST (image classification)
- **Scenario**: 1 (simple disagreement pattern)
- **Clients**: 0, 1, 2 (three clients)
- **Rounds**: 2 (two FL training rounds)
- **Local epochs**: 3 (each client trains for 3 epochs per round)

## Basic Command Structure

```bash
uv run scripts/run_fl.py [OPTIONS]
```

### Required Parameters

At minimum, you need to specify:
- **Dataset type** (`-e`): Which dataset to use
- **Scenario** (`-S`): Which disagreement pattern to simulate

### Optional Parameters

All other parameters have defaults, but you'll often want to customize:
- **Clients** (`-c`): Which clients participate
- **Rounds** (`-r`): How many FL rounds to run
- **Local epochs** (`-l`): Training intensity per round
- **Data setup** (`-s`): First-time data preparation

## Parameters Explained

### `-e, --experiment <type>`: Dataset Type

**What it does**: Selects which dataset to use for the experiment.

**Options**:
- `mnist`: Handwritten digit classification (28×28 grayscale images, 10 classes)
- `n_cmapss`: Time-series regression for aircraft engine health prediction
- `tabular`: Synthetic tabular data (automatically generated)

**Why it matters**: Different datasets require different model architectures and preprocessing:
- **MNIST**: Uses CNN/MLP models, requires image preprocessing
- **N-CMAPSS**: Uses LSTM models, requires time-series sequences
- **Tabular**: Uses MLP or tree models, requires feature extraction

**Example**:
```bash
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3
uv run scripts/run_fl.py -e tabular -S 1 -r 2 -l 3
```

**Note**: N-CMAPSS requires manual data preparation (see [Data Preparation](../README.md#-data-preparation)).

---

### `-S, --scenario <num>`: Scenario Number

**What it does**: Selects which disagreement pattern to simulate.

**Options**:
- `0`: No disagreements (baseline - all clients collaborate fully)
- `1-34`: Various disagreement patterns (see `mock_etcd/scenarios_5clients/`)
- `all`: Run all 35 scenarios sequentially

**Why it matters**: Scenarios define:
- Which clients exclude each other
- When disagreements start/end (temporal disagreements)
- What type of exclusion (inbound, outbound, bidirectional, full)

**Common scenarios**:
- **Scenario 0**: Baseline (no disagreements) - useful for comparison
- **Scenario 1**: Simple inbound exclusion (Client 0 excludes Client 1)
- **Scenario 2**: Mixed disagreements (multiple clients excluding each other)
- **Scenario 5**: Temporal disagreements (disagreements change per round)

**Example**:
```bash
# Baseline (no disagreements)
uv run scripts/run_fl.py -e mnist -S 0 -r 2 -l 3

# Simple disagreement
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3

# all scenarios (slow)
uv run scripts/run_fl.py -e mnist -S all -r 2 -l 3
```

**Note**: You can inspect scenario definitions in `mock_etcd/scenarios_5clients/scenario<N>.json`.

---

### `-c, --clients <ids>`: Client Selection

**What it does**: Specifies which clients participate in the federation.

**Options**:
- Single number: `-c 3` -> Creates clients 0, 1, 2 (3 clients total)
- Multiple IDs: `-c 0 -c 1 -c 2` -> Explicitly selects clients 0, 1, 2
- Omitted: Uses `num_clients` from the scenario file

**Why it matters**:
- **Fewer clients**: Faster runs, less data diversity
- **More clients**: More realistic, but slower
- **Specific IDs**: Useful for testing specific client combinations

**Example**:
```bash
# 3 clients (0, 1, 2)
uv run scripts/run_fl.py -e mnist -S 1 -c 3 -r 2 -l 3

# Explicit client selection
uv run scripts/run_fl.py -e mnist -S 1 -c 0 -c 1 -c 2 -r 2 -l 3

# Use scenario's default (if scenario defines num_clients)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3
```

**Note**: N-CMAPSS is limited to ≤6 clients due to data availability.

---

### `-r, --rounds <num>`: Number of FL Rounds

**What it does**: Sets how many federated learning rounds to execute.

**Default**: 3 rounds

**Why it matters**:
- **Fewer rounds (1-3)**: Quick tests, but models may not converge
- **More rounds (5-10)**: Better model quality, but slower
- **Many rounds (10+)**: Full convergence, but very slow

**Trade-off**: more rounds improve accuracy up to convergence, at higher training cost.

**Example**:
```bash
# Quick test (2 rounds)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3

# Standard run (5 rounds)
uv run scripts/run_fl.py -e mnist -S 1 -r 5 -l 3

# Full training (10 rounds)
uv run scripts/run_fl.py -e mnist -S 1 -r 10 -l 3
```

**Note**: Each round involves:
1. Client local training
2. Model aggregation
3. Disagreement resolution (if applicable)
4. Unlearning (if enabled and disagreements exist)

---

### `-l, --local-epochs <num>`: Local Training Epochs

**What it does**: Sets how many epochs each client trains locally before sending updates.

**Default**: 5 epochs

**Why it matters**:
- **Fewer epochs (1-3)**: Faster, but less local learning
- **More epochs (5-10)**: Better local optimization, but slower
- **Many epochs (10+)**: Overfitting risk, very slow

**Trade-off**: More epochs = better local models, but slower per round.

**Example**:
```bash
# Quick training (1 epoch per round)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 1

# Standard training (5 epochs per round)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 5

# Intensive training (10 epochs per round)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 10
```

**Note**: training time scales with the number of rounds, local epochs, clients, and per-client data size.

---

### `-b, --batch-size <num>`: Batch Size

**What it does**: Sets the batch size for local client training.

**Default**: 64

**Why it matters**:
- **Smaller batches (16-32)**: More gradient updates, slower but potentially better
- **Larger batches (64-128)**: Faster, but may reduce model quality
- **Very large batches (256+)**: Memory issues, poor convergence

**Example**:
```bash
# Small batches (more updates)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -b 32

# Default batches
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -b 64

# Large batches (faster)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -b 128
```

---

### `-s, --setup-data`: Data Setup Flag

**What it does**: Triggers automatic data preparation (download/generation).

**When to use**:
- **First run with MNIST**: Always use `-s` to download data
- **First run with tabular**: Always use `-s` to generate synthetic data
- **Subsequent runs**: Not needed (data already exists)

**Why it matters**: Without `-s`, the run will fail if data doesn't exist.

**Example**:
```bash
# First run - setup data
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -s

# Subsequent runs - skip setup
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3
```

**Note**: N-CMAPSS requires manual data preparation (see [Data Preparation](../README.md#-data-preparation)).

---

### `-f, --force-setup`: Force Data Setup

**What it does**: Forces data setup even if data already exists (overwrites existing data).

**When to use**: When you want to regenerate/download fresh data.

**Example**:
```bash
# Force fresh data download
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -s -f
```

---

### `-i, --iid`: IID Data Distribution

**What it does**: Uses Independent and Identically Distributed (IID) data distribution.

**Default**: Non-IID (realistic but harder)

**Why it matters**:
- **IID**: Each client gets a random sample of all classes (easier, faster convergence)
- **Non-IID**: Each client gets specific classes (realistic, harder convergence)

**Example**:
```bash
# IID distribution (easier)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -i

# Non-IID distribution (realistic, default)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3
```

**Note**: Only applies to MNIST and tabular datasets.

---

### `-d, --results-dir <dir>`: Custom Results Directory

**What it does**: Sets a custom directory name for results (instead of auto-generated timestamp).

**Default**: `results/fl_simulation_YYYYMMDD_HHMMSS_<dataset>_s<scenario>/`

**Why it matters**: Useful for organizing experiments or comparing specific runs.

**Example**:
```bash
# Custom results directory
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -d results/my_experiment
```

---

### `-C, --config <file>`: Custom Configuration File

**What it does**: Uses a custom configuration file instead of `mock_etcd/configuration.json`.

**When to use**: When you have dataset-specific configurations (e.g., `configuration_adult_full.json`).

**Example**:
```bash
# Use custom config
uv run scripts/run_fl.py -e adult -S 1 -r 2 -l 3 -C mock_etcd/configuration_adult_full.json
```

---

### `--no-viz`: Skip Visualization

**What it does**: Skips automatic plot generation (faster runs).

**When to use**: When you only care about metrics, not visualizations.

**Example**:
```bash
# Skip plots (faster)
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 --no-viz
```

---

### `--verbose-plots`: Comprehensive Visualizations

**What it does**: Generates all plots for all rounds (not just the last round).

**Default**: Only generates plots for the last round (faster)

**When to use**: When you want detailed visualizations for analysis.

**Example**:
```bash
# Generate all plots
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 --verbose-plots
```

---

## What Happens During a Run

A run has two phases: a baseline FL run, then unlearning applied to that baseline.

### Phase 1: Baseline FL Run (Without Unlearning)

1. **Initialization**:
   - Loads configuration from `mock_etcd/configuration.json`
   - Loads scenario from `mock_etcd/scenarios_5clients/scenario<N>.json`
   - Initializes server and clients
   - Prepares data (if `-s` flag used)

2. **Federated Learning Rounds**:
   For each round (1 to `-r`):
   - **Client Training**: Each client trains locally for `-l` epochs
   - **Model Aggregation**: Server aggregates client updates (FedAvg)
   - **Disagreement Resolution**: Creates tracks for excluded clients
   - **Checkpointing**: Saves models at each stage

3. **Result Storage**:
   - All checkpoints saved to `baseline/model_storage/round_<N>/`
   - Global model: `global_model_aggregated/model.pt`
   - Track models: `tracks/<track_name>/model.pt`

**Why this phase exists**: Ensures all unlearning strategies are evaluated on the **exact same** FL training trajectory (fair comparison).

### Phase 2: Unlearning Strategy Application

For each strategy (`exact_retraining`, `sisa`, `distillation`):

1. **Copy Baseline**:
   - Creates `strategy_<name>/` directory
   - Copies all baseline checkpoints

2. **Apply Unlearning**:
   For each round and track that had exclusions:
   - Loads pre-unlearning checkpoint
   - Identifies excluded clients' data
   - Applies unlearning strategy
   - Saves unlearned model to branch

3. **Evaluation**:
   - Evaluates unlearned model on:
     - **Forget set**: Data from excluded clients (should be forgotten)
     - **Retain set**: Data from remaining clients (should be preserved)
     - **Test set**: General performance
   - Calculates metrics (accuracy, RMSE, MIA, behavioral distance, efficiency)

4. **Result Storage**:
   - Metrics saved to `strategy_<name>/model_storage/round_<N>/tracks/<track>/unlearning/branches/<strategy>/metrics.json`

### Phase 3: Strategy Comparison

1. **Aggregation**:
   - Collects metrics from all strategies
   - Compares performance across strategies

2. **Comparison File**:
   - Saves `strategy_comparison.json` with:
     - Per-strategy metrics
     - Delta from exact retraining (golden standard)
     - Best strategy identification

3. **Visualization**:
   - Generates comparison plots (unless `--no-viz`)
   - Saves to `output/plots/`

### Total Execution Time

Approximate time breakdown:
- **Baseline FL**: ~70% of total time (full training)
- **Unlearning**: ~25% of total time (3 strategies × tracks)
- **Comparison**: ~5% of total time (aggregation + plots)

**Example**: A 5-round experiment with 3 clients might take:
- Baseline: ~5 minutes
- Unlearning: ~2 minutes
- Comparison: ~30 seconds
- **Total**: ~7.5 minutes

---

## Understanding Results

### Results Directory Structure

```
results/fl_simulation_YYYYMMDD_HHMMSS_mnist_s1/
├── baseline/                              # Baseline FL run (no unlearning)
│   ├── model_storage/
│   │   ├── round_1/
│   │   │   ├── global_model_aggregated/   # Global model (all clients)
│   │   │   │   └── model.pt
│   │   │   └── tracks/                    # Per-track models
│   │   │       ├── global/                # Track for clients without disagreements
│   │   │       │   └── model.pt
│   │   │       └── track_0_no1/           # Track where Client 0 excludes Client 1
│   │   │           └── model.pt
│   │   └── round_2/
│   │       └── ...
│   └── output/
│       └── plots/                         # Baseline visualizations
│
├── strategy_exact_retraining/             # Exact retraining strategy
│   ├── model_storage/
│   │   └── round_X/tracks/<track>/unlearning/
│   │       └── branches/
│   │           └── exact_retraining/
│   │               ├── model.pt            # Unlearned model
│   │               └── metrics.json        # ← METRICS HERE
│   └── output/
│
├── strategy_sisa/                         # SISA strategy
│   └── model_storage/.../branches/sisa/
│       └── metrics.json                    # ← METRICS HERE
│
├── strategy_distillation/                 # Distillation strategy
│   └── model_storage/.../branches/distillation/
│       └── metrics.json                    # ← METRICS HERE
│
└── strategy_comparison.json                # ← COMPARISON OF ALL STRATEGIES
```

### Key Metrics Files

#### 1. Per-Strategy Metrics (`metrics.json`)

Location: `strategy_<name>/model_storage/round_<N>/tracks/<track>/unlearning/branches/<strategy>/metrics.json`

**Contents**:
```json
{
  "efficiency_metrics": {
    "avg_unlearning_time_s": 2.5,
    "total_unlearning_time_s": 2.5,
    "retrain_fraction": 0.8,
    "total_storage_mb": 15.2,
    "num_branches": 1
  },
  "forget_set_comparison": {
    "original_accuracy": 0.95,
    "unlearned_accuracy": 0.10,
    "unlearning_score": 0.85
  },
  "retain_set_metrics": {
    "accuracy": 0.92,
    "rmse": 0.15
  },
  "behavioral_distance": {
    "logit_mse": 0.02,
    "kl_divergence": 0.05
  },
  "mia_metrics": {
    "mia_accuracy": 0.55,
    "mia_improvement": 0.10
  }
}
```

**What to look for**:
- **`retrain_fraction`**: Should be < 1.0 (efficiency metric - lower is better)
- **`unlearning_score`**: Should be high (forget set accuracy should drop)
- **`retain_set_metrics.accuracy`**: Should be high (utility preservation)
- **`mia_improvement`**: Should be positive (privacy improvement)

#### 2. Strategy Comparison (`strategy_comparison.json`)

Location: `results/fl_simulation_*/strategy_comparison.json`

**Contents**:
```json
{
  "comparison_summary": {
    "best_strategy": "exact_retraining",
    "best_retain_accuracy": 0.92
  },
  "strategies": {
    "exact_retraining": { ... },
    "sisa": { ... },
    "distillation": { ... }
  },
  "deltas_from_exact_retraining": {
    "sisa": { "retain_accuracy_delta": -0.02 },
    "distillation": { "retain_accuracy_delta": -0.05 }
  }
}
```

**What to look for**:
- **`best_strategy`**: Which strategy performed best
- **`deltas_from_exact_retraining`**: How other strategies compare to golden standard

### Viewing Results

#### Quick Metrics Check

```bash
# View strategy comparison
cat results/fl_simulation_*/strategy_comparison.json | python3 -m json.tool

# View specific strategy metrics
cat results/fl_simulation_*/strategy_exact_retraining/model_storage/round_*/tracks/*/unlearning/branches/exact_retraining/metrics.json | python3 -m json.tool
```

#### Find Latest Results

```bash
# List all result directories (sorted by time)
ls -lt results/ | head -10

# Get latest result directory
LATEST=$(ls -t results/ | head -1)
echo "Latest: $LATEST"

# View comparison
cat "results/$LATEST/strategy_comparison.json" | python3 -m json.tool
```

---

## Common Use Cases

### Use Case 1: Quick Test Run

**Goal**: Verify the system works, don't care about accuracy.

```bash
uv run scripts/run_fl.py -e mnist -S 1 -c 3 -r 2 -l 1 --no-viz
```

**Why these parameters**:
- `-r 2`: Only 2 rounds (fast)
- `-l 1`: Only 1 epoch per round (fast)
- `-c 3`: Only 3 clients (fast)
- `--no-viz`: Skip plots (faster)

**Expected time**: ~2-3 minutes

---

### Use Case 2: Standard Experiment

**Goal**: Get meaningful results for analysis.

```bash
uv run scripts/run_fl.py -e mnist -S 1 -c 0 -c 1 -c 2 -r 5 -l 5
```

**Why these parameters**:
- `-r 5`: Enough rounds for convergence
- `-l 5`: Enough epochs for good local training
- `-c 0 -c 1 -c 2`: Standard 3-client setup

**Expected time**: ~10-15 minutes

---

### Use Case 3: Full Training Run

**Goal**: Best possible model quality.

```bash
uv run scripts/run_fl.py -e mnist -S 1 -c 0 -c 1 -c 2 -c 3 -c 4 -c 5 -r 10 -l 10
```

**Why these parameters**:
- `-r 10`: Many rounds for full convergence
- `-l 10`: Many epochs for intensive training
- `-c 0 ... -c 5`: All 6 clients (more data diversity)

**Expected time**: ~1-2 hours

---

### Use Case 4: Compare Multiple Scenarios

**Goal**: Test how different disagreement patterns affect results.

```bash
# Run scenario 0 (baseline)
uv run scripts/run_fl.py -e mnist -S 0 -r 5 -l 5

# Run scenario 1 (simple disagreement)
uv run scripts/run_fl.py -e mnist -S 1 -r 5 -l 5

# Run scenario 2 (mixed disagreements)
uv run scripts/run_fl.py -e mnist -S 2 -r 5 -l 5
```

**Then compare**:
```bash
# Compare results
uv run scripts/compare_fl_runs.py results/fl_simulation_*_s0 results/fl_simulation_*_s1 results/fl_simulation_*_s2
```

---

### Use Case 5: Test Different Datasets

**Goal**: Verify framework works with different data types.

```bash
# MNIST (images)
uv run scripts/run_fl.py -e mnist -S 1 -r 3 -l 3

# Tabular (synthetic)
uv run scripts/run_fl.py -e tabular -S 1 -r 3 -l 3

# N-CMAPSS (time-series, requires data prep)
uv run scripts/run_fl.py -e n_cmapss -S 1 -r 3 -l 3
```

---

## Troubleshooting

### Problem: "No such file or directory" errors

**Cause**: Missing data or wrong working directory.

**Solution**:
```bash
# Check you're in the right directory
cd fl-disagreement-resolution

# Check if data exists
ls data/mnist/train/  # Should show client directories

# If missing, run with -s flag
uv run scripts/run_fl.py -e mnist -S 1 -r 2 -l 3 -s
```

---

### Problem: "Module not found" errors

**Cause**: Missing dependencies.

**Solution**:
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate      # Windows
```

---

### Problem: Very long runtime

**Cause**: Too many rounds/epochs/clients.

**Solution**:
- Reduce rounds: `-r 2` instead of `-r 10`
- Reduce epochs: `-l 3` instead of `-l 10`
- Reduce clients: `-c 3` instead of `-c 6`
- Skip visualization: `--no-viz`

---

### Problem: Low accuracy in results

**Cause**: Insufficient training or data issues.

**Solution**:
- Increase rounds: `-r 10` instead of `-r 2`
- Increase epochs: `-l 10` instead of `-l 3`
- Check data quality: Inspect `data/<dataset>/train/`
- Use IID distribution: Add `-i` flag (easier convergence)

---

### Problem: Empty `forget_set_comparison` in metrics

**Cause**: No unlearning was applied (track had no excluded clients).

**Solution**: This is expected if:
- Scenario has no disagreements (scenario 0)
- Track has no excluded clients
- Check scenario file: `mock_etcd/scenarios_5clients/scenario<N>.json`

---

### Problem: "Track has no clients" warning

**Cause**: All clients in a track were excluded.

**Solution**: This is expected for certain disagreement patterns. The system will skip unlearning for empty tracks and log skip metrics.

---

### Problem: Out of memory errors

**Cause**: Too large batch size or too many clients.

**Solution**:
- Reduce batch size: `-b 32` instead of `-b 64`
- Reduce clients: `-c 3` instead of `-c 6`
- Reduce model size: Check `mock_etcd/configuration.json` -> `unlearning.model_params`

---

## Next Steps

After running experiments:

1. **Analyze Results**: Check `strategy_comparison.json` for performance metrics
2. **Compare Scenarios**: Use `scripts/compare_fl_runs.py` to compare multiple runs
3. **Generate Visualizations**: Use `scripts/visualize_track_contributions.py` for detailed plots
4. **Run Grid Experiments**: Use `scripts/grid_run.py` for systematic evaluation

For more information, see the [main README](../README.md).

