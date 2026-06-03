# Evaluating Machine Unlearning Strategies in a Policy-Aware Federated Learning Framework

Code and experiments for the BSc Artificial Intelligence thesis "Evaluating Machine
Unlearning Strategies in a Policy-Aware Federated Learning Framework: Across Datasets of
Increasing Complexity" (University of Amsterdam, 2026).

The framework turns client-level disagreements and data-sharing policy changes in
federated learning into traceable, isolated model versions, and uses that mechanism to
compare five machine unlearning strategies under identical conditions across three
datasets.

## Overview

When a client withdraws from a federated collaboration or changes its data-sharing policy,
its contribution must be removed from the global model. This framework:

- detects each policy change (an exclusion event) and writes a shared pre-unlearning
  checkpoint.
- branches every unlearning strategy from that identical checkpoint as an isolated,
  policy-tagged model version, so strategies are compared from the same starting point.
- evaluates each branch along four dimensions: forgetting quality, utility preservation,
  privacy protection, and computational cost.

The full motivation, method, and results are described in the accompanying thesis.

## Built on

The federated-learning base and the multi-track disagreement-resolution layer build on the
work of Daan Rosendal, https://github.com/DaanRosendal/fl-disagreement-resolution (MSc
thesis, University of Amsterdam, 2025): the FedAvg base, the disagreement scenarios, and
the multi-track branching originate there. The machine-unlearning strategy library
(`machine_unlearning_tool/`) provides the strategy implementations. The contribution of
this thesis is the policy-aware unlearning layer on top: the checkpoint-and-branch
mechanism, the integration of all five strategies behind one interface, and the controlled
cross-dataset evaluation.

## Unlearning strategies

1. Centralized exact retraining (reference): retrain from scratch on the pooled retain set.
2. Federated exact retraining (reference): replay the full FedAvg loop, excluding the
   forgotten clients from every round.
3. SISA: client-aware sharded and sliced retraining, ensembled at inference.
4. Knowledge distillation: subtract the forgotten clients' stored updates, then distil from
   the pre-unlearning teacher on retain-set inputs.
5. Noise distillation (MF): the data-free variant, distilled on Gaussian noise instead of
   retain data.

## Datasets

- MNIST (grayscale image classification, CNN).
- Adult Income (tabular binary classification, MLP).
- CIFAR-10 (RGB image classification, ResNet-20 with GroupNorm).

Data is partitioned across five clients. The headline experiments use IID partitions.
MNIST and Adult Income are additionally evaluated under a non-IID partition (Dirichlet,
alpha = 0.5).

## Installation

Requires Python 3.12 or higher and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jeffkroon/Federated_unlearning_evaluation.git
cd Federated_unlearning_evaluation/fl-disagreement-resolution
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv sync
```

## Reproducing the thesis experiments

MNIST and Adult Income are run over ten seeds (42-51). CIFAR-10 is run on a single seed (42).

```bash
# IID grid (all datasets, all scenarios):
python3 scripts/run_full_data_thesis.py

# non-IID grid (MNIST and Adult Income):
python3 scripts/run_full_data_noniid.py -w 6

# aggregate the raw runs into the result tables used in the thesis:
python3 scripts/consolidate_v3.py        # -> THESIS_RESULTS_FINAL/
python3 scripts/consolidate_noniid.py    # -> THESIS_RESULTS_NONIID/
```

Aggregated results (mean and standard deviation over the permanent scenarios S1, S2, S3,
S8) are written to `THESIS_RESULTS_FINAL/` (IID) and `THESIS_RESULTS_NONIID/` (non-IID),
each containing the per-seed values, the aggregated metrics, and ready-to-input LaTeX
tables.

## Extending the framework

The framework is dataset- and strategy-agnostic. A new dataset is added by implementing a
dataset adapter (data loading, input and output dimensionality, and a classification or
regression flag) and registering it. The orchestrator, branching mechanism, and evaluation
pipeline then apply unchanged. A new unlearning strategy is added behind the single
unlearning interface used by the existing five.

## Repository structure

```
.
├── fl-disagreement-resolution/   # the framework
│   ├── fl_orchestrator.py        # experiment orchestration (FL rounds + unlearning)
│   ├── fl_server/                # aggregation, disagreement resolution, branching, evaluation, strategies
│   ├── fl_client/                # client-side training
│   ├── fl_module/                # dataset adapters + model/dataset registries (mnist, adult, cifar10, tabular, custom)
│   ├── mock_etcd/                # configurations and disagreement scenarios
│   └── scripts/                  # experiment runners and result consolidation
├── machine_unlearning_tool/      # unlearning strategy implementations
├── THESIS_RESULTS_FINAL/         # aggregated IID results (per-seed values, metrics, LaTeX tables)
└── THESIS_RESULTS_NONIID/        # aggregated non-IID results
```
