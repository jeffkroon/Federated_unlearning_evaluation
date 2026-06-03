#!/usr/bin/env python3
"""Non-IID full-data MNIST + Adult grid (Dirichlet alpha=0.5 label skew).

Same experiment as the IID full-data grid (run_full_data_parallel); the ONLY
difference is the client partition. Every sample is still used (full data), but
distributed with Dirichlet(alpha=0.5) label skew instead of IID. Results are
isolated in results/full_data_noniid/; the IID results are untouched.

NOTE: the dataset adapters hardcode data/mnist and data/adult, so this run
(re)generates NON-IID per-client data into those dirs. Run it on a dedicated
machine/pod; do not mix with IID runs on the same box without regenerating.
The Dirichlet partition is deterministic (seed 42), so all run-seeds share the
same data split and only training stochasticity varies across seeds.

Run from fl-disagreement-resolution/:
    python3 scripts/run_full_data_noniid.py -w 6

Resumable: re-run the same command; completed runs are skipped via
results/full_data_noniid/full_data_runs.json.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_full_data_parallel as grid  # tested grid logic

# --- overrides (functions resolve these as module globals at call time) ---
grid.DATASETS = {
    "mnist": Path("mock_etcd/configuration_mnist_full_noniid.json"),
    "adult": Path("mock_etcd/configuration_adult_full_noniid.json"),
}
grid.SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   #match the IID n=10 for comparison
grid.SCENARIOS = [0, 1, 2, 3, 4, 8]
grid.RESULTS_BASE = Path("results/full_data_noniid")
grid.RUN_DIRS_PATH = grid.RESULTS_BASE / "full_data_runs.json"


def pregenerate_noniid_data() -> None:
    """Partition non-IID data ONCE before the pool, so the parallel runs find it
    and skip setup (no race). Forces a fresh non-IID split into data/mnist and
    data/adult regardless of any pre-existing IID data on the box."""
    #repo root holds the fl_module package; ensure it's importable (Python only
    # puts the script dir, scripts/, on sys.path by default).
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import fl_module
    from fl_module.adult.utils import setup_federated_data as setup_adult

    n = grid.NUM_CLIENTS
    mnist_spc = json.loads(grid.DATASETS["mnist"].read_text())["data"]["client_sample_size"]
    adult_spc = json.loads(grid.DATASETS["adult"].read_text())["data"]["client_sample_size"]

    # MNIST setup has no force flag, so clear the train dir to force a non-IID regen.
    shutil.rmtree("data/mnist/train", ignore_errors=True)
    print(f"[pre-gen] MNIST non-IID (Dirichlet alpha=0.5), {n} clients", flush=True)
    fl_module.setup_mnist_federated_data(num_clients=n, samples_per_client=mnist_spc, iid=False)

    print(f"[pre-gen] Adult non-IID (Dirichlet alpha=0.5), {n} clients", flush=True)
    setup_adult(num_clients=n, samples_per_client=adult_spc, iid=False, data_dir="data/adult", force=True)


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    grid.RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    pregenerate_noniid_data()
    grid.main()
