#!/usr/bin/env python3
"""Re-run MNIST + Adult on the FULL dataset (all data distributed across clients).

Motivation: the original thesis runs used a 1,000-sample subset per client (~8-15%
of each dataset). The FL benchmarking norm is to partition the entire training set.
This grid re-runs both datasets with the full partition (MNIST 12,000/client,
Adult 7,235/client) so absolute performance is comparable to prior work, while
keeping the within-dataset strategy comparison identical.

Isolated from everything else:
  - configs:    mock_etcd/configuration_{mnist,adult}_full.json
  - results:    results/full_data/fl_simulation_*  (separate base_dir)
  - run map:    results/full_data/full_data_runs.json (resumable)
  - log:        /tmp/full_data_thesis.log

Run under caffeinate to keep the laptop awake:
    caffeinate -i -m -s python3 scripts/run_full_data_thesis.py 2>&1 | tee results/full_data/run.log
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

#Matches the original thesis grid (9 scenarios x 3 seeds), so MNIST/Adult keep
# full statistical power (mean +/- std over 3 seeds) after the full-data re-run.
SCENARIOS = [0, 1, 2, 3, 4, 8, 10, 20, 34]
SEEDS = [42, 43, 44]

FL_ROUNDS = 10
LOCAL_EPOCHS = 5
NUM_CLIENTS = 5

# dataset -> isolated config
DATASETS = {
    "mnist": Path("mock_etcd/configuration_mnist_full.json"),
    "adult": Path("mock_etcd/configuration_adult_full.json"),
}

RESULTS_BASE = Path("results/full_data")
LOG_PATH = Path("/tmp/full_data_thesis.log")
RUN_DIRS_PATH = RESULTS_BASE / "full_data_runs.json"


def load_run_dirs() -> dict:
    if RUN_DIRS_PATH.exists():
        return json.loads(RUN_DIRS_PATH.read_text())
    return {}


def save_run_dirs(run_dirs: dict) -> None:
    RUN_DIRS_PATH.write_text(json.dumps(run_dirs, indent=2))


def make_key(dataset: str, seed: int, scenario: int) -> str:
    return f"{dataset}_seed{seed}_s{scenario}"


def run_one(dataset: str, seed: int, scenario: int, base_cfg: dict) -> Optional[str]:
    """Run one experiment. Returns the created results directory name (relative to RESULTS_BASE)."""
    cfg = json.loads(json.dumps(base_cfg))
    cfg["experiment"]["type"] = dataset
    cfg["experiment"]["random_seed"] = seed
    cfg["experiment"]["fl_rounds"] = FL_ROUNDS

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir="mock_etcd",
        prefix=f".tmp_full_{dataset}_{seed}_{scenario}_",
        delete=False,
    ) as f:
        json.dump(cfg, f, indent=2)
        tmp = f.name

    before = {
        p.name
        for p in RESULTS_BASE.glob(f"fl_simulation_*_{dataset}_s{scenario}")
        if (p / "run_metadata.json").exists()
    }

    try:
        with open(LOG_PATH, "a") as log:
            log.write(f"\n\n--- start {dataset} seed={seed} scenario={scenario} ---\n")
            log.flush()
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_fl.py",
                    "-S", str(scenario),
                    "-e", dataset,
                    "-r", str(FL_ROUNDS),
                    "-l", str(LOCAL_EPOCHS),
                    "-c", str(NUM_CLIENTS),
                    "-C", tmp,
                    "--no-viz",
                ],
                stdout=log,
                stderr=log,
                check=False,
            )
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass

    after = {
        p.name
        for p in RESULTS_BASE.glob(f"fl_simulation_*_{dataset}_s{scenario}")
        if (p / "run_metadata.json").exists()
    }
    new = sorted(after - before, reverse=True)
    if not new:
        return None

    dir_name = new[0]
    # run_fl.py's built-in consolidation hardcodes Path("results") and cannot see
    #our custom base_dir, so we consolidate the correct dir ourselves.
    _consolidate(RESULTS_BASE / dir_name, scenario, dataset)
    return dir_name


def _consolidate(sim_path: Path, scenario: int, dataset: str) -> None:
    """Write consolidated_results.json into the (custom base_dir) sim dir."""
    saved_argv = sys.argv
    try:
        sys.argv = ["run_fl"]  #guard run_fl.py's module-level argparse on import
        from run_fl import consolidate_results
        consolidate_results(str(sim_path), scenario=str(scenario), experiment=dataset)
    except Exception as e:  # noqa: BLE001, log and continue; verify() will flag it
        print(f"  WARNING: consolidation failed for {sim_path}: {e}", flush=True)
    finally:
        sys.argv = saved_argv


def verify(dir_name: Optional[str]) -> bool:
    """True if the run looks complete (consolidated_results + >=5 strategy subdirs)."""
    if not dir_name:
        return False
    p = RESULTS_BASE / dir_name
    if not (p / "consolidated_results.json").exists():
        return False
    strategy_dirs = [x for x in p.iterdir() if x.name.startswith("strategy_")]
    return len(strategy_dirs) >= 5


def main() -> None:
    os.chdir(Path(__file__).parent.parent)
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")

    base_cfgs = {}
    for dataset, cfg_path in DATASETS.items():
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}")
            sys.exit(1)
        base_cfgs[dataset] = json.loads(cfg_path.read_text())

    run_dirs = load_run_dirs()
    total = len(DATASETS) * len(SEEDS) * len(SCENARIOS)
    done = sum(1 for k in run_dirs if run_dirs[k].get("status") == "OK")

    print(
        f"\nFULL-DATA thesis grid: {total} runs total ({done} already done). "
        f"{FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs x {NUM_CLIENTS} clients."
    )
    print(f"  datasets: {list(DATASETS)}")
    print(f"  results:  {RESULTS_BASE}/")
    print(f"  map:      {RUN_DIRS_PATH}")
    print(f"  log:      {LOG_PATH}\n")

    start = time.time()
    for dataset in DATASETS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                key = make_key(dataset, seed, scenario)

                if run_dirs.get(key, {}).get("status") == "OK":
                    print(f"[SKIP done] {key} -> {run_dirs[key]['dir']}")
                    continue

                done += 1
                run_start = time.time()
                print(f"[{done}/{total}] {dataset} seed={seed} s={scenario} ...", flush=True)

                dir_name = run_one(dataset, seed, scenario, base_cfgs[dataset])
                ok = verify(dir_name)
                status = "OK" if ok else "INCOMPLETE"
                elapsed = time.time() - run_start

                run_dirs[key] = {
                    "dir": dir_name or "MISSING",
                    "status": status,
                    "dataset": dataset,
                    "seed": seed,
                    "scenario": scenario,
                    "elapsed_seconds": round(elapsed, 1),
                }
                save_run_dirs(run_dirs)

                print(
                    f"  -> {dir_name or 'MISSING'}  [{status}]  ({elapsed/60:.1f}min)",
                    flush=True,
                )

    total_elapsed = time.time() - start
    n_ok = sum(1 for v in run_dirs.values() if v.get("status") == "OK")
    n_fail = sum(1 for v in run_dirs.values() if v.get("status") != "OK")
    print(f"\nDone in {total_elapsed/3600:.1f}h, OK: {n_ok}/{total}, incomplete: {n_fail}")
    print(f"Run mapping: {RUN_DIRS_PATH}")


if __name__ == "__main__":
    main()
