#!/usr/bin/env python3
"""Parallel full-data MNIST + Adult grid.

Identical experiment to run_full_data_thesis.py (full dataset distributed across
5 clients, 9 scenarios x 3 seeds = 54 runs), but runs several independent runs
CONCURRENTLY. The 54 runs are embarrassingly parallel; the only shared state that
must be isolated per run is:

  - the results directory  -> each run writes into results/full_data/<key>/
  - the SISA checkpoint dir -> each run gets results/full_data/<key>/sisa_checkpoints

Sim-dir name collisions (the orchestrator names dirs fl_simulation_<HHMMSS>_<type>_s<N>
without the seed, at 1-second granularity) are avoided by giving every run its own
base_dir, so two runs of the same scenario/different seed can never share a path.

Per-run torch/BLAS threading is capped (THREADS_PER_RUN) so NUM_WORKERS runs do not
oversubscribe the CPU. On a 12-core machine, 5 workers x 2 threads ~= 10 threads.

Resumable: completed runs (status OK in the run-map) are skipped.

Run under caffeinate to keep the laptop awake:
    caffeinate -i -m -s python3 scripts/run_full_data_parallel.py 2>&1 | tee results/full_data/run_parallel.log
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

# s20 dropped: degenerate near-baseline (a single 1-round exclusion that leaves no
#track, so functionally identical to the s0 empty baseline, and mislabeled "20 Clients").
SCENARIOS = [0, 1, 2, 3, 4, 8, 10, 34]
SEEDS = [42]  #single seed (complexity stress-test, not full statistical replication)

FL_ROUNDS = 10
LOCAL_EPOCHS = 5
NUM_CLIENTS = 5

DATASETS = {
    "mnist": Path("mock_etcd/configuration_mnist_full.json"),
    "adult": Path("mock_etcd/configuration_adult_full.json"),
}

DEFAULT_WORKERS = 5
THREADS_PER_RUN = 2

RESULTS_BASE = Path("results/full_data")
RUN_DIRS_PATH = RESULTS_BASE / "full_data_runs.json"

_map_lock = Lock()
_consolidate_fn = None  # imported once in main(), before the pool starts


def make_key(dataset: str, seed: int, scenario: int) -> str:
    return f"{dataset}_seed{seed}_s{scenario}"


def load_run_dirs() -> dict:
    if RUN_DIRS_PATH.exists():
        return json.loads(RUN_DIRS_PATH.read_text())
    return {}


def save_run_dirs(run_dirs: dict) -> None:
    RUN_DIRS_PATH.write_text(json.dumps(run_dirs, indent=2))


def _thread_limited_env() -> dict:
    """Cap intra-run threading so concurrent runs share cores without thrashing."""
    env = dict(os.environ)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[var] = str(THREADS_PER_RUN)
    return env


def _find_sim_dir(run_base: Path, dataset: str, scenario: int) -> Optional[Path]:
    """The real sim dir is the one carrying run_metadata.json (orchestrator may
    leave empty timestamped dirs behind during strategy branching)."""
    candidates = [
        p
        for p in run_base.glob(f"fl_simulation_*_{dataset}_s{scenario}")
        if (p / "run_metadata.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _verify(sim_dir: Optional[Path]) -> bool:
    if not sim_dir or not (sim_dir / "consolidated_results.json").exists():
        return False
    strategy_dirs = [x for x in sim_dir.iterdir() if x.name.startswith("strategy_")]
    return len(strategy_dirs) >= 5


def run_one(dataset: str, seed: int, scenario: int, base_cfg: dict) -> dict:
    """Run one isolated experiment and return its run-map entry."""
    key = make_key(dataset, seed, scenario)
    run_base = RESULTS_BASE / key
    run_base.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(json.dumps(base_cfg))
    cfg["experiment"]["type"] = dataset
    cfg["experiment"]["random_seed"] = seed
    cfg["experiment"]["fl_rounds"] = FL_ROUNDS
    cfg["results"]["base_dir"] = str(run_base)
    cfg["unlearning"]["train_params"]["checkpoint_dir"] = str(run_base / "sisa_checkpoints")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir="mock_etcd",
        prefix=f".tmp_par_{dataset}_{seed}_{scenario}_",
        delete=False,
    ) as f:
        json.dump(cfg, f, indent=2)
        tmp = f.name

    run_start = time.time()
    # Per-run ISOLATED disagreements file: concurrent runs can never clobber a shared
    # mock_etcd/disagreements.json (the bug that corrupted the earlier parallel grid).
    env = _thread_limited_env()
    env["FL_DISAGREEMENTS_PATH"] = str((run_base / "disagreements.json").resolve())
    try:
        log_path = run_base / "run.log"
        with open(log_path, "w") as log:
            subprocess.run(
                [
                    sys.executable, "scripts/run_fl.py",
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
                env=env,
                check=False,
            )
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass

    sim_dir = _find_sim_dir(run_base, dataset, scenario)
    if sim_dir is not None and _consolidate_fn is not None:
        try:
            _consolidate_fn(str(sim_dir), scenario=str(scenario), experiment=dataset)
        except Exception as e:  # noqa: BLE001, log; _verify will flag the gap
            print(f"  WARNING: consolidation failed for {sim_dir}: {e}", flush=True)

    ok = _verify(sim_dir)
    return {
        "dir": str(sim_dir) if sim_dir else "MISSING",
        "status": "OK" if ok else "INCOMPLETE",
        "dataset": dataset,
        "seed": seed,
        "scenario": scenario,
        "elapsed_seconds": round(time.time() - run_start, 1),
    }


def main() -> None:
    os.chdir(Path(__file__).parent.parent)

    parser = argparse.ArgumentParser(description="Parallel full-data MNIST+Adult grid")
    parser.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"concurrent runs (default {DEFAULT_WORKERS})")
    args = parser.parse_args()

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    #Import the consolidation helper once, before any worker thread starts, so the
    # one-time module import (and its sys.argv touch) cannot race across threads.
    global _consolidate_fn
    saved_argv = sys.argv
    try:
        sys.argv = ["run_fl"]
        from run_fl import consolidate_results
        _consolidate_fn = consolidate_results
    finally:
        sys.argv = saved_argv

    base_cfgs = {}
    for dataset, cfg_path in DATASETS.items():
        if not cfg_path.exists():
            print(f"ERROR: config not found: {cfg_path}")
            sys.exit(1)
        base_cfgs[dataset] = json.loads(cfg_path.read_text())

    run_dirs = load_run_dirs()
    tasks = [
        (dataset, seed, scenario)
        for dataset in DATASETS
        for seed in SEEDS
        for scenario in SCENARIOS
        if run_dirs.get(make_key(dataset, seed, scenario), {}).get("status") != "OK"
    ]
    total = len(DATASETS) * len(SEEDS) * len(SCENARIOS)
    already = total - len(tasks)

    print(
        f"\nPARALLEL full-data grid: {total} runs ({already} already OK, {len(tasks)} to run).\n"
        f"  workers:  {args.workers}  ({THREADS_PER_RUN} threads/run)\n"
        f"  datasets: {list(DATASETS)}\n"
        f"  results:  {RESULTS_BASE}/<key>/\n"
        f"  map:      {RUN_DIRS_PATH}\n",
        flush=True,
    )

    start = time.time()
    done = already
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, d, s, sc, base_cfgs[d]): make_key(d, s, sc)
            for (d, s, sc) in tasks
        }
        print(f"Launched {len(futures)} runs across {args.workers} workers...\n", flush=True)
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                entry = fut.result()
            except Exception as e:  # noqa: BLE001
                entry = {"dir": "ERROR", "status": "ERROR", "error": str(e)}
            done += 1
            with _map_lock:
                run_dirs[key] = entry
                save_run_dirs(run_dirs)
            print(
                f"[{done}/{total}] {key}  [{entry['status']}]  "
                f"({entry.get('elapsed_seconds', 0) / 60:.1f}min)",
                flush=True,
            )

    elapsed = time.time() - start
    n_ok = sum(1 for v in run_dirs.values() if v.get("status") == "OK")
    print(f"\nDone in {elapsed / 3600:.2f}h, OK: {n_ok}/{total}")
    print(f"Run mapping: {RUN_DIRS_PATH}")


if __name__ == "__main__":
    main()
