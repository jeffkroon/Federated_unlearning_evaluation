#!/usr/bin/env python3
"""Multi-seed full-data MNIST + Adult grid (seeds 43, 44, 45) for statistical
replication of the reported results, using the corrected from-scratch
exact_retraining (Conv2d + GroupNorm reset).

Reuses the tested run_full_data_parallel logic verbatim; only overrides:
  - SEEDS      -> 43, 44, 45  (seed 42 already done)
  - SCENARIOS  -> 0,1,2,3,4,8 (the reported set; S10/S34 are excluded from the
                  cross-dataset comparison, so they are not needed for error bars)
  - output dir -> results/full_data_seeds345/  (canonical seed-42 in
                  results/full_data/ is left completely untouched)

Every per-run safety property of the parent script is preserved: isolated
FL_DISAGREEMENTS_PATH per run (no race), per-run base_dir, resumable run-map.

Run from fl-disagreement-resolution/:
    python3 scripts/run_full_data_multiseed.py            # default workers
    python3 scripts/run_full_data_multiseed.py -w 3       # cap concurrency
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_full_data_parallel as grid  #tested logic

# --- overrides (functions resolve these as module globals at call time) ---
grid.SEEDS = [43, 44, 45]
grid.SCENARIOS = [0, 1, 2, 3, 4, 8]
grid.RESULTS_BASE = Path("results/full_data_seeds345")
grid.RUN_DIRS_PATH = grid.RESULTS_BASE / "full_data_runs.json"

if __name__ == "__main__":
    grid.main()
