# Consolidated thesis results (n=10): provenance & method

Single authoritative dataset for the thesis tables. All values **recomputed from raw
metrics** (`metrics.json` / `consolidated_results.json`), not from any earlier aggregation
CSV. MNIST/Adult are aggregated over seeds 42-51 (n=10). CIFAR-10 is single-seed (42).

Built by `fl-disagreement-resolution/scripts/consolidate_v3.py` (re-runnable). v3 is
identical in method to v2. The only change is generalised seed→results-dir routing so the
46-51 batch is included. Self-test: `SEEDS=42,43,44,45` reproduces the n=4 build byte-for-byte.

Aggregation: per (dataset, strategy), for each seed take the mean over the unlearned tracks
of each scenario at its final active round, then the mean over scenarios, giving one value per
seed. Report mean ± sample-std across seeds. Permanent table = **{S1, S2, S3, S8}** at the final
round (10 MNIST/Adult, 35 CIFAR). **S4 is separate** (round 3, its temporary exclusion expires
after round 3) in `aggregated_s4_temporary.csv`; never pooled with the permanent table.

## Per-seed source routing

| Seed(s) | Results directory |
|---------|-------------------|
| 42 | `results/full_data/` |
| 43, 44, 45 | `results/full_data_seeds345/` |
| 46-51 | `results/full_data_seeds_46_51/` |

The 46-51 batch ran on RunPod with the fully corrected code (from-scratch exact_retraining.
matplotlib/brokenaxes/xgboost present so distillation actually trains). All 72 runs (6 seeds ×
{mnist,adult} × {S0,S1,S2,S3,S4,S8}) completed with status OK.

## Supersession (which source wins per cell)

```
MNIST  exact_retraining  seed42  -> exact_refix  (true from-scratch re-run)
MNIST  exact_retraining  43-51   -> that seed's full_data dir (grid ran fixed code)
ADULT  exact_retraining  ALL     -> that seed's full_data dir (MLP: warm-start bug never applied)
MNIST/ADULT  fed-exact/sisa/distill/mf  ALL -> that seed's full_data dir
CIFAR  exact_retraining          -> exact_refix
CIFAR  fed-exact forget-metrics  -> gap-fill
CIFAR  sisa/distill/mf/baseline  -> THESIS_RESULTS_VERIFIED
```

## Why the corrections were needed (unchanged from the n=4 build)

1. **exact_retraining warm-start defect**: reset only `Linear`, leaving `Conv2d`/`GroupNorm`
   warm-started, not a true from-scratch retrain on CNN/ResNet (Adult MLP unaffected). Fixed
   (`machine_unlearning_tool/unlearning.py`) and used for every seed. Effect negligible (< 0.008).
2. **CIFAR federated_exact_retraining missing forget-metrics**: input-reshape defect silently
   skipped them. Fixed and re-run (gap-fill). Validated equivalent to the original campaign.

## "n/a" cells (empty in CSV): by design, not missing data

- `federated_exact_retraining`: no `activation_cosine_similarity`, no `retrain_fraction` (not exposed).
- `sisa`: no `activation_cosine_similarity` (not exposed).

## Scope notes

- Scenarios reported: S1, S2, S3, S4, S8 (S0 = baseline only). MNIST/Adult additionally have complete
  S10/S34, excluded from cross-dataset comparison (no CIFAR counterpart). CIFAR coverage = S0-S4, S8.
- S4 is read at round 3 (early training). Its absolute values are lower than the final-round
  scenarios, a property of S4, applied equally to all strategies.

## Source directories are read-only

This consolidation only reads from the sources and writes here. The originals are untouched.
