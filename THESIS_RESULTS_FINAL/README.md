# THESIS_RESULTS_FINAL: authoritative consolidated results (n=10)

> Canonical results directory. The previous single-batch n=4 build is preserved
> read-only in `THESIS_RESULTS_FINAL_n4_archive/`.

One clean directory with the final numbers for the thesis. All values **recomputed
from raw metrics** (not from intermediate aggregation CSVs). This is the multi-seed
build: MNIST and Adult are now replicated across **10 seeds (42-51)**. CIFAR-10 stays
single-seed (42).

## Files

| File | What |
|------|------|
| `aggregated_mean_std.csv` / `.json` | **PRIMARY.** Permanent scenarios {S1,S2,S3,S8} only. Per (dataset, strategy): mean ± std across seeds. |
| `aggregated_s4_temporary.csv` | **S4 reported SEPARATELY** (temporary exclusion, read at round 3). Do NOT pool with the permanent table. |
| `per_seed_values.csv` | Transparency: each individual seed's value per (dataset, strategy, metric), permanent set. |
| `baseline_quality.csv` | Global pre-unlearning model quality (accuracy, F1, precision, recall), mean ± std, permanent set. |
| `latex/` | Ready-to-`\input` tables (trade-off, forgetting, MIA, S4, baseline). Captions auto-state the seed count. |
| `confusion_matrices/` | Global-model confusion matrices (PNG), seed 42, final round, per scenario. |
| `PROVENANCE.md` | Per-cell source + supersession rules + re-run rationale. |

> **CRITICAL, S4 is not pooled.** S4 is a *temporary* exclusion read at **round 3** (mid-training),
> while the permanent scenarios are read at the final round (10 MNIST/Adult, 35 CIFAR, fully trained).
> Averaging S4 into the permanent mean mixes training stages and drags every strategy toward its
> round-3 value. The main table therefore uses **{S1, S2, S3, S8}**. S4 has its own table.

## Seeds & statistics

| Dataset | Seeds | Reporting |
|---------|-------|-----------|
| MNIST | 42-51 (n=10) | **mean ± std** |
| Adult Income | 42-51 (n=10) | **mean ± std** |
| CIFAR-10 | 42 (n=1) | point estimate (std = 0) |

MNIST/Adult are replicated across 10 seeds, so mean ± std is reported there. CIFAR is a
single-seed complexity stress-test (35-round ResNet × 5 strategies × scenarios is the
expensive one) and is reported as a point estimate.

## Aggregation method

Per (dataset, strategy, metric): for each seed, take the mean over the unlearned tracks of
each scenario at its final round, then the mean over scenarios. That gives one value per seed.
the file reports mean ± sample-std (ddof=1) across seeds.

- `aggregated_mean_std.csv`: permanent scenarios **{S1, S2, S3, S8}** at the final round
  (10 MNIST/Adult, 35 CIFAR, fully trained).
- `aggregated_s4_temporary.csv`: **S4 only, at round 3** (its exclusion expires after round 3).

**Validation (self-test passed).** Re-running the builder restricted to seeds 42-45
(`SEEDS=42,43,44,45`) reproduces the previous n=4 `THESIS_RESULTS_FINAL` **byte-for-byte**
(aggregated, S4, and baseline CSVs identical). This proves the seed-generalised routing
changed nothing in the aggregation method, only more seeds were added. The n=10 means sit
within the n=4 spread (e.g. MNIST MF 0.412±0.075 → 0.447±0.071, Adult Exact RT 0.842 → 0.850):
no shift, tighter error bars.

## Metrics

- **Utility**: `utility_accuracy_test`: unlearned-model accuracy on the global held-out test set.
- **Forgetting**: `unlearning_score` (forget-acc before − after), `forget_confidence_mean_unlearned`,
  `forget_entropy_mean_unlearned`.
- **Privacy**: `mia_accuracy_unlearned` (≈0.5 = good), `mia_improvement`.
- **Behavioral change**: `js_divergence_mean` (vs pre-unlearning model), `activation_cosine_similarity`.
- **Cost**: `retrain_fraction`. (Wall-clock time is hardware-dependent and excluded from mean ± std.)
- Per-strategy F1/precision/recall are **not** in the data (only global-model level → `baseline_quality.csv`).

## exact_retraining is the corrected from-scratch version everywhere

All `exact_retraining` cells use the corrected gold standard (Conv2d + GroupNorm reset, true
from-scratch). MNIST seed-42 from the RunPod re-run. Adult seed-42 from the original run (a pure
MLP, the warm-start defect never applied). **seeds 43-45 and 46-51 from the multi-seed grids
(fixed code throughout)**. CIFAR from the RunPod re-run. Effect of the fix was negligible
(< 0.008 on every metric), conclusions unchanged, reference now methodologically correct.

## Confusion matrices: availability

`confusion_matrices/` holds the **global model's** confusion matrix (PNG) at the final round,
seed 42, for each scenario, all three datasets (18 files). Seed 42 only, seeds 43-51 plots were
not downloaded (only JSON metrics), and raw predictions are not stored, so per-seed confusion
matrices cannot be regenerated without re-running.

## Source directories (read-only: untouched by this build)

- `fl-disagreement-resolution/results/full_data/`: MNIST + Adult, seed 42, all strategies
- `fl-disagreement-resolution/results/full_data_seeds345/`: MNIST + Adult, seeds 43-45
- `fl-disagreement-resolution/results/full_data_seeds_46_51/`: MNIST + Adult, **seeds 46-51 (this batch)**
- `fl-disagreement-resolution/results/exact_refix/`: MNIST + CIFAR exact_retraining (from-scratch)
- `fl-disagreement-resolution/results/cifar_gapfill/`: CIFAR fed-exact forget-metrics
- `THESIS_RESULTS_VERIFIED/cifar/raw/`: CIFAR sisa/distillation/mf/baseline (seed 42)

Rebuild: `python3 fl-disagreement-resolution/scripts/consolidate_v3.py` (from the repo root).
Self-test: `SEEDS=42,43,44,45 OUT=../_selftest python3 fl-disagreement-resolution/scripts/consolidate_v3.py`.
LaTeX: `python3 fl-disagreement-resolution/scripts/gen_latex_tables.py` (defaults to THESIS_RESULTS_FINAL).
