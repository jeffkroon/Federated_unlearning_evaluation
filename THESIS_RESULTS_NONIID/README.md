# THESIS_RESULTS_NONIID: non-IID consolidated results (n=10)

Companion to `THESIS_RESULTS_FINAL/` (the IID thesis results). Same experiment,
same aggregation method, the ONLY difference is the client data partition:

- **IID** (`THESIS_RESULTS_FINAL/`): each client gets a representative slice.
- **non-IID** (this dir): clients partitioned by **Dirichlet(alpha=0.5) label skew**,
  using every sample (same data volume, only the split differs).

MNIST + Adult, **seeds 42-51 (n=10)**. No CIFAR (non-IID CIFAR was out of scope.
the IID thesis keeps CIFAR as a single-seed stress-test).

## Files

| File | What |
|------|------|
| `aggregated_mean_std.csv` | **PRIMARY.** Permanent {S1,S2,S3,S8}, mean ± std over 10 seeds. |
| `aggregated_s4_temporary.csv` | S4 separate (round 3). Not pooled with permanent. |
| `per_seed_values.csv` | Per-seed values (transparency). |
| `baseline_quality.csv` | Pre-unlearning global model quality, mean ± std. |
| `latex/` | Ready-to-`\input` tables (captions auto-state 10 seeds, no CIFAR). |

## Isolation

Built by `fl-disagreement-resolution/scripts/consolidate_noniid.py`, which reads ONLY
`results/full_data_noniid/` and writes ONLY here. The script **hard-refuses** any output
path containing "FINAL", so the IID thesis (`THESIS_RESULTS_FINAL/`) can never be touched.

## Headline IID → non-IID effect (utility, permanent set)

The federated methods degrade under heterogeneity (FedAvg struggles with non-IID), while
centralized exact retraining stays robust:

| dataset / strategy | utility IID | utility non-IID |
|---|---|---|
| mnist / fed_exact | 0.992 | 0.987 |
| mnist / sisa | 0.986 | 0.963 |
| adult / fed_exact | 0.848 | 0.823 |
| adult / sisa | 0.849 | 0.826 |
| adult / exact_retraining | 0.850 | 0.843 |

Aggregation method, validation harness, and the S4 / from-scratch-exact conventions are
identical to `THESIS_RESULTS_FINAL/`: see that directory's README for the method details.

Rebuild: `python3 fl-disagreement-resolution/scripts/consolidate_noniid.py`
LaTeX: `FIN=THESIS_RESULTS_NONIID python3 fl-disagreement-resolution/scripts/gen_latex_tables.py`
