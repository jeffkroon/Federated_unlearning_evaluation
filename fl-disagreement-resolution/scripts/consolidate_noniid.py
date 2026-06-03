#!/usr/bin/env python3
"""Aggregate the NON-IID full-data grid into THESIS_RESULTS_NONIID/ (n=10).

Strict isolation from the IID thesis results:
  - reads ONLY results/full_data_noniid/  (the non-IID grid output)
  - writes ONLY ../THESIS_RESULTS_NONIID/  (hard-guarded: refuses any path
    containing "FINAL", so it can never touch THESIS_RESULTS_FINAL)
Same aggregation method as consolidate_v3.py (permanent {S1,S2,S3,S8} mean +- std;
S4 separate at round 3), so IID and non-IID tables are directly comparable. MNIST
and Adult only; every strategy is read from the non-IID grid (no exact_refix, no CIFAR).

Run from fl-disagreement-resolution/:
    python3 scripts/consolidate_noniid.py
"""
import json, glob, math, os, statistics as st
from collections import defaultdict

OUT = os.environ.get("OUT", "../THESIS_RESULTS_NONIID")
if "FINAL" in OUT.upper():
    raise SystemExit(f"REFUSING to write to '{OUT}': non-IID results must never touch the IID thesis dir.")

SRC = "results/full_data_noniid"
PERMANENT = [1, 2, 3, 8]
S4_SET = [4]
METRICS = ["utility_accuracy_test", "unlearning_score", "forget_confidence_mean_unlearned",
           "forget_entropy_mean_unlearned", "mia_accuracy_unlearned", "mia_improvement",
           "js_divergence_mean", "activation_cosine_similarity", "retrain_fraction"]
STRATS = ["exact_retraining", "federated_exact_retraining", "sisa", "distillation", "mf"]
_env = os.environ.get("SEEDS")
SEEDS = [int(x) for x in _env.split(",")] if _env else list(range(42, 52))


def fr(s):
    return 3 if s == 4 else 10  # mnist/adult: round 3 for S4, else final round 10


def from_consolidated(path, strat, Rn):
    r = json.load(open(path))
    rd = r["strategies"].get(strat, {}).get("rounds", {}).get(str(Rn), {})
    return [t["unlearning_metrics"] for t in rd.get("tracks", {}).values()
            if isinstance(t, dict) and t.get("unlearning_metrics")]


def get_tracks(ds, strat, seed, s):
    p = glob.glob(f"{SRC}/{ds}_seed{seed}_s{s}/fl_simulation_*/consolidated_results.json")
    return from_consolidated(p[0], strat, fr(s)) if p else []


def seed_value(ds, strat, seed, scen_set):
    per = defaultdict(list)
    for s in scen_set:
        tr = get_tracks(ds, strat, seed, s)
        if not tr:
            continue
        for m in METRICS:
            vals = [t[m] for t in tr if isinstance(t.get(m), (int, float)) and math.isfinite(t[m])]
            if vals:
                per[m].append(sum(vals) / len(vals))
    return {m: (sum(v) / len(v)) for m, v in per.items() if v}


os.makedirs(OUT, exist_ok=True)


def build(scen_set):
    agg, perseed = [], []
    for ds in ["mnist", "adult"]:
        for strat in STRATS:
            sv = {seed: seed_value(ds, strat, seed, scen_set) for seed in SEEDS}
            present = [seed for seed in SEEDS if sv.get(seed)]
            for seed in present:
                for m, v in sv[seed].items():
                    perseed.append((ds, strat, seed, m, v))
            row = {"dataset": ds, "strategy": strat, "n_seeds": len(present)}
            for m in METRICS:
                xs = [sv[seed][m] for seed in SEEDS if m in sv.get(seed, {})]
                if xs:
                    row[m + "_mean"] = sum(xs) / len(xs)
                    row[m + "_std"] = st.stdev(xs) if len(xs) > 1 else 0.0
            agg.append(row)
    return agg, perseed


agg_rows, perseed_rows = build(PERMANENT)
s4_rows, _ = build(S4_SET)

cols = ["dataset", "strategy", "n_seeds"] + [m + x for m in METRICS for x in ("_mean", "_std")]


def write_agg(path, rows):
    L = [",".join(cols)]
    for r in rows:
        L.append(",".join([str(r.get("dataset")), str(r.get("strategy")), str(r.get("n_seeds"))] +
                          [f"{r[m + x]:.4f}" if (m + x) in r else "" for m in METRICS for x in ("_mean", "_std")]))
    open(path, "w").write("\n".join(L) + "\n")


write_agg(f"{OUT}/aggregated_mean_std.csv", agg_rows)
write_agg(f"{OUT}/aggregated_s4_temporary.csv", s4_rows)

L2 = ["dataset,strategy,seed,metric,value"]
for ds, strat, seed, m, v in perseed_rows:
    L2.append(f"{ds},{strat},{seed},{m},{v:.4f}")
open(f"{OUT}/per_seed_values.csv", "w").write("\n".join(L2) + "\n")

# baseline global quality (pre-unlearning original_model), mean +- std over seeds
base_rows = []
for ds in ["mnist", "adult"]:
    perm = defaultdict(list)
    for seed in SEEDS:
        for s in PERMANENT:
            paths = glob.glob(f"{SRC}/{ds}_seed{seed}_s{s}/fl_simulation_*/consolidated_results.json")
            if not paths:
                continue
            r = json.load(open(paths[0]))
            rd = r["strategies"].get("original_model", {}).get("rounds", {}).get("10", {})
            g = rd.get("global") or rd.get("tracks", {}).get("global", {})
            if g:
                for k in ("accuracy", "f1", "precision", "recall"):
                    if k in g:
                        perm[(seed, k)].append(g[k])
    for k in ("accuracy", "f1", "precision", "recall"):
        seedmeans = [sum(perm[(seed, k)]) / len(perm[(seed, k)]) for seed in SEEDS if perm.get((seed, k))]
        if seedmeans:
            base_rows.append((ds, k, sum(seedmeans) / len(seedmeans),
                              st.stdev(seedmeans) if len(seedmeans) > 1 else 0.0, len(seedmeans)))
L3 = ["dataset,metric,mean,std,n_seeds"]
for ds, k, mn, sd, n in base_rows:
    L3.append(f"{ds},{k},{mn:.4f},{sd:.4f},{n}")
open(f"{OUT}/baseline_quality.csv", "w").write("\n".join(L3) + "\n")

print(f"OUT={OUT}  SRC={SRC}  SEEDS={SEEDS}")
print("NON-IID AGGREGATED, PERMANENT {S1,S2,S3,S8} (mean +- std):")
print(f"{'ds':7s}{'strategy':28s}{'n':>3s}  {'utility':>15s}{'unlScore':>15s}{'MIA':>15s}")
for r in agg_rows:
    def f(m):
        return f"{r[m + '_mean']:.3f}±{r[m + '_std']:.3f}" if (m + '_mean') in r else "   -   "
    print(f"{r['dataset']:7s}{r['strategy']:28s}{r['n_seeds']:>3d}  "
          f"{f('utility_accuracy_test'):>15s}{f('unlearning_score'):>15s}{f('mia_accuracy_unlearned'):>15s}")
print(f"\nWritten to {OUT}/ (IID THESIS_RESULTS_FINAL untouched).")
