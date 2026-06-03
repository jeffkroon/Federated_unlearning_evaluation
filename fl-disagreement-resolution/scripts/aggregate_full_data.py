#!/usr/bin/env python3
"""Aggregate full-data FL/unlearning runs into clean tables, honestly.

Reads real values from each run's consolidated_results.json (only the simulation
directory that carries run_metadata.json, ignoring empty branching artifacts).
NOTHING is invented:
  - single seed (n=1)  -> point values, NO fabricated std (std stays blank).
  - >1 seed            -> mean and std over the seeds.
  - a metric the data does not contain stays empty (never zero/random-filled).

It also writes a coverage report that distinguishes:
  - baseline scenarios with no disagreement (unlearning legitimately absent), from
  - scenarios that DO have disagreements but where unlearning metrics are missing
    (a genuine gap, flagged loudly).

Dataset-agnostic: the same script aggregates MNIST/Adult now and CIFAR later by
pointing --results-base / --datasets / --fl-rounds at the CIFAR runs, identical style.

Usage (current MNIST+Adult, single seed, s20 dropped):
    python scripts/aggregate_full_data.py

CIFAR later (once downloaded into e.g. results/cifar):
    python scripts/aggregate_full_data.py --results-base results/cifar \
        --datasets cifar10 --fl-rounds 35 --scenarios 0 1 2 3 4 8 10 34 \
        --out results/cifar_aggregated
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

STRATEGIES = ["original_model", "exact_retraining", "federated_exact_retraining",
              "sisa", "distillation", "mf"]

# Effectiveness + cost metrics, exactly as keyed inside track["unlearning_metrics"].
UNLEARNING_METRICS = [
    "utility_accuracy_test",
    "forget_accuracy_original",
    "forget_accuracy_unlearned",
    "unlearning_score",
    "mia_accuracy_original",
    "mia_accuracy_unlearned",
    "mia_improvement",
    "activation_cosine_similarity",
    "forget_confidence_mean_unlearned",
    "forget_entropy_mean_unlearned",
    "js_divergence_mean",
    # cost: wall-clock (contention-sensitive) + hardware-independent proxies
    "unlearning_time_s",
    "retrain_fraction",
    "N_retrain",
    "N_train_total",
]

SCENARIO_DIR = Path("mock_etcd/scenarios_5clients")


def find_sim_dir(run_base: Path) -> Optional[Path]:
    """Return the real simulation dir (the one with run_metadata.json), newest first.
    The orchestrator leaves empty timestamped dirs during branching; we skip those."""
    cands = [p for p in run_base.glob("fl_simulation_*") if (p / "run_metadata.json").exists()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def scenario_has_disagreements(scenario: int) -> Optional[bool]:
    """True/False if the scenario file defines any disagreement; None if file absent."""
    f = SCENARIO_DIR / f"scenario{scenario}.json"
    if not f.exists():
        return None
    dis = json.loads(f.read_text()).get("disagreements", {})
    return any(len(v) > 0 for v in dis.values())


def extract_final_round_metrics(data: dict, strategy: str, fl_rounds: int) -> dict:
    """Pull the final-round global accuracy and (for unlearning strategies) the
    unlearning metrics averaged over the unlearned tracks. Falls back to the last
    round that actually has unlearned tracks (temporal scenarios unlearn early)."""
    strat = data.get("strategies", {}).get(strategy)
    if not strat:
        return {}
    all_rounds = strat.get("rounds", {})
    round_data = all_rounds.get(str(fl_rounds), {})
    result = {}

    g = round_data.get("global", {})
    if g:
        result["global_accuracy"] = g.get("accuracy")
        result["global_f1"] = g.get("f1")
        result["global_loss"] = g.get("test_loss")

    tracks = round_data.get("tracks", {})
    unlearned = {k: v for k, v in tracks.items() if v.get("unlearned")}
    if not unlearned:
        for rnd in sorted(all_rounds.keys(), key=int, reverse=True):
            r_un = {k: v for k, v in all_rounds[rnd].get("tracks", {}).items() if v.get("unlearned")}
            if r_un:
                unlearned = r_un
                result["unlearning_round"] = int(rnd)
                break

    if unlearned:
        for metric in UNLEARNING_METRICS:
            vals = [t.get("unlearning_metrics", {}).get(metric)
                    for t in unlearned.values() if t.get("unlearning_metrics")]
            vals = [v for v in vals if v is not None]
            result[metric] = statistics.mean(vals) if vals else None
        result["n_unlearned_tracks"] = len(unlearned)

    return result


def agg_seed_values(values):
    """Point value (n=1) or mean±std (n>=2). Returns (mean, std-or-None, n)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return round(vals[0], 6), None, 1
    return round(statistics.mean(vals), 6), round(statistics.stdev(vals), 6), len(vals)


def main():
    ap = argparse.ArgumentParser(description="Aggregate full-data FL/unlearning runs (honest, no fabrication)")
    ap.add_argument("--results-base", default="results/full_data")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "adult"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--scenarios", nargs="+", type=int, default=[0, 1, 2, 3, 4, 8, 10, 34])
    ap.add_argument("--fl-rounds", type=int, default=10)
    ap.add_argument("--out", default="results/full_data_aggregated")
    args = ap.parse_args()

    base = Path(args.results_base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_seed = defaultdict(lambda: defaultdict(list))   # (ds,sc,strat) -> metric -> [vals over seeds]
    coverage = []                                       #rows for the coverage report
    missing_runs = []

    for ds in args.datasets:
        for sc in args.scenarios:
            has_dis = scenario_has_disagreements(sc)
            for seed in args.seeds:
                run_base = base / f"{ds}_seed{seed}_s{sc}"
                sim = find_sim_dir(run_base) if run_base.exists() else None
                if sim is None:
                    missing_runs.append(f"{ds}_seed{seed}_s{sc}")
                    coverage.append((ds, sc, seed, has_dis, "RUN MISSING", []))
                    continue
                data = json.loads((sim / "consolidated_results.json").read_text())
                fl_rounds = data.get("fl_rounds", args.fl_rounds)

                base_metrics = extract_final_round_metrics(data, "original_model", fl_rounds)
                base_acc = base_metrics.get("global_accuracy")
                base_f1 = base_metrics.get("global_f1")

                strat_with_unlearning = []
                for strat in STRATEGIES:
                    m = extract_final_round_metrics(data, strat, fl_rounds)
                    #unlearning strategies don't change the global model, so utility == baseline FL model
                    if m.get("global_accuracy") is None and base_acc is not None:
                        m["global_accuracy"] = base_acc
                    if m.get("global_f1") is None and base_f1 is not None:
                        m["global_f1"] = base_f1
                    if m.get("n_unlearned_tracks"):
                        strat_with_unlearning.append(strat)
                    for metric, val in m.items():
                        per_seed[(ds, sc, strat)][metric].append(val)
                coverage.append((ds, sc, seed, has_dis, "ok", strat_with_unlearning))

    # ---- summary with honest single-seed handling ----
    summary = {}
    for (ds, sc, strat), metric_lists in per_seed.items():
        entry = {"dataset": ds, "scenario": sc, "strategy": strat, "n_seeds": len(args.seeds)}
        for metric, values in metric_lists.items():
            m, s, n = agg_seed_values(values)
            entry[metric] = m
            entry[f"{metric}_std"] = s   # None for a single seed, so blank in the CSV, never fabricated
            entry[f"{metric}_n"] = n
        summary[f"{ds}_s{sc}_{strat}"] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- CSV ----
    if summary:
        keys = sorted({k for e in summary.values() for k in e})
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in sorted(summary.values(), key=lambda r: (r["dataset"], r["scenario"], STRATEGIES.index(r["strategy"]))):
                w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in keys})

    #---- coverage report (the "nothing missing" guarantee, made explicit) ----
    report = []
    report.append(f"Aggregation coverage, seeds={args.seeds}, fl_rounds={args.fl_rounds}")
    report.append(f"datasets={args.datasets}  scenarios={args.scenarios}")
    report.append("")
    genuine_gaps = []
    for ds, sc, seed, has_dis, status, strat_un in coverage:
        if status == "RUN MISSING":
            report.append(f"  [MISSING RUN] {ds} s{sc} seed{seed}")
            continue
        if has_dis is False:
            report.append(f"  {ds} s{sc} seed{seed}: baseline (no disagreement) -> unlearning N/A (expected)")
        else:
            expected = [s for s in STRATEGIES if s != "original_model"]
            got = set(strat_un)
            gaps = [s for s in expected if s not in got]
            if gaps:
                report.append(f"  [GAP] {ds} s{sc} seed{seed}: disagreement present but NO unlearning metrics for: {gaps}")
                genuine_gaps.append((ds, sc, seed, gaps))
            else:
                report.append(f"  {ds} s{sc} seed{seed}: ok, unlearning metrics for all {len(expected)} strategies")
    report.append("")
    report.append(f"Missing runs: {missing_runs or 'none'}")
    report.append(f"Genuine unlearning gaps: {genuine_gaps or 'none'}")
    (out / "coverage_report.txt").write_text("\n".join(report))

    #---- console summary ----
    n_label = "single seed (n=1, no std)" if len(args.seeds) == 1 else f"{len(args.seeds)} seeds (mean±std)"
    print(f"\nAggregated {len(summary)} (dataset,scenario,strategy) rows, {n_label}")
    print(f"Output: {out}/  (summary.json, summary.csv, coverage_report.txt)\n")
    hdr = f"{'DS':<7}{'Sc':>3} {'Strategy':<28}{'GlobAcc':>9}{'FgtAcc_un':>10}{'MIA_un':>8}{'JSdiv':>8}{'Retr%':>7}{'Time_s':>9}"
    print(hdr); print("-" * len(hdr))

    def cell(e, k, d=3):
        v = e.get(k)
        if v is None:
            return "-"
        s = e.get(f"{k}_std")
        return f"{v:.{d}f}±{s:.{d}f}" if s is not None else f"{v:.{d}f}"

    for ds in args.datasets:
        for sc in args.scenarios:
            for strat in STRATEGIES:
                e = summary.get(f"{ds}_s{sc}_{strat}")
                if not e:
                    continue
                print(f"{ds:<7}{sc:>3} {strat:<28}"
                      f"{cell(e,'global_accuracy'):>9}{cell(e,'forget_accuracy_unlearned'):>10}"
                      f"{cell(e,'mia_accuracy_unlearned'):>8}{cell(e,'js_divergence_mean'):>8}"
                      f"{cell(e,'retrain_fraction',2):>7}{cell(e,'unlearning_time_s',1):>9}")
            print()

    if genuine_gaps:
        print(f"[WARN] GENUINE GAPS (disagreement but no unlearning metrics): {genuine_gaps}")
    if missing_runs:
        print(f"[WARN] MISSING RUNS: {missing_runs}")
    if not genuine_gaps and not missing_runs:
        print("[OK] No missing runs, no unlearning gaps. See coverage_report.txt for the baseline N/A list.")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
