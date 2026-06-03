#!/usr/bin/env python3
"""CLI tool for listing, viewing, and comparing FL experiment runs."""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
sys.path.insert(0, parent_dir)

from scripts.results_index import load_index, rebuild_index, query_runs


def cmd_list(args):
    """List all runs with optional filters."""
    entries = query_runs(
        experiment=args.experiment,
        scenario=args.scenario,
        last_n=args.last,
    )

    if not entries:
        print("No runs found. Try: python scripts/fl_runs.py rebuild-index")
        return

    # Print table
    header = f"{'ID':>5}  {'Date':10}  {'Experiment':12}  {'Scenario':>8}  {'Clients':>7}  {'Rounds':>6}  {'Unlearn':>7}"
    print(header)
    print("-" * len(header))

    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime("%Y-%m-%d")
            except ValueError:
                date_str = ts[:10]
        else:
            date_str = ""

        scenario = e.get("scenario")
        scenario_str = f"S{scenario}" if scenario is not None else ""
        unlearn = "yes" if e.get("unlearning_enabled") else "no"

        print(
            f"{e['id']:>5}  {date_str:10}  {e.get('experiment_type', ''):12}  "
            f"{scenario_str:>8}  {e.get('num_clients', ''):>7}  "
            f"{e.get('fl_rounds', ''):>6}  {unlearn:>7}"
        )

    print(f"\n{len(entries)} run(s) shown")


def cmd_show(args):
    """Show details for a specific run."""
    entries = load_index()
    entry = next((e for e in entries if e["id"] == args.id), None)

    if not entry:
        print(f"Run #{args.id} not found. Available IDs: {[e['id'] for e in entries[:10]]}...")
        return

    print(f"Run #{entry['id']}")
    print(f"  Directory:   {entry['dir']}")
    print(f"  Timestamp:   {entry.get('timestamp', 'N/A')}")
    print(f"  Experiment:  {entry.get('experiment_type', 'N/A')}")
    print(f"  Scenario:    {entry.get('scenario', 'N/A')}")
    print(f"  Clients:     {entry.get('num_clients', 'N/A')}")
    print(f"  FL Rounds:   {entry.get('fl_rounds', 'N/A')}")
    print(f"  Unlearning:  {'enabled' if entry.get('unlearning_enabled') else 'disabled'}")

    #Try to load run_metadata.json for more details
    metadata_path = os.path.join(entry["dir"], "run_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n  Training config:")
        train = metadata.get("training", {})
        print(f"    Learning rate: {train.get('learning_rate', 'N/A')}")
        print(f"    Batch size:    {train.get('batch_size', 'N/A')}")
        print(f"    Local epochs:  {train.get('local_epochs', 'N/A')}")

        unl = metadata.get("unlearning", {})
        if unl.get("enabled"):
            print(f"\n  Unlearning config:")
            print(f"    Strategies:    {', '.join(unl.get('strategies', []))}")

    #Try to load consolidated_results.json for key metrics
    consolidated_path = os.path.join(entry["dir"], "consolidated_results.json")
    if os.path.exists(consolidated_path):
        with open(consolidated_path, 'r') as f:
            results = json.load(f)
        strategies = results.get("strategies", {})
        if strategies:
            print(f"\n  Results ({len(strategies)} strategy/ies):")
            for strat_name, strat_data in strategies.items():
                rounds = strat_data.get("rounds", {})
                if rounds:
                    last_round = max(rounds.keys(), key=int)
                    global_metrics = rounds[last_round].get("global", {})
                    acc = global_metrics.get("accuracy")
                    loss = global_metrics.get("test_loss") or global_metrics.get("rmse")
                    metric_str = ""
                    if acc is not None:
                        metric_str += f"acc={acc:.4f}"
                    if loss is not None:
                        if metric_str:
                            metric_str += ", "
                        metric_str += f"loss={loss:.4f}"
                    print(f"    {strat_name}: {metric_str}")


def cmd_compare(args):
    """Compare runs by ID or by scenario."""
    entries = load_index()

    if args.scenario is not None:
        # Compare all runs for a given scenario
        run_entries = [e for e in entries if str(e.get("scenario", "")) == str(args.scenario)]
        if not run_entries:
            print(f"No runs found for scenario {args.scenario}")
            return
        run_ids = [e["id"] for e in run_entries]
        print(f"Comparing {len(run_entries)} run(s) for scenario {args.scenario}: {run_ids}")
    else:
        run_ids = args.ids
        run_entries = [e for e in entries if e["id"] in run_ids]

    if len(run_entries) < 1:
        print("No matching runs found in index.")
        return

    # Resolve paths
    run_paths = []
    for e in run_entries:
        run_dir = e["dir"]
        if not os.path.isdir(run_dir):
            print(f"Warning: Directory not found for run #{e['id']}: {run_dir}")
            continue

        # The comparator expects dirs with output/fl_results.json.
        #With unlearning, the actual sub-runs are in baseline/ and strategy_*/.
        #Check if this is a multi-strategy run.
        baseline_dir = os.path.join(run_dir, "baseline")
        if os.path.isdir(baseline_dir):
            # Multi-strategy run, add each subdirectory
            for subdir in sorted(Path(run_dir).iterdir()):
                if subdir.is_dir() and (subdir.name == "baseline" or subdir.name.startswith("strategy_")):
                    run_paths.append(str(subdir))
        else:
            run_paths.append(run_dir)

    if not run_paths:
        print("No valid run directories found.")
        return

    # Delegate to FLRunComparator
    try:
        from scripts.compare_fl_runs import FLRunComparator
    except ImportError:
        print("Error: Could not import FLRunComparator from compare_fl_runs.py")
        return

    comparator = FLRunComparator()
    for path in run_paths:
        comparator.load_run(path)

    comparator.print_summary()

    if not args.no_plots:
        id_str = "_".join(str(i) for i in sorted(set(e["id"] for e in run_entries)))
        output_dir = f"results/comparisons/compare_{id_str}"
        print(f"\nGenerating comparison plots...")
        comparator.compare_performance(save_plots=True, output_dir=output_dir)
        comparator.compare_timing(save_plots=True, output_dir=output_dir)
        comparator.compare_round_progression(save_plots=True, output_dir=output_dir)
        comparator.compare_combined_metrics(save_plots=True, output_dir=output_dir)
        comparator.compare_storage_and_time(save_plots=True, output_dir=output_dir)
        print(f"Plots saved to {output_dir}/")


def cmd_rebuild_index(args):
    """Rebuild the index from existing result directories."""
    entries = rebuild_index()
    print(f"Index rebuilt: {len(entries)} run(s) found")


def main():
    parser = argparse.ArgumentParser(
        description="FL experiment run manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/fl_runs.py list --experiment mnist --last 5
  python scripts/fl_runs.py show 157
  python scripts/fl_runs.py compare 157 161
  python scripts/fl_runs.py compare --scenario 1
  python scripts/fl_runs.py rebuild-index
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List experiment runs")
    p_list.add_argument("--experiment", "-e", type=str, help="Filter by experiment type")
    p_list.add_argument("--scenario", "-s", type=int, help="Filter by scenario number")
    p_list.add_argument("--last", "-n", type=int, help="Show only last N runs")

    #show
    p_show = subparsers.add_parser("show", help="Show details of a run")
    p_show.add_argument("id", type=int, help="Run ID")

    #compare
    p_compare = subparsers.add_parser("compare", help="Compare runs")
    p_compare.add_argument("ids", type=int, nargs="*", help="Run IDs to compare")
    p_compare.add_argument("--scenario", "-s", type=int, default=None,
                           help="Compare all runs of this scenario")
    p_compare.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    # rebuild-index
    subparsers.add_parser("rebuild-index", help="Rebuild index from existing results")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "compare":
        if not args.ids and args.scenario is None:
            print("Error: provide run IDs or --scenario")
            sys.exit(1)
        cmd_compare(args)
    elif args.command == "rebuild-index":
        cmd_rebuild_index(args)


if __name__ == "__main__":
    main()
