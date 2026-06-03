#!/usr/bin/env python3
"""
Aggregate and Compare Grid Experiment Results

This script collects results from grid_run.py experiments and:
1. Extracts metrics from all runs
2. Compares performance across models and unlearning strategies
3. Identifies best combinations per scenario
4. Generates CSV files and summary reports
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_metrics_from_results_dir(results_dir: str) -> dict:
    """Load metrics from a results directory.
    
    Args:
        results_dir: Path to results directory (e.g., fl_simulation_*)
    
    Returns:
        Dictionary with metrics or None if not found
    """
    if not results_dir or not os.path.exists(results_dir):
        return None
    
    metrics = {}
    
    # Load fl_results.json
    fl_results_path = os.path.join(results_dir, "output", "fl_results.json")
    if os.path.exists(fl_results_path):
        with open(fl_results_path, 'r') as f:
            fl_results = json.load(f)
            metrics["fl_results"] = fl_results
    
    # Load timing metrics
    timing_path = os.path.join(results_dir, "output", "timing_metrics.json")
    if os.path.exists(timing_path):
        with open(timing_path, 'r') as f:
            timing = json.load(f)
            metrics["timing"] = timing
    
    # Load unlearning branch metrics
    unlearning_metrics = {}
    model_storage = os.path.join(results_dir, "model_storage")
    if os.path.exists(model_storage):
        #Find all unlearning branches
        for round_dir in glob.glob(os.path.join(model_storage, "round_*")):
            unlearning_dir = os.path.join(round_dir, "unlearning")
            if os.path.exists(unlearning_dir):
                #Load comparison
                comparison_path = os.path.join(unlearning_dir, "comparison.json")
                if os.path.exists(comparison_path):
                    with open(comparison_path, 'r') as f:
                        comparison = json.load(f)
                        round_num = int(os.path.basename(round_dir).split("_")[1])
                        unlearning_metrics[f"round_{round_num}"] = comparison
                
                # Load individual branch metrics
                branches_dir = os.path.join(unlearning_dir, "branches")
                if os.path.exists(branches_dir):
                    for branch_dir in glob.glob(os.path.join(branches_dir, "*")):
                        if os.path.isdir(branch_dir):
                            branch_name = os.path.basename(branch_dir)
                            metrics_path = os.path.join(branch_dir, "metrics.json")
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r') as f:
                                    branch_metrics = json.load(f)
                                    round_num = int(os.path.basename(round_dir).split("_")[1])
                                    key = f"round_{round_num}_branch_{branch_name}"
                                    unlearning_metrics[key] = branch_metrics
    
    if unlearning_metrics:
        metrics["unlearning"] = unlearning_metrics
    
    return metrics if metrics else None


def extract_performance_metrics(metrics: dict, experiment_type: str) -> dict:
    """Extract performance metrics from loaded metrics.
    
    Returns:
        Dictionary with extracted metrics
    """
    extracted = {
        "test_accuracy": None,
        "test_loss": None,
        "test_rmse": None,
        "test_mae": None,
        "test_r2": None,
        "unlearning_time_s": None,
        "total_time_s": None
    }
    
    # Extract from fl_results
    if "fl_results" in metrics:
        fl_results = metrics["fl_results"]
        
        # Get last round results
        if "rounds" in fl_results:
            rounds = fl_results["rounds"]
            if rounds:
                last_round = rounds[-1]
                
                try:
                    from fl_module.registry import DatasetAdapterRegistry
                    adapter = DatasetAdapterRegistry.get_adapter(experiment_type)
                    is_cls = adapter.is_classification() if adapter is not None else (experiment_type == "mnist" or experiment_type == "cifar10" or experiment_type == "tabular")
                except Exception:
                    is_cls = experiment_type in ("mnist", "cifar10", "tabular")
                if is_cls:
                    if "global_test_accuracy" in last_round:
                        extracted["test_accuracy"] = last_round["global_test_accuracy"]
                    if "global_test_loss" in last_round:
                        extracted["test_loss"] = last_round["global_test_loss"]
                else:
                    if "global_test_rmse" in last_round:
                        extracted["test_rmse"] = last_round["global_test_rmse"]
                    if "global_test_mae" in last_round:
                        extracted["test_mae"] = last_round["global_test_mae"]
                    if "global_test_r2" in last_round:
                        extracted["test_r2"] = last_round["global_test_r2"]
    
    #Extract unlearning time from unlearning metrics
    if "unlearning" in metrics:
        unlearning = metrics["unlearning"]
        times = []
        for key, value in unlearning.items():
            if isinstance(value, dict) and "unlearning_time_s" in value:
                times.append(value["unlearning_time_s"])
        if times:
            extracted["unlearning_time_s"] = sum(times)  #Total unlearning time
    
    # Extract total time from timing metrics
    if "timing" in metrics:
        timing = metrics["timing"]
        if "total_running_time_seconds" in timing:
            extracted["total_time_s"] = timing["total_running_time_seconds"]
    
    return extracted


def aggregate_grid_results(grid_dir: str) -> pd.DataFrame:
    """Aggregate all results from grid experiments.
    
    Args:
        grid_dir: Directory containing grid experiment results
    
    Returns:
        DataFrame with all results
    """
    rows = []
    
    # Find all meta.json files
    meta_files = glob.glob(os.path.join(grid_dir, "**", "meta.json"), recursive=True)
    
    print(f"Found {len(meta_files)} experiment runs")
    
    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Skip if not successful
            if meta.get("status") != "success":
                continue
            
            #Load metrics from results directory
            results_dir = meta.get("results_dir")
            if not results_dir or not os.path.exists(results_dir):
                continue
            
            metrics = load_metrics_from_results_dir(results_dir)
            if not metrics:
                continue
            
            #Extract performance metrics
            experiment_type = meta.get("experiment_type", "mnist")
            perf_metrics = extract_performance_metrics(metrics, experiment_type)
            
            # Create row
            row = {
                "scenario": meta.get("scenario"),
                "experiment_type": experiment_type,
                "model_type": meta.get("model_type"),
                "unlearning_strategy": meta.get("unlearning_strategy"),
                "fl_rounds": meta.get("fl_rounds"),
                "local_epochs": meta.get("local_epochs"),
                "num_clients": meta.get("num_clients"),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "tag": meta.get("tag"),
                "results_dir": results_dir,
                **perf_metrics
            }
            
            rows.append(row)
        
        except Exception as e:
            print(f"Warning: Could not process {meta_path}: {e}")
            continue
    
    if not rows:
        print("No valid results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    return df


def calculate_scores(df: pd.DataFrame, experiment_type: str) -> pd.DataFrame:
    """Calculate composite scores for ranking.
    
    Score function: lower is better
    - For MNIST: test_loss + 0.01 * unlearning_time_s
    - For N-CMAPSS: test_rmse + 0.01 * unlearning_time_s
    
    Args:
        df: DataFrame with results
        experiment_type: 'mnist' or 'n_cmapss'
    
    Returns:
        DataFrame with added 'score' column
    """
    df = df.copy()
    
    try:
        from fl_module.registry import DatasetAdapterRegistry
        adapter = DatasetAdapterRegistry.get_adapter(experiment_type)
        is_cls = adapter.is_classification() if adapter is not None else (experiment_type in ("mnist", "cifar10", "tabular"))
    except Exception:
        is_cls = experiment_type in ("mnist", "cifar10", "tabular")
    if is_cls:
        if "test_loss" in df.columns and df["test_loss"].notna().any():
            primary_metric = df["test_loss"].fillna(1.0)
        elif "test_accuracy" in df.columns and df["test_accuracy"].notna().any():
            primary_metric = 1.0 - df["test_accuracy"].fillna(0.0)
        else:
            primary_metric = pd.Series([1.0] * len(df))
    else:
        if "test_rmse" in df.columns and df["test_rmse"].notna().any():
            primary_metric = df["test_rmse"].fillna(float('inf'))
        else:
            primary_metric = pd.Series([float('inf')] * len(df))
    
    # Unlearning time (normalize to seconds, default to 0 if missing)
    unlearning_time = df.get("unlearning_time_s", pd.Series([0.0] * len(df))).fillna(0.0)
    
    # Calculate score: primary_metric + 0.01 * unlearning_time
    df["score"] = primary_metric + 0.01 * unlearning_time
    
    return df


def find_best_per_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Find best combination per scenario.
    
    Args:
        df: DataFrame with results and scores
    
    Returns:
        DataFrame with best result per scenario
    """
    if "score" not in df.columns:
        print("Warning: No score column found. Calculating scores...")
        experiment_type = df["experiment_type"].iloc[0] if len(df) > 0 else "mnist"
        df = calculate_scores(df, experiment_type)
    
    #Group by scenario and find best (lowest score)
    best_per_scenario = df.loc[df.groupby("scenario")["score"].idxmin()].copy()
    
    #Sort by scenario
    best_per_scenario = best_per_scenario.sort_values("scenario")
    
    return best_per_scenario


def generate_summary_report(df: pd.DataFrame, best_df: pd.DataFrame, output_dir: str):
    """Generate summary report.
    
    Args:
        df: Full results DataFrame
        best_df: Best results per scenario DataFrame
        output_dir: Output directory for reports
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall statistics
    stats = {
        "total_runs": len(df),
        "scenarios_covered": df["scenario"].nunique(),
        "models_tested": df["model_type"].nunique(),
        "strategies_tested": df["unlearning_strategy"].nunique(),
    }
    
    # Model performance summary
    if "test_accuracy" in df.columns:
        model_summary = df.groupby("model_type")["test_accuracy"].agg(["mean", "std", "count"])
    elif "test_rmse" in df.columns:
        model_summary = df.groupby("model_type")["test_rmse"].agg(["mean", "std", "count"])
    else:
        model_summary = pd.DataFrame()
    
    # Strategy performance summary
    if "test_accuracy" in df.columns:
        strategy_summary = df.groupby("unlearning_strategy")["test_accuracy"].agg(["mean", "std", "count"])
    elif "test_rmse" in df.columns:
        strategy_summary = df.groupby("unlearning_strategy")["test_rmse"].agg(["mean", "std", "count"])
    else:
        strategy_summary = pd.DataFrame()
    
    #Save summary
    summary = {
        "statistics": stats,
        "model_summary": model_summary.to_dict() if not model_summary.empty else {},
        "strategy_summary": strategy_summary.to_dict() if not strategy_summary.empty else {},
        "best_per_scenario_count": best_df["model_type"].value_counts().to_dict(),
        "best_per_scenario_strategy_count": best_df["unlearning_strategy"].value_counts().to_dict()
    }
    
    summary_path = os.path.join(output_dir, "summary_report.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Summary report saved to: {summary_path}")


def main():
    """Main function to aggregate grid results."""
    parser = argparse.ArgumentParser(
        description="Aggregate and compare grid experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate results from default grid directory
  python aggregate_grid_results.py
  
  # Aggregate from specific directory
  python aggregate_grid_results.py -d results/grid/mnist
  
  # Custom output directory
  python aggregate_grid_results.py -d results/grid/mnist -o results/analysis
        """
    )
    
    parser.add_argument(
        "-d", "--grid-dir",
        type=str,
        default="results/grid",
        help="Grid experiment results directory (default: results/grid)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: same as grid-dir)"
    )
    
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        choices=["mnist", "n_cmapss"],
        default=None,
        help="Filter by experiment type (default: all)"
    )
    
    args = parser.parse_args()
    
    #Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    # Determine grid directory
    if args.experiment:
        grid_dir = os.path.join(args.grid_dir, args.experiment)
    else:
        grid_dir = args.grid_dir
    
    if not os.path.exists(grid_dir):
        print(f"Error: Grid directory not found: {grid_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(grid_dir, "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Aggregating Grid Results")
    print("=" * 80)
    print(f"Grid directory: {grid_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Aggregate results
    print("\nLoading results...")
    df = aggregate_grid_results(grid_dir)
    
    if df.empty:
        print("No results found!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} successful runs")
    
    #Filter by experiment type if specified
    if args.experiment:
        df = df[df["experiment_type"] == args.experiment]
        print(f"Filtered to {len(df)} runs for {args.experiment}")
    
    #Calculate scores
    print("\nCalculating scores...")
    experiment_type = df["experiment_type"].iloc[0] if len(df) > 0 else "mnist"
    df = calculate_scores(df, experiment_type)
    
    # Find best per scenario
    print("\nFinding best combinations per scenario...")
    best_df = find_best_per_scenario(df)
    
    # Save CSV files
    print("\nSaving results...")
    
    # Full results
    csv_path = os.path.join(output_dir, "grid_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Full results saved to: {csv_path}")
    
    #Best per scenario
    best_csv_path = os.path.join(output_dir, "winners_per_scenario.csv")
    best_df.to_csv(best_csv_path, index=False)
    print(f"Best per scenario saved to: {best_csv_path}")
    
    #Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, best_df, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total runs: {len(df)}")
    print(f"Scenarios covered: {df['scenario'].nunique()}")
    print(f"Models tested: {', '.join(df['model_type'].unique())}")
    print(f"Strategies tested: {', '.join(df['unlearning_strategy'].unique())}")
    print("\nBest model per scenario:")
    print(best_df[["scenario", "model_type", "unlearning_strategy", "score"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

