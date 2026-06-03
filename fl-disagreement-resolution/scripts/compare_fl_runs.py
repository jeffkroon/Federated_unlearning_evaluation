#!/usr/bin/env python3
"""
Federated Learning Run Comparison Tool

This script compares multiple federated learning simulation runs,
analyzing performance metrics, timing data, and scenario characteristics.
Can handle multiple runs per scenario and average the results.
"""

import json
import os
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def usage():
    """Print usage information."""
    print("""Usage: compare_fl_runs.py <runs...> [options]

Arguments:
  runs                     Paths to FL simulation result directories (multiple runs of the same scenario will be averaged).

Options:
  -o, --output-dir <dir>   Output directory for plots (default: auto-generated timestamped directory).
  --no-plots               Skip generating plots, only show summary.
  --names <names...>       Custom names for the runs (optional).
  -h, --help               Display this help and exit.

Examples:
  compare_fl_runs.py results/fl_simulation_mnist_s1 results/fl_simulation_mnist_s2
  compare_fl_runs.py results/fl_simulation_mnist_s1 results/fl_simulation_mnist_s2 --output-dir comparison_s1_s2
  compare_fl_runs.py results/fl_simulation_mnist_s1 results/fl_simulation_mnist_s2 --no-plots
  compare_fl_runs.py results/fl_simulation_mnist_s1 results/fl_simulation_mnist_s2 --names "Scenario 1" "Scenario 2"

The script groups runs by scenario and averages their metrics.
      If you have multiple runs of the same scenario, they will be averaged together in the comparison charts.
""")


class FLRunComparator:
    def __init__(self):
        self.runs = {}
        self.scenario_runs = defaultdict(list)  # Group runs by scenario
        self.num_runs_per_scenario = {}  # Track number of runs per scenario

    def _get_scenario_description(self, scenario, clients):
        """Generate descriptive scenario labels based on scenario number."""
        if scenario is None or scenario == 'Unknown':
            return f"({clients} clients)"

        scenario_num = int(scenario) if isinstance(scenario, (int, str)) and str(scenario).isdigit() else scenario

        #Handle scenario 7 (10 clients, no exclusion)
        if scenario_num == 7:
            return f"({clients} clients,\nno excl.)"

        #Handle scenarios 8-12 (ring of 10)
        elif 8 <= scenario_num <= 12:
            return f"({clients} clients,\nring of 10)"

        # Handle scenarios 13-19 (6 clients with different exclusion patterns)
        elif 13 <= scenario_num <= 19:
            if scenario_num == 13:
                return "(no excl.)"
            else:
                # S14 = next 1, S15 = next 2, etc.
                next_count = scenario_num - 13
                return f"(next {next_count})"

        # Handle scenarios 20-24 (20 clients with ring patterns)
        elif 20 <= scenario_num <= 24:
            if scenario_num == 20:
                return "(no excl.)"
            elif scenario_num == 24:
                return f"({clients} clients,\nring of 20)"
            else:
                #S21 = ring of 5, S22 = ring of 10, S23 = ring of 15
                ring_size = (scenario_num - 20) * 5
                return f"(ring of {ring_size})"

        #Handle scenarios 25-28 (5 clients with ring patterns)
        elif 25 <= scenario_num <= 28:
            if scenario_num == 25:
                return f"({clients} clients,\nno excl.)"
            elif scenario_num == 26:
                return "(ring of 5,\nnext 1)"
            else:
                # S27 = ring of 10, S28 = ring of 15
                ring_size = (scenario_num - 25) * 5
                return f"({clients} clients,\nring of {ring_size})"

        # Handle scenarios 29-31 (ring patterns with next exclusions)
        elif 29 <= scenario_num <= 31:
            if scenario_num == 29:
                return "(ring of 10,\nnext 2)"
            elif scenario_num == 30:
                return "(ring of 10,\nnext 3)"
            elif scenario_num == 31:
                return "(ring of 10,\nnext 4)"

        # Fallback for other scenarios
        else:
            return f"({clients} clients)"

    def _generate_chart_label(self, scenario, clients):
        """Generate a label for charts (with newline)."""
        desc = self._get_scenario_description(scenario, clients)
        return f"S{scenario}\n{desc}"

    def _generate_legend_label(self, scenario, clients):
        """Generate a label for legends (without newline)."""
        desc = self._get_scenario_description(scenario, clients)
        return f"S{scenario} {desc}"

    def _calculate_directory_size(self, directory_path):
        """Calculate the total size of a directory in MiB.

        Args:
            directory_path: Path to the directory

        Returns:
            float: Size in MiB, or None if directory doesn't exist
        """
        try:
            if not os.path.exists(directory_path):
                return None

            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        #Skip files that can't be accessed
                        continue

            #Convert bytes to MiB
            return total_size / (1024 * 1024)
        except Exception:
            return None

    def load_run(self, run_path, run_name=None):
        """Load a federated learning run from results directory."""
        run_path = Path(run_path)

        if run_name is None:
            run_name = run_path.name

        print(f"Loading run: {run_name}")

        # Load main results
        fl_results_path = run_path / "output" / "fl_results.json"
        timing_metrics_path = run_path / "output" / "timing_metrics.json"

        run_data = {
            "name": run_name,
            "path": str(run_path),
            "loaded_at": datetime.now()
        }

        # Load FL results
        if fl_results_path.exists():
            with open(fl_results_path, 'r') as f:
                fl_results = json.load(f)
                run_data["fl_results"] = fl_results
                run_data["experiment_type"] = fl_results.get("experiment_type", "unknown")
        else:
            print(f"Warning: No fl_results.json found in {run_path}")
            return None

        # Load timing metrics
        if timing_metrics_path.exists():
            with open(timing_metrics_path, 'r') as f:
                timing_data = json.load(f)
                #Handle timing metrics structure
                if isinstance(timing_data, dict) and "aggregation_timing_history" in timing_data:
                    run_data["timing_metrics"] = timing_data["aggregation_timing_history"]
                    run_data["total_running_time"] = timing_data.get("total_running_time_seconds")
                    run_data["round_timing_metrics"] = timing_data.get("round_timing_history", [])
                    run_data["experiment_init_time"] = timing_data.get("experiment_init_time_seconds")
                    run_data["evaluation_timing_metrics"] = timing_data.get("evaluation_timing_history", [])
                else:
                    run_data["timing_metrics"] = timing_data if isinstance(timing_data, list) else []
                    run_data["round_timing_metrics"] = []
                    run_data["experiment_init_time"] = None
                    run_data["evaluation_timing_metrics"] = []
        else:
            print(f"Warning: No timing metrics found in {run_path}")
            run_data["timing_metrics"] = []
            run_data["experiment_init_time"] = None
            run_data["evaluation_timing_metrics"] = []

        #Calculate model storage directory size
        model_storage_path = run_path / "model_storage"
        run_data["model_storage_size_mib"] = self._calculate_directory_size(model_storage_path)

        # Extract scenario info from directory name
        dir_name = run_path.name
        if "_s" in dir_name:
            scenario_part = dir_name.split("_s")[-1]
            scenario_num = ""
            for char in scenario_part:
                if char.isdigit():
                    scenario_num += char
                else:
                    break
            run_data["scenario"] = int(scenario_num) if scenario_num else None
        else:
            run_data["scenario"] = None

        # Determine number of clients from timing data
        if run_data["timing_metrics"]:
            run_data["num_clients"] = run_data["timing_metrics"][0].get("num_clients", "unknown")
        else:
            run_data["num_clients"] = "unknown"

        # Extract basic metrics
        self._extract_summary_metrics(run_data)

        self.runs[run_name] = run_data

        #Group by scenario for averaging
        scenario = run_data.get('scenario')
        if scenario is not None:
            self.scenario_runs[scenario].append(run_data)

        print(f"Loaded run: {run_name} (Scenario {run_data['scenario']}, {run_data['num_clients']} clients)")

        return run_data

    def _extract_summary_metrics(self, run_data):
        """Extract summary metrics from the run data."""
        fl_results = run_data.get("fl_results", {})
        timing_metrics = run_data.get("timing_metrics", [])

        #Extract total running time from fl_results (fallback if not in timing metrics)
        if "total_running_time" not in run_data:
            run_data["total_running_time"] = fl_results.get("total_running_time")

        # Performance metrics
        rounds_data = fl_results.get("rounds", [])
        if rounds_data:
            final_round = rounds_data[-1]
            run_data["final_accuracy"] = final_round.get("test_accuracy")
            run_data["final_loss"] = final_round.get("test_loss")
            run_data["final_precision"] = final_round.get("mean_precision")
            run_data["final_recall"] = final_round.get("mean_recall")
            run_data["final_f1"] = final_round.get("mean_f1")
            run_data["total_rounds"] = len([r for r in rounds_data if r["round"] > 0])

            # Calculate average track performance for each round
            experiment_type = run_data.get("experiment_type", "mnist")
            run_data["avg_track_performance"] = self._calculate_average_track_performance(rounds_data, experiment_type)

            # Calculate final average track performance metrics
            if run_data["avg_track_performance"]:
                final_round_num = max(run_data["avg_track_performance"].keys())
                run_data["final_avg_track_accuracy"] = run_data["avg_track_performance"][final_round_num]

                #Also calculate average track metrics from the final round's track results
                final_round = rounds_data[-1]
                track_results = final_round.get("track_results", {})

                if track_results:
                    #Collect all track metrics (including global)
                    track_precisions = []
                    track_recalls = []
                    track_f1s = []

                    # Add global metrics
                    if final_round.get("mean_precision") is not None:
                        track_precisions.append(final_round.get("mean_precision"))
                    if final_round.get("mean_recall") is not None:
                        track_recalls.append(final_round.get("mean_recall"))
                    if final_round.get("mean_f1") is not None:
                        track_f1s.append(final_round.get("mean_f1"))

                    # Add track-specific metrics
                    for track_name, track_data in track_results.items():
                        if track_data.get("precision") is not None:
                            track_precisions.append(track_data.get("precision"))
                        if track_data.get("recall") is not None:
                            track_recalls.append(track_data.get("recall"))
                        if track_data.get("f1") is not None:
                            track_f1s.append(track_data.get("f1"))

                    # Calculate averages
                    run_data["final_avg_track_precision"] = np.mean(track_precisions) if track_precisions else None
                    run_data["final_avg_track_recall"] = np.mean(track_recalls) if track_recalls else None
                    run_data["final_avg_track_f1"] = np.mean(track_f1s) if track_f1s else None

        #Timing metrics summary
        if timing_metrics:
            total_times = [entry["total_aggregation_time_seconds"] for entry in timing_metrics]
            aggregation_times = [entry["aggregation_time_seconds"] for entry in timing_metrics]
            resolution_times = [entry["resolution_time_seconds"] * 1000 for entry in timing_metrics]  #in ms
            has_disagreements = [entry["has_disagreements"] for entry in timing_metrics]

            run_data["avg_total_time"] = np.mean(total_times)
            run_data["avg_aggregation_time"] = np.mean(aggregation_times)
            run_data["avg_resolution_time_ms"] = np.mean(resolution_times)
            run_data["rounds_with_disagreements"] = sum(has_disagreements)
            run_data["total_timing_rounds"] = len(timing_metrics)

            # Calculate overhead
            with_disag_times = [t for t, has_disag in zip(total_times, has_disagreements) if has_disag]
            without_disag_times = [t for t, has_disag in zip(total_times, has_disagreements) if not has_disag]

            if with_disag_times and without_disag_times:
                avg_with = np.mean(with_disag_times)
                avg_without = np.mean(without_disag_times)
                run_data["disagreement_overhead_pct"] = ((avg_with - avg_without) / avg_without) * 100
            else:
                run_data["disagreement_overhead_pct"] = None

        # Round timing metrics summary
        round_timing_metrics = run_data.get("round_timing_metrics", [])
        if round_timing_metrics:
            track_init_times = [entry["track_model_initialization_time_seconds"] for entry in round_timing_metrics]
            client_training_times = [entry["total_client_training_time_seconds"] for entry in round_timing_metrics]
            total_round_times = [entry["total_round_time_seconds"] for entry in round_timing_metrics]

            # Extract aggregation timing data from round timing
            aggregation_times = [entry.get("aggregation_time_seconds", 0) for entry in round_timing_metrics]
            resolution_times = [entry.get("resolution_time_seconds", 0) for entry in round_timing_metrics]
            total_aggregation_times = [entry.get("total_aggregation_time_seconds", 0) for entry in round_timing_metrics]

            #Extract evaluation timing data from round timing
            evaluation_times = [entry.get("evaluation_time_seconds", 0) for entry in round_timing_metrics]

            run_data["avg_track_init_time"] = np.mean(track_init_times)
            run_data["avg_client_training_time"] = np.mean(client_training_times)
            run_data["avg_total_round_time"] = np.mean(total_round_times)
            run_data["avg_round_aggregation_time"] = np.mean(aggregation_times)
            run_data["avg_round_resolution_time"] = np.mean(resolution_times)
            run_data["avg_round_total_aggregation_time"] = np.mean(total_aggregation_times)
            run_data["avg_evaluation_time"] = np.mean(evaluation_times)

            #Calculate individual client training statistics
            all_client_times = []
            for round_entry in round_timing_metrics:
                client_times = round_entry.get("client_training_times", {})
                for client_id, client_data in client_times.items():
                    all_client_times.append(client_data.get("training_time_seconds", 0))

            if all_client_times:
                run_data["avg_individual_client_training_time"] = np.mean(all_client_times)
                run_data["max_individual_client_training_time"] = np.max(all_client_times)
                run_data["min_individual_client_training_time"] = np.min(all_client_times)

    def _calculate_average_track_performance(self, rounds_data, experiment_type=None):
        """Calculate average performance across all tracks (including global) for each round.

        Args:
            rounds_data: List of round data from fl_results
            experiment_type: Type of experiment ('mnist' or 'n_cmapss')

        Returns:
            dict: Dictionary with round numbers as keys and average performance as values
        """
        avg_performance = {}

        for round_data in rounds_data:
            if round_data["round"] <= 0:
                continue

            round_num = round_data["round"]
            track_results = round_data.get("track_results", {})

            # Collect all track performances (including global)
            performances = []

            # Handle different experiment types
            if experiment_type == "n_cmapss":
                # For N-CMAPSS, use test_loss (RMSE) instead of accuracy
                global_rmse = round_data.get("test_loss")
                if global_rmse is not None:
                    performances.append(global_rmse)

                #Add track-specific RMSE
                for track_name, track_data in track_results.items():
                    track_rmse = track_data.get("rmse")
                    if track_rmse is not None:
                        performances.append(track_rmse)

                #Calculate average if we have any performances
                if performances:
                    avg_performance[round_num] = np.mean(performances)
                else:
                    # Fallback to global performance if no track results
                    avg_performance[round_num] = global_rmse
            else:
                # Default to MNIST behavior (accuracy)
                global_accuracy = round_data.get("test_accuracy")
                if global_accuracy is not None:
                    performances.append(global_accuracy)

                # Add track-specific performances
                for track_name, track_data in track_results.items():
                    track_accuracy = track_data.get("accuracy")
                    if track_accuracy is not None:
                        performances.append(track_accuracy)

                #Calculate average if we have any performances
                if performances:
                    avg_performance[round_num] = np.mean(performances)
                else:
                    #Fallback to global performance if no track results
                    avg_performance[round_num] = global_accuracy

        return avg_performance

    def compare_performance(self, save_plots=True, output_dir=None):
        """Compare performance metrics across runs."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        # Performance comparison
        plt.figure(figsize=(15, 10))

        # Prepare data for comparison
        metrics = ['final_avg_track_accuracy', 'final_avg_track_precision', 'final_avg_track_recall', 'final_avg_track_f1']
        metric_titles = ['Final Avg Track Accuracy', 'Final Avg Track Precision', 'Final Avg Track Recall', 'Final Avg Track F1 Score']
        fallback_metrics = ['final_accuracy', 'final_precision', 'final_recall', 'final_f1']

        for i, (metric, title, fallback) in enumerate(zip(metrics, metric_titles, fallback_metrics)):
            plt.subplot(2, 2, i+1)

            values = []
            errors = []
            labels = []
            colors = []

            for scenario, run_data in averaged_data.items():
                #Try to get average track metric first, otherwise fallback to global metric
                value = run_data.get(metric)
                error = run_data.get(f'{metric}_std', 0)
                if value is None:
                    value = run_data.get(fallback)
                    error = run_data.get(f'{fallback}_std', 0)

                if value is not None:
                    values.append(value)
                    errors.append(error)
                    clients = run_data.get('num_clients', 'Unknown')
                    labels.append(self._generate_chart_label(scenario, clients))

                    #Color by scenario
                    if scenario is not None:
                        colors.append(plt.cm.Set1(scenario % 10))
                    else:
                        colors.append('gray')

            if values:
                bars = plt.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7,
                             capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
                plt.xticks(range(len(values)), labels, rotation=45, ha='center')
                plt.tick_params(axis='x', pad=1)
                plt.ylabel(title)
                plt.title(f'{title} Comparison\n(across {max_runs} runs)')
                plt.grid(True, axis='y', alpha=0.3)

                # Add value labels on bars (above error bars)
                for bar, value, error in zip(bars, values, errors):
                    height = bar.get_height()
                    # Position label above error bar if error exists, otherwise above bar
                    label_y = height + error if error > 0 else height
                    plt.annotate(f'{value:.4f}',
                                xy=(bar.get_x() + bar.get_width() / 2, label_y),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"Saved performance comparison to {output_dir}/performance_comparison.png")
        plt.show()

    def compare_timing(self, save_plots=True, output_dir=None):
        """Compare timing metrics across runs."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        plt.figure(figsize=(18, 6))

        #Timing metrics comparison
        timing_metrics = ['avg_total_time', 'avg_resolution_time_ms', 'avg_aggregation_time']
        timing_titles = ['Avg Total Time (s)', 'Avg Resolution Time (ms)', 'Avg Aggregation Time (s)']

        for i, (metric, title) in enumerate(zip(timing_metrics, timing_titles)):
            plt.subplot(1, 3, i+1)

            values = []
            errors = []
            labels = []

            for scenario, run_data in averaged_data.items():
                value = run_data.get(metric)
                error = run_data.get(f'{metric}_std', 0)
                if value is not None:
                    values.append(value)
                    errors.append(error)
                    clients = run_data.get('num_clients', 'Unknown')
                    labels.append(self._generate_chart_label(scenario, clients))

            if values:
                bars = plt.bar(range(len(values)), values, yerr=errors, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1,
                             capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
                plt.xticks(range(len(values)), labels, rotation=45, ha='center')
                plt.tick_params(axis='x', pad=1)
                plt.ylabel(title)
                plt.title(f'{title} Comparison\n(across {max_runs} runs)')
                plt.grid(True, axis='y', alpha=0.3)

                #Add value labels on bars (above error bars)
                for bar, value, error in zip(bars, values, errors):
                    height = bar.get_height()
                    # Position label above error bar if error exists, otherwise above bar
                    label_y = height + error if error > 0 else height
                    if metric == 'avg_resolution_time_ms':
                        plt.annotate(f'{value:.3f}ms',
                                    xy=(bar.get_x() + bar.get_width() / 2, label_y),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
                    else:
                        plt.annotate(f'{value:.3f}s',
                                    xy=(bar.get_x() + bar.get_width() / 2, label_y),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'timing_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"Saved timing comparison to {output_dir}/timing_comparison.png")
        plt.show()

    def compare_round_progression(self, save_plots=True, output_dir=None):
        """Compare average track accuracy progression across rounds."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        # Detect experiment type from the first scenario
        first_scenario_data = next(iter(averaged_data.values())) if averaged_data else {}
        experiment_type = first_scenario_data.get("experiment_type", "mnist")

        plt.figure(figsize=(12, 8))

        for scenario, run_data in averaged_data.items():
            avg_track_performance = run_data.get("avg_track_performance", {})
            std_track_performance = run_data.get("std_track_performance", {})

            if avg_track_performance:
                rounds = list(avg_track_performance.keys())
                values = list(avg_track_performance.values())
                errors = [std_track_performance.get(round_num, 0) for round_num in rounds]

                clients = run_data.get('num_clients', 'Unknown')
                label = self._generate_legend_label(scenario, clients)

                plt.errorbar(rounds, values, yerr=errors, marker='o', linewidth=2, markersize=6,
                           label=label, capsize=4, capthick=1)

        plt.xlabel('Round')

        if experiment_type == "n_cmapss":
            plt.ylabel('Average Track RMSE')
            plt.title(f'Average Track RMSE Progression Across Rounds\n(across {max_runs} runs)')
        else:
            plt.ylabel('Average Track Accuracy')
            plt.title(f'Average Track Accuracy Progression Across Rounds\n(across {max_runs} runs)')

        plt.grid(True, alpha=0.3)
        plt.legend()

        #Set x-axis to show only whole numbers
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if save_plots:
            plt.savefig(os.path.join(output_dir, 'accuracy_progression.png'),
                       bbox_inches='tight', dpi=150)
            print(f"Saved accuracy progression to {output_dir}/accuracy_progression.png")
        plt.show()

    def compare_combined_metrics(self, save_plots=True, output_dir=None):
        """Create a comprehensive 4x2 grid comparison with all timing metrics and performance."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        #Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        # Detect experiment type
        first_scenario_data = next(iter(averaged_data.values())) if averaged_data else {}
        experiment_type = first_scenario_data.get("experiment_type", "mnist")

        # Create 4x2 grid
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Scalability comparison (across {max_runs} runs)', fontsize=16, fontweight='bold')

        # Define custom color palette
        custom_colors = ['#E91E63', '#00BCD4', '#4CAF50', '#3F51B5', '#607D8B', '#F44336', '#2E7D32', '#1976D2', '#AD1457', '#00695C']
        #Colors: pink, cyan, green, indigo, blue-gray, red, dark green, blue, deep pink, teal

        #Prepare common data for all plots
        scenarios = []
        labels = []
        colors = []

        for i, (scenario, run_data) in enumerate(averaged_data.items()):
            scenarios.append(scenario)
            clients = run_data.get('num_clients', 'Unknown')
            labels.append(self._generate_chart_label(scenario, clients))
            # Use custom color palette
            colors.append(custom_colors[i % len(custom_colors)])

        # Top row plots
        # 1. Average Resolution Time
        ax = axes[0, 0]
        values = [averaged_data[s].get('avg_round_resolution_time', 0) * 1000 for s in scenarios]  #Convert to ms
        errors = [averaged_data[s].get('avg_round_resolution_time_std', 0) * 1000 for s in scenarios]  #Convert to ms
        if any(v > 0 for v in values):
            bars = ax.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
            ax.tick_params(axis='x', pad=2)
            ax.set_ylabel('Time (ms)')
            ax.set_title('Avg Resolution Time')
            ax.grid(True, axis='y', alpha=0.3)
            # Add value labels (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                # Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                ax.annotate(f'{value:.2f}ms', xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 2. Average Track Model Initialization Time
        ax = axes[0, 1]
        values = [averaged_data[s].get('avg_track_init_time', 0) * 1000 for s in scenarios]  #Convert to ms
        errors = [averaged_data[s].get('avg_track_init_time_std', 0) * 1000 for s in scenarios]  #Convert to ms
        if any(v > 0 for v in values):
            bars = ax.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
            ax.tick_params(axis='x', pad=2)
            ax.set_ylabel('Time (ms)')
            ax.set_title('Avg Track Model Init Time')
            ax.grid(True, axis='y', alpha=0.3)
            # Add value labels (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                # Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                ax.annotate(f'{value:.1f}ms', xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 3. Average Client Training Time
        ax = axes[0, 2]
        values = [averaged_data[s].get('avg_client_training_time', 0) for s in scenarios]
        errors = [averaged_data[s].get('avg_client_training_time_std', 0) for s in scenarios]
        if any(v > 0 for v in values):
            bars = ax.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
            ax.tick_params(axis='x', pad=2)
            ax.set_ylabel('Time (s)')
            ax.set_title('Avg Client Training Time')
            ax.grid(True, axis='y', alpha=0.3)
            #Add value labels (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                #Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                ax.annotate(f'{value:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 4. Average Aggregation Time
        ax = axes[0, 3]
        values = [averaged_data[s].get('avg_round_aggregation_time', 0) for s in scenarios]
        errors = [averaged_data[s].get('avg_round_aggregation_time_std', 0) for s in scenarios]
        if any(v > 0 for v in values):
            bars = ax.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
            ax.tick_params(axis='x', pad=2)
            ax.set_ylabel('Time (s)')
            ax.set_title('Avg Aggregation Time')
            ax.grid(True, axis='y', alpha=0.3)
            # Add value labels (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                # Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                ax.annotate(f'{value:.3f}s', xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        #Bottom row plots
        #5. Accuracy Progression
        ax = axes[1, 0]
        for i, scenario in enumerate(scenarios):
            run_data = averaged_data[scenario]
            avg_track_performance = run_data.get("avg_track_performance", {})
            std_track_performance = run_data.get("std_track_performance", {})

            if avg_track_performance:
                rounds = list(avg_track_performance.keys())
                values = list(avg_track_performance.values())
                errors = [std_track_performance.get(round_num, 0) for round_num in rounds]

                clients = run_data.get('num_clients', 'Unknown')
                label = self._generate_legend_label(scenario, clients)

                ax.errorbar(rounds, values, yerr=errors, marker='o', linewidth=2, markersize=4,
                          label=label, color=colors[i], capsize=3, capthick=1)

        ax.set_xlabel('Round')
        if experiment_type == "n_cmapss":
            ax.set_ylabel('Average Track RMSE')
            ax.set_title('Avg Track RMSE Progression')
        else:
            ax.set_ylabel('Average Track Accuracy')
            ax.set_title('Avg Track Accuracy Progression')

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        # Set x-axis to show only whole numbers
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # 6. Model Storage Size
        ax = axes[1, 1]
        values = [averaged_data[s].get('model_storage_size_mib', 0) for s in scenarios]
        errors = [averaged_data[s].get('model_storage_size_mib_std', 0) for s in scenarios]
        if any(v > 0 for v in values):
            bars = ax.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
            ax.tick_params(axis='x', pad=2)
            ax.set_ylabel('Size (MiB)')
            ax.set_title('Model Storage Size')
            ax.grid(True, axis='y', alpha=0.3)
            # Add value labels (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                #Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                ax.annotate(f'{value:.1f}MiB', xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        #7. Average Total Round Time
        ax = axes[1, 2]
        values = [averaged_data[s].get('avg_total_round_time', 0) for s in scenarios]
        errors = [averaged_data[s].get('avg_total_round_time_std', 0) for s in scenarios]
        if any(v > 0 for v in values):
            bars = ax.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
            ax.tick_params(axis='x', pad=2)
            ax.set_ylabel('Time (s)')
            ax.set_title('Avg Round Time')
            ax.grid(True, axis='y', alpha=0.3)
            # Add value labels (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                # Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                ax.annotate(f'{value:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, label_y),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 8. Cumulative Time Breakdown
        ax = axes[1, 3]
        self._plot_cumulative_time_breakdown(ax, averaged_data, scenarios, labels)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"Saved comprehensive comparison to {output_dir}/comprehensive_comparison.png")
        plt.show()

    def _plot_cumulative_time_breakdown(self, ax, averaged_data, scenarios, labels):
        """Plot a stacked bar chart showing time breakdown for each scenario."""
        #Prepare data for stacked bars
        experiment_init_times = []
        track_init_times = []
        client_training_times = []
        resolution_times = []
        aggregation_times = []
        evaluation_times = []
        other_times = []
        total_time_errors = []  #Standard deviations for total running time

        for scenario in scenarios:
            run_data = averaged_data[scenario]
            total_time = run_data.get('total_running_time', 0)
            total_time_std = run_data.get('total_running_time_std', 0)

            # Get experiment initialization time (one-time cost)
            experiment_init = run_data.get('experiment_init_time', 0)

            # Get component times (multiply by number of rounds to get total time spent)
            track_init = run_data.get('avg_track_init_time', 0) * run_data.get('total_rounds', 1)
            client_training = run_data.get('avg_client_training_time', 0) * run_data.get('total_rounds', 1)
            resolution = run_data.get('avg_round_resolution_time', 0) * run_data.get('total_rounds', 1)
            aggregation = run_data.get('avg_round_aggregation_time', 0) * run_data.get('total_rounds', 1)
            evaluation = run_data.get('avg_evaluation_time', 0) * run_data.get('total_rounds', 1)

            # Calculate "other" time (remaining unaccounted time)
            accounted_time = experiment_init + track_init + client_training + resolution + aggregation + evaluation
            other = max(0, total_time - accounted_time)

            experiment_init_times.append(experiment_init)
            track_init_times.append(track_init)
            client_training_times.append(client_training)
            resolution_times.append(resolution)
            aggregation_times.append(aggregation)
            evaluation_times.append(evaluation)
            other_times.append(other)
            total_time_errors.append(total_time_std)

        #Create stacked bar chart
        width = 0.8
        x_pos = range(len(scenarios))

        #Stack the bars with distinct time breakdown colors and black borders.
        # Each layer's bottom= is the cumulative height of the layers below it.
        ax.bar(x_pos, experiment_init_times, width, label='Experiment Init', color=plt.cm.Set1(7), alpha=0.8, edgecolor='black', linewidth=0.8)  # Gray

        bottom1 = experiment_init_times
        ax.bar(x_pos, track_init_times, width, bottom=bottom1,
               label='Track Init', color=plt.cm.Set1(4), alpha=0.8, edgecolor='black', linewidth=0.8)  # Orange

        bottom2 = [e + t for e, t in zip(bottom1, track_init_times)]
        ax.bar(x_pos, client_training_times, width, bottom=bottom2,
               label='Client Training', color=plt.cm.Set1(2), alpha=0.8, edgecolor='black', linewidth=0.8)  #Green - largest component

        bottom3 = [b + c for b, c in zip(bottom2, client_training_times)]
        ax.bar(x_pos, resolution_times, width, bottom=bottom3,
               label='Resolution', color=plt.cm.Set1(3), alpha=0.8, edgecolor='black', linewidth=0.8)  #Purple

        bottom4 = [b + r for b, r in zip(bottom3, resolution_times)]
        ax.bar(x_pos, aggregation_times, width, bottom=bottom4,
               label='Aggregation', color=plt.cm.Set1(0), alpha=0.8, edgecolor='black', linewidth=0.8)  # Red - large component

        bottom5 = [b + a for b, a in zip(bottom4, aggregation_times)]
        ax.bar(x_pos, evaluation_times, width, bottom=bottom5,
               label='Evaluation', color=plt.cm.Set1(1), alpha=0.8, edgecolor='black', linewidth=0.8)  # Bright Blue - large component

        bottom6 = [b + e for b, e in zip(bottom5, evaluation_times)]
        ax.bar(x_pos, other_times, width, bottom=bottom6,
               label='Other', color=plt.cm.Set1(8), alpha=0.8, edgecolor='black', linewidth=0.8)  # Brown

        #Add error bars for total running time on top of stacked bars
        totals = [sum(x) for x in zip(experiment_init_times, track_init_times, client_training_times, resolution_times, aggregation_times, evaluation_times, other_times)]
        ax.errorbar(x_pos, totals, yerr=total_time_errors, fmt='none', capsize=4, capthick=1,
                   ecolor='black', elinewidth=1.5, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=90, ha='center', va='top')
        ax.tick_params(axis='x', pad=2)
        ax.set_ylabel('Time (s)')
        ax.set_title('Avg Total FL Run Time Breakdown')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        #Add total time labels on top of bars (adjust position to account for error bars)
        for i, (total, error) in enumerate(zip(totals, total_time_errors)):
            # Position label above error bar if error exists, otherwise above bar
            label_y = total + error if error > 0 else total
            ax.annotate(f'{total:.1f}s', xy=(i, label_y), xytext=(0, 3),
                       textcoords="offset points", ha='center', va='bottom', fontsize=8)

    def compare_storage_and_time(self, save_plots=True, output_dir=None):
        """Compare total running time and model storage size across runs."""
        if len(self.scenario_runs) < 2:
            print("Need at least 2 scenarios to compare")
            return

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f'results/comparisons/comparison_{timestamp}'

        if save_plots:
            os.makedirs(output_dir, exist_ok=True)

        # Use averaged scenario data
        averaged_data = self.get_averaged_scenario_data()
        max_runs = max(self.num_runs_per_scenario.values()) if self.num_runs_per_scenario else 0

        plt.figure(figsize=(15, 6))

        # Subplot 1: Total Running Time
        plt.subplot(1, 2, 1)
        values = []
        errors = []
        labels = []
        colors = []

        for scenario, run_data in averaged_data.items():
            value = run_data.get('total_running_time')
            error = run_data.get('total_running_time_std', 0)
            if value is not None:
                values.append(value)
                errors.append(error)
                clients = run_data.get('num_clients', 'Unknown')
                labels.append(self._generate_chart_label(scenario, clients))

                #Color by scenario
                if scenario is not None:
                    colors.append(plt.cm.Set1(scenario % 10))
                else:
                    colors.append('gray')

        if values:
            bars = plt.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            plt.xticks(range(len(values)), labels, rotation=45, ha='center')
            plt.tick_params(axis='x', pad=1)
            plt.ylabel('Total Running Time (seconds)')
            plt.title(f'Total Running Time Comparison\n(across {max_runs} runs)')
            plt.grid(True, axis='y', alpha=0.3)

            #Add value labels on bars (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                # Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                plt.annotate(f'{value:.1f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, label_y),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Subplot 2: Model Storage Size
        plt.subplot(1, 2, 2)
        values = []
        errors = []
        labels = []
        colors = []

        for scenario, run_data in averaged_data.items():
            value = run_data.get('model_storage_size_mib')
            error = run_data.get('model_storage_size_mib_std', 0)
            if value is not None:
                values.append(value)
                errors.append(error)
                clients = run_data.get('num_clients', 'Unknown')
                labels.append(self._generate_chart_label(scenario, clients))

                # Color by scenario (same as time plot)
                if scenario is not None:
                    colors.append(plt.cm.Set1(scenario % 10))
                else:
                    colors.append('gray')

        if values:
            bars = plt.bar(range(len(values)), values, yerr=errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                         capsize=5, error_kw={'elinewidth': 1, 'capthick': 1})
            plt.xticks(range(len(values)), labels, rotation=45, ha='center')
            plt.tick_params(axis='x', pad=1)
            plt.ylabel('Model Storage Size (MiB)')
            plt.title(f'Model Storage Size Comparison\n(across {max_runs} runs)')
            plt.grid(True, axis='y', alpha=0.3)

            #Add value labels on bars (above error bars)
            for bar, value, error in zip(bars, values, errors):
                height = bar.get_height()
                #Position label above error bar if error exists, otherwise above bar
                label_y = height + error if error > 0 else height
                plt.annotate(f'{value:.1f}MiB',
                            xy=(bar.get_x() + bar.get_width() / 2, label_y),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(output_dir, 'storage_and_time_comparison.png'),
                       bbox_inches='tight', dpi=150)
            print(f"Saved storage and time comparison to {output_dir}/storage_and_time_comparison.png")
        plt.show()

    def print_summary(self):
        """Print a summary comparison of all loaded runs."""
        if not self.runs:
            print("No runs loaded")
            return

        print("\n" + "="*80)
        print("FEDERATED LEARNING RUNS COMPARISON SUMMARY (AVERAGED BY SCENARIO)")
        print("="*80)

        # Print individual runs first
        print("\nIndividual runs loaded:")
        for run_name, run_data in self.runs.items():
            scenario = run_data.get('scenario', 'Unknown')
            print(f"   - {run_name} (Scenario {scenario})")

        # Then print averaged summary
        averaged_data = self.get_averaged_scenario_data()

        for scenario, run_data in averaged_data.items():
            num_runs = self.num_runs_per_scenario.get(scenario, 0)
            print(f"\n[Summary] Scenario {scenario} (averaged across {num_runs} runs)")
            print(f"   Clients: {run_data.get('num_clients', 'Unknown')}")
            print(f"   Experiment: {run_data.get('experiment_type', 'Unknown')}")
            print(f"   Total Rounds: {run_data.get('total_rounds', 'Unknown'):.1f}")

            # Performance
            accuracy = run_data.get('final_accuracy')
            avg_track_accuracy = run_data.get('final_avg_track_accuracy')

            if accuracy:
                print(f"   Final Global Accuracy: {accuracy:.4f}")
            if avg_track_accuracy and avg_track_accuracy != accuracy:
                print(f"   Final Avg Track Accuracy: {avg_track_accuracy:.4f}")
            elif avg_track_accuracy:
                print(f"   Final Accuracy: {avg_track_accuracy:.4f}")

            #Timing and Storage
            total_time = run_data.get('total_running_time')
            storage_size = run_data.get('model_storage_size_mib')
            resolution_time = run_data.get('avg_resolution_time_ms')
            overhead = run_data.get('disagreement_overhead_pct')

            if total_time:
                print(f"   Total Running Time: {total_time:.1f}s")
            if storage_size:
                print(f"   Model Storage Size: {storage_size:.1f}MiB")
            if resolution_time:
                print(f"   Avg Resolution Time: {resolution_time:.3f}ms")
            if overhead is not None:
                print(f"   Disagreement Overhead: {overhead:.1f}%")

            rounds_with_disag = run_data.get('rounds_with_disagreements', 0)
            total_timing_rounds = run_data.get('total_timing_rounds', 0)
            if total_timing_rounds > 0:
                print(f"   Rounds with Disagreements: {rounds_with_disag:.1f}/{total_timing_rounds:.1f}")

    def _average_scenario_metrics(self, scenario_runs):
        """Average metrics across multiple runs of the same scenario."""
        if not scenario_runs:
            return None

        #Initialize aggregated data with the first run's structure
        first_run = scenario_runs[0]
        avg_data = {
            "scenario": first_run.get("scenario"),
            "num_clients": first_run.get("num_clients"),
            "experiment_type": first_run.get("experiment_type"),
            "num_runs": len(scenario_runs)
        }

        # Metrics to average (numeric values)
        numeric_metrics = [
            'final_accuracy', 'final_loss', 'final_precision', 'final_recall', 'final_f1',
            'final_avg_track_accuracy', 'final_avg_track_precision', 'final_avg_track_recall', 'final_avg_track_f1',
            'avg_total_time', 'avg_aggregation_time', 'avg_resolution_time_ms',
            'disagreement_overhead_pct', 'total_rounds', 'total_running_time', 'model_storage_size_mib',
            'experiment_init_time', 'avg_track_init_time', 'avg_client_training_time', 'avg_total_round_time',
            'avg_individual_client_training_time', 'max_individual_client_training_time', 'min_individual_client_training_time',
            'avg_round_aggregation_time', 'avg_round_resolution_time', 'avg_round_total_aggregation_time', 'avg_evaluation_time'
        ]

        # Average numeric metrics and calculate standard deviations
        for metric in numeric_metrics:
            values = [run.get(metric) for run in scenario_runs if run.get(metric) is not None]
            if values:
                avg_data[metric] = np.mean(values)
                # Calculate standard deviation (0 if only one value)
                avg_data[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        #Sum integer metrics
        integer_sum_metrics = ['rounds_with_disagreements', 'total_timing_rounds']
        for metric in integer_sum_metrics:
            values = [run.get(metric, 0) for run in scenario_runs if run.get(metric) is not None]
            if values:
                avg_data[metric] = int(np.mean(values))
                #Also calculate std for integer metrics that might be used in plots
                avg_data[f'{metric}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0

        # Average track performance across rounds
        experiment_type = first_run.get("experiment_type", "mnist")
        avg_data["avg_track_performance"], avg_data["std_track_performance"] = self._average_track_performance_across_runs(scenario_runs, experiment_type)

        return avg_data

    def _average_track_performance_across_runs(self, scenario_runs, experiment_type="mnist"):
        """Average track performance across multiple runs for each round."""
        if not scenario_runs:
            return {}

        # Collect all rounds from all runs
        all_rounds = set()
        for run in scenario_runs:
            track_perf = run.get("avg_track_performance", {})
            all_rounds.update(track_perf.keys())

        if not all_rounds:
            return {}

        # Average performance for each round and calculate standard deviations
        avg_track_performance = {}
        std_track_performance = {}
        for round_num in sorted(all_rounds):
            round_values = []
            for run in scenario_runs:
                track_perf = run.get("avg_track_performance", {})
                if round_num in track_perf:
                    round_values.append(track_perf[round_num])

            if round_values:
                avg_track_performance[round_num] = np.mean(round_values)
                std_track_performance[round_num] = np.std(round_values, ddof=1) if len(round_values) > 1 else 0.0

        return avg_track_performance, std_track_performance

    def get_averaged_scenario_data(self):
        """Get averaged data for each scenario."""
        averaged_data = {}

        for scenario, runs in self.scenario_runs.items():
            self.num_runs_per_scenario[scenario] = len(runs)
            averaged_data[scenario] = self._average_scenario_metrics(runs)

        return averaged_data

def main():
    #Check for help flag before parsing to avoid required argument errors
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()
        return

    parser = argparse.ArgumentParser(
        description='Compare Federated Learning Runs - Automatically averages multiple runs of the same scenario',
        add_help=False) #We handle help manually

    parser.add_argument('runs', nargs='+', help='Paths to FL simulation result directories (multiple runs of the same scenario will be averaged)')

    # Generate timestamped default directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f'results/comparisons/comparison_{timestamp}'

    parser.add_argument('--output-dir', '-o', default=default_output_dir,
                       help=f'Output directory for plots (default: {default_output_dir})')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots, only show summary')
    parser.add_argument('--names', nargs='+',
                       help='Custom names for the runs (optional)')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Display this help and exit')

    args = parser.parse_args()

    if args.help:
        usage()
        return

    # Initialize comparator
    comparator = FLRunComparator()

    # Load runs
    for i, run_path in enumerate(args.runs):
        custom_name = args.names[i] if args.names and i < len(args.names) else None
        comparator.load_run(run_path, custom_name)

    #Print summary
    comparator.print_summary()

    if not args.no_plots:
        print("\nGenerating comparison plots (averaged by scenario)...")

        #Generate comparisons
        comparator.compare_performance(save_plots=True, output_dir=args.output_dir)
        comparator.compare_timing(save_plots=True, output_dir=args.output_dir)
        comparator.compare_round_progression(save_plots=True, output_dir=args.output_dir)
        comparator.compare_combined_metrics(save_plots=True, output_dir=args.output_dir)
        comparator.compare_storage_and_time(save_plots=True, output_dir=args.output_dir)

        print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
