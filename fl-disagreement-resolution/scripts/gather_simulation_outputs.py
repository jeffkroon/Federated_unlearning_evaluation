#!/usr/bin/env python3
"""
Script to gather output files from FL simulation results folder
and organize them by scenario.
"""

import shutil
import re
import argparse
import sys
from pathlib import Path


def usage():
    """Print usage information."""
    print("""Usage: gather_simulation_outputs.py [options]

Options:
  --collect <type>         Specify what to collect: 'track_contributions' for track contributions plots,
                           'model_performance' for model performance plots, or 'both' for everything (default: both).
  -h, --help               Display this help and exit.

Examples:
  gather_simulation_outputs.py
  gather_simulation_outputs.py --collect track_contributions
  gather_simulation_outputs.py --collect model_performance
  gather_simulation_outputs.py --collect both

All collected files are saved to results/collected_outputs/ and organized by scenario.
""")

def extract_scenario_from_path(path):
    """Extract scenario number from the simulation directory name."""
    # Pattern to match scenario number from directory name like 'fl_simulation_20250615_003242_mnist_s1'
    match = re.search(r'_s(\d+)$', path.name)
    if match:
        return int(match.group(1))
    return None

def extract_dataset_from_path(path):
    """Extract dataset name from the simulation directory name."""
    # Pattern to match dataset name from directory name like 'fl_simulation_20250615_003242_mnist_s1'
    #Look for known dataset names right before _s<number>
    if 'n_cmapss' in path.name:
        return 'n_cmapss'
    elif 'mnist' in path.name:
        return 'mnist'
    else:
        #Fallback: try to extract the last word before _s<number>
        match = re.search(r'_([a-zA-Z_]+)_s\d+$', path.name)
        if match:
            return match.group(1)
    return "unknown"

def find_last_round_file(directory, pattern):
    """Find the file with the highest round number matching the pattern."""
    files = list(directory.glob(pattern))
    if not files:
        return None

    # Extract round numbers and find the maximum
    max_round = -1
    max_file = None

    for file in files:
        match = re.search(r'round_(\d+)', file.name)
        if match:
            round_num = int(match.group(1))
            if round_num > max_round:
                max_round = round_num
                max_file = file

    return max_file

def collect_track_contributions(sim_dir, scenario_num, dataset, output_dir, copied_files):
    """Collect track contributions files."""
    track_contrib_file = sim_dir / "output" / f"track_contributions_{sim_dir.name}.png"
    if track_contrib_file.exists():
        new_filename = f"s{scenario_num}_{dataset}_track_contributions.png"
        destination = output_dir / new_filename
        shutil.copy2(track_contrib_file, destination)
        copied_files.append(new_filename)
        print(f"  Copied track contributions: {new_filename}")
        return True
    else:
        print(f"  Track contributions file not found: {track_contrib_file}")
        return False

def collect_model_performance(sim_dir, scenario_num, dataset, output_dir, copied_files):
    """Collect model performance (track metrics comparison) files."""
    plots_dir = sim_dir / "output" / "server" / "plots"
    if plots_dir.exists():
        track_metrics_file = find_last_round_file(plots_dir, "track_metrics_comparison_round_*.png")
        if track_metrics_file:
            new_filename = f"s{scenario_num}_{dataset}_track_metrics_comparison.png"
            destination = output_dir / new_filename
            shutil.copy2(track_metrics_file, destination)
            copied_files.append(new_filename)
            print(f"  Copied track metrics comparison: {new_filename}")
            return True
        else:
            print(f"  Track metrics comparison file not found in: {plots_dir}")
            return False
    else:
        print(f"  Plots directory not found: {plots_dir}")
        return False

def main():
    # Check for help flag before parsing to avoid required argument errors
    if '-h' in sys.argv or '--help' in sys.argv:
        usage()
        return

    parser = argparse.ArgumentParser(
        description="Gather output files from FL simulation results folder and organize them by scenario.",
        add_help=False)  # We handle help manually

    parser.add_argument(
        "--collect",
        choices=["track_contributions", "model_performance", "both"],
        default="both",
        help="Specify what to collect: 'track_contributions' for track contributions plots, "
             "'model_performance' for model performance plots, or 'both' for everything (default: both)"
    )
    parser.add_argument('-h', '--help', action='store_true',
                       help='Display this help and exit')

    args = parser.parse_args()

    if args.help:
        usage()
        return

    #Define paths
    results_dir = Path("results")
    output_dir = Path("results/collected_outputs")

    #Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Determine what to collect based on arguments
    collect_contributions = args.collect in ["track_contributions", "both"]
    collect_performance = args.collect in ["model_performance", "both"]

    # Find all simulation directories
    print("Scanning for simulation output files...")
    print(f"Collection mode: {args.collect}")
    if collect_contributions:
        print("  - Will collect track contributions plots")
    if collect_performance:
        print("  - Will collect model performance plots")

    copied_files = []

    for sim_dir in results_dir.iterdir():
        if sim_dir.is_dir() and sim_dir.name.startswith("fl_simulation_"):
            scenario_num = extract_scenario_from_path(sim_dir)
            dataset = extract_dataset_from_path(sim_dir)

            if scenario_num is not None:
                print(f"\nProcessing scenario {scenario_num} ({dataset}):")

                # Collect files based on user selection
                if collect_contributions:
                    collect_track_contributions(sim_dir, scenario_num, dataset, output_dir, copied_files)

                if collect_performance:
                    collect_model_performance(sim_dir, scenario_num, dataset, output_dir, copied_files)

    print(f"\nDone! All collected files are in: {output_dir}")
    print(f"Total files copied: {len(copied_files)}")

    #Print summary by file type
    print("\nSummary:")
    contrib_files = [f for f in copied_files if "track_contributions" in f]
    metrics_files = [f for f in copied_files if "track_metrics_comparison" in f]

    if contrib_files:
        print(f"  Track contributions files: {len(contrib_files)}")
    if metrics_files:
        print(f"  Track metrics comparison files: {len(metrics_files)}")

    #Print summary by dataset
    mnist_files = [f for f in copied_files if "mnist" in f]
    ncmapss_files = [f for f in copied_files if "n_cmapss" in f]

    if mnist_files:
        print(f"  MNIST files: {len(mnist_files)}")
    if ncmapss_files:
        print(f"  N-CMAPSS files: {len(ncmapss_files)}")

    print(f"\nAll files are now in a single folder: {output_dir}")

if __name__ == "__main__":
    main()
