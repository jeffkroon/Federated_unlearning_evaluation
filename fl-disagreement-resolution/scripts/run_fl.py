#!/usr/bin/env python3
"""
Federated Learning Experiment Runner

This script runs federated learning experiments with support for different
scenarios, experiment types, and configuration overrides.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import glob
from datetime import datetime
from pathlib import Path


def usage():
    """Print usage information."""
    print("""Usage: run_fl.py [options]

Options:
  -e, --experiment <n>     Experiment type (n_cmapss, mnist, or cifar10).
  -c, --clients <ids>      Number of clients or a list of IDs (e.g., 6 or '0 1 3 5').
                           If not specified, uses 'num_clients' from the scenario file.
  -r, --rounds <num>       Number of FL rounds (default: 3).
  -l, --local-epochs <num> Number of local training epochs (default: 5).
  -b, --batch-size <num>   Batch size (default: 64).
  -s, --setup-data         Setup data (for MNIST only).
  -f, --force-setup        Force data setup even if it exists (for MNIST only).
  -i, --iid                Use IID data distribution (for MNIST only).
  -d, --results-dir <dir>  Custom results directory (default: auto-generated).
  -S, --scenario [num ...] Scenario(s) to run, e.g. -S 0 1 2 3 4 5 for six runs, or 'all' (default: 0).
                           Scenarios can define 'num_clients' (N-CMAPSS is limited to <= 6).
  -C, --config <file>      Path to configuration file (default: mock_etcd/configuration.json).
  --no-viz                 Skip automatic track visualization generation.
  --verbose-plots          Generate all plots (default: only last round track metrics + track contributions).
  -h, --help               Display this help and exit.

Examples:
  run_fl.py -e n_cmapss -c 3 -r 5
  run_fl.py -e mnist -c 6 -s -i
  run_fl.py -e mnist -c 4 -d 'results/my_experiment'
  run_fl.py -e mnist -c 4 -s -f -i
  run_fl.py -e mnist -S 1
  run_fl.py -e mnist -S 1 -c 4
  run_fl.py -e mnist -S all
  run_fl.py -C custom_config.json
  run_fl.py -e mnist -S 1 --no-viz
  run_fl.py -e n_cmapss -c 3 -r 5 --verbose-plots

By default, only track metrics comparison plots for the last round are generated
      to improve performance. Use --verbose-plots to generate all plots for all rounds.
      Track contributions visualization is automatically generated after each
      experiment completion and saved to the simulation's output/ directory.
""")


def get_all_scenarios():
    """Get all available scenario files sorted naturally."""
    scenario_dir = "mock_etcd/scenarios_5clients"
    scenario_files = glob.glob(os.path.join(scenario_dir, "scenario*.json"))
    # Sort naturally by scenario number
    return sorted(scenario_files, key=lambda f: int(os.path.basename(f).replace("scenario", "").replace(".json", "")))


def run_single_scenario(scenario, original_args):
    """Run a single scenario with the given arguments."""
    print("=" * 65)
    print(f"Running experiment with scenario {scenario}")
    print("=" * 65)

    #Create new args with this specific scenario
    new_args = copy.deepcopy(original_args)
    new_args.scenario = scenario

    #Run the experiment and capture simulation dir
    sim_dir = run_experiment(new_args)

    # Consolidate results if we have a simulation dir
    if sim_dir:
        consolidate_results(sim_dir, scenario, new_args.experiment)

    print(f"Experiment with scenario {scenario} completed.")
    print()


def process_clients_parameter(clients_input):
    """Process the clients parameter, converting number to client list if needed."""
    if not clients_input:
        return ""
    
    # If it's a list (from nargs='+'), join it
    if isinstance(clients_input, list):
        clients_str = " ".join(clients_input)
    else:
        clients_str = clients_input

    # Check if it's a single number (no spaces)
    if clients_str.strip().isdigit():
        num_clients = int(clients_str.strip())
        clients_list = [str(i) for i in range(num_clients)]
        clients_result = " ".join(clients_list)
        print(f"Using {num_clients} clients: {clients_result}")
        return clients_result
    else:
        print(f"Using specified client IDs: {clients_str}")
        return clients_str


def setup_scenario(scenario, config_file, experiment_type, clients):
    """Setup scenario by extracting disagreements and determining clients."""
    #Determine scenario path
    if os.path.isfile(scenario):
        scenario_path = scenario
    elif os.path.isfile(f"mock_etcd/scenarios_5clients/scenario{scenario}.json"):
        scenario_path = f"mock_etcd/scenarios_5clients/scenario{scenario}.json"
    else:
        print(f"Error: Scenario file not found for scenario {scenario}")
        sys.exit(1)

    print(f"Using scenario: {scenario_path}")

    #Load scenario and extract disagreements
    try:
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)

        # Save disagreements. A unique FL_DISAGREEMENTS_PATH (set by parallel runners) keeps
        # concurrent runs from overwriting each other's shared mock_etcd/disagreements.json.
        # Unset (the default) preserves the original behaviour exactly.
        disag_path = os.environ.get("FL_DISAGREEMENTS_PATH", 'mock_etcd/disagreements.json')
        os.makedirs(os.path.dirname(disag_path) or '.', exist_ok=True)
        with open(disag_path, 'w') as f:
            json.dump(scenario_data.get('disagreements', {}), f, indent=2)

        print(f'Copied disagreements from scenario to {disag_path}')

        #Use scenario's num_clients if not set
        if not clients:
            scenario_clients = scenario_data.get('num_clients', 10)

            #Validate client count for n_cmapss
            if experiment_type == 'n_cmapss' and scenario_clients > 6:
                print(f'Error: N-CMAPSS experiment cannot use more than 6 clients (scenario requests {scenario_clients})', file=sys.stderr)
                sys.exit(1)

            # Generate client list
            client_list = ' '.join(str(i) for i in range(scenario_clients))
            print(f'Using {scenario_clients} clients from scenario: {client_list}')
            clients = client_list
        else:
            print('Using explicitly specified clients')

        # Compute scenario tag for directory suffix (passed via CLI, not written to config)
        scenario_tag = os.path.splitext(os.path.basename(str(scenario)))[0]
        directory_suffix = f'_s{scenario_tag}'

        return clients, scenario_path, directory_suffix

    except Exception as e:
        print(f"Error: Failed to process scenario: {e}")
        sys.exit(1)


def find_latest_simulation_dir(experiment_type=None, scenario=None):
    """Find the most recent FL simulation directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None

    # Build pattern to match simulation directories
    if experiment_type and scenario and scenario != "0":
        pattern = f"fl_simulation_*_{experiment_type}_s{scenario}"
    elif experiment_type:
        pattern = f"fl_simulation_*_{experiment_type}*"
    else:
        pattern = "fl_simulation_*"

    #Find all matching directories
    sim_dirs = list(results_dir.glob(pattern))

    if not sim_dirs:
        #Try without scenario suffix
        sim_dirs = list(results_dir.glob("fl_simulation_*"))

    if not sim_dirs:
        return None

    # Return most recent (by timestamp)
    return str(max(sim_dirs, key=lambda x: x.stat().st_mtime))

def consolidate_results(simulation_path, scenario=None, experiment=None):
    """Collect baseline + strategy metrics into one compact JSON."""
    sim_path = Path(simulation_path)
    if not sim_path.exists():
        print(f"Warning: Cannot consolidate; missing {simulation_path}")
        return None

    summary = {
        "simulation_dir": str(sim_path),
        "scenario": scenario,
        "experiment": experiment,
        "strategies": {}
    }
    experiment_type = None
    fl_rounds = None

    # strategy dirs
    strat_dirs = []
    for entry in sim_path.iterdir():
        if entry.is_dir() and (entry.name == "baseline" or entry.name.startswith("strategy_")):
            strat_dirs.append(entry)

    strat_dirs_sorted = sorted(strat_dirs, key=lambda p: (p.name != "baseline", p.name))

    for strat_dir in strat_dirs_sorted:
        strat_name = "original_model" if strat_dir.name == "baseline" else strat_dir.name.replace("strategy_", "")
        strat_entry = {"rounds": {}}

        fl_results_path = strat_dir / "output" / "fl_results.json"
        if fl_results_path.exists():
            try:
                data = json.loads(fl_results_path.read_text())
                experiment_type = experiment_type or data.get("experiment_type")
                fl_rounds = fl_rounds or data.get("fl_rounds")
                for round_info in data.get("rounds", []):
                    rnd = round_info.get("round")
                    if rnd is None or rnd == 0:
                        continue
                    metrics = {}
                    if data.get("experiment_type") in ("mnist", "cifar10", "tabular", "adult"):
                        metrics = {
                            "accuracy": round_info.get("test_accuracy"),
                            "precision": round_info.get("mean_precision"),
                            "recall": round_info.get("mean_recall"),
                            "f1": round_info.get("mean_f1"),
                            "test_loss": round_info.get("test_loss"),
                        }
                    else:
                        metrics = {
                            "rmse": round_info.get("test_loss"),
                            "test_loss": round_info.get("test_loss"),
                        }
                    strat_entry["rounds"].setdefault(str(rnd), {})["global"] = metrics
            except Exception as e:
                print(f"Warning: Failed to read {fl_results_path}: {e}")

        # track evaluations (sort by round number for consistent ordering)
        track_eval_files = list(strat_dir.glob("**/track_evaluation_round_*.json"))
        track_eval_files.sort(key=lambda f: int(f.stem.split("_")[-1]))
        for track_file in track_eval_files:
            try:
                round_tag = track_file.stem.split("_")[-1]
                tdata = json.loads(track_file.read_text())
                #Attach branch metrics if present
                round_dir = strat_dir / "model_storage" / f"round_{round_tag}"
                for tname, tmetrics in tdata.items():
                    metrics_path = round_dir / "tracks" / tname / "unlearning" / "branches" / strat_name / "metrics.json"
                    if metrics_path.exists():
                        try:
                            branch_metrics = json.loads(metrics_path.read_text())
                            tmetrics["unlearning_branch"] = strat_name
                            tmetrics["unlearning_metrics"] = branch_metrics
                        except Exception:
                            pass
                    #mark unlearned
                    tmetrics["unlearned"] = tmetrics.get("model_source", "").startswith("branch:")
                    if tname == "global" and tmetrics.get("model_source") == "global_track_baseline_copy":
                        tmetrics["unlearned"] = False
                        tmetrics["note"] = "Global track copy; no unlearning applied"
                strat_entry["rounds"].setdefault(round_tag, {})["tracks"] = tdata
            except Exception as e:
                print(f"Warning: Failed to read track eval {track_file}: {e}")

        # Sort rounds by numerical order before adding to summary
        strat_entry["rounds"] = dict(sorted(strat_entry["rounds"].items(), key=lambda x: int(x[0])))
        summary["strategies"][strat_name] = strat_entry

    if experiment_type:
        summary["experiment_type"] = experiment_type
    if fl_rounds:
        summary["fl_rounds"] = fl_rounds

    out_path = sim_path / "consolidated_results.json"
    try:
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Consolidated results written to: {out_path}")
    except Exception as e:
        print(f"Warning: Failed to write consolidated results: {e}")
    return out_path

def run_track_visualization(simulation_path, fl_rounds=None):
    """Run the track contributions visualization for a completed experiment."""
    if not simulation_path or not os.path.exists(simulation_path):
        print("Warning: No valid simulation directory found for visualization")
        return

    model_storage = Path(simulation_path) / "model_storage"
    track_files = []
    viz_path = simulation_path

    if model_storage.exists():
        track_files = list(model_storage.glob("round_*/tracks/track_metadata.json"))
    
    # Try subdirectories (baseline, strategy_*)
    if not track_files:
        for subdir in ["baseline", "strategy_exact_retraining", "strategy_sisa", "strategy_distillation"]:
            subdir_storage = Path(simulation_path) / subdir / "model_storage"
            if subdir_storage.exists():
                track_files = list(subdir_storage.glob("round_*/tracks/track_metadata.json"))
                if track_files:
                    viz_path = str(Path(simulation_path) / subdir)
                    break
    
    if not track_files:
        return

    print("\n" + "="*50)
    print("Generating track contributions visualization...")
    print("="*50)

    try:
        # Run the visualization script
        viz_script = "scripts/visualize_track_contributions.py"
        cmd = ["python", viz_script, viz_path]

        #Add rounds parameter if available
        if fl_rounds:
            cmd.extend(["--rounds", str(fl_rounds)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("Track visualization completed successfully!")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if "Visualization saved to:" in line:
                    print(line)
        else:
            print(f"Warning: Visualization failed with error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("Warning: Visualization timed out")
    except Exception as e:
        print(f"Warning: Could not run visualization: {e}")


def run_experiment(args):
    """Run a federated learning experiment with the given arguments."""
    scenarios = args.scenario if isinstance(args.scenario, list) else [args.scenario]

    #Process clients parameter
    if args.clients:
        clients = process_clients_parameter(args.clients)
    else:
        clients = ""

    # Handle 'all' scenarios
    if scenarios == ["all"]:
        print("Running all available scenarios...")

        scenario_files = get_all_scenarios()
        if not scenario_files:
            print("Error: No scenario files found in mock_etcd/scenarios_5clients/")
            sys.exit(1)

        for scenario_file in scenario_files:
            scenario_num = os.path.basename(scenario_file).replace("scenario", "").replace(".json", "")
            run_single_scenario(scenario_num, args)

        print("All scenarios completed.")
        return

    # Handle multiple scenarios (e.g. -S 0 1 2 3 4 5)
    if len(scenarios) > 1:
        print(f"Running {len(scenarios)} scenarios: {', '.join(scenarios)}")
        for s in scenarios:
            run_single_scenario(s, args)
        print("All requested scenarios completed.")
        return

    # Single scenario
    scenario_str = scenarios[0]
    directory_suffix = None
    if scenario_str:
        clients, scenario_path, directory_suffix = setup_scenario(scenario_str, args.config, args.experiment, clients)

    #Build the command to run fl_orchestrator.py.
    #Use the same interpreter that launched this script (works with or without uv);
    # FL_PYTHON env var overrides if a specific interpreter is needed.
    _py = os.environ.get("FL_PYTHON", sys.executable)
    cmd = [_py, "fl_orchestrator.py", "--config", args.config]

    # Add override flag if any parameters are specified
    if any([args.experiment, clients, args.rounds, args.local_epochs, args.batch_size,
            args.setup_data, args.force_setup, args.iid, args.results_dir, args.verbose_plots]):
        cmd.append("--override")

    # Add override arguments
    if args.experiment:
        cmd.extend(["--experiment", args.experiment])

    if clients:
        #Split clients string and add each ID separately
        client_ids = clients.split()
        cmd.extend(["--clients"] + client_ids)

    if args.rounds:
        cmd.extend(["--fl_rounds", str(args.rounds)])

    if args.local_epochs:
        cmd.extend(["--local_epochs", str(args.local_epochs)])

    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])

    if args.setup_data:
        cmd.append("--setup_data")

    if args.force_setup:
        cmd.append("--force_setup_data")

    if args.iid:
        cmd.append("--iid")

    if args.results_dir:
        cmd.extend(["--results_dir", args.results_dir])

    if args.verbose_plots:
        cmd.append("--verbose_plots")

    if directory_suffix and not args.results_dir:
        cmd.extend(["--directory_suffix", directory_suffix])
    if scenario_str is not None and scenario_str != "":
        cmd.extend(["--scenario", str(scenario_str)])

    #Print experiment details
    if args.experiment:
        print(f"Running {args.experiment} federated learning experiment")

    if clients:
        print(f"Using clients: {clients}")

    print(f"Using scenario: {scenario_str}")
    print(f"Parameters will override configuration in {args.config}")
    print(f"Running command: {' '.join(cmd)}")

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Define log file path with timestamp (sanitize scenario for filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment or "all"
    raw_scenario = scenario_str or "none"
    scenario_name = os.path.splitext(os.path.basename(str(raw_scenario)))[0]
    log_file = f"logs/experiment_{timestamp}_{experiment_name}_s{scenario_name}.log"

    print(f"Logging output to: {log_file}")

    # Execute the command and log output
    try:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1)

            for line in iter(process.stdout.readline, ''):
                print(line, end='')  #Print to console
                log.write(line)      #Write to log file
                log.flush()

            process.wait()

            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                sys.exit(process.returncode)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)

    finally:
        pass

    simulation_dir = find_latest_simulation_dir(args.experiment, scenario_name)

    # Consolidate results into a single file for this simulation
    if simulation_dir:
        consolidate_results(simulation_dir, scenario_name, args.experiment)

    if not args.no_viz:
        if simulation_dir:
            run_track_visualization(simulation_dir, args.rounds)
    else:
        print("Skipping track visualization (--no-viz specified)")

    print("Experiment completed.")
    return simulation_dir


def main():
    """Main function to parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(description='Run federated learning experiments',
                                   add_help=False)  # We handle help manually

    parser.add_argument('-e', '--experiment', type=str,
                       help='Experiment type (n_cmapss or mnist)')
    parser.add_argument('-c', '--clients', type=str, action='append',
                       help='Client IDs (e.g., -c 0 -c 1 -c 2 or as string "0 1 2") or number of clients (e.g., 6)')
    parser.add_argument('-r', '--rounds', type=int,
                       help='Number of FL rounds (default: 3)')
    parser.add_argument('-l', '--local-epochs', type=int, dest='local_epochs',
                       help='Number of local training epochs (default: 5)')
    parser.add_argument('-b', '--batch-size', type=int, dest='batch_size',
                       help='Batch size (default: 64)')
    parser.add_argument('-s', '--setup-data', action='store_true', dest='setup_data',
                       help='Setup data (for MNIST only)')
    parser.add_argument('-f', '--force-setup', action='store_true', dest='force_setup',
                       help='Force data setup even if it exists (for MNIST only)')
    parser.add_argument('-i', '--iid', action='store_true',
                       help='Use IID data distribution (for MNIST only)')
    parser.add_argument('-d', '--results-dir', type=str, dest='results_dir',
                       help='Custom results directory (default: auto-generated)')
    parser.add_argument('-S', '--scenario', type=str, nargs='*', default=['0'],
                       help='Scenario number(s) to run, e.g. -S 0 1 2 3 4 5 for 6 scenarios, or "all" for all 35')
    parser.add_argument('-C', '--config', type=str, default="mock_etcd/configuration.json",
                       help='Path to configuration file (default: mock_etcd/configuration.json)')
    parser.add_argument('--no-viz', action='store_true', dest='no_viz',
                       help='Skip automatic track visualization generation')
    parser.add_argument('--verbose-plots', action='store_true', dest='verbose_plots',
                       help='Generate all plots (default: only last round track metrics + track contributions)')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Display this help and exit')

    args = parser.parse_args()

    if args.help:
        usage()
        sys.exit(0)

    # This ensures relative paths work correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  #Go up one level from scripts/
    os.chdir(parent_dir)

    run_experiment(args)


if __name__ == "__main__":
    main()
