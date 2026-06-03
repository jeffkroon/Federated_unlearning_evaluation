#!/usr/bin/env python3
"""
Grid Runner for Federated Learning + Unlearning Experiments

This script runs all combinations of:
- Scenarios (0-34)
- Models (PyTorch: lstm, mlp, cnn | sklearn: random_forest, xgboost)
- Unlearning strategies (exact_retraining, sisa, distillation)

Each combination creates a separate experiment run with its own results directory.
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from itertools import product
import glob


#Define all combinations
SCENARIOS = list(range(35))  #0-34
PYTORCH_MODELS = ["lstm", "mlp", "cnn"]  # For PyTorch orchestrator
SKLEARN_MODELS = ["random_forest", "xgboost"]  # For sklearn orchestrator
UNLEARNING_STRATEGIES = ["exact_retraining", "sisa", "distillation"]
EXPERIMENT_TYPES = ["mnist", "n_cmapss", "cifar10"]  # Can be filtered


def get_all_scenarios():
    """Get all available scenario files."""
    scenario_dir = "mock_etcd/scenarios_5clients"
    scenario_files = glob.glob(os.path.join(scenario_dir, "scenario*.json"))
    #Sort naturally by scenario number
    return sorted(scenario_files, key=lambda f: int(os.path.basename(f).replace("scenario", "").replace(".json", "")))


def create_config_for_run(
    base_config_path: str,
    experiment_type: str,
    model_type: str,
    unlearning_strategy: str,
    scenario: int,
    output_dir: str,
    fl_rounds: int = 5,
    local_epochs: int = 1
) -> str:
    """Create a temporary config file for a specific run.
    
    Returns:
        Path to temporary config file
    """
    #Load base config
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Update experiment type
    config["experiment"]["type"] = experiment_type
    
    # Update unlearning config
    config["unlearning"]["enabled"] = True
    config["unlearning"]["model_type"] = model_type
    config["unlearning"]["strategies"] = [unlearning_strategy]
    config["unlearning"]["use_strategy"] = unlearning_strategy
    
    # Update training config
    config["training"]["local_epochs"] = local_epochs
    
    #Update results config to use custom directory
    config["results"]["use_timestamped_dir"] = True
    config["results"]["base_dir"] = output_dir
    
    #Create temporary config file
    temp_config_dir = os.path.join(output_dir, "configs")
    os.makedirs(temp_config_dir, exist_ok=True)
    temp_config_path = os.path.join(
        temp_config_dir,
        f"config_s{scenario}_{model_type}_{unlearning_strategy}.json"
    )
    
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_config_path


def run_single_experiment(
    experiment_type: str,
    model_type: str,
    unlearning_strategy: str,
    scenario: int,
    base_config_path: str,
    grid_output_dir: str,
    fl_rounds: int = 5,
    local_epochs: int = 1,
    setup_data: bool = False,
    iid: bool = False,
    verbose: bool = False
) -> dict:
    """Run a single experiment combination.
    
    Returns:
        Dictionary with run metadata and status
    """
    # Create unique tag for this run
    tag = f"s{scenario}_{model_type}_{unlearning_strategy}"
    run_output_dir = os.path.join(grid_output_dir, tag)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Create config for this run
    temp_config = create_config_for_run(
        base_config_path=base_config_path,
        experiment_type=experiment_type,
        model_type=model_type,
        unlearning_strategy=unlearning_strategy,
        scenario=scenario,
        output_dir=run_output_dir,
        fl_rounds=fl_rounds,
        local_epochs=local_epochs
    )
    
    # Determine which orchestrator to use
    is_sklearn = model_type in SKLEARN_MODELS
    orchestrator_script = "fl_orchestrator_sklearn.py" if is_sklearn else "fl_orchestrator.py"
    
    #Update config file with all necessary parameters
    with open(temp_config, 'r') as f:
        config = json.load(f)
    
    #Update experiment config
    config["experiment"]["type"] = experiment_type
    config["experiment"]["fl_rounds"] = fl_rounds
    if setup_data:
        config["data"]["setup_data"] = True
    if iid and experiment_type == "mnist":
        config["experiment"]["iid"] = True
    
    # Update training config
    config["training"]["local_epochs"] = local_epochs
    
    # For sklearn, add model_type to config
    if is_sklearn:
        config["model_type"] = model_type
    
    # Write updated config
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    #Build command
    cmd = ["uv", "run", orchestrator_script, "--config", temp_config]
    
    #Add override arguments for PyTorch orchestrator
    if not is_sklearn:
        cmd.append("--override")
        cmd.extend(["--experiment", experiment_type])
        cmd.extend(["--fl_rounds", str(fl_rounds)])
        cmd.extend(["--local_epochs", str(local_epochs)])
        
        if setup_data:
            cmd.append("--setup_data")
        
        if iid and experiment_type == "mnist":
            cmd.append("--iid")
    
    # For sklearn, pass model_type as argument
    if is_sklearn:
        cmd.extend(["--model_type", model_type])
    
    # Setup scenario
    scenario_path = f"mock_etcd/scenarios_5clients/scenario{scenario}.json"
    if not os.path.exists(scenario_path):
        return {
            "status": "error",
            "error": f"Scenario file not found: {scenario_path}",
            "tag": tag,
            "scenario": scenario,
            "model_type": model_type,
            "unlearning_strategy": unlearning_strategy
        }
    
    # Copy scenario disagreements to disagreements.json
    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)
    
    disagreements_path = "mock_etcd/disagreements.json"
    with open(disagreements_path, 'w') as f:
        json.dump(scenario_data.get("disagreements", {}), f, indent=2)
    
    #Get client IDs from scenario
    num_clients = scenario_data.get("num_clients", 6)
    if experiment_type == "n_cmapss" and num_clients > 6:
        num_clients = 6  #Limit for N-CMAPSS
    client_ids = list(range(num_clients))
    
    # Update config with client IDs
    with open(temp_config, 'r') as f:
        config = json.load(f)
    config["experiment"]["client_ids"] = client_ids
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Add clients to command for PyTorch orchestrator
    if not is_sklearn:
        cmd.extend(["--clients"] + [str(cid) for cid in client_ids])
    
    # Create log files
    stdout_path = os.path.join(run_output_dir, "stdout.txt")
    stderr_path = os.path.join(run_output_dir, "stderr.txt")
    
    #Run experiment
    start_time = datetime.now()
    status = "success"
    error_msg = None
    
    try:
        if verbose:
            print(f"Running: {tag}")
            print(f"  Command: {' '.join(cmd)}")
        
        with open(stdout_path, 'w') as stdout_file, open(stderr_path, 'w') as stderr_file:
            result = subprocess.run(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  #Go to fl-disagreement-resolution/
                timeout=3600  # 1 hour timeout per run
            )
            
            if result.returncode != 0:
                status = "error"
                error_msg = f"Command failed with return code {result.returncode}"
    
    except subprocess.TimeoutExpired:
        status = "timeout"
        error_msg = "Experiment timed out after 1 hour"
    except Exception as e:
        status = "error"
        error_msg = str(e)
    
    end_time = datetime.now()
    elapsed_seconds = (end_time - start_time).total_seconds()
    
    # Find the actual results directory (created by orchestrator)
    # The orchestrator creates a timestamped directory
    results_pattern = os.path.join(run_output_dir, "fl_simulation_*")
    results_dirs = glob.glob(results_pattern)
    actual_results_dir = max(results_dirs, key=os.path.getmtime) if results_dirs else None
    
    #Create metadata
    metadata = {
        "tag": tag,
        "scenario": scenario,
        "experiment_type": experiment_type,
        "model_type": model_type,
        "unlearning_strategy": unlearning_strategy,
        "fl_rounds": fl_rounds,
        "local_epochs": local_epochs,
        "num_clients": num_clients,
        "status": status,
        "error": error_msg,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "config_path": temp_config,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "results_dir": actual_results_dir
    }
    
    #Save metadata
    metadata_path = os.path.join(run_output_dir, "meta.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    """Main function to run grid experiments."""
    parser = argparse.ArgumentParser(
        description="Run grid of FL + unlearning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all combinations for MNIST
  python grid_run.py -e mnist -o results/grid_mnist
  
  # Run specific scenarios and models
  python grid_run.py -e mnist -s 1 2 3 -m lstm cnn -o results/grid_test
  
  # Run with custom rounds and epochs
  python grid_run.py -e mnist -r 3 -l 2 -o results/grid_short
        """
    )
    
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        choices=["mnist", "n_cmapss", "cifar10"],
        required=True,
        help="Experiment type (mnist, n_cmapss, or cifar10)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="results/grid",
        help="Output directory for grid results (default: results/grid)"
    )
    
    parser.add_argument(
        "-s", "--scenarios",
        type=int,
        nargs="+",
        default=None,
        help="Specific scenarios to run (default: all 0-34)"
    )
    
    parser.add_argument(
        "-m", "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to run (default: all)"
    )
    
    parser.add_argument(
        "-u", "--unlearning-strategies",
        type=str,
        nargs="+",
        choices=UNLEARNING_STRATEGIES,
        default=None,
        help="Specific unlearning strategies (default: all)"
    )
    
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=5,
        help="Number of FL rounds (default: 5)"
    )
    
    parser.add_argument(
        "-l", "--local-epochs",
        type=int,
        default=1,
        help="Number of local training epochs (default: 1)"
    )
    
    parser.add_argument(
        "--setup-data",
        action="store_true",
        help="Setup data (for MNIST only)"
    )
    
    parser.add_argument(
        "--iid",
        action="store_true",
        help="Use IID data distribution (for MNIST only)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="mock_etcd/configuration.json",
        help="Base configuration file (default: mock_etcd/configuration.json)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually running"
    )
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    # Determine which scenarios to run
    if args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = SCENARIOS
    
    # Determine which models to run
    if args.models:
        models = args.models
    else:
        #All models for this experiment type
        if args.experiment == "mnist":
            models = PYTORCH_MODELS + SKLEARN_MODELS
        else:  #n_cmapss
            models = ["lstm", "mlp"] + SKLEARN_MODELS  # CNN not for N-CMAPSS
    
    # Determine which unlearning strategies
    if args.unlearning_strategies:
        unlearning_strategies = args.unlearning_strategies
    else:
        unlearning_strategies = UNLEARNING_STRATEGIES
    
    # Create output directory
    grid_output_dir = os.path.join(args.output_dir, args.experiment)
    os.makedirs(grid_output_dir, exist_ok=True)
    
    #Generate all combinations
    combinations = list(product(scenarios, models, unlearning_strategies))
    
    print("=" * 80)
    print("Grid Experiment Runner")
    print("=" * 80)
    print(f"Experiment type: {args.experiment}")
    print(f"Scenarios: {len(scenarios)} ({min(scenarios)}-{max(scenarios)})")
    print(f"Models: {len(models)} ({', '.join(models)})")
    print(f"Unlearning strategies: {len(unlearning_strategies)} ({', '.join(unlearning_strategies)})")
    print(f"Total combinations: {len(combinations)}")
    print(f"Output directory: {grid_output_dir}")
    print("=" * 80)
    
    if args.dry_run:
        print("\nDry run - would execute:")
        for scenario, model, strategy in combinations[:10]:  #Show first 10
            print(f"  s{scenario} {model} {strategy}")
        if len(combinations) > 10:
            print(f"  ... and {len(combinations) - 10} more")
        return
    
    # Save grid configuration
    grid_config = {
        "experiment_type": args.experiment,
        "scenarios": scenarios,
        "models": models,
        "unlearning_strategies": unlearning_strategies,
        "fl_rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "total_combinations": len(combinations),
        "start_time": datetime.now().isoformat()
    }
    
    grid_config_path = os.path.join(grid_output_dir, "grid_config.json")
    with open(grid_config_path, 'w') as f:
        json.dump(grid_config, f, indent=2)
    
    # Run all combinations
    results = []
    for idx, (scenario, model, strategy) in enumerate(combinations, 1):
        print(f"\n[{idx}/{len(combinations)}] Running scenario {scenario}, model {model}, strategy {strategy}...")
        
        try:
            metadata = run_single_experiment(
                experiment_type=args.experiment,
                model_type=model,
                unlearning_strategy=strategy,
                scenario=scenario,
                base_config_path=args.config,
                grid_output_dir=grid_output_dir,
                fl_rounds=args.rounds,
                local_epochs=args.local_epochs,
                setup_data=args.setup_data,
                iid=args.iid,
                verbose=args.verbose
            )
            results.append(metadata)
            
            if metadata["status"] == "success":
                print(f"  OK: Success ({metadata['elapsed_seconds']:.1f}s)")
            else:
                print(f"  FAIL {metadata['status']}: {metadata.get('error', 'Unknown error')}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress...")
            break
        except Exception as e:
            print(f"  FAIL Exception: {e}")
            results.append({
                "status": "error",
                "error": str(e),
                "scenario": scenario,
                "model_type": model,
                "unlearning_strategy": strategy
            })
    
    # Save summary
    summary = {
        "grid_config": grid_config,
        "total_runs": len(results),
        "successful": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "timeout": len([r for r in results if r.get("status") == "timeout"]),
        "end_time": datetime.now().isoformat(),
        "results": results
    }
    
    summary_path = os.path.join(grid_output_dir, "grid_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Grid Run Complete")
    print("=" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Timeout: {summary['timeout']}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

