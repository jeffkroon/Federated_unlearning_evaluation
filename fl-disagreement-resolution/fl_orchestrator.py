"""Federated learning orchestrator implementation."""

import os
import sys
import argparse
import json
import time
import shutil
import tempfile
from datetime import datetime
import fl_module
import numpy as np
import matplotlib.pyplot as plt
import torch
from fl_server.evaluation import evaluate_track_models

from fl_client import FederatedClient
from fl_server import FederatedServer
from mock_etcd.etcd_loader import MockEtcdLoader

class FederatedOrchestrator:
    """Orchestrator for coordinating clients and server in federated learning."""

    def __init__(self, config_path="mock_etcd/configuration.json"):
        """Initialize the federated learning orchestrator.

        Args:
            config_path: Path to the configuration file
        """
        #Load configuration
        self.config = MockEtcdLoader(config_path)

        #Get configuration sections
        self.exp_config = self.config.get_experiment_config()
        self.data_config = self.config.get_data_config()
        self.train_config = self.config.get_training_config()
        self.results_config = self.config.get_results_config()

        # Extract common parameters
        self.experiment_type = self.exp_config.get("type")
        self.client_ids_in_experiment = self.exp_config.get("client_ids")
        self.fl_rounds = self.exp_config.get("fl_rounds")
        self.results_dir = self.results_config.get("results_dir")

        # Setup MNIST data if needed
        self._setup_data_if_needed()

        # Create model_storage directory
        if self.results_dir:
            structure = self.results_config.get("structure", {})
            model_storage_dir = structure.get("model_storage_dir", "model_storage")
            os.makedirs(os.path.join(self.results_dir, model_storage_dir), exist_ok=True)

        #Initialize server
        self.server = self._init_server()

        #Initialize clients
        self.clients = self._init_clients()
        
        # Set clients in server for unlearning
        self.server.set_clients(self.clients)
    
    def _get_structure_config(self):
        """Get the directory structure configuration.
        
        Returns:
            dict: Directory structure configuration
        """
        # Default structure configuration
        default_structure = {
            "round_template": "round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "client_prefix": "client_"
        }
        
        # Try to load from configuration
        structure = self.results_config.get("structure", default_structure)
        return structure

    def _setup_data_if_needed(self):
        """Setup data if needed via adapter (plug-and-play)."""
        if not self.data_config.get("setup_data", False):
            return
        from fl_module.registry import DatasetAdapterRegistry
        adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
        if adapter is None:
            return
        setup_fn = getattr(adapter, "setup_data", None)
        if not callable(setup_fn):
            return
        data_config = {**self.data_config, "iid": self.exp_config.get("iid", True)}
        try:
            n = max(self.client_ids_in_experiment) + 1 if self.client_ids_in_experiment else 1
            client_ids = list(range(n))
        except (TypeError, ValueError):
            client_ids = [0]
        adapter.setup_data(data_config, client_ids)

    def _init_server(self):
        """Initialize the federated learning server.

        Returns:
            FederatedServer: Initialized server
        """
        server = FederatedServer(
            experiment_type=self.experiment_type,
            test_dir=self.config.get_test_dir(),
            test_units=self.data_config.get("test_units"),
            results_dir=self.results_dir,
            verbose_plots=self.results_config.get("verbose_plots", False),
            config_path=self.config.config_path
        )

        #Load test data for evaluation
        if self.experiment_type == "n_cmapss":
            server.load_test_data(sample_size=self.data_config.get("test_sample_size", 500))
        else:
            server.load_test_data()

        #Initialize the server with experiment metadata
        server.init_experiment(
            fl_rounds=self.fl_rounds,
            client_ids=self.client_ids_in_experiment,
            iid=self.exp_config.get("iid", False) if self.experiment_type == "mnist" else None
        )

        return server

    def _init_clients(self):
        """Initialize federated learning clients.

        Returns:
            dict: Dictionary mapping client IDs to client instances
        """
        clients = {}
        for client_id in self.client_ids_in_experiment:
            clients[client_id] = FederatedClient(
                client_id=client_id,
                experiment_type=self.experiment_type,
                data_dir=self.config.get_train_dir(),
                batch_size=self.train_config.get("batch_size", 64),
                epochs=self.train_config.get("local_epochs", 5),
                learning_rate=self.train_config.get("learning_rate", 0.001),
                results_dir=self.results_dir,
                config_path=self.config.config_path
            )
            # Load client data
            clients[client_id].load_data(sample_size=self.data_config.get("client_sample_size", 1000))

        return clients
    
    def _init_server_for_strategy(self, strategy_name, results_dir):
        """Initialize a server instance for a specific unlearning strategy.
        
        Args:
            strategy_name: Name of the unlearning strategy
            results_dir: Results directory for this strategy
            
        Returns:
            FederatedServer: Initialized server with single_strategy set
        """
        server = FederatedServer(
            experiment_type=self.experiment_type,
            test_dir=self.config.get_test_dir(),
            test_units=self.data_config.get("test_units"),
            results_dir=results_dir,
            verbose_plots=self.results_config.get("verbose_plots", False),
            single_strategy=strategy_name,  # Set the single strategy for this server
            config_path=self.config.config_path
        )

        # Load test data for evaluation
        if self.experiment_type == "n_cmapss":
            server.load_test_data(sample_size=self.data_config.get("test_sample_size", 500))
        else:
            server.load_test_data()

        #Initialize the server with experiment metadata
        server.init_experiment(
            fl_rounds=self.fl_rounds,
            client_ids=self.client_ids_in_experiment,
            iid=self.exp_config.get("iid", False) if self.experiment_type == "mnist" else None
        )

        return server
    
    def _init_clients_for_strategy(self, results_dir):
        """Initialize clients for a specific unlearning strategy.
        
        Args:
            results_dir: Results directory for this strategy
            
        Returns:
            dict: Dictionary mapping client IDs to client instances
        """
        clients = {}
        for client_id in self.client_ids_in_experiment:
            clients[client_id] = FederatedClient(
                client_id=client_id,
                experiment_type=self.experiment_type,
                data_dir=self.config.get_train_dir(),
                batch_size=self.train_config.get("batch_size", 64),
                epochs=self.train_config.get("local_epochs", 5),
                learning_rate=self.train_config.get("learning_rate", 0.001),
                results_dir=results_dir,
                config_path=self.config.config_path
            )
            #Load client data (same data as original clients)
            clients[client_id].load_data(sample_size=self.data_config.get("client_sample_size", 1000))

        return clients
    
    def _save_run_metadata(self):
        """Save run metadata to the results directory for reproducibility."""
        # Try to determine scenario from config or directory name
        scenario = self.config.config.get("disagreement", {}).get("active_scenario")
        if scenario is None:
            # Extract from results dir name (e.g. ..._s1 or ..._sscenario1)
            import re
            m = re.search(r"_s(?:scenario)?(\d+)$", os.path.basename(self.results_dir))
            if m:
                scenario = int(m.group(1))

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "experiment_type": self.experiment_type,
            "scenario": scenario,
            "num_clients": len(self.client_ids_in_experiment),
            "client_ids": self.client_ids_in_experiment,
            "fl_rounds": self.fl_rounds,
            "training": self.train_config,
            "unlearning": self.config.config.get("unlearning", {}),
            "iid": self.exp_config.get("iid"),
            "results_dir": self.results_dir,
            "config": self.config.config,
        }
        metadata_path = os.path.join(self.results_dir, "run_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Run metadata saved to {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save run metadata: {e}")

    def _print_run_summary(self):
        """Print a compact summary after the experiment completes."""
        unlearning_config = self.config.config.get("unlearning", {})
        unlearning_enabled = unlearning_config.get("enabled", False)
        strategies = unlearning_config.get("strategies", []) if unlearning_enabled else []

        # Try to get the run ID from the index
        run_id = ""
        try:
            from scripts.results_index import load_index
            entries = load_index()
            for e in entries:
                if e.get("dir") == self.results_dir:
                    run_id = e["id"]
                    break
        except Exception:
            pass

        scenario = self.config.config.get("disagreement", {}).get("active_scenario", "N/A")

        print(f"\n{'=' * 60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Results:     {self.results_dir}")
        print(f"  Experiment:  {self.experiment_type} | Scenario: {scenario} | "
              f"Rounds: {self.fl_rounds} | Clients: {len(self.client_ids_in_experiment)}")
        if unlearning_enabled and strategies:
            print(f"  Unlearning:  {', '.join(strategies)}")
        else:
            print(f"  Unlearning:  disabled")
        print()
        print("  Key files:")
        print("    consolidated_results.json  - All metrics")
        if unlearning_enabled:
            print("    strategy_comparison.json   - Strategy comparison")
        print("    run_metadata.json          - Config for reproducibility")
        if run_id:
            print()
            print(f"  View:    python scripts/fl_runs.py show {run_id}")
            print(f"  Compare: python scripts/fl_runs.py compare {run_id} <other_id>")
        print(f"{'=' * 60}")

    def _register_in_index(self):
        """Register this run in the results index."""
        try:
            from scripts.results_index import register_run
            metadata_path = os.path.join(self.results_dir, "run_metadata.json")
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            register_run(self.results_dir, metadata)
        except Exception as e:
            print(f"Warning: Could not register run in index: {e}")

    def _compare_all_strategies(self, strategies, original_results_dir, strategy_results):
        """Compare results from all unlearning strategies, including track-by-track performance.
        
        This function compares:
        1. Overall metrics (time, final accuracy/loss)
        2. Track performance metrics (per track comparison across strategies)
        3. Identifies best strategy per track and overall
        
        Args:
            strategies: List of strategy names that were run
            original_results_dir: Base results directory
            strategy_results: Dictionary mapping strategy names to their results
        """
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON")
        print(f"{'='*70}\n")
        
        comparison = {
            "strategies": strategies,
            "comparison_timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "strategy_results": {},
            "track_comparison": {},
            "overall_average": {},
            "forget_set_comparison": {}  #New: forget-set metrics comparison
        }
        
        #Load track evaluation results for each strategy
        track_data_by_strategy = {}
        experiment_type = None
        
        for strategy_name in strategies:
            if strategy_name in strategy_results:
                results = strategy_results[strategy_name]
                final_round = results.get("final_round", 0)
                
                # Extract overall metrics
                comparison["strategy_results"][strategy_name] = {
                    "total_time_seconds": results.get("total_time", 0),
                    "experiment_init_time_seconds": results.get("experiment_init_time", 0),
                    "final_round": final_round
                }
                
                # Try to extract final evaluation metrics if available
                if "results" in results and "rounds" in results["results"]:
                    final_round_data = results["results"]["rounds"][-1] if results["results"]["rounds"] else {}
                    if "global_test_accuracy" in final_round_data:
                        comparison["strategy_results"][strategy_name]["final_accuracy"] = final_round_data["global_test_accuracy"]
                    if "global_test_loss" in final_round_data:
                        comparison["strategy_results"][strategy_name]["final_loss"] = final_round_data["global_test_loss"]
                
                # latest post-unlearning track eval from output/server/
                strategy_dir = os.path.join(original_results_dir, f"strategy_{strategy_name}")
                track_eval_dir = os.path.join(strategy_dir, "output", "server")
                track_eval_files = []
                if os.path.exists(track_eval_dir):
                    for fname in os.listdir(track_eval_dir):
                        if fname.startswith("track_evaluation_round_") and fname.endswith(".json"):
                            try:
                                rnd = int(fname.replace("track_evaluation_round_", "").replace(".json", ""))
                                track_eval_files.append((rnd, os.path.join(track_eval_dir, fname)))
                            except ValueError:
                                continue
                
                if track_eval_files:
                    #Use the highest round available
                    track_eval_files.sort(key=lambda x: x[0])
                    final_round = track_eval_files[-1][0]
                    comparison["strategy_results"][strategy_name]["final_round"] = final_round
                    track_eval_path = track_eval_files[-1][1]
                    try:
                        with open(track_eval_path, 'r') as f:
                            track_data = json.load(f)
                            track_data_by_strategy[strategy_name] = track_data
                            
                            #Determine experiment type from track data
                            if not experiment_type:
                                first_track = list(track_data.values())[0] if track_data else {}
                                if "accuracy" in first_track:
                                    experiment_type = "classification"  # MNIST or tabular
                                elif "rmse" in first_track:
                                    experiment_type = "regression"  # N-CMAPSS
                            
                            print(f"  Loaded track evaluation results for {strategy_name} (round {final_round})")
                    except Exception as e:
                        print(f"  Warning: Could not load track evaluation for {strategy_name}: {e}")
                else:
                    print(f"  Warning: Track evaluation file not found for {strategy_name} in {track_eval_dir}")
        
        # Compare track performance across strategies
        if track_data_by_strategy:
            #Get all unique track names across all strategies
            all_tracks = set()
            for strategy_name, track_data in track_data_by_strategy.items():
                all_tracks.update(track_data.keys())
            
            #Compare each track across strategies
            for track_name in sorted(all_tracks):
                track_comparison = {}
                
                for strategy_name in strategies:
                    if strategy_name in track_data_by_strategy:
                        track_data = track_data_by_strategy[strategy_name]
                        if track_name in track_data:
                            track_comparison[strategy_name] = track_data[track_name]
                
                if track_comparison:
                    comparison["track_comparison"][track_name] = track_comparison
                    
                    # best strategy for this track (global = baseline, not unlearning)
                    if track_name == "global":
                        comparison["track_comparison"][track_name]["best_strategy"] = "(baseline)"
                    elif experiment_type == "classification":
                        best_strategy = max(track_comparison.keys(),
                                            key=lambda s: track_comparison[s].get("accuracy", 0))
                        comparison["track_comparison"][track_name]["best_strategy"] = best_strategy
                    else:
                        best_strategy = min(track_comparison.keys(),
                                            key=lambda s: track_comparison[s].get("rmse", float('inf')))
                        comparison["track_comparison"][track_name]["best_strategy"] = best_strategy
            
            # avg performance per strategy across all tracks
            for strategy_name in strategies:
                if strategy_name in track_data_by_strategy:
                    track_data = track_data_by_strategy[strategy_name]
                    
                    if experiment_type == "classification":
                        accuracies = [track_data[track].get("accuracy", 0) 
                                    for track in track_data.keys() 
                                    if "accuracy" in track_data[track]]
                        f1_scores = [track_data[track].get("f1", 0) 
                                   for track in track_data.keys() 
                                   if "f1" in track_data[track]]
                        
                        if accuracies:
                            comparison["overall_average"][strategy_name] = {
                                "average_accuracy": np.mean(accuracies),
                                "average_f1": np.mean(f1_scores) if f1_scores else None
                            }
                    else:
                        rmses = [track_data[track].get("rmse", float('inf')) 
                               for track in track_data.keys() 
                               if "rmse" in track_data[track]]
                        maes = [track_data[track].get("mae", float('inf')) 
                              for track in track_data.keys() 
                              if "mae" in track_data[track]]
                        
                        if rmses:
                            comparison["overall_average"][strategy_name] = {
                                "average_rmse": np.mean(rmses),
                                "average_mae": np.mean(maes) if maes else None
                            }
            
            # best strategy by utility (highest acc / lowest RMSE)
            if experiment_type == "classification":
                best_overall = max(comparison["overall_average"].keys(),
                                  key=lambda s: comparison["overall_average"][s].get("average_accuracy", 0))
            else:
                best_overall = min(comparison["overall_average"].keys(),
                                  key=lambda s: comparison["overall_average"][s].get("average_rmse", float('inf')))
            
            comparison["best_overall_strategy"] = best_overall

            #Efficiency metrics first (needed for best_by_efficiency on training-only time)
            self._calculate_efficiency_metrics(comparison, strategies, original_results_dir)

            #Best by efficiency: fastest by training-only time among those within 90% of best utility
            best_by_efficiency = None
            eff = comparison.get("efficiency_metrics", {})
            sr = comparison.get("strategy_results", {})
            def _training_time(s):
                return eff.get(s, {}).get("total_training_time_s") or sr.get(s, {}).get("total_time_seconds") or float("inf")
            if experiment_type == "classification":
                best_acc = comparison["overall_average"].get(best_overall, {}).get("average_accuracy", 0) or 1e-9
                min_acc = 0.9 * best_acc
                candidates = [s for s in comparison["overall_average"]
                             if comparison["overall_average"][s].get("average_accuracy", 0) >= min_acc
                             and _training_time(s) < float("inf")]
            else:
                best_rmse = comparison["overall_average"].get(best_overall, {}).get("average_rmse", float("inf")) or 1e9
                max_rmse = 1.1 * best_rmse
                candidates = [s for s in comparison["overall_average"]
                             if comparison["overall_average"][s].get("average_rmse", float("inf")) <= max_rmse
                             and _training_time(s) < float("inf")]
            if candidates:
                best_by_efficiency = min(candidates, key=_training_time)
            comparison["best_by_efficiency_strategy"] = best_by_efficiency

            # Load and compare forget-set metrics
            self._compare_forget_set_metrics(comparison, strategies, original_results_dir, experiment_type)
            
            # Load and compare MIA metrics
            self._compare_mia_metrics(comparison, strategies, original_results_dir, experiment_type)
            
            # Calculate behavioral distance metrics
            self._calculate_behavioral_distance(comparison, strategies, original_results_dir, experiment_type)

            #Create visualizations
            self._plot_strategy_comparison(comparison, original_results_dir, experiment_type)
            self._plot_forget_set_comparison(comparison, original_results_dir, experiment_type)
            self._plot_mia_comparison(comparison, original_results_dir, experiment_type)
            self._plot_behavioral_distance(comparison, original_results_dir, experiment_type)
            self._plot_efficiency_metrics(comparison, original_results_dir)
        
        #Save comparison
        comparison_path = os.path.join(original_results_dir, "strategy_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print("Strategy comparison saved to:", comparison_path)
        print("\nStrategy comparison summary:")
        for strategy_name in strategies:
            if strategy_name in comparison["strategy_results"]:
                metrics = comparison["strategy_results"][strategy_name]
                print(f"  {strategy_name}:")
                print(f"    Total time: {metrics.get('total_time_seconds', 0):.2f}s")
                if "final_accuracy" in metrics:
                    print(f"    Final accuracy: {metrics['final_accuracy']:.4f}")
                if "final_loss" in metrics:
                    print(f"    Final loss: {metrics['final_loss']:.4f}")
        
        if comparison.get("overall_average"):
            print("\nTrack performance averages:")
            for strategy_name in strategies:
                if strategy_name in comparison["overall_average"]:
                    avg_metrics = comparison["overall_average"][strategy_name]
                    if "average_accuracy" in avg_metrics:
                        print(f"  {strategy_name}:")
                        print(f"    Average accuracy: {avg_metrics['average_accuracy']:.4f}")
                        if avg_metrics.get("average_f1"):
                            print(f"    Average F1: {avg_metrics['average_f1']:.4f}")
                    elif "average_rmse" in avg_metrics:
                        print(f"  {strategy_name}:")
                        print(f"    Average RMSE: {avg_metrics['average_rmse']:.4f}")
                        if avg_metrics.get("average_mae"):
                            print(f"    Average MAE: {avg_metrics['average_mae']:.4f}")
        
        if comparison.get("best_overall_strategy"):
            print("\nBest overall strategy (by utility: accuracy / RMSE):")
            print(f"  {comparison['best_overall_strategy']}")
        if comparison.get("best_by_efficiency_strategy"):
            print("\nFastest strategy (by training-only time, within 90% of best utility):")
            print(f"  {comparison['best_by_efficiency_strategy']}")
        
        if comparison.get("track_comparison"):
            print("\nBest strategy per track:")
            for track_name, track_data in comparison["track_comparison"].items():
                if "best_strategy" in track_data:
                    print(f"  {track_name}: {track_data['best_strategy']}")
        
        print(f"\n{'='*70}")
        print("ALL STRATEGY RUNS COMPLETED")
        print(f"{'='*70}\n")
    
    def _plot_strategy_comparison(self, comparison, output_dir, experiment_type):
        """Create visualization comparing strategies across tracks.
        
        Args:
            comparison: Comparison dictionary with track_comparison data
            output_dir: Directory to save plots
            experiment_type: "classification" or "regression"
        """
        if not comparison.get("track_comparison"):
            return
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        track_comparison = comparison["track_comparison"]
        strategies = comparison["strategies"]
        
        if experiment_type == "classification":
            # Plot accuracy comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            track_names = sorted(track_comparison.keys())
            x = np.arange(len(track_names))
            width = 0.8 / len(strategies)
            
            for i, strategy_name in enumerate(strategies):
                accuracies = []
                for track_name in track_names:
                    if strategy_name in track_comparison[track_name]:
                        acc = track_comparison[track_name][strategy_name].get("accuracy", 0)
                        accuracies.append(acc)
                    else:
                        accuracies.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, accuracies, width, label=strategy_name, alpha=0.8)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.3f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Track', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Strategy Comparison: Accuracy per Track', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(track_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "strategy_comparison_accuracy.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot F1 comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, strategy_name in enumerate(strategies):
                f1_scores = []
                for track_name in track_names:
                    if strategy_name in track_comparison[track_name]:
                        f1 = track_comparison[track_name][strategy_name].get("f1", 0)
                        f1_scores.append(f1)
                    else:
                        f1_scores.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, f1_scores, width, label=strategy_name, alpha=0.8)
                
                #Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.3f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Track', fontsize=12)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.set_title('Strategy Comparison: F1 Score per Track', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(track_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "strategy_comparison_f1.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        else:  #regression
            # Plot RMSE comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            track_names = sorted(track_comparison.keys())
            x = np.arange(len(track_names))
            width = 0.8 / len(strategies)
            
            for i, strategy_name in enumerate(strategies):
                rmses = []
                for track_name in track_names:
                    if strategy_name in track_comparison[track_name]:
                        rmse = track_comparison[track_name][strategy_name].get("rmse", float('inf'))
                        rmses.append(rmse)
                    else:
                        rmses.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, rmses, width, label=strategy_name, alpha=0.8)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0 and height != float('inf'):
                        ax.annotate(f'{height:.2f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Track', fontsize=12)
            ax.set_ylabel('RMSE', fontsize=12)
            ax.set_title('Strategy Comparison: RMSE per Track (lower is better)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(track_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "strategy_comparison_rmse.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot R² comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, strategy_name in enumerate(strategies):
                r2_scores = []
                for track_name in track_names:
                    if strategy_name in track_comparison[track_name]:
                        r2 = track_comparison[track_name][strategy_name].get("r_squared", -float('inf'))
                        r2_scores.append(r2)
                    else:
                        r2_scores.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, r2_scores, width, label=strategy_name, alpha=0.8)
                
                #Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height != 0:
                        ax.annotate(f'{height:.3f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Track', fontsize=12)
            ax.set_ylabel('R²', fontsize=12)
            ax.set_title('Strategy Comparison: R² per Track (higher is better)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(track_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "strategy_comparison_r2.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Strategy comparison plots saved to {plots_dir}")
    
    def _compare_forget_set_metrics(self, comparison, strategies, results_dir, experiment_type):
        """Vergelijkt forget-set metrics tussen strategieën, met exact_retraining als baseline."""
        if "exact_retraining" not in strategies:
            print("  Warning: exact_retraining not found in strategies. Cannot compare forget-set metrics.")
            return
        
        print("\nForget-set metrics comparison:")

        #Forget-set metrics laden voor alle strategieën
        forget_metrics_by_strategy = {}
        structure = self._get_structure_config()
        
        for strategy_name in strategies:
            strategy_dir = os.path.join(results_dir, f"strategy_{strategy_name}")
            forget_metrics_by_strategy[strategy_name] = {}
            
            # Alle rounds zoeken met tracks die unlearning hebben
            for round_num in range(1, 100):
                round_dir = os.path.join(
                    strategy_dir,
                    structure["round_template"].format(round=round_num)
                )
                
                if not os.path.exists(round_dir):
                    continue
                
                tracks_dir = os.path.join(round_dir, "tracks")
                if not os.path.exists(tracks_dir):
                    continue
                
                # Alle tracks met branches zoeken
                for track_name in os.listdir(tracks_dir):
                    track_dir = os.path.join(tracks_dir, track_name)
                    if not os.path.isdir(track_dir):
                        continue
                    
                    branches_dir = os.path.join(track_dir, "unlearning", "branches", strategy_name)
                    metrics_path = os.path.join(branches_dir, "metrics.json")
                    
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                            
                            # Forget-set metrics ophalen
                            forget_key = "forget_accuracy_original" if experiment_type == "classification" else "forget_rmse_original"
                            if forget_key in metrics:
                                track_key = f"round_{round_num}_{track_name}"
                                if track_key not in forget_metrics_by_strategy[strategy_name]:
                                    forget_metrics_by_strategy[strategy_name][track_key] = {}
                                
                                if experiment_type == "classification":
                                    forget_metrics_by_strategy[strategy_name][track_key] = {
                                        "forget_accuracy_original": metrics.get("forget_accuracy_original", 0),
                                        "forget_accuracy_unlearned": metrics.get("forget_accuracy_unlearned", 0),
                                        "unlearning_score": metrics.get("unlearning_score", 0)
                                    }
                                else:
                                    forget_metrics_by_strategy[strategy_name][track_key] = {
                                        "forget_rmse_original": metrics.get("forget_rmse_original", float('inf')),
                                        "forget_rmse_unlearned": metrics.get("forget_rmse_unlearned", float('inf')),
                                        "unlearning_score": metrics.get("unlearning_score", 0)
                                    }
                        except Exception as e:
                            print(f"  Warning: Could not load forget-set metrics from {metrics_path}: {e}")
        
        #Elke strategie vergelijken met exact_retraining baseline
        exact_retraining_metrics = forget_metrics_by_strategy.get("exact_retraining", {})
        
        if not exact_retraining_metrics:
            print("  Warning: No forget-set metrics found for exact_retraining baseline.")
            return
        
        forget_set_comparison = {}
        
        #Voor elke track/round combinatie
        all_track_keys = set()
        for strategy_name in strategies:
            all_track_keys.update(forget_metrics_by_strategy.get(strategy_name, {}).keys())
        
        for track_key in sorted(all_track_keys):
            exact_metrics = exact_retraining_metrics.get(track_key, {})
            if not exact_metrics:
                continue
            
            track_comparison = {
                "exact_retraining": exact_metrics
            }
            
            # Elke strategie vergelijken met exact_retraining
            for strategy_name in strategies:
                if strategy_name == "exact_retraining":
                    continue
                
                strategy_metrics = forget_metrics_by_strategy.get(strategy_name, {}).get(track_key, {})
                if not strategy_metrics:
                    continue
                
                if experiment_type == "classification":
                    exact_acc = exact_metrics.get("forget_accuracy_unlearned", 0)
                    strategy_acc = strategy_metrics.get("forget_accuracy_unlearned", 0)
                    
                    delta = strategy_acc - exact_acc
                    pct_diff = (delta / exact_acc * 100) if exact_acc > 0 else 0
                    
                    track_comparison[strategy_name] = {
                        **strategy_metrics,
                        "delta_from_exact_retraining": delta,
                        "pct_diff_from_exact_retraining": pct_diff
                    }
                else:
                    exact_rmse = exact_metrics.get("forget_rmse_unlearned", float('inf'))
                    strategy_rmse = strategy_metrics.get("forget_rmse_unlearned", float('inf'))
                    
                    delta = strategy_rmse - exact_rmse
                    pct_diff = (delta / exact_rmse * 100) if exact_rmse > 0 and exact_rmse != float('inf') else 0
                    
                    track_comparison[strategy_name] = {
                        **strategy_metrics,
                        "delta_from_exact_retraining": delta,
                        "pct_diff_from_exact_retraining": pct_diff
                    }
            
            if len(track_comparison) > 1:  # Meer dan alleen exact_retraining
                forget_set_comparison[track_key] = track_comparison
        
        comparison["forget_set_comparison"] = forget_set_comparison
        
        # Overall averages berekenen
        if forget_set_comparison:
            overall_forget_metrics = {}
            
            for strategy_name in strategies:
                if strategy_name == "exact_retraining":
                    continue
                
                deltas = []
                pct_diffs = []
                
                for track_key, track_data in forget_set_comparison.items():
                    if strategy_name in track_data:
                        delta = track_data[strategy_name].get("delta_from_exact_retraining", 0)
                        pct_diff = track_data[strategy_name].get("pct_diff_from_exact_retraining", 0)
                        if delta != 0 or pct_diff != 0:  #Alleen meenemen als vergelijking gemaakt is
                            deltas.append(delta)
                            pct_diffs.append(pct_diff)
                
                if deltas:
                    overall_forget_metrics[strategy_name] = {
                        "average_delta_from_exact_retraining": np.mean(deltas),
                        "average_pct_diff_from_exact_retraining": np.mean(pct_diffs),
                        "std_delta": np.std(deltas),
                        "std_pct_diff": np.std(pct_diffs)
                    }
            
            comparison["forget_set_overall"] = overall_forget_metrics
            
            #Print summary
            print("\nForget-set comparison summary:")
            for strategy_name in strategies:
                if strategy_name == "exact_retraining":
                    continue
                if strategy_name in overall_forget_metrics:
                    metrics = overall_forget_metrics[strategy_name]
                    print(f"  {strategy_name}:")
                    if experiment_type == "classification":
                        print(f"    Average delta from exact_retraining (accuracy): {metrics['average_delta_from_exact_retraining']:.4f}")
                    else:
                        print(f"    Average delta from exact_retraining (RMSE): {metrics['average_delta_from_exact_retraining']:.4f}")
                    print(f"    Average % difference: {metrics['average_pct_diff_from_exact_retraining']:.2f}%")
        
        print(f"  Forget-set comparison completed for {len(forget_set_comparison)} track/round combinations")
    
    def _plot_forget_set_comparison(self, comparison, output_dir, experiment_type):
        """Create visualization comparing forget-set metrics with exact_retraining baseline.
        
        Args:
            comparison: Comparison dictionary with forget_set_comparison data
            output_dir: Directory to save plots
            experiment_type: "classification" or "regression"
        """
        if not comparison.get("forget_set_comparison"):
            return
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        forget_set_comparison = comparison["forget_set_comparison"]
        strategies = [s for s in comparison["strategies"] if s != "exact_retraining"]
        
        if experiment_type == "classification":
            # Plot forget-set accuracy comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            track_keys = sorted(forget_set_comparison.keys())
            x = np.arange(len(track_keys))
            width = 0.8 / (len(strategies) + 1)  # +1 for exact_retraining baseline
            
            # Plot exact_retraining baseline
            exact_accs = []
            for track_key in track_keys:
                exact_metrics = forget_set_comparison[track_key].get("exact_retraining", {})
                exact_accs.append(exact_metrics.get("forget_accuracy_unlearned", 0))
            
            offset = -len(strategies) / 2 * width
            bars = ax.bar(x + offset, exact_accs, width, label="exact_retraining (baseline)", 
                         alpha=0.8, color='gold', edgecolor='black', linewidth=1.5)
            
            #Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=7)
            
            #Plot other strategies
            for i, strategy_name in enumerate(strategies):
                strategy_accs = []
                deltas = []
                for track_key in track_keys:
                    if strategy_name in forget_set_comparison[track_key]:
                        strategy_metrics = forget_set_comparison[track_key][strategy_name]
                        strategy_accs.append(strategy_metrics.get("forget_accuracy_unlearned", 0))
                        deltas.append(strategy_metrics.get("delta_from_exact_retraining", 0))
                    else:
                        strategy_accs.append(0)
                        deltas.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, strategy_accs, width, label=strategy_name, alpha=0.8)
                
                # Add value labels with delta
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0:
                        delta = deltas[j]
                        label = f'{height:.3f}\n(Δ{delta:+.3f})'
                        ax.annotate(label,
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Track/Round', fontsize=12)
            ax.set_ylabel('Forget-Set Accuracy (Unlearned)', fontsize=12)
            ax.set_title('Forget-Set Accuracy Comparison (exact_retraining = golden standard)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                             rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "forget_set_accuracy_comparison.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot unlearning score comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, strategy_name in enumerate(["exact_retraining"] + strategies):
                scores = []
                for track_key in track_keys:
                    if strategy_name in forget_set_comparison[track_key]:
                        metrics = forget_set_comparison[track_key][strategy_name]
                        scores.append(metrics.get("unlearning_score", 0))
                    else:
                        scores.append(0)
                
                offset = (i - len(strategies) / 2) * width if i > 0 else -len(strategies) / 2 * width
                color = 'gold' if strategy_name == "exact_retraining" else None
                edgecolor = 'black' if strategy_name == "exact_retraining" else None
                linewidth = 1.5 if strategy_name == "exact_retraining" else 1
                bars = ax.bar(x + offset, scores, width, label=strategy_name, 
                             alpha=0.8, color=color, edgecolor=edgecolor, linewidth=linewidth)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.3f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Track/Round', fontsize=12)
            ax.set_ylabel('Unlearning Score', fontsize=12)
            ax.set_title('Unlearning Score Comparison (higher = better forgetting)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                             rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "forget_set_unlearning_score_comparison.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        else:  #regression
            #Plot forget-set RMSE comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            track_keys = sorted(forget_set_comparison.keys())
            x = np.arange(len(track_keys))
            width = 0.8 / (len(strategies) + 1)
            
            # Plot exact_retraining baseline
            exact_rmses = []
            for track_key in track_keys:
                exact_metrics = forget_set_comparison[track_key].get("exact_retraining", {})
                exact_rmses.append(exact_metrics.get("forget_rmse_unlearned", float('inf')))
            
            offset = -len(strategies) / 2 * width
            bars = ax.bar(x + offset, exact_rmses, width, label="exact_retraining (baseline)", 
                         alpha=0.8, color='gold', edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height != float('inf') and height > 0:
                    ax.annotate(f'{height:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=7)
            
            # Plot other strategies
            for i, strategy_name in enumerate(strategies):
                strategy_rmses = []
                deltas = []
                for track_key in track_keys:
                    if strategy_name in forget_set_comparison[track_key]:
                        strategy_metrics = forget_set_comparison[track_key][strategy_name]
                        strategy_rmses.append(strategy_metrics.get("forget_rmse_unlearned", float('inf')))
                        deltas.append(strategy_metrics.get("delta_from_exact_retraining", 0))
                    else:
                        strategy_rmses.append(float('inf'))
                        deltas.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, strategy_rmses, width, label=strategy_name, alpha=0.8)
                
                #Add value labels with delta
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height != float('inf') and height > 0:
                        delta = deltas[j]
                        label = f'{height:.2f}\n(Δ{delta:+.2f})'
                        ax.annotate(label,
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Track/Round', fontsize=12)
            ax.set_ylabel('Forget-Set RMSE (Unlearned)', fontsize=12)
            ax.set_title('Forget-Set RMSE Comparison (exact_retraining = golden standard, higher = better forgetting)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                             rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "forget_set_rmse_comparison.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Forget-set comparison plots saved to {plots_dir}")
    
    def _compare_mia_metrics(self, comparison, strategies, results_dir, experiment_type):
        """Vergelijkt MIA metrics tussen strategieën, met exact_retraining als baseline."""
        if "exact_retraining" not in strategies:
            print("  Warning: exact_retraining not found in strategies. Cannot compare MIA metrics.")
            return
        
        print("\nMIA metrics comparison:")

        #MIA metrics laden voor alle strategieën
        mia_metrics_by_strategy = {}
        structure = self._get_structure_config()
        
        for strategy_name in strategies:
            strategy_dir = os.path.join(results_dir, f"strategy_{strategy_name}")
            mia_metrics_by_strategy[strategy_name] = {}
            
            # Alle rounds zoeken met tracks die unlearning hebben
            for round_num in range(1, 100):
                round_dir = os.path.join(
                    strategy_dir,
                    structure["round_template"].format(round=round_num)
                )
                
                if not os.path.exists(round_dir):
                    continue
                
                tracks_dir = os.path.join(round_dir, "tracks")
                if not os.path.exists(tracks_dir):
                    continue
                
                # Alle tracks met branches zoeken
                for track_name in os.listdir(tracks_dir):
                    track_dir = os.path.join(tracks_dir, track_name)
                    if not os.path.isdir(track_dir):
                        continue
                    
                    branches_dir = os.path.join(track_dir, "unlearning", "branches", strategy_name)
                    metrics_path = os.path.join(branches_dir, "metrics.json")
                    
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                            
                            # MIA metrics ophalen
                            if "mia_accuracy_unlearned" in metrics:
                                track_key = f"round_{round_num}_{track_name}"
                                if track_key not in mia_metrics_by_strategy[strategy_name]:
                                    mia_metrics_by_strategy[strategy_name][track_key] = {}
                                
                                mia_metrics_by_strategy[strategy_name][track_key] = {
                                    "mia_accuracy_original": metrics.get("mia_accuracy_original", float('nan')),
                                    "mia_accuracy_unlearned": metrics.get("mia_accuracy_unlearned", float('nan')),
                                    "mia_improvement": metrics.get("mia_improvement", float('nan'))
                                }
                        except Exception as e:
                            print(f"  Warning: Could not load MIA metrics from {metrics_path}: {e}")
        
        #Elke strategie vergelijken met exact_retraining baseline
        exact_retraining_metrics = mia_metrics_by_strategy.get("exact_retraining", {})
        
        if not exact_retraining_metrics:
            print("  Warning: No MIA metrics found for exact_retraining baseline.")
            return
        
        mia_comparison = {}
        
        #Voor elke track/round combinatie
        all_track_keys = set()
        for strategy_name in strategies:
            all_track_keys.update(mia_metrics_by_strategy.get(strategy_name, {}).keys())
        
        for track_key in sorted(all_track_keys):
            exact_metrics = exact_retraining_metrics.get(track_key, {})
            if not exact_metrics:
                continue
            
            track_comparison = {
                "exact_retraining": exact_metrics
            }
            
            # Elke strategie vergelijken met exact_retraining
            for strategy_name in strategies:
                if strategy_name == "exact_retraining":
                    continue
                
                strategy_metrics = mia_metrics_by_strategy.get(strategy_name, {}).get(track_key, {})
                if not strategy_metrics:
                    continue
                
                exact_mia = exact_metrics.get("mia_accuracy_unlearned", float('nan'))
                strategy_mia = strategy_metrics.get("mia_accuracy_unlearned", float('nan'))
                
                # Delta: verschil t.o.v. exact_retraining (idealiter beide ~0.5)
                delta = strategy_mia - exact_mia
                pct_diff = (delta / exact_mia * 100) if not np.isnan(exact_mia) and exact_mia > 0 else 0
                
                track_comparison[strategy_name] = {
                    **strategy_metrics,
                    "delta_from_exact_retraining": delta,
                    "pct_diff_from_exact_retraining": pct_diff
                }
            
            if len(track_comparison) > 1:  # Meer dan alleen exact_retraining
                mia_comparison[track_key] = track_comparison
        
        comparison["mia_comparison"] = mia_comparison
        
        #Overall averages berekenen
        if mia_comparison:
            overall_mia_metrics = {}
            
            for strategy_name in strategies:
                if strategy_name == "exact_retraining":
                    continue
                
                deltas = []
                pct_diffs = []
                improvements = []
                
                for track_key, track_data in mia_comparison.items():
                    if strategy_name in track_data:
                        delta = track_data[strategy_name].get("delta_from_exact_retraining", 0)
                        pct_diff = track_data[strategy_name].get("pct_diff_from_exact_retraining", 0)
                        improvement = track_data[strategy_name].get("mia_improvement", 0)
                        if not np.isnan(delta) and not np.isnan(pct_diff):
                            deltas.append(delta)
                            pct_diffs.append(pct_diff)
                        if not np.isnan(improvement):
                            improvements.append(improvement)
                
                if deltas:
                    overall_mia_metrics[strategy_name] = {
                        "average_delta_from_exact_retraining": float(np.mean(deltas)),
                        "std_delta_from_exact_retraining": float(np.std(deltas)),
                        "average_pct_diff_from_exact_retraining": float(np.mean(pct_diffs)),
                        "average_mia_improvement": float(np.mean(improvements)) if improvements else float('nan')
                    }
            
            comparison["mia_overall_average"] = overall_mia_metrics
        
        print(f"  MIA comparison completed for {len(mia_comparison)} track/round combinations")
    
    def _plot_mia_comparison(self, comparison, output_dir, experiment_type):
        """Create visualization comparing MIA metrics with exact_retraining baseline."""
        if not comparison.get("mia_comparison"):
            return
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        mia_comparison = comparison["mia_comparison"]
        strategies = [s for s in comparison["strategies"] if s != "exact_retraining"]
        
        #Plot MIA accuracy comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        track_keys = sorted(mia_comparison.keys())
        x = np.arange(len(track_keys))
        width = 0.8 / (len(strategies) + 1)  # +1 for exact_retraining baseline
        
        # Plot exact_retraining baseline (idealiter ~0.5)
        exact_mias = []
        for track_key in track_keys:
            exact_metrics = mia_comparison[track_key].get("exact_retraining", {})
            exact_mias.append(exact_metrics.get("mia_accuracy_unlearned", float('nan')))
        
        offset = -len(strategies) / 2 * width
        bars = ax.bar(x + offset, exact_mias, width, label="exact_retraining (baseline)", 
                     alpha=0.8, color='gold', edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=7)
        
        #Plot other strategies
        for i, strategy_name in enumerate(strategies):
            strategy_mias = []
            deltas = []
            for track_key in track_keys:
                if strategy_name in mia_comparison[track_key]:
                    strategy_metrics = mia_comparison[track_key][strategy_name]
                    strategy_mias.append(strategy_metrics.get("mia_accuracy_unlearned", float('nan')))
                    deltas.append(strategy_metrics.get("delta_from_exact_retraining", 0))
                else:
                    strategy_mias.append(float('nan'))
                    deltas.append(0)
            
            offset = (i - len(strategies) / 2 + 0.5) * width
            bars = ax.bar(x + offset, strategy_mias, width, label=strategy_name, alpha=0.8)
            
            #Add value labels with delta
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    delta = deltas[j]
                    label = f'{height:.3f}\n(Δ{delta:+.3f})'
                    ax.annotate(label,
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=7)
        
        # Add reference line at 0.5 (random guess)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random guess (0.5)')
        
        ax.set_xlabel('Track/Round', fontsize=12)
        ax.set_ylabel('MIA Accuracy (Unlearned)', fontsize=12)
        ax.set_title('MIA Accuracy Comparison (exact_retraining = golden standard, ~0.5 = perfect unlearning)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                         rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mia_accuracy_comparison.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot MIA improvement (mia_orig - mia_unl)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, strategy_name in enumerate(strategies):
            improvements = []
            for track_key in track_keys:
                if strategy_name in mia_comparison[track_key]:
                    strategy_metrics = mia_comparison[track_key][strategy_name]
                    improvements.append(strategy_metrics.get("mia_improvement", float('nan')))
                else:
                    improvements.append(float('nan'))
            
            offset = (i - len(strategies) / 2 + 0.5) * width
            bars = ax.bar(x + offset, improvements, width, label=strategy_name, alpha=0.8)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax.annotate(f'{height:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Track/Round', fontsize=12)
        ax.set_ylabel('MIA Improvement (Original - Unlearned)', fontsize=12)
        ax.set_title('MIA Improvement (higher = better unlearning)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                         rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mia_improvement_comparison.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  MIA comparison plots saved to {plots_dir}")
    
    def _calculate_behavioral_distance(self, comparison, strategies, results_dir, experiment_type):
        """Berekent behavioral distance (logit-MSE, KL-divergence) tussen unlearned en exact_retraining modellen."""
        if "exact_retraining" not in strategies:
            print("  Warning: exact_retraining not found in strategies. Cannot calculate behavioral distance.")
            return
        
        print("\nBehavioral distance calculation:")

        #Import behavioral distance functions
        import sys
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        from machine_unlearning_tool.evaluation import calculate_behavioral_distance
        
        #Use first strategy's server config for test loader
        strategy_dir = os.path.join(results_dir, f"strategy_exact_retraining")
        if not os.path.exists(strategy_dir):
            print("  Warning: exact_retraining strategy directory not found.")
            return
        
        # Try to load server config from the first strategy
        # Calculate during unlearning
        
        # Calculate behavioral distance
        #test loader from a server instance
        behavioral_distances = {}
        structure = self._get_structure_config()
        
        #Create a temporary server to get test loader
        try:
            temp_server = self._init_server()
            if not hasattr(temp_server, 'test_loader') or temp_server.test_loader is None:
                print("  Warning: Cannot get test loader. Behavioral distance will not be calculated.")
                return
            
            is_classification = experiment_type == "classification"
            
            # Import behavioral distance function
            import sys
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            from machine_unlearning_tool.evaluation import calculate_behavioral_distance
            from fl_module import create_model
            
            print("  Calculating behavioral distance metrics by comparing models on test set...")
            
            # Load models and calculate distance
            for round_num in range(1, 100):
                # Try to find tracks with branches
                exact_retraining_dir = os.path.join(results_dir, f"strategy_exact_retraining")
                round_dir = os.path.join(
                    exact_retraining_dir,
                    structure["round_template"].format(round=round_num)
                )
                
                if not os.path.exists(round_dir):
                    continue
                
                tracks_dir = os.path.join(round_dir, "tracks")
                if not os.path.exists(tracks_dir):
                    continue
                
                #Find all tracks
                for track_name in os.listdir(tracks_dir):
                    track_dir = os.path.join(tracks_dir, track_name)
                    if not os.path.isdir(track_dir):
                        continue
                    
                    #Load exact_retraining model
                    exact_branch_dir = os.path.join(track_dir, "unlearning", "branches", "exact_retraining")
                    exact_model_path = os.path.join(exact_branch_dir, "model.pt")
                    
                    if not os.path.exists(exact_model_path):
                        continue
                    
                    track_key = f"round_{round_num}_{track_name}"
                    
                    # Create exact_retraining model
                    exact_model = create_model(
                        experiment_type=self.experiment_type,
                        input_dim=getattr(temp_server, 'input_dim', None),
                        output_dim=getattr(temp_server, 'output_dim', None),
                        hidden_dim=getattr(temp_server, 'hidden_dim', None)
                    ).to(temp_server.device)
                    exact_model.load_state_dict(torch.load(exact_model_path, map_location=temp_server.device))
                    
                    # Compare each strategy with exact_retraining
                    for strategy_name in strategies:
                        if strategy_name == "exact_retraining":
                            continue
                        
                        if strategy_name not in behavioral_distances:
                            behavioral_distances[strategy_name] = {}
                        
                        strategy_branch_dir = os.path.join(track_dir, "unlearning", "branches", strategy_name)
                        strategy_model_path = os.path.join(strategy_branch_dir, "model.pt")
                        
                        if not os.path.exists(strategy_model_path):
                            continue
                        
                        # Create strategy model
                        strategy_model = create_model(
                            experiment_type=self.experiment_type,
                            input_dim=getattr(temp_server, 'input_dim', None),
                            output_dim=getattr(temp_server, 'output_dim', None),
                            hidden_dim=getattr(temp_server, 'hidden_dim', None)
                        ).to(temp_server.device)
                        strategy_model.load_state_dict(torch.load(strategy_model_path, map_location=temp_server.device))
                        
                        #Calculate behavioral distance
                        try:
                            distance_metrics = calculate_behavioral_distance(
                                model_unlearned=strategy_model,
                                model_exact_retrain=exact_model,
                                loader=temp_server.test_loader,
                                device=temp_server.device,
                                is_classification=is_classification
                            )
                            
                            if distance_metrics:
                                behavioral_distances[strategy_name][track_key] = distance_metrics
                        except Exception as e:
                            print(f"  Warning: Could not calculate behavioral distance for {strategy_name} on {track_key}: {e}")
        
        except Exception as e:
            print(f"  Warning: Could not calculate behavioral distance: {e}")
            import traceback
            traceback.print_exc()
            return
        
        for strategy_name in strategies:
            if strategy_name == "exact_retraining":
                continue  #Skip exact_retraining zelf
            
            strategy_dir = os.path.join(results_dir, f"strategy_{strategy_name}")
            behavioral_distances[strategy_name] = {}
            
            # Alle rounds zoeken met tracks die unlearning hebben
            for round_num in range(1, 100):
                round_dir = os.path.join(
                    strategy_dir,
                    structure["round_template"].format(round=round_num)
                )
                
                if not os.path.exists(round_dir):
                    continue
                
                tracks_dir = os.path.join(round_dir, "tracks")
                if not os.path.exists(tracks_dir):
                    continue
                
                # Alle tracks met branches zoeken
                for track_name in os.listdir(tracks_dir):
                    track_dir = os.path.join(tracks_dir, track_name)
                    if not os.path.isdir(track_dir):
                        continue
                    
                    branches_dir = os.path.join(track_dir, "unlearning", "branches", strategy_name)
                    metrics_path = os.path.join(branches_dir, "metrics.json")
                    
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                            
                            # Behavioral distance metrics ophalen als beschikbaar
                            if "logit_mse" in metrics or "kl_divergence" in metrics or "output_mse" in metrics:
                                track_key = f"round_{round_num}_{track_name}"
                                behavioral_distances[strategy_name][track_key] = {
                                    "logit_mse": metrics.get("logit_mse", None),
                                    "kl_divergence": metrics.get("kl_divergence", None),
                                    "output_mse": metrics.get("output_mse", None)
                                }
                        except Exception as e:
                            pass  #Silently skip if metrics don't have behavioral distance
        
        if behavioral_distances:
            comparison["behavioral_distance"] = behavioral_distances
            print(f"  Loaded behavioral distance metrics for {len(behavioral_distances)} strategies")
        else:
            print("  Warning: No behavioral distance metrics found in branch metadata.")
            print("  These metrics will be calculated during unlearning in future runs.")
    
    def _plot_behavioral_distance(self, comparison, output_dir, experiment_type):
        """Maakt visualisaties voor behavioral distance metrics."""
        if not comparison.get("behavioral_distance"):
            return
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        behavioral_distance = comparison["behavioral_distance"]
        strategies = list(behavioral_distance.keys())
        
        if experiment_type == "classification":
            #Plot logit-MSE comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Collect all track keys
            all_track_keys = set()
            for strategy_name in strategies:
                all_track_keys.update(behavioral_distance[strategy_name].keys())
            
            track_keys = sorted(all_track_keys)
            x = np.arange(len(track_keys))
            width = 0.8 / len(strategies)
            
            for i, strategy_name in enumerate(strategies):
                logit_mses = []
                for track_key in track_keys:
                    if track_key in behavioral_distance[strategy_name]:
                        mse = behavioral_distance[strategy_name][track_key].get("logit_mse")
                        logit_mses.append(mse if mse is not None else 0)
                    else:
                        logit_mses.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, logit_mses, width, label=strategy_name, alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.4f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Track/Round', fontsize=12)
            ax.set_ylabel('Logit MSE', fontsize=12)
            ax.set_title('Behavioral Distance: Logit MSE (lower = closer to exact_retraining)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                             rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "behavioral_distance_logit_mse.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot KL-divergence comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, strategy_name in enumerate(strategies):
                kl_divs = []
                for track_key in track_keys:
                    if track_key in behavioral_distance[strategy_name]:
                        kl = behavioral_distance[strategy_name][track_key].get("kl_divergence")
                        kl_divs.append(kl if kl is not None else 0)
                    else:
                        kl_divs.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, kl_divs, width, label=strategy_name, alpha=0.8)
                
                #Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.4f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Track/Round', fontsize=12)
            ax.set_ylabel('KL Divergence', fontsize=12)
            ax.set_title('Behavioral Distance: KL Divergence (lower = closer to exact_retraining)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                             rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "behavioral_distance_kl_divergence.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        else:  #regression
            # Plot output-MSE comparison
            fig, ax = plt.subplots(figsize=(14, 8))
            
            all_track_keys = set()
            for strategy_name in strategies:
                all_track_keys.update(behavioral_distance[strategy_name].keys())
            
            track_keys = sorted(all_track_keys)
            x = np.arange(len(track_keys))
            width = 0.8 / len(strategies)
            
            for i, strategy_name in enumerate(strategies):
                output_mses = []
                for track_key in track_keys:
                    if track_key in behavioral_distance[strategy_name]:
                        mse = behavioral_distance[strategy_name][track_key].get("output_mse")
                        output_mses.append(mse if mse is not None else 0)
                    else:
                        output_mses.append(0)
                
                offset = (i - len(strategies) / 2 + 0.5) * width
                bars = ax.bar(x + offset, output_mses, width, label=strategy_name, alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.4f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=7)
            
            ax.set_xlabel('Track/Round', fontsize=12)
            ax.set_ylabel('Output MSE', fontsize=12)
            ax.set_title('Behavioral Distance: Output MSE (lower = closer to exact_retraining)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([tk.replace('round_', 'R').replace('_', ' ') for tk in track_keys], 
                             rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "behavioral_distance_output_mse.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Behavioral distance plots saved to {plots_dir}")
    
    def _calculate_efficiency_metrics(self, comparison, strategies, results_dir):
        """Berekent efficiency en cost metrics: unlearning time, retrain fraction, storage overhead."""
        print("\nEfficiency & cost metrics:")
        
        structure = self._get_structure_config()
        efficiency_data = {}
        
        for strategy_name in strategies:
            strategy_dir = os.path.join(results_dir, f"strategy_{strategy_name}")
            efficiency_data[strategy_name] = {
                "unlearning_times": [],
                "training_times": [],
                "retrain_fractions": [],
                "storage_sizes": [],
                "num_branches": 0
            }
            
            # Alle rounds zoeken met tracks die unlearning hebben
            for round_num in range(1, 100):
                round_dir = os.path.join(
                    strategy_dir,
                    structure["round_template"].format(round=round_num)
                )
                
                if not os.path.exists(round_dir):
                    continue
                
                tracks_dir = os.path.join(round_dir, "tracks")
                if not os.path.exists(tracks_dir):
                    continue
                
                #Alle tracks met branches zoeken
                for track_name in os.listdir(tracks_dir):
                    track_dir = os.path.join(tracks_dir, track_name)
                    if not os.path.isdir(track_dir):
                        continue
                    
                    branches_dir = os.path.join(track_dir, "unlearning", "branches", strategy_name)
                    metrics_path = os.path.join(branches_dir, "metrics.json")
                    model_path = os.path.join(branches_dir, "model.pt")
                    
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                            
                            #Unlearning time (total wall clock) ophalen
                            if "unlearning_time_s" in metrics:
                                efficiency_data[strategy_name]["unlearning_times"].append(metrics["unlearning_time_s"])
                            # Training-only time (excludes I/O, setup, aggregation)
                            if "training_time_s" in metrics:
                                efficiency_data[strategy_name]["training_times"].append(metrics["training_time_s"])

                            # Retrain fraction ophalen
                            if "retrain_fraction" in metrics:
                                efficiency_data[strategy_name]["retrain_fractions"].append(metrics["retrain_fraction"])
                            
                            # Model storage size berekenen
                            if os.path.exists(model_path):
                                try:
                                    model_size_bytes = os.path.getsize(model_path)
                                    model_size_mb = model_size_bytes / (1024 * 1024)
                                    efficiency_data[strategy_name]["storage_sizes"].append(model_size_mb)
                                    efficiency_data[strategy_name]["num_branches"] += 1
                                except Exception:
                                    pass
                        except Exception as e:
                            pass  #Stil overslaan
        
        #Averages en totals berekenen
        efficiency_summary = {}
        for strategy_name in strategies:
            data = efficiency_data[strategy_name]
            
            summary = {}
            
            # Unlearning time (total) statistieken
            if data["unlearning_times"]:
                summary["avg_unlearning_time_s"] = np.mean(data["unlearning_times"])
                summary["total_unlearning_time_s"] = np.sum(data["unlearning_times"])
                summary["min_unlearning_time_s"] = np.min(data["unlearning_times"])
                summary["max_unlearning_time_s"] = np.max(data["unlearning_times"])
            # Training-only time statistieken
            if data["training_times"]:
                summary["avg_training_time_s"] = np.mean(data["training_times"])
                summary["total_training_time_s"] = np.sum(data["training_times"])

            # Retrain fraction statistieken
            if data["retrain_fractions"]:
                summary["avg_retrain_fraction"] = np.mean(data["retrain_fractions"])
                summary["min_retrain_fraction"] = np.min(data["retrain_fractions"])
                summary["max_retrain_fraction"] = np.max(data["retrain_fractions"])
            
            #Storage statistieken
            if data["storage_sizes"]:
                summary["total_storage_mb"] = np.sum(data["storage_sizes"])
                summary["avg_model_size_mb"] = np.mean(data["storage_sizes"])
                summary["num_branches"] = data["num_branches"]
            
            if summary:
                efficiency_summary[strategy_name] = summary
        
        comparison["efficiency_metrics"] = efficiency_summary
        
        #Print summary
        print("\nEfficiency summary:")
        for strategy_name in strategies:
            if strategy_name in efficiency_summary:
                metrics = efficiency_summary[strategy_name]
                print(f"  {strategy_name}:")
                if "avg_unlearning_time_s" in metrics:
                    print(f"    Avg unlearning time (total): {metrics['avg_unlearning_time_s']:.4f}s")
                    print(f"    Total unlearning time: {metrics['total_unlearning_time_s']:.4f}s")
                if "avg_training_time_s" in metrics:
                    print(f"    Avg training-only time: {metrics['avg_training_time_s']:.4f}s")
                    print(f"    Total training-only time: {metrics['total_training_time_s']:.4f}s")
                if "avg_retrain_fraction" in metrics:
                    print(f"    Avg retrain fraction (ρ): {metrics['avg_retrain_fraction']:.4f}")
                if "total_storage_mb" in metrics:
                    print(f"    Total storage: {metrics['total_storage_mb']:.2f} MB")
                    print(f"    Number of branches: {metrics['num_branches']}")
        
        print(f"  Efficiency metrics calculated for {len(efficiency_summary)} strategies")
    
    def _plot_efficiency_metrics(self, comparison, output_dir):
        """Maakt visualisaties voor efficiency metrics."""
        if not comparison.get("efficiency_metrics"):
            return
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        efficiency_metrics = comparison["efficiency_metrics"]
        strategies = list(efficiency_metrics.keys())
        
        # Plot 1: Unlearning Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategy_names = []
        avg_times = []
        for strategy_name in strategies:
            if "avg_unlearning_time_s" in efficiency_metrics[strategy_name]:
                strategy_names.append(strategy_name)
                avg_times.append(efficiency_metrics[strategy_name]["avg_unlearning_time_s"])
        
        if strategy_names:
            bars = ax.bar(strategy_names, avg_times, alpha=0.8, color=['gold', 'lightblue', 'lightcoral'][:len(strategy_names)])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}s',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Strategy', fontsize=12)
            ax.set_ylabel('Average Unlearning Time (seconds)', fontsize=12)
            ax.set_title('Unlearning Time Comparison', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "efficiency_unlearning_time.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Retrain Fraction Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategy_names = []
        retrain_fractions = []
        for strategy_name in strategies:
            if "avg_retrain_fraction" in efficiency_metrics[strategy_name]:
                strategy_names.append(strategy_name)
                retrain_fractions.append(efficiency_metrics[strategy_name]["avg_retrain_fraction"])
        
        if strategy_names:
            bars = ax.bar(strategy_names, retrain_fractions, alpha=0.8, color=['gold', 'lightblue', 'lightcoral'][:len(strategy_names)])
            
            #Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Strategy', fontsize=12)
            ax.set_ylabel('Retrain Fraction (ρ)', fontsize=12)
            ax.set_title('Data Reuse Efficiency: Retrain Fraction (lower = more efficient)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "efficiency_retrain_fraction.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        #Plot 3: Storage Overhead Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategy_names = []
        storage_sizes = []
        num_branches_list = []
        for strategy_name in strategies:
            if "total_storage_mb" in efficiency_metrics[strategy_name]:
                strategy_names.append(strategy_name)
                storage_sizes.append(efficiency_metrics[strategy_name]["total_storage_mb"])
                num_branches_list.append(efficiency_metrics[strategy_name].get("num_branches", 0))
        
        if strategy_names:
            bars = ax.bar(strategy_names, storage_sizes, alpha=0.8, color=['gold', 'lightblue', 'lightcoral'][:len(strategy_names)])
            
            # Add value labels with branch count
            for i, bar in enumerate(bars):
                height = bar.get_height()
                label = f'{height:.2f} MB\n({num_branches_list[i]} branches)'
                ax.annotate(label,
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Strategy', fontsize=12)
            ax.set_ylabel('Total Storage (MB)', fontsize=12)
            ax.set_title('Storage Overhead: Total Model Storage per Strategy', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "efficiency_storage_overhead.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Efficiency plots saved to {plots_dir}")

    def run_federated_learning(self):
        """Execute federated learning with sophisticated disagreement resolution.

        This process implements a disagreement-aware federated learning system that:
        1. Analyzes active client disagreements for each round
        2. Creates separate model tracks for conflicting client groups
        3. Enables clients to train on multiple tracks (primary + background participation)
        4. Performs track-based aggregation with optional deep rewind/incremental finetuning
        5. Evaluates both global and track-specific model performance

        The process follows these steps:
        1. Initialize the global model (global_model_initial)
        2. For each round:
           a. Server analyzes disagreements and prepares track-specific models
           b. Clients load their assigned primary track model + any background track models
           c. Clients train on multiple tracks based on disagreement participation
           d. Server aggregates models using disagreement-aware track algorithm
           e. Server evaluates global model and individual track performance
        
        If unlearning is enabled, this will run a separate complete FL run for each
        unlearning strategy (exact_retraining, sisa, distillation), allowing comparison
        of different unlearning methods across the full FL trajectory.
        """
        # Save run metadata for reproducibility
        self._save_run_metadata()

        # run each strategy as its own pass when unlearning is on
        unlearning_enabled = self.server.unlearning_config.get("enabled", False)
        
        if unlearning_enabled:
            #Get strategies to run
            strategies = self.server.unlearning_config.get("strategies", ["exact_retraining", "sisa", "distillation"])
            
            #Include exact_retraining as baseline
            if "exact_retraining" not in strategies:
                strategies.insert(0, "exact_retraining")
                print("exact_retraining added as baseline (golden standard)")
            
            # Store original results directory
            original_results_dir = self.results_dir
            
           
            print("\nStep 1: running baseline FL run (all strategies share this).")

            baseline_results_dir = os.path.join(original_results_dir, "baseline")
            
 
            original_unlearning_enabled = self.server.unlearning_config.get("enabled", False)
            self.server.unlearning_config["enabled"] = False
            
            # Temporarily change results_dir to baseline directory
            original_results_dir_server = self.server.results_dir
            original_results_dir_clients = {}
            self.server.results_dir = baseline_results_dir
            self.results_dir = baseline_results_dir
            
        
            for client_id, client in self.clients.items():
                original_results_dir_clients[client_id] = client.results_dir
                client.results_dir = baseline_results_dir
            
            # Run baseline FL loop (no unlearning)
            baseline_total_time, baseline_results = self._run_single_strategy_fl_loop(
                None,  #No strategy name for baseline
                self.server,
                self.clients
            )
            
            #Restore unlearning config and results_dir
            self.server.unlearning_config["enabled"] = original_unlearning_enabled
            self.server.results_dir = original_results_dir_server
            self.results_dir = original_results_dir
            
            # Restore clients' results_dir
            for client_id, client in self.clients.items():
                client.results_dir = original_results_dir_clients[client_id]
            
            print(f"Baseline FL run completed and saved to {baseline_results_dir}")
            
            # For each strategy, copy baseline and apply unlearning
            print("\nStep 2: applying unlearning strategies to baseline checkpoints.")

            strategy_results = {}
            for strategy_name in strategies:
                print(f"\nApplying strategy: {strategy_name}")
                
                # Create separate results directory for this strategy
                strategy_results_dir = os.path.join(
                    original_results_dir,
                    f"strategy_{strategy_name}"
                )
                
                #Copy baseline to strategy directory
                if os.path.exists(strategy_results_dir):
                    shutil.rmtree(strategy_results_dir)
                shutil.copytree(baseline_results_dir, strategy_results_dir)
                print(f"Copied baseline checkpoints to {strategy_results_dir}")
                
                #Initialize server for this strategy (with same checkpoints)
                strategy_server = self._init_server_for_strategy(strategy_name, strategy_results_dir)
                
                # Initialize clients for this strategy (same data, different results_dir)
                strategy_clients = self._init_clients_for_strategy(strategy_results_dir)
                strategy_server.set_clients(strategy_clients)
                
                # Apply unlearning to existing checkpoints (no new FL training)
                strategy_total_time, strategy_results[strategy_name] = self._apply_unlearning_to_baseline(
                    strategy_name,
                    strategy_server,
                    strategy_clients
                )
                
                print(f"\n{'='*70}")
                print(f"COMPLETED UNLEARNING FOR STRATEGY: {strategy_name.upper()}")
                print(f"Total time: {strategy_total_time:.2f} seconds")
                print(f"{'='*70}\n")
            
            # Compare all strategies at the end
            try:
                self._compare_all_strategies(strategies, original_results_dir, strategy_results)
            except Exception as e:
                print(f"Warning: Strategy comparison failed: {e}")

        else:
            #Original single FL run (no unlearning or unlearning disabled)
            self._run_single_strategy_fl_loop(None, self.server, self.clients)

        #Register in results index and print summary
        self._register_in_index()
        self._print_run_summary()
    
    def _apply_unlearning_to_baseline(self, strategy_name, server, clients):
        """Apply unlearning strategy to existing baseline checkpoints (no new FL training).
        
        This method iterates through all rounds and tracks from the baseline run,
        and applies unlearning to each track that had exclusions.
        
        Args:
            strategy_name: Name of the unlearning strategy to apply
            server: FederatedServer instance (with results_dir pointing to strategy directory)
            clients: Dictionary of FederatedClient instances
            
        Returns:
            tuple: (total_time, results_dict)
        """
        unlearning_start_time = time.time()
        
        print(f"Applying unlearning strategy '{strategy_name}' to baseline checkpoints...")
        
        # Set the strategy in server config
        original_use_strategy = server.unlearning_config.get("use_strategy", None)
        server.unlearning_config["use_strategy"] = strategy_name
        server.unlearning_config["enabled"] = True
        
        structure = self._get_structure_config()
        results = {}

        # Optional gate: FL_UNLEARN_ROUNDS="35" or "3,10" restricts unlearning to
        # those rounds only (targeted gap-fill runs). Unset = all rounds (default).
        unlearn_rounds_env = os.environ.get("FL_UNLEARN_ROUNDS", "").strip()
        allowed_unlearn_rounds = (
            {int(x) for x in unlearn_rounds_env.split(",") if x.strip()}
            if unlearn_rounds_env else None
        )

        #Iterate through all rounds
        for round_num in range(1, server.fl_rounds + 1):
            if allowed_unlearn_rounds is not None and round_num not in allowed_unlearn_rounds:
                continue
            round_dir = os.path.join(
                server.results_dir,
                structure["round_template"].format(round=round_num)
            )

            if not os.path.exists(round_dir):
                continue
            
            tracks_dir = os.path.join(round_dir, "tracks")
            if not os.path.exists(tracks_dir):
                continue
            
            print(f"\n--- Applying unlearning to Round {round_num} ---")
            
            #Load track metadata to get track info
            track_metadata_path = os.path.join(tracks_dir, "track_metadata.json")
            if not os.path.exists(track_metadata_path):
                continue
            
            try:
                with open(track_metadata_path, 'r') as f:
                    track_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load track metadata for round {round_num}: {e}")
                continue
            
            tracks = track_metadata.get("tracks", {})
            client_tracks = track_metadata.get("client_tracks", {})
            
            # Get all client IDs that participated in this round
            all_client_ids = set()
            for track_clients in tracks.values():
                all_client_ids.update(track_clients)
            
            # Apply unlearning to each track
            for track_name, track_clients_list in tracks.items():
                track_dir = os.path.join(tracks_dir, track_name)
                if not os.path.isdir(track_dir):
                    continue
                
                track_clients_set = set(track_clients_list)
                
                # Only apply unlearning if track excludes some clients
                if track_name == "global" and len(track_clients_set) >= len(all_client_ids):
                    print(f"  Skipping track '{track_name}' (global track with all clients)")
                    continue
                
                #Get excluded clients (clients not in this track)
                excluded_clients = all_client_ids - track_clients_set
                
                if not excluded_clients:
                    print(f"  Skipping track '{track_name}' (no excluded clients)")
                    continue
                
                print(f"  Applying unlearning to track '{track_name}' (excluding clients: {sorted(excluded_clients)})")
                
                #Load track model
                track_model_path = os.path.join(track_dir, "model.pt")
                if not os.path.exists(track_model_path):
                    print(f"    Warning: Track model not found at {track_model_path}")
                    continue
                
                # Apply unlearning for this track
                try:
                    server._apply_unlearning_for_track(
                        round_num=round_num,
                        track_name=track_name,
                        track_clients=track_clients_set,
                        all_client_ids=all_client_ids,
                        track_model_path=track_model_path
                    )
                except Exception as e:
                    print(f"    Error applying unlearning to track '{track_name}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Evaluate unlearned tracks for this round and store metrics
            try:
                print(f"  Evaluating unlearned track models for round {round_num}")
                track_eval = evaluate_track_models(server, round_num)
                if track_eval:
                    results.setdefault("rounds", {})[round_num] = track_eval
            except Exception as e:
                print(f"  Warning: Could not evaluate unlearned tracks for round {round_num}: {e}")
        
        # Restore original strategy setting
        if original_use_strategy is not None:
            server.unlearning_config["use_strategy"] = original_use_strategy
        
        total_time = time.time() - unlearning_start_time
        
        #Add timing to results
        results["total_time"] = total_time
        results["experiment_init_time"] = 0  #Unlearning has no init time, only unlearning time
        
        return total_time, results
    
    def _run_single_strategy_fl_loop(self, strategy_name, server, clients):
        """Run a single complete FL loop for a specific unlearning strategy.
        
        Args:
            strategy_name: Name of the unlearning strategy (None if unlearning disabled)
            server: FederatedServer instance for this strategy
            clients: Dictionary of FederatedClient instances for this strategy
            
        Returns:
            tuple: (total_time, results_dict)
        """
        fl_start_time = time.time()

        strategy_label = f" ({strategy_name})" if strategy_name else ""
        print(f"Starting federated learning with {server.fl_rounds} rounds{strategy_label}...")

        experiment_init_start_time = time.time()

        # Initialize and save the initial global model
        server.initialize_model(round_num=0)

        # Initial evaluation of the global model (round 0)
        server.evaluate_model(fl_round=0)
        print("Initial model evaluation completed")

        experiment_init_time = time.time() - experiment_init_start_time

        # Initialize round timing history
        if not hasattr(server, 'round_timing_history'):
            server.round_timing_history = []

        #Initialize evaluation timing history
        if not hasattr(server, 'evaluation_timing_history'):
            server.evaluation_timing_history = []

        #Main federated learning loop
        for fl_round in range(1, server.fl_rounds + 1):
            # Start timing the entire round
            round_start_time = time.time()

            print(f"\n--- Federated Learning Round {fl_round}/{server.fl_rounds}{strategy_label} ---")

            print("Analyzing disagreements and preparing track-specific models...")
            if fl_round == 1:
                # Create initial tracks from global model
                training_model_dir, track_init_time = server.prepare_training_model(fl_round, use_initial=True)
                print("Created initial track models from global_model_initial for round 1")
            else:
                # Update tracks based on disagreement evolution
                training_model_dir, track_init_time = server.prepare_training_model(fl_round, use_initial=False)
                print(f"Updated track models based on disagreement changes from round {fl_round-1}")

            print("Starting disagreement-aware multi-track client training...")
            client_training_start_time = time.time()
            client_training_times = {}

            #Get fully excluded clients from the server
            for client_id in self.client_ids_in_experiment:
                if client_id not in clients:
                    print(f"Warning: Client {client_id} configured in experiment but not initialized. Skipping.")
                    continue

                client = clients[client_id]
                print(f"Client {client_id}: Loading track models and training with disagreement resolution...")

                #Time individual client training
                client_start_time = time.time()

                client.load_track_models_for_round(fl_round)
                training_results = client.train_with_disagreement_resolution(epochs=self.train_config.get("local_epochs", 5), round_num=fl_round)
                client.save_trained_track_models(fl_round)

                # Record individual client training time
                client_training_time = time.time() - client_start_time
                client_training_times[client_id] = {
                    "training_time_seconds": client_training_time,
                    "epochs": self.train_config.get("local_epochs", 5),
                    "total_training_time_from_results": training_results.get("training_time", {}).get("total_seconds", 0) if training_results else 0
                }
                print(f"Client {client_id} completed training in {client_training_time:.4f} seconds")

            # Calculate total client training phase time
            total_client_training_time = time.time() - client_training_start_time

            # Track-based aggregation
            print("Performing disagreement-aware track-based model aggregation...")

            server.aggregate_with_disagreement_resolution(fl_round)

            #Evaluate models
            print("Evaluating global model and track-specific performance...")

            evaluation_start_time = time.time()

            #Server evaluates both global and track models
            server.evaluate_model(fl_round=fl_round)

            evaluation_time = time.time() - evaluation_start_time

            # Store evaluation timing
            server.evaluation_timing_history.append({
                "round": fl_round,
                "evaluation_time_seconds": evaluation_time
            })

            # Calculate total round time
            total_round_time = time.time() - round_start_time

            # Extract aggregation timing data for this round
            aggregation_timing = None
            if hasattr(server, 'aggregation_timing_history') and server.aggregation_timing_history:
                #Get most recent aggregation timing entry
                for timing_entry in reversed(server.aggregation_timing_history):
                    if timing_entry.get("round") == fl_round:
                        aggregation_timing = timing_entry
                        break

            #Store round timing metrics
            round_timing = {
                "round": fl_round,
                "track_model_initialization_time_seconds": track_init_time,
                "client_training_times": client_training_times,
                "total_client_training_time_seconds": total_client_training_time,
                "evaluation_time_seconds": evaluation_time,
                "total_round_time_seconds": total_round_time,
                "num_participating_clients": len(client_training_times)
            }

            # Add aggregation timing data if available
            if aggregation_timing:
                round_timing.update({
                    "aggregation_time_seconds": aggregation_timing.get("aggregation_time_seconds", 0),
                    "resolution_time_seconds": aggregation_timing.get("resolution_time_seconds", 0),
                    "total_aggregation_time_seconds": aggregation_timing.get("total_aggregation_time_seconds", 0),
                    "has_disagreements": aggregation_timing.get("has_disagreements", False)
                })

            server.round_timing_history.append(round_timing)

            print(f"Round {fl_round} completed with disagreement resolution.")
            print(f"Round {fl_round} timing summary:")
            print(f"  Track initialization: {track_init_time:.4f}s")
            print(f"  Client training phase: {total_client_training_time:.4f}s")
            if aggregation_timing:
                print(f"  Resolution time: {aggregation_timing.get('resolution_time_seconds', 0):.4f}s")
                print(f"  Aggregation time: {aggregation_timing.get('aggregation_time_seconds', 0):.4f}s")
                print(f"  Total aggregation time: {aggregation_timing.get('total_aggregation_time_seconds', 0):.4f}s")
            print(f"  Evaluation time: {evaluation_time:.4f}s")
            print(f"  Total round time: {total_round_time:.4f}s")

        # Calculate total running time
        total_running_time = time.time() - fl_start_time

        # Add timing information to server results and save final results
        server.set_total_running_time(total_running_time)
        server.experiment_init_time = experiment_init_time
        server._save_experiment_results()

        #Extract results for comparison
        results_dict = {
            "total_time": total_running_time,
            "experiment_init_time": experiment_init_time,
            "final_round": server.fl_rounds,
            "results": server.results if hasattr(server, 'results') else {}
        }

        print(f"\nFederated learning with disagreement resolution completed{strategy_label}!")
        print(f"Experiment initialization time: {experiment_init_time:.2f} seconds")
        print(f"Total running time: {total_running_time:.2f} seconds")
        
        return total_running_time, results_dict


def main():
    """Run the orchestrator as a standalone application."""
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument("--config", type=str, default="mock_etcd/configuration.json",
                        help="Path to configuration file")
    parser.add_argument("--override", action="store_true",
                        help="Override configuration with command line arguments")

    #Optional override arguments (only used if --override is specified)
    parser.add_argument("--experiment", type=str, choices=["n_cmapss", "mnist", "cifar10", "tabular", "adult"],
                        help="Experiment type")
    parser.add_argument("--clients", type=int, nargs="+",
                        help="Client IDs")
    parser.add_argument("--fl_rounds", type=int,
                        help="Number of federated learning rounds")
    parser.add_argument("--local_epochs", type=int,
                        help="Number of local training epochs")
    parser.add_argument("--setup_data", action="store_true",
                        help="Set up experiment data (for MNIST)")
    parser.add_argument("--force_setup_data", action="store_true",
                        help="Force data setup even if it exists")
    parser.add_argument("--iid", action="store_true",
                        help="Use IID data distribution (for MNIST)")
    parser.add_argument("--results_dir", type=str,
                        help="Results directory for models and outputs")
    parser.add_argument("--verbose_plots", action="store_true",
                        help="Generate all plots (default: only last round track metrics + track contributions)")
    parser.add_argument("--directory_suffix", type=str, default=None,
                        help="Suffix to append to results directory name (e.g. '_s1')")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario id for display (e.g. 5); written to config when using temp config")

    args = parser.parse_args()

    temp_config_path = None
    try:
        config_path = args.config

        # write a temp config with the overrides applied
        if args.override or args.directory_suffix:
            with open(args.config, 'r') as f:
                config = json.load(f)

            # Apply CLI overrides
            if args.override:
                if args.experiment:
                    config["experiment"]["type"] = args.experiment
                if args.clients:
                    config["experiment"]["client_ids"] = args.clients
                if args.fl_rounds:
                    config["experiment"]["fl_rounds"] = args.fl_rounds
                if args.local_epochs:
                    config["training"]["local_epochs"] = args.local_epochs
                if args.setup_data:
                    config["data"]["setup_data"] = True
                if args.force_setup_data:
                    config["data"]["force_setup_data"] = True
                if args.iid:
                    config["experiment"]["iid"] = True
                if args.results_dir:
                    config["results"]["custom_dir"] = args.results_dir
                if args.verbose_plots:
                    config["results"]["verbose_plots"] = True

            # Apply directory suffix
            if args.directory_suffix:
                config.setdefault("results", {})["directory_suffix"] = args.directory_suffix
            if args.scenario is not None:
                config.setdefault("disagreement", {})["active_scenario"] = args.scenario

            #Write to a temp file so downstream components read correct values
            temp_config_path = os.path.join(
                os.path.dirname(args.config),
                f".tmp_config_{os.getpid()}.json"
            )
            with open(temp_config_path, 'w') as f:
                json.dump(config, f, indent=2)

            config_path = temp_config_path
            print(f"Created temporary config with overrides: {temp_config_path}")

        #Create and run orchestrator
        orchestrator = FederatedOrchestrator(config_path=config_path)

        # Run federated learning
        orchestrator.run_federated_learning()
    finally:
        # Clean up temp config file
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print(f"Cleaned up temporary config: {temp_config_path}")


if __name__ == "__main__":
    main()
