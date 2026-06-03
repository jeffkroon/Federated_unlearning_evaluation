"""Sklearn-based federated learning server implementation."""

import os
import json
import pickle
import time
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import glob

# Import machine_unlearning_tool for model creation
import sys
#Path: fl_server_sklearn/server.py -> fl-disagreement-resolution -> Thesis -> machine_unlearning_tool
#So we need to go up 3 levels from fl_server_sklearn
current_file = os.path.abspath(__file__)  # /.../Thesis/fl-disagreement-resolution/fl_server_sklearn/server.py
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # /.../Thesis
# Add Thesis directory to path so we can import machine_unlearning_tool
sys.path.insert(0, base_dir)
from machine_unlearning_tool import create_model

import fl_module
from fl_server_sklearn.aggregation import (
    aggregate_sklearn_models,
    aggregate_models_from_files,
    get_structure_config,
)
from fl_server_sklearn.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks,
)
from fl_server_sklearn.evaluation import evaluate_sklearn_model

#Import unlearning components
from fl_server.branching import BranchRegistry
from fl_server.unlearning_strategies import (
    get_strategy,
    collect_all_client_data
)


class SklearnFederatedServer:
    """Server-side implementation for sklearn-based federated learning."""

    def __init__(
        self,
        experiment_type: str,
        model_type: str = "random_forest",  #"random_forest" or "xgboost"
        test_dir=None,
        test_units=None,
        results_dir=None,
        verbose_plots=False,
        model_params: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the sklearn federated learning server.

        Args:
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            model_type: Type of sklearn model ('random_forest' or 'xgboost')
            test_dir: Directory containing test data
            test_units: List of unit IDs to use for testing (for N-CMAPSS)
            results_dir: Directory for storing models and results
            verbose_plots: Whether to generate all plots
            model_params: Parameters for sklearn model
        """
        self.experiment_type = experiment_type
        self.model_type = model_type
        self.test_dir = test_dir
        self.test_units = test_units
        self.results_dir = results_dir
        self.verbose_plots = verbose_plots
        self.round = 0
        self.global_model = None
        self.model_params = model_params or {}
        self.config_path = config_path  # Store config path for unlearning config loading
        
        # Results tracking
        self.results = {
            "experiment_type": experiment_type,
            "model_type": model_type,
            "rounds": []
        }
        
        self.training_history = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": []
        }
        
        # Disagreement resolution
        self.fully_excluded_clients_for_current_round = set()
        self.disagreement_settings = self._get_disagreement_config()
        
        #Unlearning configuration
        self.unlearning_config = self._get_unlearning_config()
        self.clients = None  #Will be set by orchestrator
        
        # Create output directories
        if results_dir:
            structure = self._get_structure_config()
            self.output_dir = os.path.join(results_dir, "output", "server_sklearn")
            os.makedirs(self.output_dir, exist_ok=True)
            model_storage_path = os.path.join(results_dir, structure["model_storage_dir"])
            os.makedirs(model_storage_path, exist_ok=True)
        else:
            os.makedirs("output/server_sklearn_results", exist_ok=True)
        
        # Initialize model
        self._init_model()
        
        # Load test data
        if test_dir:
            try:
                self.load_test_data()
            except Exception as e:
                print(f"Warning: Could not load test data: {e}")
                print("Test data will be loaded later if needed.")
    
    def _get_structure_config(self):
        """Get directory structure configuration."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "mock_etcd/configuration.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        return config["results"]["structure"]
        except Exception:
            pass
        
        #Default structure
        return {
            "model_storage_dir": "model_storage_sklearn",
            "global_model_initial": "model_storage_sklearn/global_model_initial",
            "round_template": "model_storage_sklearn/round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "global_model_aggregated": "global_model_aggregated",
            "output_dir": "output",
            "client_prefix": "client_"
        }
    
    def _init_model(self):
        """Initialize global sklearn model."""
        if self.experiment_type == "n_cmapss":
            self.seq_len = 50
            self.n_features = 20
            self.input_dim = self.seq_len * self.n_features
            #For sklearn, we flatten sequences
            input_size = self.input_dim
            # For regression, use Regressor
            sklearn_model_type = self.model_type  # random_forest or xgboost
        elif self.experiment_type == "mnist":
            # MNIST: 28x28 = 784 pixels
            input_size = 784
            #For classification, use Classifier
            if self.model_type == "random_forest":
                sklearn_model_type = "random_forest_classifier"
            elif self.model_type == "xgboost":
                sklearn_model_type = "xgboost_classifier"
            else:
                sklearn_model_type = self.model_type
        else:
            raise ValueError(f"Unsupported experiment type: {self.experiment_type}")
        
        #Create sklearn model - need to handle classification vs regression
        if sklearn_model_type in ["random_forest_classifier", "xgboost_classifier"]:
            # Import classifiers
            if sklearn_model_type == "random_forest_classifier":
                from sklearn.ensemble import RandomForestClassifier
                self.global_model = RandomForestClassifier(**self.model_params)
            elif sklearn_model_type == "xgboost_classifier":
                from xgboost import XGBClassifier
                self.global_model = XGBClassifier(**self.model_params)
        else:
            # Use create_model for regression models
            self.global_model = create_model(
                model_type=self.model_type,
                input_size=input_size,  # Not used for sklearn but kept for compatibility
                **self.model_params
            )
        
        print(f"Initialized global {self.model_type} model for {self.experiment_type}")
    
    def load_test_data(self, sample_size=500):
        """Load test data for model evaluation.
        
        Args:
            sample_size: Maximum number of samples to load per test unit
        """
        if self.experiment_type == "n_cmapss":
            if not self.test_dir or not self.test_units:
                raise ValueError("Test directory and test units must be provided for N-CMAPSS")
            
            #Load test data
            test_samples, test_labels = fl_module.load_ncmapss_test_data(
                self.test_dir,
                self.test_units,
                sample_size=sample_size
            )
            
            #Preprocess
            _, test_normalized, _ = fl_module.preprocess_ncmapss_data(test_samples, test_samples)
            
            # Convert to numpy arrays (flatten sequences for sklearn)
            n_samples, seq_len, n_features = test_normalized.shape
            self.test_X = test_normalized.reshape(n_samples, -1)  # Flatten to [N, seq_len * n_features]
            self.test_y = test_labels
            
            print(f"Loaded test data with {len(self.test_X)} samples")
            
        elif self.experiment_type == "mnist":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for MNIST")
            
            # Load MNIST test data
            test_images, test_labels = fl_module.load_mnist_test_data(
                test_dir=self.test_dir
            )
            
            #Flatten images for sklearn
            self.test_X = test_images.reshape(len(test_images), -1)  #[N, 784]
            self.test_y = test_labels
            
            print(f"Loaded MNIST test data with {len(self.test_X)} samples")
        else:
            raise NotImplementedError(f"{self.experiment_type} test data loading not implemented")
    
    def save_model(self, model_dir):
        """Save global sklearn model to a directory.
        
        Args:
            model_dir: Directory to save the model to
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save sklearn model using pickle
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.global_model, f)
        
        # Save metadata
        metadata = {
            "experiment_type": self.experiment_type,
            "model_type": self.model_type,
            "round": self.round,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved global sklearn model to {model_dir}")
    
    def load_model(self, model_dir):
        """Load global sklearn model from a directory.
        
        Args:
            model_dir: Directory containing the model
        """
        model_path = os.path.join(model_dir, "model.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        # Load sklearn model
        with open(model_path, 'rb') as f:
            self.global_model = pickle.load(f)
        
        #Load metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "round" in metadata:
                    print(f"Loaded model from round {metadata['round']}")
        
        print(f"Loaded global sklearn model from {model_dir}")
    
    def initialize_model(self, round_num=0):
        """Initialize and save the initial model.
        
        Args:
            round_num: Usually 0 for the initial model
        """
        if not self.results_dir:
            return
        
        structure = self._get_structure_config()
        initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
        os.makedirs(initial_model_dir, exist_ok=True)
        
        self.save_model(initial_model_dir)
        print("Initialized and saved the initial global sklearn model")
    
    def init_experiment(self, fl_rounds, client_ids, iid=False):
        """Initialize experiment metadata.
        
        Args:
            fl_rounds: Number of federated learning rounds
            client_ids: List of client IDs
            iid: Whether the data distribution is IID (for MNIST)
        """
        self.fl_rounds = fl_rounds
        self.client_ids = client_ids
        self.iid = iid
        
        #Update results with experiment metadata
        self.results["fl_rounds"] = fl_rounds
        self.results["client_ids"] = client_ids
        self.results["iid"] = iid if self.experiment_type == "mnist" else None
    
    def aggregate_models(self, round_num, clients_dir, aggregation_weights=None):
        """Aggregate client sklearn models using ensemble method with disagreement resolution.
        
        Args:
            round_num: Current round number
            clients_dir: Directory containing client model directories
            aggregation_weights: Optional dictionary mapping client IDs to weights
            
        Returns:
            Aggregated sklearn model (ensemble)
        """
        # Use the full aggregation with disagreement resolution
        return aggregate_models_from_files(self, clients_dir, aggregation_weights)
    
    def evaluate_model(self, fl_round=0):
        """Evaluate the global sklearn model on test data.
        
        Args:
            fl_round: Current federated learning round
        """
        if not hasattr(self, 'test_X') or not hasattr(self, 'test_y'):
            print("Warning: Test data not loaded. Skipping evaluation.")
            return
        
        metrics = evaluate_sklearn_model(
            self.global_model,
            self.test_X,
            self.test_y,
            experiment_type=self.experiment_type
        )
        
        # Store results
        self.training_history["rounds"].append(fl_round)
        self.training_history["global_test_loss"].append(metrics.get("loss", 0.0))
        if "accuracy" in metrics:
            self.training_history["global_test_accuracy"].append(metrics["accuracy"])
        
        print(f"Round {fl_round} - Test Metrics: {metrics}")
        
        return metrics
    
    def _get_disagreement_config(self):
        """Get the disagreement configuration.
        
        Returns:
            dict: Disagreement configuration
        """
        default_disagreement_config = {
            "initiation_mechanism": "shallow",
            "lifting_mechanism": "shallow",
            "deep_lifting_finetune_rounds": 3
        }
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "mock_etcd/configuration.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "disagreement" in config:
                        loaded_disagreement_config = config["disagreement"]
                        for key, value in default_disagreement_config.items():
                            if key not in loaded_disagreement_config:
                                loaded_disagreement_config[key] = value
                        return loaded_disagreement_config
        except Exception as e:
            print(f"Error loading disagreement configuration: {e}. Using default config.")
        
        return default_disagreement_config
    
    def _get_unlearning_config(self):
        """Get unlearning configuration.
        
        Returns:
            dict: Unlearning configuration
        """
        default_unlearning_config = {
            "enabled": False,
            "model_type": "random_forest",  # Default to sklearn model type
            "strategies": ["exact_retraining", "sisa", "distillation"],
            "model_params": {},
            "train_params": {"epochs": 10, "batch_size": 64, "lr": 1e-3}
        }
        
        #Try to use the config path passed to server, otherwise fall back to default
        config_paths_to_try = []
        if self.config_path:
            config_paths_to_try.append(self.config_path)
        
        #Also try default location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config_path = os.path.join(base_dir, "mock_etcd/configuration.json")
        if default_config_path not in config_paths_to_try:
            config_paths_to_try.append(default_config_path)
        
        for config_path in config_paths_to_try:
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if "unlearning" in config:
                            loaded_config = config["unlearning"]
                            for key, value in default_unlearning_config.items():
                                if key not in loaded_config:
                                    loaded_config[key] = value
                            return loaded_config
            except Exception as e:
                print(f"Error loading unlearning configuration from {config_path}: {e}. Trying next...")
                continue
        
        return default_unlearning_config
    
    def set_clients(self, clients):
        """Set clients dictionary for unlearning.
        
        Args:
            clients: Dictionary mapping client IDs to SklearnFederatedClient instances
        """
        self.clients = clients
    
    def _apply_unlearning(self, round_num: int, forget_ids: List[int]):
        """Apply unlearning strategies when clients are excluded (sklearn version).
        
        Args:
            round_num: Current round number
            forget_ids: List of client IDs to forget
        """
        if not self.clients:
            print("Warning: Clients not set. Cannot apply unlearning.")
            return
        
        print(f"\n=== APPLYING UNLEARNING FOR ROUND {round_num} ===")
        print(f"Forgetting clients: {forget_ids}")
        
        # Initialize branch registry
        branch_registry = BranchRegistry(self.results_dir, round_num)
        
        # Save pre-unlearning checkpoint (sklearn model with pickle)
        pre_checkpoint = {"type": "sklearn", "model": self.global_model}
        branch_registry.save_pre_unlearning_checkpoint(
            pre_checkpoint,
            forget_ids,
            metadata={"round": round_num, "experiment_type": self.experiment_type, "is_sklearn": True}
        )
        
        # Validate forget_ids
        available_client_ids = set(self.clients.keys())
        invalid_forget_ids = [fid for fid in forget_ids if fid not in available_client_ids]
        if invalid_forget_ids:
            print(f"Error: Invalid forget_ids {invalid_forget_ids}. Available client IDs: {sorted(available_client_ids)}")
            return
        
        if not forget_ids:
            print("Warning: No forget_ids provided. Skipping unlearning.")
            return
        
        #Collect all client data
        print("Collecting all client data for unlearning...")
        all_data = collect_all_client_data(self.clients, self.experiment_type)
        
        #Validate that we have data
        if all_data is None or len(all_data.get("df", [])) == 0:
            print("Error: No data collected from clients. Cannot perform unlearning.")
            return
        
        # Get strategies to run
        strategies_to_run = self.unlearning_config.get("strategies", ["exact_retraining"])
        model_type = self.unlearning_config.get("model_type", self.model_type)  # Use server's model_type as default
        model_params = self.unlearning_config.get("model_params", self.model_params)
        train_params = self.unlearning_config.get("train_params", {})
        
        # Check if we can reuse existing branches (same policy situation)
        reuse_enabled = self.unlearning_config.get("reuse_existing_branches", True)
        
        #Run each strategy
        branch_results = {}
        for strategy_name in strategies_to_run:
            print(f"\nRunning unlearning strategy: {strategy_name}")
            
            #Check if we can reuse an existing branch
            if reuse_enabled:
                existing_branch = branch_registry.find_existing_branch(
                    forget_ids=forget_ids,
                    strategy_name=strategy_name,
                    model_type=model_type,
                    max_search_rounds=10
                )
                
                if existing_branch:
                    print(f"Found existing branch from round {existing_branch['round']} with same policy!")
                    print(f"  Forget IDs: {forget_ids}")
                    print(f"  Strategy: {strategy_name}")
                    print(f"  Model type: {model_type}")
                    print(f"  Reusing model from round {existing_branch['round']}")
                    
                    # Load the existing model (pickle for sklearn)
                    existing_model_path = existing_branch["path"]
                    with open(existing_model_path, 'rb') as f:
                        existing_model_state = pickle.load(f)
                    
                    # If it's wrapped in dict, extract model
                    if isinstance(existing_model_state, dict) and "model" in existing_model_state:
                        existing_model = existing_model_state["model"]
                    else:
                        existing_model = existing_model_state
                    
                    # Load metrics from existing branch
                    existing_metrics_path = os.path.join(existing_branch["branch_dir"], "metrics.json")
                    if os.path.exists(existing_metrics_path):
                        with open(existing_metrics_path, 'r') as f:
                            existing_metrics = json.load(f)
                    else:
                        existing_metrics = {"note": "reused from previous round", "unlearning_time_s": 0.0}
                    
                    #Save as new branch checkpoint (reference to old one)
                    branch_registry.save_branch_checkpoint(
                        strategy_name,
                        {"type": "sklearn", "model": existing_model},
                        existing_metrics,
                        metadata={
                            "round": round_num,
                            "forget_ids": forget_ids,
                            "reused_from_round": existing_branch["round"],
                            "reused": True,
                            "model_type": model_type,
                            "is_sklearn": True
                        }
                    )
                    
                    branch_results[strategy_name] = existing_metrics
                    print(f"Reused branch '{strategy_name}' from round {existing_branch['round']}")
                    continue
            
            #No existing branch found, run unlearning
            try:
                # Get strategy
                from fl_server.unlearning_strategies import get_strategy
                strategy = get_strategy(
                    strategy_name,
                    model_type=model_type,
                    model_params=model_params,
                    train_params=train_params,
                    device=None  # sklearn doesn't use device
                )
                
                # Apply unlearning
                result = strategy.unlearn(
                    pretrained_model=self.global_model,
                    all_data=all_data,
                    forget_ids=forget_ids,
                    round_num=round_num
                )
                
                #Save branch checkpoint (sklearn model with pickle)
                unlearned_model = result["model"]
                model_state = {"type": "sklearn", "model": unlearned_model}
                
                branch_registry.save_branch_checkpoint(
                    strategy_name,
                    model_state,
                    result["metrics"],
                    metadata={
                        "round": round_num,
                        "forget_ids": forget_ids,
                        "model_type": model_type,
                        "reused": False,
                        "is_sklearn": True
                    }
                )
                
                branch_results[strategy_name] = result["metrics"]
                print(f"Strategy '{strategy_name}' completed. Metrics: {result['metrics']}")
                
            except Exception as e:
                print(f"Error running strategy '{strategy_name}': {e}")
                import traceback
                traceback.print_exc()
                #Continue to next strategy instead of failing completely
        
        # Handle case where all strategies failed
        if not branch_results:
            print("\nWARNING:  WARNING: All unlearning strategies failed!")
            print("Falling back to pre-unlearning model state.")
            
            # Load pre-unlearning checkpoint as fallback
            pre_checkpoint = branch_registry.load_pre_unlearning_checkpoint()
            if pre_checkpoint:
                if isinstance(pre_checkpoint, dict) and "model" in pre_checkpoint:
                    self.global_model = pre_checkpoint["model"]
                else:
                    self.global_model = pre_checkpoint
                print("Loaded pre-unlearning model as fallback.")
            else:
                print("WARNING:  Could not load pre-unlearning checkpoint. Model state unchanged.")
            
            print(f"=== UNLEARNING COMPLETE FOR ROUND {round_num} (FALLBACK) ===\n")
            return
        
        # Save comparison
        if branch_results:
            comparison = {
                "round": round_num,
                "forget_ids": forget_ids,
                "strategies": branch_results,
                "best_strategy": min(branch_results.keys(), key=lambda k: branch_results[k].get("rmse", float('inf')))
            }
            branch_registry.save_comparison(comparison)
            
            #Use best strategy's model (or exact_retraining as default)
            use_strategy = self.unlearning_config.get("use_strategy", "exact_retraining")
            if use_strategy not in branch_results:
                use_strategy = "exact_retraining"
            
            print(f"\nUsing unlearned model from strategy: {use_strategy}")
            #Load the unlearned model
            unlearned_state = branch_registry.load_branch_checkpoint(use_strategy)
            
            if unlearned_state:
                # Extract sklearn model from dict
                if isinstance(unlearned_state, dict) and "model" in unlearned_state:
                    self.global_model = unlearned_state["model"]
                    print(f"Loaded unlearned sklearn model from '{use_strategy}' branch")
                elif isinstance(unlearned_state, dict) and "type" in unlearned_state:
                    # Should have model key
                    if "model" in unlearned_state:
                        self.global_model = unlearned_state["model"]
                        print(f"Loaded unlearned sklearn model from '{use_strategy}' branch")
                    else:
                        print(f"WARNING:  Warning: Could not extract model from checkpoint. Using pre-unlearning model.")
                        pre_checkpoint = branch_registry.load_pre_unlearning_checkpoint()
                        if pre_checkpoint and isinstance(pre_checkpoint, dict) and "model" in pre_checkpoint:
                            self.global_model = pre_checkpoint["model"]
                else:
                    # Direct model object
                    self.global_model = unlearned_state
                    print(f"Loaded unlearned sklearn model from '{use_strategy}' branch")
            else:
                print(f"WARNING:  Warning: Could not load model from strategy '{use_strategy}'. Using pre-unlearning model.")
                pre_checkpoint = branch_registry.load_pre_unlearning_checkpoint()
                if pre_checkpoint:
                    if isinstance(pre_checkpoint, dict) and "model" in pre_checkpoint:
                        self.global_model = pre_checkpoint["model"]
                    else:
                        self.global_model = pre_checkpoint
                    print("Loaded pre-unlearning model as fallback.")
        
        print(f"=== UNLEARNING COMPLETE FOR ROUND {round_num} ===\n")
    
    def prepare_training_model(self, round_num, use_initial=False):
        """Prepare the global model for a specific round with disagreement resolution.
        
        Args:
            round_num: The current round number
            use_initial: Whether to use the initial model (for round 1)
        
        Returns:
            tuple: (Path to the prepared model directory, preparation time in seconds)
        """
        import time
        
        if not self.results_dir:
            return None, 0.0
        
        preparation_start_time = time.time()
        
        print(f"\n=== SERVER PREPARATION FOR ROUND {round_num} ===")
        self.fully_excluded_clients_for_current_round = set()
        
        structure = self._get_structure_config()
        disagreement_settings = self._get_disagreement_config()
        initiation_mechanism = disagreement_settings.get("initiation_mechanism", "shallow")
        lifting_mechanism = disagreement_settings.get("lifting_mechanism", "shallow")
        finetune_total_rounds = disagreement_settings.get("deep_lifting_finetune_rounds", 3)
        print(f"  Using initiation_mechanism: {initiation_mechanism}, lifting_mechanism: {lifting_mechanism}, deep_lifting_finetune_rounds: {finetune_total_rounds}")
        
        round_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num))
        os.makedirs(round_dir, exist_ok=True)
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)
        
        etcd_dir = "mock_etcd"
        disagreements = load_disagreements(etcd_dir)
        active_disagreements = get_active_disagreements(disagreements, round_num)
        
        if active_disagreements:
            for client_id_str, disags_list in active_disagreements.items():
                for disag_item in disags_list:
                    if disag_item.get('type') == 'full':
                        try:
                            numeric_id = int(client_id_str.split('_')[-1]) if '_' in client_id_str else int(client_id_str)
                            self.fully_excluded_clients_for_current_round.add(numeric_id)
                        except ValueError:
                            print(f"Warning: Could not parse numeric ID from client_id_str '{client_id_str}' for full exclusion.")
            if self.fully_excluded_clients_for_current_round:
                print(f"  Fully excluded clients identified for round {round_num}: {sorted(list(self.fully_excluded_clients_for_current_round))}")
        
        #Apply unlearning if enabled and clients are excluded
        if self.fully_excluded_clients_for_current_round and self.unlearning_config.get("enabled", False):
            self._apply_unlearning(round_num, list(self.fully_excluded_clients_for_current_round))
        
        if use_initial:
            source_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            print(f"Round {round_num} starting with initial global model from {source_model_dir}")
            self.load_model(source_model_dir)
            
            if active_disagreements:
                print(f"Creating initial tracks for round {round_num} from global initial model")
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks:
                    print("Warning: client_ids not found in self.results, attempting to infer from client_dirs.")
                    import glob
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients_sklearn", "client_*")
                    client_dirs = glob.glob(client_dirs_pattern)
                    client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs]) if client_dirs else []
                    if not client_ids_for_tracks:
                        print(f"No client directories found at {client_dirs_pattern}. Track creation might be affected.")
                
                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)
                tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(tracks_dir, exist_ok=True)
                metadata_path = os.path.join(tracks_dir, "track_metadata.json")
                track_metadata_content = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }
                with open(metadata_path, "w") as f:
                    json.dump(track_metadata_content, f, indent=2)
                
                for track_name_iter in track_info.get("tracks", {}):
                    track_dir_iter = os.path.join(tracks_dir, track_name_iter)
                    os.makedirs(track_dir_iter, exist_ok=True)
                    #Save sklearn model with pickle
                    model_path = os.path.join(track_dir_iter, "model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.global_model, f)
                    individual_track_meta = {
                        "track_name": track_name_iter,
                        "round": round_num,
                        "client_ids": list(track_info.get("tracks", {}).get(track_name_iter, [])),
                        "rewound_this_round": False,
                        "finetuning_status": {}
                    }
                    with open(os.path.join(track_dir_iter, "metadata.json"), "w") as f_meta_track:
                        json.dump(individual_track_meta, f_meta_track, indent=2)
                    print(f"Created initial track '{track_name_iter}' for round {round_num}")
            
            self.save_model(training_model_dir)
            print(f"Saved global model for training at {training_model_dir}")
        else:
            # Not use_initial (round_num > 1 typically)
            prev_global_aggregated_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1), structure["global_model_aggregated"])
            if os.path.exists(os.path.join(prev_global_aggregated_dir, "model.pkl")):
                print(f"Loading main global model from previous round {round_num-1}'s aggregated model: {prev_global_aggregated_dir}")
                self.load_model(prev_global_aggregated_dir)
            else:
                print(f"Warning: Previous round's aggregated model not found at {prev_global_aggregated_dir}. Falling back to initial global model.")
                initial_model_dir_fallback = os.path.join(self.results_dir, structure["global_model_initial"])
                self.load_model(initial_model_dir_fallback)
            self.save_model(training_model_dir)
            print(f"Saved main global model for training in round {round_num} at {training_model_dir}")
            
            if active_disagreements:
                print(f"Active disagreements found for round {round_num}. Initiation mechanism: {initiation_mechanism}")
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks:
                    print("Warning: client_ids not found in self.results for track creation in round > 1.")
                    import glob
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients_sklearn", "client_*")
                    client_dirs = glob.glob(client_dirs_pattern)
                    client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs]) if client_dirs else []
                
                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)
                current_round_tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(current_round_tracks_dir, exist_ok=True)
                prev_round_tracks_main_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1), "tracks")
                
                for track_name, clients_in_this_track_set in track_info.get("tracks", {}).items():
                    clients_in_this_track_list = list(clients_in_this_track_set)
                    current_specific_track_dir = os.path.join(current_round_tracks_dir, track_name)
                    os.makedirs(current_specific_track_dir, exist_ok=True)
                    prev_specific_track_dir = os.path.join(prev_round_tracks_main_dir, track_name)
                    track_existed_previously = os.path.exists(os.path.join(prev_specific_track_dir, "model.pkl"))
                    
                    if track_existed_previously:
                        # Load previous track model
                        with open(os.path.join(prev_specific_track_dir, "model.pkl"), 'rb') as f:
                            prev_track_model = pickle.load(f)
                        # Save to current round
                        with open(os.path.join(current_specific_track_dir, "model.pkl"), 'wb') as f:
                            pickle.dump(prev_track_model, f)
                        print(f"    Track '{track_name}' continued from previous round")
                    else:
                        #Create new track from global model
                        with open(os.path.join(current_specific_track_dir, "model.pkl"), 'wb') as f:
                            pickle.dump(self.global_model, f)
                        print(f"    Track '{track_name}' created from global model")
                    
                    #Save track metadata
                    individual_track_meta = {
                        "track_name": track_name,
                        "round": round_num,
                        "client_ids": clients_in_this_track_list,
                        "rewound_this_round": not track_existed_previously,
                        "finetuning_status": {}
                    }
                    with open(os.path.join(current_specific_track_dir, "metadata.json"), "w") as f_meta_track:
                        json.dump(individual_track_meta, f_meta_track, indent=2)
                
                # Save track metadata
                track_metadata_content = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }
                metadata_path = os.path.join(current_round_tracks_dir, "track_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(track_metadata_content, f, indent=2)
        
        preparation_time = time.time() - preparation_start_time
        print(f"=== SERVER PREPARATION COMPLETE FOR ROUND {round_num} ({preparation_time:.4f}s) ===\n")
        
        return training_model_dir, preparation_time

