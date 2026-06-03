"""Main federated learning server implementation."""

import os
import torch
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict
import glob

from fl_module import create_model
import fl_module
from fl_server.evaluation import evaluate_model
from fl_server.aggregation import aggregate_models_from_files
from fl_server.utils import make_json_serializable
from fl_server.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks
)
from fl_server.branching import BranchRegistry
from fl_server.unlearning_strategies import (
    get_strategy,
    collect_all_client_data
)

# Import is_pytorch_model from machine_unlearning_tool
import sys
machine_unlearning_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "machine_unlearning_tool")
if os.path.exists(machine_unlearning_path):
    sys.path.insert(0, machine_unlearning_path)
    try:
        from machine_unlearning_tool import is_pytorch_model
    except ImportError:
        def is_pytorch_model(model):
            return isinstance(model, torch.nn.Module)
else:
    def is_pytorch_model(model):
        return isinstance(model, torch.nn.Module)

class FederatedServer:
    """Server-side implementation for federated learning."""

    def __init__(
        self,
        experiment_type,
        test_dir=None,
        test_units=None,
        device=None,
        results_dir=None,
        verbose_plots=False,
        single_strategy=None,
        config_path=None
    ):
        """Set up the FL server: config, output dirs and training-history buffers."""
        self.experiment_type = experiment_type
        self.test_dir = test_dir
        self.test_units = test_units
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.results_dir = results_dir
        self.verbose_plots = verbose_plots
        self.single_strategy = single_strategy  # Single strategy for this server instance
        self.config_path = config_path  #Store config path for reading model dimensions
        self.round = 0
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.training_history = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": []  #For classification tasks like MNIST
        }

        # Experiment metadata (is initialized in init_experiment)
        self.fl_rounds = None
        self.client_ids = None
        self.iid = None

        # Results tracking
        self.results = {
            "experiment_type": experiment_type,
            "rounds": []
        }

        self.aggregation_timing_history = []
        self.disagreement_settings = self._get_disagreement_config()
        
        # Unlearning configuration
        self.unlearning_config = self._get_unlearning_config()
        self.clients = None  #Will be set by orchestrator

        #Create output directories
        if results_dir:
            # Get structure config
            structure = self._get_structure_config()

            # Create output directory
            self.output_dir = os.path.join(results_dir, "output", "server")
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

            # Create model_storage directory
            model_storage_path = os.path.join(results_dir, structure["model_storage_dir"])
            os.makedirs(model_storage_path, exist_ok=True)
        else:
            os.makedirs("output/server_results", exist_ok=True)
            os.makedirs("output/plots", exist_ok=True)

        #Initialize model based on experiment type
        self._init_model()

    def _init_model(self):
        """Initialize global model via adapter and model registry (plug-and-play)."""
        from fl_module.registry import DatasetAdapterRegistry
        from fl_module.model_registry import ModelRegistry

        adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
        model_params = self.unlearning_config.get("model_params", {})

        if adapter is not None:
            self._is_classification = adapter.is_classification()
            self.seq_len = adapter.get_sequence_length()
            self.input_dim = adapter.get_input_dim()
            self.output_dim = adapter.get_output_dim()
            if self.input_dim is None:
                self.input_dim = model_params.get("input_dim", 20)
            if self.output_dim is None:
                self.output_dim = model_params.get("output_dim", 2)
        else:
            self._is_classification = self.experiment_type not in ("n_cmapss",)
            self.input_dim = model_params.get("input_dim", 20)
            self.output_dim = model_params.get("output_dim", 2)
            self.seq_len = 50 if self.experiment_type == "n_cmapss" else 1

        if self.experiment_type == "n_cmapss":
            self.n_features = 20
            self.hidden_dim = 32
            kwargs = dict(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        elif self.experiment_type in ("mnist", "cifar10"):
            kwargs = {}
        else:
            kwargs = dict(input_dim=self.input_dim, output_dim=self.output_dim)

        model_type = self.experiment_type
        if model_type.startswith("custom") and ModelRegistry.get_factory(model_type) is None:
            model_type = "tabular"
        self.global_model = create_model(model_type, **kwargs).to(self.device)
        print(f"Initialized global {self.experiment_type} model")

    def load_test_data(self, sample_size=500):
        """Load the test set and build server.test_loader."""
        if self.experiment_type == "n_cmapss":
            if not self.test_dir or not self.test_units:
                raise ValueError("Test directory and test units must be provided for N-CMAPSS")

            #Load test data
            test_samples, test_labels = fl_module.load_ncmapss_test_data(
                self.test_dir,
                self.test_units,
                sample_size=sample_size
            )

            # Preprocess test data
            _, test_normalized, _ = fl_module.preprocess_ncmapss_data(test_samples, test_samples)

            # Create test dataloader
            self.test_loader = fl_module.create_ncmapss_test_dataloader(
                test_normalized,
                test_labels,
                batch_size=64
            )

            print(f"Loaded test data with {len(test_samples)} samples")
        elif self.experiment_type == "mnist":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for MNIST")

            test_images, test_labels = fl_module.load_mnist_test_data(
                test_dir=self.test_dir
            )

            self.test_loader = fl_module.create_mnist_test_dataloader(
                test_images,
                test_labels,
                batch_size=64
            )

            print(f"Loaded MNIST test data with {len(test_images)} samples")
        elif self.experiment_type == "cifar10":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for CIFAR-10")

            test_images, test_labels = fl_module.load_cifar10_test_data(
                test_dir=self.test_dir
            )

            self.test_loader = fl_module.create_cifar10_test_dataloader(
                test_images,
                test_labels,
                batch_size=128
            )

            print(f"Loaded CIFAR-10 test data with {len(test_images)} samples")
        elif self.experiment_type == "tabular":
            if not self.test_dir:
                raise ValueError("Test directory must be provided for tabular")

            # Load tabular test data
            test_features, test_labels = fl_module.load_tabular_test_data(
                test_dir=self.test_dir
            )

            #Create test dataloader
            self.test_loader = fl_module.create_tabular_test_dataloader(
                test_features,
                test_labels,
                batch_size=64
            )

            print(f"Loaded tabular test data with {len(test_features)} samples")
        elif self.experiment_type == "adult":
            from fl_module.adult.utils import load_test_data as load_adult_test_data
            from fl_module.tabular.utils import create_test_dataloader as create_adult_test_dataloader
            test_features, test_labels = load_adult_test_data(test_dir=self.test_dir)
            if sample_size and len(test_features) > sample_size:
                idx = np.random.choice(len(test_features), sample_size, replace=False)
                test_features, test_labels = test_features[idx], test_labels[idx]
            self.test_loader = create_adult_test_dataloader(test_features, test_labels, batch_size=64)
            print(f"Loaded Adult test data: {len(test_features)} samples, {test_features.shape[1]} features")
        elif self.experiment_type.startswith("custom"):
            #Custom dataset: load test data using adapter
            try:
                from fl_module.registry import DatasetAdapterRegistry
                from fl_module.custom.utils import load_custom_test_data, create_custom_test_dataloader
                
                adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
                if adapter is None:
                    raise ValueError(f"No adapter registered for experiment_type='{self.experiment_type}'. "
                                   f"Make sure to call register_custom_dataset() first.")
                
                # Load test data
                test_samples, test_labels = load_custom_test_data(self.experiment_type, sample_size=sample_size)
                
                # Determine if classification or regression
                # int labels = classification, float = regression
                unique_labels = np.unique(test_labels)
                try:
                    labels_as_int = test_labels.astype(int)
                    is_classification = (len(unique_labels) <= 10 and 
                                       np.allclose(test_labels, labels_as_int))
                except:
                    is_classification = len(unique_labels) <= 10
                
                #Create test dataloader
                self.test_loader = create_custom_test_dataloader(
                    test_samples,
                    test_labels,
                    batch_size=64,
                    is_classification=is_classification
                )
                
                print(f"Loaded custom test data with {len(test_samples)} samples")
                print(f"  Input dim: {test_samples.shape[1]}")
                print(f"  Task type: {'Classification' if is_classification else 'Regression'}")
            except ImportError:
                raise ImportError("Custom dataset support requires fl_module.custom.utils. "
                                "Make sure machine_unlearning_tool is available.")
        else:
            #To be implemented for other experiments
            raise NotImplementedError(f"{self.experiment_type} test data loading not implemented yet")

    def get_model_parameters(self):
        """Current global model parameters to send to clients."""
        return self.global_model.get_parameters()

    def save_model(self, model_dir):
        """Save the global model state dict and a metadata file."""
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.global_model.state_dict(), model_path)

        # Save model metadata
        metadata = {
            "experiment_type": self.experiment_type,
            "round": self.round,
            "timestamp": datetime.now().isoformat(),
            "disagreement_settings_active_this_round": self.disagreement_settings,
            "fully_excluded_clients_this_round": sorted(list(getattr(self, 'fully_excluded_clients_for_current_round', set())))
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved global model to {model_dir}")

    def load_model(self, model_dir):
        """Load the global model state dict (and log the round from metadata if present)."""
        model_path = os.path.join(model_dir, "model.pt")
        metadata_path = os.path.join(model_dir, "metadata.json")

        self.global_model.load_state_dict(torch.load(model_path, map_location=self.device))

        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "round" in metadata:
                    print(f"Loaded model from round {metadata['round']}")

        print(f"Loaded global model from {model_dir}")

    def create_model_dirs(self, round_num=None, structure=None):
        """Create the model directories for a round (or the initial-model dir if round_num is None)."""
        if not self.results_dir or not structure:
            return None

        if round_num is None:
            initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            os.makedirs(initial_model_dir, exist_ok=True)
            return initial_model_dir

        # Create round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        os.makedirs(round_dir, exist_ok=True)

        # Create directory for the global model for training
        training_model_dir = os.path.join(round_dir, structure["global_model"])
        os.makedirs(training_model_dir, exist_ok=True)

        #Create directory for the aggregated model
        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        return {
            "round_dir": round_dir,
            "training_model_dir": training_model_dir,
            "aggregated_model_dir": aggregated_model_dir
        }

    def get_model_dir_paths(self, round_num=None, aggregated=False, structure=None):
        """Path to a round's global-model dir (aggregated or for-training), or the initial-model dir."""
        if not self.results_dir or not structure:
            return None

        if round_num is None:
            dir_path = os.path.join(self.results_dir, structure["global_model_initial"])
            os.makedirs(dir_path, exist_ok=True)
            return dir_path

        #Round-specific global model
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        if aggregated:
            dir_path = os.path.join(round_dir, structure["global_model_aggregated"])
        else:
            dir_path = os.path.join(round_dir, structure["global_model"])

        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def init_experiment(self, fl_rounds, client_ids, iid=False):
        """Record experiment metadata (rounds, clients, iid) and save initial results."""
        self.fl_rounds = fl_rounds
        self.client_ids = client_ids
        self.iid = iid

        # Update results with experiment metadata
        self.results["fl_rounds"] = fl_rounds
        self.results["client_ids"] = client_ids
        self.results["iid"] = iid if self.experiment_type in ("mnist", "cifar10") else None

        # Save initial results
        self._save_experiment_results()

    def set_total_running_time(self, total_time_seconds):
        """Store the total wall-clock time of the whole FL run."""
        if not hasattr(self, 'aggregation_timing_history'):
            self.aggregation_timing_history = []

        self.total_running_time = total_time_seconds

    def _save_experiment_results(self):
        """Save experiment results to a JSON file."""
        # results directory present?
        if not self.results_dir:
            return

        #Create the output directory if it doesn't exist
        output_dir = os.path.join(self.results_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        #Add timing metrics to results if available
        if hasattr(self, 'aggregation_timing_history'):
            self.results["aggregation_timing_metrics"] = self.aggregation_timing_history

        # Convert NumPy types to Python native types
        serializable_results = make_json_serializable(self.results)

        # Save the results
        results_path = os.path.join(output_dir, "fl_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Saved experiment results to {results_path}")

        # Save timing metrics separately
        if hasattr(self, 'aggregation_timing_history') and self.aggregation_timing_history:
            timing_path = os.path.join(output_dir, "timing_metrics.json")

            #Create timing metrics structure
            timing_data = {
                "total_running_time_seconds": getattr(self, 'total_running_time', None),
                "experiment_init_time_seconds": getattr(self, 'experiment_init_time', None),
                "aggregation_timing_history": self.aggregation_timing_history,
                "round_timing_history": getattr(self, 'round_timing_history', []),
                "evaluation_timing_history": getattr(self, 'evaluation_timing_history', [])
            }

            serializable_timing = make_json_serializable(timing_data)
            with open(timing_path, "w") as f:
                json.dump(serializable_timing, f, indent=2)
            print(f"Saved timing metrics to {timing_path}")

    def initialize_model(self, round_num=0):
        """Save the freshly initialized global model as the round-0 checkpoint."""
        if not self.results_dir:
            return

        structure = self._get_structure_config()

        initial_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
        os.makedirs(initial_model_dir, exist_ok=True)

        #Save the initial model
        self.save_model(initial_model_dir)
        print("Initialized and saved the initial global model")

    def prepare_training_model(self, round_num, use_initial=False):
        """Place the right global model in the round's training dir; returns (dir, prep_time_s)."""
        if not self.results_dir:
            return None, 0.0

        # Timing the track model initialization
        preparation_start_time = time.time()

        print(f"\nPreparing server for round {round_num}.")
        # logging only; not dropped from baseline training
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
                print(f"  Fully excluded clients identified for round {round_num}: {sorted(list(self.fully_excluded_clients_for_current_round))} (kept in baseline; will be forgotten via unlearning)")

        if use_initial:
            source_model_dir = os.path.join(self.results_dir, structure["global_model_initial"])
            print(f"Round {round_num} starting with initial global model from {source_model_dir}")
            self.load_model(source_model_dir)

            if active_disagreements:
                print(f"Creating initial tracks for round {round_num} from global initial model")
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks:
                    print("Warning: client_ids not found in self.results, attempting to infer from client_dirs.")
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients", "client_*")
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
                    track_model_path = os.path.join(track_dir_iter, "model.pt")
                    torch.save(self.global_model.state_dict(), track_model_path)
                    individual_track_meta = {
                        "track_name": track_name_iter,
                        "round": round_num,
                        "client_ids": list(track_info.get("tracks", {}).get(track_name_iter, [])),
                        "rewound_this_round": False, # Cannot be rewound in round 1 from initial
                        "finetuning_status": {}
                    }
                    with open(os.path.join(track_dir_iter, "metadata.json"), "w") as f_meta_track:
                        json.dump(individual_track_meta, f_meta_track, indent=2)
                    print(f"Created initial track '{track_name_iter}' for round {round_num}")
                    
                    #Apply unlearning for this track if enabled
                    unlearning_applied_round1 = False
                    unlearning_strategy_round1 = None
                    if self.unlearning_config.get("enabled", False):
                        track_clients_set = set(track_info.get("tracks", {}).get(track_name_iter, []))
                        all_clients_set = set(client_ids_for_tracks)
                        #unlearn only if track excludes clients (not all-client global)
                        if track_name_iter != "global" or len(track_clients_set) < len(all_clients_set):
                            self._apply_unlearning_for_track(
                                round_num=round_num,
                                track_name=track_name_iter,
                                track_clients=track_clients_set,
                                all_client_ids=all_clients_set,
                                track_model_path=track_model_path
                            )
                            unlearning_applied_round1 = True
                            # Check which strategy was used
                            track_unlearning_dir = os.path.join(track_dir_iter, "unlearning")
                            comparison_path = os.path.join(track_unlearning_dir, "comparison.json")
                            branches_dir = os.path.join(track_unlearning_dir, "branches")
                            
                            if os.path.exists(comparison_path):
                                # Multi-strategy mode: check comparison.json
                                try:
                                    with open(comparison_path, 'r') as f:
                                        comparison = json.load(f)
                                        unlearning_strategy_round1 = comparison.get("best_strategy", None)
                                except:
                                    pass
                            elif os.path.exists(branches_dir):
                                # Single-strategy mode: check if branches exist
                                branches = [d for d in os.listdir(branches_dir) if os.path.isdir(os.path.join(branches_dir, d))]
                                if branches:
                                    #In single-strategy mode, use the single strategy name
                                    unlearning_strategy_round1 = self.single_strategy if self.single_strategy else branches[0]
                                    print(f"    Detected unlearning strategy (single-strategy mode): {unlearning_strategy_round1}")
                    
                    #Update metadata with unlearning info
                    individual_track_meta["unlearning_applied"] = unlearning_applied_round1
                    individual_track_meta["unlearning_strategy"] = unlearning_strategy_round1
                    with open(os.path.join(track_dir_iter, "metadata.json"), "w") as f_meta_track:
                        json.dump(individual_track_meta, f_meta_track, indent=2)

            self.save_model(training_model_dir)
            print(f"Saved global model for training at {training_model_dir}")
            
            # unlearn after baseline if enabled (even with "full" clients)
            if self.unlearning_config.get("enabled", False):
                self._apply_unlearning(round_num, list(self.fully_excluded_clients_for_current_round))
        else: # Not use_initial (round_num > 1 typically)
            prev_global_aggregated_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1), structure["global_model_aggregated"])
            if os.path.exists(os.path.join(prev_global_aggregated_dir, "model.pt")):
                print(f"Loading main global model from previous round {round_num-1}'s aggregated model: {prev_global_aggregated_dir}")
                self.load_model(prev_global_aggregated_dir)
            else:
                print(f"""Warning: Previous round's aggregated model not found at {prev_global_aggregated_dir}. Falling back to initial global model for main training model of round {round_num}.""")
                initial_model_dir_fallback = os.path.join(self.results_dir, structure["global_model_initial"])
                self.load_model(initial_model_dir_fallback)
            self.save_model(training_model_dir)
            print(f"Saved main global model for training in round {round_num} at {training_model_dir}")
            
            # Apply unlearning after baseline if enabled
            if self.unlearning_config.get("enabled", False):
                self._apply_unlearning(round_num, list(self.fully_excluded_clients_for_current_round))

            if active_disagreements:
                print(f"Active disagreements found for round {round_num}. Initiation mechanism: {initiation_mechanism}")
                client_ids_for_tracks = self.results.get("client_ids", [])
                if not client_ids_for_tracks: #Fallback for client_ids
                    print("Warning: client_ids not found in self.results for track creation in round > 1.")
                    client_dirs_pattern = os.path.join(self.results_dir, "output", "clients", "client_*")
                    client_dirs = glob.glob(client_dirs_pattern)
                    client_ids_for_tracks = sorted([int(os.path.basename(d).split("_")[-1]) for d in client_dirs]) if client_dirs else []

                track_info = create_model_tracks(active_disagreements, client_ids_for_tracks)
                current_round_tracks_dir = os.path.join(round_dir, "tracks")
                os.makedirs(current_round_tracks_dir, exist_ok=True)
                prev_round_tracks_main_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1), "tracks")

                current_round_track_metadata_content = {
                    "round": round_num,
                    "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
                    "client_tracks": track_info.get("client_tracks", {})
                }

                for track_name, clients_in_this_track_set in track_info.get("tracks", {}).items():
                    clients_in_this_track_list = list(clients_in_this_track_set)
                    current_specific_track_dir = os.path.join(current_round_tracks_dir, track_name)
                    os.makedirs(current_specific_track_dir, exist_ok=True)
                    prev_specific_track_dir = os.path.join(prev_round_tracks_main_dir, track_name)
                    track_existed_previously_as_specific_dir = os.path.exists(os.path.join(prev_specific_track_dir, "model.pt"))

                    current_track_finetuning_status = {}
                    if lifting_mechanism == "deep_incr_finetune":
                        print(f"    Deep incremental finetuning analysis for track '{track_name}':")
                        print(f"      Total finetuning rounds: {finetune_total_rounds}")
                        print(f"      Clients in track: {sorted(clients_in_this_track_list)}")

                        prev_finetune_status_path = os.path.join(prev_specific_track_dir, "finetuning_status.json")
                        prev_track_finetuning_status_loaded = {}
                        if os.path.exists(prev_finetune_status_path):
                            try:
                                with open(prev_finetune_status_path, 'r') as f_fs:
                                    prev_track_finetuning_status_loaded = json.load(f_fs)
                                print(f"      Previous finetuning status: {prev_track_finetuning_status_loaded}")
                            except Exception as e:
                                print(f"      Warning: Could not load previous finetuning status: {e}")
                        else:
                            print("      No previous finetuning status file found")

                        prev_track_clients_metadata = set()
                        prev_track_metadata_path_iter = os.path.join(prev_specific_track_dir, "metadata.json")
                        if os.path.exists(prev_track_metadata_path_iter):
                            try:
                                with open(prev_track_metadata_path_iter, 'r') as f_meta:
                                    prev_track_clients_metadata = set(json.load(f_meta).get("client_ids", []))
                                print(f"      Previous track clients: {sorted(list(prev_track_clients_metadata))}")
                            except Exception as e:
                                print(f"      Warning: Could not load previous metadata: {e}")
                        else:
                            print("      No previous track metadata found")

                        #Analyze client finetuning needs
                        clients_new_to_track = []
                        clients_continuing_ft = []
                        clients_completed_ft = []
                        clients_no_ft = []

                        print("      Analyzing finetuning for each client:")
                        for client_id_numeric in clients_in_this_track_list:
                            client_id_str_iter = str(client_id_numeric)
                            if client_id_numeric not in prev_track_clients_metadata and track_existed_previously_as_specific_dir:
                                print(f"        Client {client_id_str_iter}: New to track, starting finetuning (1/{finetune_total_rounds})")
                                current_track_finetuning_status[client_id_str_iter] = 1
                                clients_new_to_track.append(client_id_str_iter)
                            elif client_id_str_iter in prev_track_finetuning_status_loaded:
                                progress = prev_track_finetuning_status_loaded[client_id_str_iter] + 1
                                if progress <= finetune_total_rounds:
                                    print(f"        Client {client_id_str_iter}: Continuing finetuning, round {progress}/{finetune_total_rounds}")
                                    current_track_finetuning_status[client_id_str_iter] = progress
                                    clients_continuing_ft.append(f"{client_id_str_iter}({progress}/{finetune_total_rounds})")
                                else:
                                    print(f"        Client {client_id_str_iter}: Completed finetuning, no further action")
                                    clients_completed_ft.append(client_id_str_iter)
                            else:
                                print(f"        Client {client_id_str_iter}: No finetuning required")
                                clients_no_ft.append(client_id_str_iter)

                        # Summary of finetuning actions for this track
                        print(f"      Track '{track_name}' finetuning summary:")
                        if clients_new_to_track:
                            print(f"        New to track: {clients_new_to_track}")
                        if clients_continuing_ft:
                            print(f"        Continuing: {clients_continuing_ft}")
                        if clients_completed_ft:
                            print(f"        Completed: {clients_completed_ft}")
                        if clients_no_ft:
                            print(f"        No action: {clients_no_ft}")

                    if current_track_finetuning_status:
                        with open(os.path.join(current_specific_track_dir, "finetuning_status.json"), 'w') as f_fs_curr:
                            json.dump(current_track_finetuning_status, f_fs_curr, indent=2)
                        print(f"    Saved finetuning status for track '{track_name}': {current_track_finetuning_status}")
                    else:
                        print(f"    No clients require finetuning in track '{track_name}'")

                    print(f"  Evaluating track: '{track_name}'. Existed previously: {track_existed_previously_as_specific_dir}")
                    is_new_track_to_dir_structure = not track_existed_previously_as_specific_dir
                    composition_has_changed = False
                    if is_new_track_to_dir_structure:
                        if track_name == "global":
                            all_system_clients = set(self.results.get("client_ids", []))
                            active_disags_prev_round = get_active_disagreements(disagreements, round_num - 1 if round_num > 0 else 0)
                            fully_excluded_prev = set()
                            if active_disags_prev_round:
                                for cid_str, d_list in active_disags_prev_round.items():
                                    for d_item in d_list:
                                        if d_item.get('type') == 'full':
                                            try:
                                                fully_excluded_prev.add(int(cid_str.split('_')[-1]) if '_' in cid_str else int(cid_str))
                                            except ValueError:
                                                print(f"Warning: Could not parse client ID '{cid_str}' during rewind check for global track.")
                            conceptual_prev_global_clients = all_system_clients - fully_excluded_prev
                            if clients_in_this_track_set != conceptual_prev_global_clients:
                                composition_has_changed = True
                                print("    Global track is new to 'tracks' dir. Composition changed.")
                            else:
                                print("    Global track is new to 'tracks' dir. Composition UNCHANGED.")
                        else: # Non-global track, new to dir structure
                            composition_has_changed = True
                            print(f"    Non-global track '{track_name}' is new. Marking composition changed.")
                    else: # Track existed previously
                        prev_meta_path = os.path.join(prev_specific_track_dir, "metadata.json")
                        if os.path.exists(prev_meta_path):
                            with open(prev_meta_path, 'r') as f_m:
                                prev_clients = set(json.load(f_m).get("client_ids", []))
                            if prev_clients != clients_in_this_track_set:
                                composition_has_changed = True
                                print(f"    Track '{track_name}' existed and composition changed.")
                            else:
                                print(f"    Track '{track_name}' existed and composition UNCHANGED.")
                        else:
                            composition_has_changed = True
                            print(f"    Track '{track_name}' existed but no prev metadata. Marking changed.")

                    perform_rewind_for_this_track = (initiation_mechanism == "deep_rewind" and composition_has_changed)
                    print(f"    Perform rewind for '{track_name}': {perform_rewind_for_this_track}")

                    if perform_rewind_for_this_track:
                        print(f"    Performing deep rewind for track '{track_name}':")
                        self.load_model(os.path.join(self.results_dir, structure["global_model_initial"]))
                        current_rewound_model_state = self.global_model.state_dict()
                        for hist_round in range(1, round_num):
                            hist_clients_dir = os.path.join(self.results_dir, structure["round_template"].format(round=hist_round), structure["clients_dir"])
                            client_model_files_hist = [os.path.join(hist_clients_dir, f"{structure['client_prefix']}{cid}", "model.pt") for cid in clients_in_this_track_list if os.path.exists(os.path.join(hist_clients_dir, f"{structure['client_prefix']}{cid}", "model.pt"))]
                            if client_model_files_hist:
                                #Extract client IDs from file paths for better logging
                                client_ids_in_round = []
                                for file_path in client_model_files_hist:
                                    client_dir = os.path.basename(os.path.dirname(file_path))
                                    client_id = client_dir.replace(structure['client_prefix'], "")
                                    client_ids_in_round.append(client_id)
                                client_ids_str = ", ".join(sorted(client_ids_in_round))
                                print(f"      Round {hist_round}: Aggregating {len(client_model_files_hist)} models from clients [{client_ids_str}]")
                                aggregated_state = self._aggregate_model_states_from_files_for_rewind(client_model_files_hist, self.device)
                                if aggregated_state:
                                    current_rewound_model_state = aggregated_state
                                else:
                                    print(f"        Warning: Aggregation failed for '{track_name}' in round {hist_round}.")
                            else:
                                print(f"      Round {hist_round}: No models available for '{track_name}' in rewind.")
                        self.global_model.load_state_dict(current_rewound_model_state)
                        self.save_model(current_specific_track_dir)
                        print(f"    Deep rewind complete for '{track_name}'. Saved to {current_specific_track_dir}")
                    else: #Not rewinding this track
                        source_model_for_track_path = prev_specific_track_dir if track_existed_previously_as_specific_dir else prev_global_aggregated_dir
                        print(f"Loading model for track '{track_name}' from '{source_model_for_track_path}'.")
                        if os.path.exists(os.path.join(source_model_for_track_path, "model.pt")):
                            self.load_model(source_model_for_track_path)
                        else:
                            print(f"Warning: Source model '{source_model_for_track_path}' for track '{track_name}' not found. Using initial global model.")
                            self.load_model(os.path.join(self.results_dir, structure["global_model_initial"]))
                        self.save_model(current_specific_track_dir)

                        # Apply unlearning for this track if enabled and composition changed
                        if self.unlearning_config.get("enabled", False) and composition_has_changed:
                            track_model_path = os.path.join(current_specific_track_dir, "model.pt")
                            self._apply_unlearning_for_track(
                                round_num=round_num,
                                track_name=track_name,
                                track_clients=clients_in_this_track_set,
                                all_client_ids=set(client_ids_for_tracks),
                                track_model_path=track_model_path
                            )

                    # Check if unlearning was applied
                    unlearning_applied = False
                    unlearning_strategy = None
                    if self.unlearning_config.get("enabled", False):
                        track_unlearning_dir = os.path.join(current_specific_track_dir, "unlearning")
                        if os.path.exists(track_unlearning_dir):
                            # comparison.json (multi-strategy) or branches/ (single-strategy)
                            comparison_path = os.path.join(track_unlearning_dir, "comparison.json")
                            branches_dir = os.path.join(track_unlearning_dir, "branches")
                            
                            if os.path.exists(comparison_path):
                                #Multi-strategy mode: check comparison.json
                                unlearning_applied = True
                                try:
                                    with open(comparison_path, 'r') as f:
                                        comparison = json.load(f)
                                        unlearning_strategy = comparison.get("best_strategy", None)
                                except:
                                    pass
                            elif os.path.exists(branches_dir):
                                #Single-strategy mode: check if branches exist
                                branches = [d for d in os.listdir(branches_dir) if os.path.isdir(os.path.join(branches_dir, d))]
                                if branches:
                                    unlearning_applied = True
                                    # In single-strategy mode, use the single strategy name
                                    unlearning_strategy = self.single_strategy if self.single_strategy else branches[0]
                                    print(f"    Detected unlearning applied (single-strategy mode): {unlearning_strategy}")
                    
                    individual_track_meta = {
                        "track_name": track_name, 
                        "round": round_num, 
                        "client_ids": clients_in_this_track_list, 
                        "rewound_this_round": perform_rewind_for_this_track,
                        "finetuning_status": {cid_str: f"{prog}/{finetune_total_rounds}" for cid_str, prog in current_track_finetuning_status.items()} if current_track_finetuning_status else {},
                        "unlearning_applied": unlearning_applied,
                        "unlearning_strategy": unlearning_strategy
                    }
                    with open(os.path.join(current_specific_track_dir, "metadata.json"), "w") as f_track_meta:
                        json.dump(individual_track_meta, f_track_meta, indent=2)

                with open(os.path.join(current_round_tracks_dir, "track_metadata.json"), "w") as f_meta_overall:
                    json.dump(current_round_track_metadata_content, f_meta_overall, indent=2)
                print(f"Saved track metadata for round {round_num}.")
            else: # No active disagreements
                print(f"No active disagreements for round {round_num}. Using standard global model from {prev_global_aggregated_dir}")
                if round_num > 1 and lifting_mechanism == "deep_incr_finetune":
                    print(f"    Deep incremental finetuning check for round {round_num}:")
                    print(f"      - Mechanism: {lifting_mechanism}")
                    print(f"      - Total finetuning rounds: {finetune_total_rounds}")
                    print("      - No active tracks detected")
                    current_global_finetuning_status = {}
                    prev_round_main_dir = os.path.join(self.results_dir, structure["round_template"].format(round=round_num - 1))
                    prev_global_finetune_status_path = os.path.join(prev_round_main_dir, "global_finetuning_status.json")
                    prev_global_finetuning_status_loaded = {}
                    if os.path.exists(prev_global_finetune_status_path):
                        try:
                            with open(prev_global_finetune_status_path, 'r') as f_fs:
                                prev_global_finetuning_status_loaded = json.load(f_fs)
                            print(f"      Loaded previous finetuning status from R{round_num-1}: {prev_global_finetuning_status_loaded}")
                        except Exception as e:
                            print(f"      Warning: Could not load previous global finetuning status: {e}")

                    prev_round_track_metadata_path = os.path.join(prev_round_main_dir, "tracks", "track_metadata.json")
                    prev_round_had_active_tracks = os.path.exists(prev_round_track_metadata_path)
                    prev_round_client_track_map = {} # Stores client_id_str -> track_name from previous round

                    if prev_round_had_active_tracks:
                        print(f"      Tracks were active in previous round (R{round_num-1}). Checking for clients rejoining from non-global tracks.")
                        try:
                            with open(prev_round_track_metadata_path, 'r') as f_prev_meta:
                                prev_track_meta_content = json.load(f_prev_meta)
                                prev_round_client_track_map = {str(k): v for k, v in prev_track_meta_content.get("client_tracks", {}).items()}
                        except Exception as e:
                            print(f"        Warning: Could not load client_tracks from previous round's track_metadata.json: {e}")
                    else:
                        print(f"      No tracks were active in previous round (R{round_num-1}). Finetuning initiation based on client absence or status.")

                    prev_round_clients_dir = os.path.join(prev_round_main_dir, structure["clients_dir"])
                    prev_round_submitted_model_ids = set()
                    if os.path.exists(prev_round_clients_dir):
                        client_model_dirs = glob.glob(os.path.join(prev_round_clients_dir, f"{structure['client_prefix']}*"))
                        for d_path in client_model_dirs:
                            try:
                                prev_round_submitted_model_ids.add(int(os.path.basename(d_path).replace(structure['client_prefix'], "")))
                            except ValueError:
                                pass
                    print(f"      Previous round (R{round_num-1}) participants: {sorted(list(prev_round_submitted_model_ids))}")
                    #Global aggregation uses all clients; exclusions are handled via unlearning.
                    current_global_participants = set(self.results.get("client_ids", []))
                    print(f"      Current round (R{round_num}) participants: {sorted(list(current_global_participants))}")

                    #Analyze client changes
                    newly_joining = current_global_participants - prev_round_submitted_model_ids
                    continuing = current_global_participants & prev_round_submitted_model_ids
                    if newly_joining:
                        print(f"      Newly joining clients: {sorted(list(newly_joining))}")
                    if continuing:
                        print(f"      Continuing clients: {sorted(list(continuing))}")

                    print("      Analyzing finetuning requirements for each client:")

                    clients_starting_new = []
                    clients_continuing = []
                    clients_completed = []
                    clients_no_action = []

                    for client_id_numeric in current_global_participants:
                        client_id_str_gf = str(client_id_numeric)
                        start_new_finetuning_for_client = False
                        if client_id_numeric not in prev_round_submitted_model_ids:
                            print(f"        Client {client_id_str_gf}: Was absent in R{round_num-1}, now joining, starting finetuning")
                            start_new_finetuning_for_client = True
                        elif prev_round_had_active_tracks:
                            client_prev_track = prev_round_client_track_map.get(client_id_str_gf)
                            if client_prev_track and client_prev_track != "global":
                                print(f"        Client {client_id_str_gf}: Rejoining from track '{client_prev_track}', starting finetuning")
                                start_new_finetuning_for_client = True
                            else:
                                # present + tracks existed, but on 'global' or missing track info
                                # No NEW finetuning initiation due to track dissolution itself.
                                print(f"        Client {client_id_str_gf}: Was on '{client_prev_track or 'global'}' track, no new finetuning from track dissolution")

                        if start_new_finetuning_for_client:
                            current_global_finetuning_status[client_id_str_gf] = 1
                            clients_starting_new.append(client_id_str_gf)
                        elif client_id_str_gf in prev_global_finetuning_status_loaded:
                            # not new; may continue a prior global finetune cycle
                            progress_gf = prev_global_finetuning_status_loaded[client_id_str_gf] + 1
                            if progress_gf <= finetune_total_rounds:
                                print(f"        Client {client_id_str_gf}: Continuing finetuning, round {progress_gf}/{finetune_total_rounds}")
                                current_global_finetuning_status[client_id_str_gf] = progress_gf
                                clients_continuing.append(f"{client_id_str_gf}({progress_gf}/{finetune_total_rounds})")
                            else:
                                print(f"        Client {client_id_str_gf}: Completed finetuning, no further action needed")
                                clients_completed.append(client_id_str_gf)
                        else:
                            print(f"        Client {client_id_str_gf}: No finetuning action required")
                            clients_no_action.append(client_id_str_gf)
                        #present but no trigger and no prior status, so nothing to finetune

                    print("      Finetuning summary:")
                    if clients_starting_new:
                        print(f"        Starting new: {clients_starting_new}")
                    if clients_continuing:
                        print(f"        Continuing: {clients_continuing}")
                    if clients_completed:
                        print(f"        Completed: {clients_completed}")
                    if clients_no_action:
                        print(f"        No action: {clients_no_action}")

                    if current_global_finetuning_status:
                        current_global_finetune_status_path = os.path.join(round_dir, "global_finetuning_status.json")
                        try:
                            with open(current_global_finetune_status_path, 'w') as f_fs_global:
                                json.dump(current_global_finetuning_status, f_fs_global, indent=2)
                            print(f"      Saved global finetuning status: {current_global_finetuning_status}")
                        except Exception as e:
                            print(f"      Warning: Could not save global finetuning status: {e}")
                    else:
                        print("      No clients require finetuning this round")

        print(f"Server preparation for round {round_num} complete.\\n")

        #Timing the track model initialization
        preparation_time = time.time() - preparation_start_time
        print(f"Track model initialization completed in {preparation_time:.4f} seconds")

        return training_model_dir, preparation_time

    def get_client_model_path(self, round_num, client_id):
        """Path to the model a client should train from: its track model if any, else the global one."""
        if not self.results_dir:
            return None

        structure = self._get_structure_config()

        tracks_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num),
            "tracks"
        )

        # tracks dir is there, so look up this client's track
        if os.path.exists(tracks_dir):
            metadata_path = os.path.join(tracks_dir, "track_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        track_metadata = json.load(f)

                    # Get the primary track for this client
                    primary_track = track_metadata.get("client_tracks", {}).get(str(client_id))

                    if primary_track:
                        track_model_path = os.path.join(tracks_dir, primary_track, "model.pt")
                        if os.path.exists(track_model_path):
                            print(f"Using track model {primary_track} for client {client_id}")
                            return track_model_path
                except Exception as e:
                    print(f"Error loading track metadata: {e}")

        # If no tracks or track not found, use the standard model path
        standard_model_path = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num),
            structure["global_model"],
            "model.pt"
        )

        return standard_model_path

    def aggregate_with_disagreement_resolution(self, round_num):
        """Run track-based, disagreement-aware aggregation for a round. Returns success."""
        if not self.results_dir:
            return False

        structure = self._get_structure_config()

        #Get the round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])

        aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
        os.makedirs(aggregated_model_dir, exist_ok=True)

        #Aggregate client models
        aggregate_models_from_files(self, clients_dir)

        # Save the aggregated model
        self.save_model(aggregated_model_dir)

        return True



    def _get_structure_config(self):
        """Return the results directory layout from config, or sensible defaults."""
        default_structure = {
            "model_storage_dir": "model_storage",
            "global_model_initial": "model_storage/global_model_initial",
            "round_template": "model_storage/round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "global_model_aggregated": "global_model_aggregated",
            "client_prefix": "client_"
        }

        # Try to load from configuration file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "mock_etcd/configuration.json")

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        loaded_structure = config["results"]["structure"]
                        for key, value in default_structure.items():
                            if key not in loaded_structure:
                                loaded_structure[key] = value
                        return loaded_structure
        except Exception as e:
            print(f"Error loading or parsing structure configuration: {e}. Using default structure.")

        return default_structure

    def _get_disagreement_config(self):
        """Read the disagreement settings from config, falling back to defaults."""
        default_disagreement_config = {
            "initiation_mechanism": "shallow", # Default to shallow
            "lifting_mechanism": "shallow",    #Default to shallow
            "deep_lifting_finetune_rounds": 3  #Default to 3 rounds
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
            print(f"Error loading or parsing disagreement configuration: {e}. Using default disagreement config.")

        return default_disagreement_config
    
    def _get_unlearning_config(self):
        """Get unlearning configuration.
        
        Automatically determines model_type based on experiment_type if not specified:
        - n_cmapss (time-series) -> lstm (or mlp)
        - mnist (images) -> mlp (CNN not supported in unlearning framework)
        
        The FL model (e.g., CNN for MNIST) differs from the unlearning model type
        (e.g., MLP for MNIST). The unlearning framework supports: lstm, mlp, random_forest, xgboost.
        """
        # default model per experiment type for unlearning
        experiment_to_unlearning_model = {
            "n_cmapss": "lstm",  # time series
            "mnist": "mlp",  # no CNN in the unlearning framework
            "cifar10": "mlp",  #same here
            "tabular": "mlp",  #or random_forest/xgboost
        }
        
        default_unlearning_config = {
            "enabled": False,
            "model_type": experiment_to_unlearning_model.get(self.experiment_type, "lstm"),
            "strategies": ["exact_retraining", "sisa", "distillation"],
            "model_params": {},
            "train_params": {"epochs": 10, "batch_size": 64, "lr": 1e-3}
        }
        
        # Try to use the config_path passed from orchestrator first
        config = None
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    if "unlearning" in config:
                        loaded_config = config["unlearning"].copy()
                        
                        # Auto-detect model_type if not specified in config
                        if "model_type" not in loaded_config:
                            auto_model_type = experiment_to_unlearning_model.get(self.experiment_type, "lstm")
                            loaded_config["model_type"] = auto_model_type
                            print(f"Auto-detected unlearning model_type: {auto_model_type} for experiment_type: {self.experiment_type}")
                        
                        for key, value in default_unlearning_config.items():
                            if key not in loaded_config:
                                loaded_config[key] = value
                        return loaded_config
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}. Falling back to default location.")
        
        # Fallback to default config location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config_path = os.path.join(base_dir, "mock_etcd/configuration.json")
        
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, 'r') as f:
                    config = json.load(f)
            except Exception:
                pass
        
        try:
            if config and "unlearning" in config:
                loaded_config = config["unlearning"].copy()
                
                #Auto-detect model_type if not specified in config
                if "model_type" not in loaded_config:
                    auto_model_type = experiment_to_unlearning_model.get(self.experiment_type, "lstm")
                    loaded_config["model_type"] = auto_model_type
                    print(f"Auto-detected unlearning model_type: {auto_model_type} for experiment_type: {self.experiment_type}")
                
                for key, value in default_unlearning_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
        except Exception as e:
            print(f"Error loading unlearning configuration: {e}. Using default config.")
        
        #Use experiment-specific default
        default_unlearning_config["model_type"] = experiment_to_unlearning_model.get(self.experiment_type, "lstm")
        return default_unlearning_config
    
    def set_clients(self, clients):
        """Give the server the client objects (needed for unlearning)."""
        self.clients = clients
    
    def _apply_unlearning(self, round_num: int, forget_ids: List[int]):
        """Run the configured unlearning strategies for the excluded clients this round."""
        if not self.clients:
            print("Warning: Clients not set. Cannot apply unlearning.")
            return
        
        print(f"\nApplying unlearning for round {round_num}.")
        print(f"Forgetting clients: {forget_ids}")
        
        # Initialize branch registry
        branch_registry = BranchRegistry(self.results_dir, round_num)
        
        # Save pre-unlearning checkpoint
        pre_checkpoint = self.global_model.state_dict()
        branch_registry.save_pre_unlearning_checkpoint(
            pre_checkpoint,
            forget_ids,
            metadata={"round": round_num, "experiment_type": self.experiment_type}
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
        
        #Collect all client data (baseline contains all clients)
        print("Collecting all client data for unlearning...")
        all_data = collect_all_client_data(self.clients, self.experiment_type)
        
        #require data
        if all_data is None or len(all_data.get("df", [])) == 0:
            print("Error: No data collected from clients. Cannot perform unlearning.")
            return
        
        # strategy to run (single if set, else first in config)
        if self.single_strategy:
            # multi-strategy run: only this instance's single strategy
            strategy_name = self.single_strategy
            print(f"Using single strategy: {strategy_name} (multi-strategy run mode)")
        else:
            # Original mode: run all strategies (for backward compatibility)
            strategies_to_run = self.unlearning_config.get("strategies", ["exact_retraining"])
            strategy_name = strategies_to_run[0]  #Use first strategy for model update
            print(f"Running all strategies, but updating model with: {strategy_name}")
        
        model_type = self.unlearning_config.get("model_type", "lstm")
        model_params = self.unlearning_config.get("model_params", {})
        train_params = self.unlearning_config.get("train_params", {})
        
        #reuse existing branches if same policy
        reuse_enabled = self.unlearning_config.get("reuse_existing_branches", True)
        
        # Run the strategy (single strategy mode)
        branch_results = {}
        strategies_to_process = [strategy_name] if self.single_strategy else self.unlearning_config.get("strategies", ["exact_retraining"])
        
        for strategy_name in strategies_to_process:
            print(f"\nRunning unlearning strategy: {strategy_name}")
            
            # reuse an existing branch?
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
                    
                    # Load the existing model (check if sklearn or PyTorch)
                    is_sklearn = existing_branch.get("is_sklearn", False)
                    if is_sklearn:
                        import pickle
                        with open(existing_branch["path"], 'rb') as f:
                            existing_model_state = pickle.load(f)
                        #Wrap in dict for save_branch_checkpoint
                        existing_model_state = {"type": "sklearn", "model": existing_model_state}
                    else:
                        existing_model_state = torch.load(existing_branch["path"], map_location=self.device)
                    
                    #Load metrics from existing branch
                existing_metrics_path = os.path.join(existing_branch["branch_dir"], "metrics.json")
                if os.path.exists(existing_metrics_path):
                    with open(existing_metrics_path, 'r') as f:
                        existing_metrics = json.load(f)
                else:
                    existing_metrics = {"note": "reused from previous round", "unlearning_time_s": 0.0}
                    
                    # Save as new branch checkpoint (reference to old one)
                    branch_registry.save_branch_checkpoint(
                        strategy_name,
                        existing_model_state,
                        existing_metrics,
                        metadata={
                            "round": round_num,
                            "forget_ids": forget_ids,
                            "reused_from_round": existing_branch["round"],
                            "reused": True,
                            "model_type": model_type
                        }
                    )
                    
                    branch_results[strategy_name] = existing_metrics
                    print(f"Reused branch '{strategy_name}' from round {existing_branch['round']}")
                    continue
            
            # No existing branch found, run unlearning
            try:
                # Prepare FL model parameters for exact model recreation
                fl_model_params = {}
                if self.experiment_type in ("tabular", "adult"):
                    fl_model_params['input_dim'] = self.input_dim
                    fl_model_params['output_dim'] = self.output_dim
                elif self.experiment_type == "n_cmapss":
                    fl_model_params['input_dim'] = self.input_dim
                    fl_model_params['hidden_dim'] = self.hidden_dim
                    fl_model_params['output_dim'] = self.output_dim
                #MNIST doesn't need parameters (fixed architecture)

                #extra kwargs for reuse-capable strategies (SISA only)
                strategy_extra_kwargs = {}
                if strategy_name == "sisa":
                    checkpoint_dir = train_params.get("checkpoint_dir")
                    if not checkpoint_dir:
                        checkpoint_dir = os.path.join(self.results_dir, "sisa_checkpoints", "global")
                    # isolate checkpoints per experiment type (avoid state-dict mismatches)
                    checkpoint_dir = os.path.join(checkpoint_dir, self.experiment_type)
                    strategy_extra_kwargs.update({
                        "checkpoint_dir": checkpoint_dir,
                        "num_shards": train_params.get("num_shards"),
                        "num_slices": train_params.get("num_slices")
                    })

                # Get strategy
                strategy = get_strategy(
                    strategy_name,
                    model_type=model_type,
                    model_params=model_params,
                    train_params=train_params,
                    device=self.device,
                    experiment_type=self.experiment_type,  # Pass experiment_type to use FL models
                    fl_model_params=fl_model_params,  #Pass FL model params for exact recreation
                    results_dir=self.results_dir,  #NEW: Pass results directory for FedEraser
                    num_clients=len(self.client_ids),  # NEW: Pass total client count for FedEraser
                    **strategy_extra_kwargs
                )

                # Set clients_ref for FederatedExactRetrainingStrategy
                if strategy_name == "federated_exact_retraining":
                    strategy.clients_ref = self.clients
                
                # Apply unlearning
                #Pass test set for MIA (Membership Inference Attack)
                test_loader = getattr(self, 'test_loader', None)
                test_df = None
                test_input_cols = None
                test_target_col = None
                
                #sklearn models need test_df, not test_loader
                if not is_pytorch_model(self.global_model) and hasattr(self, 'test_data'):
                    test_df = self.test_data
                    test_input_cols = all_data.get("input_cols")
                    test_target_col = all_data.get("target_col")
                
                result = strategy.unlearn(
                    pretrained_model=self.global_model,
                    all_data=all_data,
                    forget_ids=forget_ids,
                    round_num=round_num,
                    test_loader=test_loader,
                    test_df=test_df,
                    test_input_cols=test_input_cols,
                    test_target_col=test_target_col
                )
                
                # Save branch checkpoint
                unlearned_model = result["model"]
                if is_pytorch_model(unlearned_model):
                    model_state = unlearned_model.state_dict()
                else:
                    # For sklearn models, wrap in dict with actual model
                    model_state = {"type": "sklearn", "model": unlearned_model}
                
                branch_registry.save_branch_checkpoint(
                    strategy_name,
                    model_state,
                    result["metrics"],
                    metadata={
                        "round": round_num,
                        "forget_ids": forget_ids,
                        "model_type": model_type,
                        "reused": False
                    }
                )
                
                branch_results[strategy_name] = result["metrics"]
                print(f"Strategy '{strategy_name}' completed. Metrics: {result['metrics']}")
                
                # single-strategy mode: global model <- unlearned model
                if self.single_strategy:
                    unlearned_model = result["model"]
                    if is_pytorch_model(unlearned_model):
                        self.global_model.load_state_dict(unlearned_model.state_dict())
                        print(f"Global model updated with {strategy_name} unlearning result")
                    else:
                        #For sklearn models, replace the model
                        self.global_model = unlearned_model
                        print(f"Global model replaced with {strategy_name} unlearned model")
                
            except Exception as e:
                print(f"Error running strategy '{strategy_name}': {e}")
                import traceback
                traceback.print_exc()
                #In single-strategy mode, this is fatal
                if self.single_strategy:
                    print(f"\nWARNING: Unlearning strategy '{strategy_name}' failed!")
                    print("Falling back to pre-unlearning model state.")
                    
                    # Load pre-unlearning checkpoint as fallback
                    pre_checkpoint = branch_registry.load_pre_unlearning_checkpoint()
                    if pre_checkpoint:
                        if is_pytorch_model(self.global_model):
                            self.global_model.load_state_dict(pre_checkpoint)
                        print("Loaded pre-unlearning model as fallback.")
                    else:
                        print("Could not load pre-unlearning checkpoint. Model state unchanged.")
                    
                    print(f"Unlearning for round {round_num} complete (fallback path).\n")
                    return
                # Continue to next strategy in multi-strategy mode
                continue
        
        # Handle case where all strategies failed
        if not branch_results:
            print("\nWARNING: All unlearning strategies failed!")
            print("Falling back to pre-unlearning model state.")
            
            #Load pre-unlearning checkpoint as fallback
            pre_checkpoint = branch_registry.load_pre_unlearning_checkpoint()
            if pre_checkpoint:
                if is_pytorch_model(self.global_model):
                    self.global_model.load_state_dict(pre_checkpoint)
                    print("Loaded pre-unlearning model as fallback.")
            else:
                print("Could not load pre-unlearning checkpoint. Model state unchanged.")
            
            print(f"Unlearning for round {round_num} complete (fallback path).\n")
            return
        
        #save comparison (multi-strategy back-compat)
        if branch_results and not self.single_strategy:
            comparison = {
                "round": round_num,
                "forget_ids": forget_ids,
                "strategies": branch_results,
                "best_strategy": min(branch_results.keys(), key=lambda k: branch_results[k].get("rmse", float('inf')))
            }
            branch_registry.save_comparison(comparison)
            print("\nUnlearning branches completed and stored for comparison.")
            print("Global model left unchanged after running all strategies (compare branches offline).")
        elif self.single_strategy:
            print(f"\nUnlearning with {strategy_name} completed. Global model updated.")
        
        print(f"Unlearning for round {round_num} complete.\n")
    
    def _apply_unlearning_for_track(
        self, 
        round_num: int, 
        track_name: str, 
        track_clients: set, 
        all_client_ids: set,
        track_model_path: str
    ):
        """Like _apply_unlearning, but remove the excluded clients from a single track's model."""
        if not self.clients:
            print(f"Warning: Clients not set. Cannot apply unlearning for track '{track_name}'.")
            return
        
        # Determine which clients are excluded from this track
        excluded_clients = all_client_ids - track_clients
        
        if not excluded_clients:
            print(f"Track '{track_name}' has no excluded clients. Skipping unlearning.")
            return
        
        # track needs clients to retrain
        if not track_clients:
            print(f"Warning: Track '{track_name}' has no clients. Cannot retrain without data.")
            print(f"   Skipping unlearning for this track (no data to retrain on).")
            # Save empty metrics to indicate unlearning was skipped
            track_unlearning_dir = os.path.join(
                os.path.dirname(track_model_path),
                "unlearning"
            )
            branches_dir = os.path.join(track_unlearning_dir, "branches")
            os.makedirs(branches_dir, exist_ok=True)
            
            #Save empty metrics for each strategy to indicate skip
            strategies_to_mark = self.unlearning_config.get("strategies", ["exact_retraining"])
            for strategy_name in strategies_to_mark:
                strategy_branch_dir = os.path.join(branches_dir, strategy_name)
                os.makedirs(strategy_branch_dir, exist_ok=True)
                skip_metrics = {
                    "unlearning_time_s": 0.0,
                    "skipped": True,
                    "reason": "Track has no clients - no data to retrain on",
                    "retrain_fraction": 0.0,
                    "N_train_total": 0,
                    "N_retrain": 0
                }
                with open(os.path.join(strategy_branch_dir, "metrics.json"), 'w') as f:
                    json.dump(skip_metrics, f, indent=2)
            return
        
        forget_ids = sorted(list(excluded_clients))
        
        print(f"\nApplying unlearning for track '{track_name}' in round {round_num}.")
        print(f"Track clients: {sorted(list(track_clients))}")
        print(f"Excluded clients (to forget): {forget_ids}")
        
        #Load the track model
        if not os.path.exists(track_model_path):
            print(f"Warning: Track model not found at {track_model_path}. Skipping unlearning.")
            return
        
        # Create track-specific branch registry
        structure = self._get_structure_config()
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        track_unlearning_dir = os.path.join(round_dir, "tracks", track_name, "unlearning")
        os.makedirs(track_unlearning_dir, exist_ok=True)
        
        # Use a modified BranchRegistry for tracks
        # simple registry structure for tracks
        track_pre_unlearning_dir = os.path.join(track_unlearning_dir, "pre_unlearning")
        track_branches_dir = os.path.join(track_unlearning_dir, "branches")
        os.makedirs(track_pre_unlearning_dir, exist_ok=True)
        os.makedirs(track_branches_dir, exist_ok=True)
        
        #Load global model (baseline_global) instead of track model
        #Track model is created by exact_retraining applied to global model
        global_model_path = os.path.join(round_dir, structure["global_model_aggregated"], "model.pt")
        if not os.path.exists(global_model_path):
            print(f"Warning: Global model not found at {global_model_path}. Skipping unlearning.")
            return
        
        global_model_state = torch.load(global_model_path, map_location=self.device)
        import copy
        baseline_original_model = None
        if is_pytorch_model(self.global_model):
            baseline_original_model = copy.deepcopy(self.global_model)
            baseline_original_model.load_state_dict(global_model_state)
            baseline_original_model.eval()
        
        # Save pre-unlearning checkpoint for this track
        pre_checkpoint_metadata = {
            "round": round_num,
            "track_name": track_name,
            "forget_ids": forget_ids,
            "track_clients": sorted(list(track_clients)),
            "experiment_type": self.experiment_type,
            "timestamp": datetime.now().isoformat()
        }
        torch.save(global_model_state, os.path.join(track_pre_unlearning_dir, "model.pt"))
        with open(os.path.join(track_pre_unlearning_dir, "metadata.json"), 'w') as f:
            json.dump(pre_checkpoint_metadata, f, indent=2)
        print(f"Saved pre-unlearning checkpoint for track '{track_name}' (using global model)")
        
        # Validate forget_ids
        available_client_ids = set(self.clients.keys())
        invalid_forget_ids = [fid for fid in forget_ids if fid not in available_client_ids]
        if invalid_forget_ids:
            print(f"Error: Invalid forget_ids {invalid_forget_ids}. Available client IDs: {sorted(available_client_ids)}")
            return
        
        # Collect all client data
        print("Collecting all client data for unlearning...")
        all_data = collect_all_client_data(self.clients, self.experiment_type)
        
        #require data
        if all_data is None or len(all_data.get("df", [])) == 0:
            print("Error: No data collected from clients. Cannot perform unlearning.")
            return
        
        #strategy to run (single if set, else first in config)
        if self.single_strategy:
            # multi-strategy run: only this instance's single strategy
            strategy_name = self.single_strategy
            strategies_to_run = [strategy_name]
            print(f"Using single strategy for track: {strategy_name} (multi-strategy run mode)")
        else:
            # Original mode: run all strategies (for backward compatibility)
            strategies_to_run = self.unlearning_config.get("strategies", ["exact_retraining"])
            strategy_name = strategies_to_run[0]  # Use first strategy for model update
            print(f"Running all strategies for track, but updating model with: {strategy_name}")
        
        model_type = self.unlearning_config.get("model_type", "lstm")
        model_params = self.unlearning_config.get("model_params", {})
        train_params = self.unlearning_config.get("train_params", {})
        
        #reuse existing branches if same policy
        reuse_enabled = self.unlearning_config.get("reuse_existing_branches", True)
        
        #Load global model into global_model for unlearning
        # Track model = exact_retraining result, other strategies also applied to global model
        original_global_model_state = self.global_model.state_dict()
        self.global_model.load_state_dict(global_model_state)
        
        # Sort strategies: exact_retraining first (baseline for comparison)
        if "exact_retraining" in strategies_to_run:
            strategies_to_run = ["exact_retraining"] + [s for s in strategies_to_run if s != "exact_retraining"]
        
        # Run the strategy/strategies
        branch_results = {}
        baseline_model = None  #Store Exact Retraining model as baseline for other strategies
        for strategy_name in strategies_to_run:
            print(f"\nRunning unlearning strategy '{strategy_name}' for track '{track_name}'")
            
            #reuse an existing branch (search previous rounds)
            if reuse_enabled:
                existing_branch = self._find_existing_track_branch(
                    track_name=track_name,
                    forget_ids=forget_ids,
                    strategy_name=strategy_name,
                    model_type=model_type,
                    max_search_rounds=10,
                    current_round=round_num
                )
                
                if existing_branch:
                    print(f"Found existing branch from round {existing_branch['round']} for track '{track_name}'!")
                    print(f"  Forget IDs: {forget_ids}")
                    print(f"  Strategy: {strategy_name}")
                    print(f"  Reusing model from round {existing_branch['round']}")
                    
                    # Load the existing model
                    existing_model_state = torch.load(existing_branch["path"], map_location=self.device)
                    
                    # Load metrics from existing branch
                    existing_metrics_path = os.path.join(existing_branch["branch_dir"], "metrics.json")
                    if os.path.exists(existing_metrics_path):
                        with open(existing_metrics_path, 'r') as f:
                            existing_metrics = json.load(f)
                    else:
                        existing_metrics = {"note": "reused from previous round", "unlearning_time_s": 0.0}
                    
                    # Save as new branch checkpoint
                    branch_dir = os.path.join(track_branches_dir, strategy_name)
                    os.makedirs(branch_dir, exist_ok=True)
                    torch.save(existing_model_state, os.path.join(branch_dir, "model.pt"))
                    
                    branch_metadata = {
                        "round": round_num,
                        "track_name": track_name,
                        "forget_ids": forget_ids,
                        "reused_from_round": existing_branch["round"],
                        "reused": True,
                        "model_type": model_type,
                        "strategy": strategy_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(os.path.join(branch_dir, "metadata.json"), 'w') as f:
                        json.dump(branch_metadata, f, indent=2)
                    with open(os.path.join(branch_dir, "metrics.json"), 'w') as f:
                        json.dump(existing_metrics, f, indent=2)
                    
                    branch_results[strategy_name] = existing_metrics
                    print(f"Reused branch '{strategy_name}' from round {existing_branch['round']}")
                    continue
                    
                    branch_results[strategy_name] = existing_metrics
                    print(f"Reused branch '{strategy_name}' from round {existing_branch['round']}")
                    continue
            
            #No existing branch found, run unlearning
            try:
                #Prepare FL model parameters for exact model recreation
                fl_model_params = {}
                if self.experiment_type in ("tabular", "adult"):
                    fl_model_params['input_dim'] = self.input_dim
                    fl_model_params['output_dim'] = self.output_dim
                elif self.experiment_type == "n_cmapss":
                    fl_model_params['input_dim'] = self.input_dim
                    fl_model_params['hidden_dim'] = self.hidden_dim
                    fl_model_params['output_dim'] = self.output_dim
                # MNIST doesn't need parameters (fixed architecture)

                # Extra kwargs for strategies (only SISA needs these)
                strategy_extra_kwargs = {}
                if strategy_name == "sisa":
                    checkpoint_dir = train_params.get("checkpoint_dir")
                    if not checkpoint_dir:
                        checkpoint_dir = os.path.join(self.results_dir, "sisa_checkpoints", track_name)
                    # isolate checkpoints per experiment type (avoid state-dict mismatches)
                    checkpoint_dir = os.path.join(checkpoint_dir, self.experiment_type)
                    strategy_extra_kwargs.update({
                        "checkpoint_dir": checkpoint_dir,
                        "num_shards": train_params.get("num_shards"),
                        "num_slices": train_params.get("num_slices")
                    })

                #Get strategy
                strategy = get_strategy(
                    strategy_name,
                    model_type=model_type,
                    model_params=model_params,
                    train_params=train_params,
                    device=self.device,
                    experiment_type=self.experiment_type,  #Pass experiment_type to use FL models
                    fl_model_params=fl_model_params,  # Pass FL model params for exact recreation
                    results_dir=self.results_dir,  # NEW: Pass results directory for FedEraser
                    num_clients=len(self.client_ids),  # NEW: Pass total client count for FedEraser
                    **strategy_extra_kwargs
                )

                #Set clients_ref for FederatedExactRetrainingStrategy
                if strategy_name == "federated_exact_retraining":
                    strategy.clients_ref = self.clients

                #Apply unlearning
                # Pass test set for MIA (Membership Inference Attack)
                test_loader = getattr(self, 'test_loader', None)
                test_df = None
                test_input_cols = None
                test_target_col = None
                
                # sklearn models need test_df, not test_loader
                if not is_pytorch_model(self.global_model) and hasattr(self, 'test_data'):
                    test_df = self.test_data
                    test_input_cols = all_data.get("input_cols")
                    test_target_col = all_data.get("target_col")
                
                # non-exact strategies: eval against baseline_model (Exact RT)
                baseline_for_eval = baseline_model if (strategy_name != "exact_retraining" and baseline_model is not None) else None
                
                result = strategy.unlearn(
                    pretrained_model=self.global_model,
                    all_data=all_data,
                    forget_ids=forget_ids,
                    round_num=round_num,
                    test_loader=test_loader,
                    test_df=test_df,
                    test_input_cols=test_input_cols,
                    test_target_col=test_target_col,
                    baseline_model=baseline_for_eval,  #Pass Exact Retraining model for evaluation
                    baseline_original_model=baseline_original_model
                )
                
                #Save Exact Retraining model as baseline for other strategies
                # Track model = exact_retraining result
                if strategy_name == "exact_retraining":
                    baseline_model = result["model"]
                    print(f"Saved Exact Retraining model as baseline for other strategies")
                    
                    # Save exact_retraining result as track model
                    unlearned_model = result["model"]
                    if is_pytorch_model(unlearned_model):
                        torch.save(unlearned_model.state_dict(), track_model_path)
                        print(f"Track model '{track_name}' = Exact Retraining result")
                    else:
                        import pickle
                        with open(track_model_path.replace('.pt', '.pkl'), 'wb') as f:
                            pickle.dump(unlearned_model, f)
                        print(f"Track model '{track_name}' = Exact Retraining result")
                # single-strategy mode (SISA/distill): also persist track model
                elif self.single_strategy and strategy_name == self.single_strategy:
                    unlearned_model = result["model"]
                    if is_pytorch_model(unlearned_model):
                        torch.save(unlearned_model.state_dict(), track_model_path)
                        print(f"Track model '{track_name}' = {strategy_name} result (single-strategy)")
                    else:
                        import pickle
                        with open(track_model_path.replace('.pt', '.pkl'), 'wb') as f:
                            pickle.dump(unlearned_model, f)
                        print(f"Track model '{track_name}' = {strategy_name} result (single-strategy)")
                
                #always save branch checkpoint (regardless of single_strategy)
                unlearned_model = result["model"]
                if is_pytorch_model(unlearned_model):
                    model_state = unlearned_model.state_dict()
                else:
                    #For sklearn models, wrap in dict
                    model_state = {"type": "sklearn", "model": unlearned_model}
                
                branch_dir = os.path.join(track_branches_dir, strategy_name)
                os.makedirs(branch_dir, exist_ok=True)
                
                try:
                    if isinstance(model_state, dict) and model_state.get("type") == "sklearn":
                        import pickle
                        with open(os.path.join(branch_dir, "model.pkl"), 'wb') as f:
                            pickle.dump(model_state["model"], f)
                    else:
                        torch.save(model_state, os.path.join(branch_dir, "model.pt"))
                    
                    branch_metadata = {
                        "round": round_num,
                        "track_name": track_name,
                        "forget_ids": forget_ids,
                        "model_type": model_type,
                        "reused": False,
                        "strategy": strategy_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(os.path.join(branch_dir, "metadata.json"), 'w') as f:
                        json.dump(branch_metadata, f, indent=2)
                    with open(os.path.join(branch_dir, "metrics.json"), 'w') as f:
                        json.dump(result["metrics"], f, indent=2)
                    
                    print(f"Branch saved for strategy '{strategy_name}' in {branch_dir}")
                except Exception as branch_error:
                    print(f"Warning: Failed to save branch for strategy '{strategy_name}': {branch_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue anyway - don't fail the whole unlearning process
                
                branch_results[strategy_name] = result["metrics"]
                print(f"Strategy '{strategy_name}' completed for track '{track_name}'. Metrics: {result['metrics']}")
                
            except Exception as e:
                print(f"Error running strategy '{strategy_name}' for track '{track_name}': {e}")
                import traceback
                traceback.print_exc()
                # In single-strategy mode, this is fatal
                if self.single_strategy:
                    print(f"\nWARNING: Unlearning strategy '{strategy_name}' failed for track '{track_name}'!")
                    print("Using pre-unlearning track model.")
                    # Restore original global model state
                    self.global_model.load_state_dict(original_global_model_state)
                    print(f"Unlearning for track '{track_name}' complete (fallback path).\n")
                    return
                #Continue to next strategy in multi-strategy mode
                continue
        
        #Restore original global model state
        self.global_model.load_state_dict(original_global_model_state)
        
        # Handle case where all strategies failed
        if not branch_results:
            print(f"\nWARNING: All unlearning strategies failed for track '{track_name}'!")
            print("Using pre-unlearning track model.")
            print(f"Unlearning for track '{track_name}' complete (fallback path).\n")
            return
        
        # save comparison (multi-strategy back-compat)
        if not self.single_strategy:
            comparison = {
                "round": round_num,
                "track_name": track_name,
                "forget_ids": forget_ids,
                "strategies": branch_results,
                "best_strategy": min(branch_results.keys(), key=lambda k: branch_results[k].get("rmse", branch_results[k].get("test_loss", float('inf'))))
            }
            comparison_path = os.path.join(track_unlearning_dir, "comparison.json")
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nUnlearning branches for track '{track_name}' completed and stored for comparison.")
            print("Track model left unchanged after running all strategies (compare branches offline).")
        elif self.single_strategy:
            print(f"\nUnlearning with {strategy_name} completed for track '{track_name}'. Track model updated.")
        
        print(f"Unlearning for track '{track_name}' complete.\n")
    
    def _find_existing_track_branch(
        self,
        track_name: str,
        forget_ids: List[int],
        strategy_name: str,
        model_type: str,
        max_search_rounds: int = 10,
        current_round: int = None
    ) -> Optional[Dict]:
        """Search previous rounds for a matching unlearning branch to reuse (None if none)."""
        structure = self._get_structure_config()

        # search backwards from the current round
        if current_round is None:
            current_round = getattr(self, 'round', 1)
        for search_round in range(current_round - 1, max(0, current_round - max_search_rounds - 1), -1):
            round_dir = os.path.join(
                self.results_dir,
                structure["round_template"].format(round=search_round)
            )
            track_unlearning_dir = os.path.join(round_dir, "tracks", track_name, "unlearning")
            branch_dir = os.path.join(track_unlearning_dir, "branches", strategy_name)
            metadata_path = os.path.join(branch_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    #Check if forget_ids match
                    existing_forget_ids = sorted(metadata.get("forget_ids", []))
                    if existing_forget_ids == sorted(forget_ids):
                        model_path = os.path.join(branch_dir, "model.pt")
                        if os.path.exists(model_path):
                            return {
                                "round": search_round,
                                "path": model_path,
                                "branch_dir": branch_dir,
                                "metadata": metadata
                            }
                except Exception as e:
                    continue
        
        return None
    
    def _transfer_model_weights(self, source_state_dict: dict, target_model: torch.nn.Module) -> bool:
        """Copy weights from the unlearned model into the FL model by matching layer names/sizes.

        The unlearning model (e.g. MLP) can differ from the FL model (TabularClassifier,
        RULPredictor, ...), so only compatible layers transfer. Returns success.
        """
        try:
            target_state_dict = target_model.state_dict().copy()
            transferred_count = 0
            skipped_count = 0
            
            #first, direct name matching
            for key in source_state_dict:
                if key in target_state_dict:
                    source_shape = source_state_dict[key].shape
                    target_shape = target_state_dict[key].shape
                    
                    if source_shape == target_shape:
                        target_state_dict[key] = source_state_dict[key].clone()
                        transferred_count += 1
                    else:
                        skipped_count += 1
            
            # for Sequential models (MLPForecast to TabularClassifier)
            # Both use nn.Sequential with structure: Linear -> ReLU -> Dropout -> Linear -> ...
            source_net_keys = [k for k in source_state_dict.keys() if 'net' in k]
            target_net_keys = [k for k in target_state_dict.keys() if 'net' in k]
            
            if source_net_keys and target_net_keys:
                # Group by layer index and type (weight/bias)
                source_layers = {}
                target_layers = {}
                
                for key in source_net_keys:
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        layer_type = parts[-1]  #'weight' or 'bias'
                        if layer_idx not in source_layers:
                            source_layers[layer_idx] = {}
                        source_layers[layer_idx][layer_type] = key
                
                for key in target_net_keys:
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        layer_type = parts[-1]
                        if layer_idx not in target_layers:
                            target_layers[layer_idx] = {}
                        target_layers[layer_idx][layer_type] = key
                
                #Match Linear layers (skip ReLU and Dropout which have no weights)
                # In Sequential: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear, 4=ReLU, 5=Dropout, ...
                # map: 0->0, 3->3, 6->6, etc. (Linear layers)
                for src_idx in sorted(source_layers.keys()):
                    # Check if this is a Linear layer (even indices: 0, 3, 6, ...)
                    #Actually, in Sequential, Linear layers are at: 0, 3, 6, 9, ... (every 3rd)
                    #verify the actual structure
                    # Match by position and check if shapes match
                    
                    # Find corresponding target layer
                    # For now, try direct index matching
                    if src_idx in target_layers:
                        for layer_type in ['weight', 'bias']:
                            if layer_type in source_layers[src_idx] and layer_type in target_layers[src_idx]:
                                src_key = source_layers[src_idx][layer_type]
                                tgt_key = target_layers[src_idx][layer_type]
                                
                                src_shape = source_state_dict[src_key].shape
                                tgt_shape = target_state_dict[tgt_key].shape
                                
                                if src_shape == tgt_shape:
                                    target_state_dict[tgt_key] = source_state_dict[src_key].clone()
                                    transferred_count += 1
            
            #for RULPredictor (fc1, fc2) from MLPForecast
            #Map: net.0 -> fc1, net.3 -> fc2 (assuming 2 hidden layers in MLPForecast)
            if any('fc1' in k for k in target_state_dict.keys()):
                # Try to find Linear layers in source and map to fc1, fc2
                source_linear_layers = []
                for key in sorted(source_state_dict.keys()):
                    if 'weight' in key and ('net.0' in key or 'net.3' in key or 'net.6' in key):
                        source_linear_layers.append(key)
                
                target_fc_layers = sorted([k for k in target_state_dict.keys() if 'fc' in k and 'weight' in k])
                
                # Map first Linear to fc1, last Linear to last fc layer
                if source_linear_layers and target_fc_layers:
                    # Map first source Linear to first target fc
                    if len(source_linear_layers) > 0 and len(target_fc_layers) > 0:
                        src_key = source_linear_layers[0]
                        tgt_key = target_fc_layers[0]
                        src_shape = source_state_dict[src_key].shape
                        tgt_shape = target_state_dict[tgt_key].shape
                        
                        if src_shape == tgt_shape:
                            target_state_dict[tgt_key] = source_state_dict[src_key].clone()
                            #Also transfer bias
                            src_bias_key = src_key.replace('.weight', '.bias')
                            tgt_bias_key = tgt_key.replace('.weight', '.bias')
                            if src_bias_key in source_state_dict and tgt_bias_key in target_state_dict:
                                if source_state_dict[src_bias_key].shape == target_state_dict[tgt_bias_key].shape:
                                    target_state_dict[tgt_bias_key] = source_state_dict[src_bias_key].clone()
                            transferred_count += 1
                    
                    #Map last source Linear to last target fc (output layer)
                    if len(source_linear_layers) > 1 and len(target_fc_layers) > 1:
                        src_key = source_linear_layers[-1]
                        tgt_key = target_fc_layers[-1]
                        src_shape = source_state_dict[src_key].shape
                        tgt_shape = target_state_dict[tgt_key].shape
                        
                        # For output layer, allow partial transfer if input dim matches
                        # (output dim might differ: 1 for regression vs N for classification)
                        if len(src_shape) == 2 and len(tgt_shape) == 2:
                            if src_shape[1] == tgt_shape[1]:  # Input dimension matches
                                #Transfer only the first output neuron's weights
                                target_state_dict[tgt_key][:1, :] = source_state_dict[src_key].clone()
                                transferred_count += 1
                                #Also transfer bias if possible
                                src_bias_key = src_key.replace('.weight', '.bias')
                                tgt_bias_key = tgt_key.replace('.weight', '.bias')
                                if src_bias_key in source_state_dict and tgt_bias_key in target_state_dict:
                                    if len(source_state_dict[src_bias_key].shape) == 1:
                                        target_state_dict[tgt_bias_key][:1] = source_state_dict[src_bias_key].clone()
            
            # update only if some weights transferred
            if transferred_count > 0:
                target_model.load_state_dict(target_state_dict, strict=False)
                print(f"   Transferred {transferred_count} compatible layer(s) (skipped {skipped_count} incompatible)")
                return True
            else:
                print(f"   No compatible layers found for weight transfer (checked {len(source_state_dict)} source layers)")
                return False
                
        except Exception as e:
            print(f"   Error during weight transfer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _aggregate_model_states_from_files_for_rewind(self, model_file_paths, device):
        """Average the state dicts at the given paths (used when rewinding); None if no files."""
        if not model_file_paths:
            return None

        aggregated_state_dict = None
        num_models = len(model_file_paths)

        # use the first model to set up the zero-initialised accumulator
        try:
            first_model_state = torch.load(model_file_paths[0], map_location=device)
            aggregated_state_dict = {name: torch.zeros_like(param) for name, param in first_model_state.items()}
        except Exception as e:
            print(f"Error loading first model for rewind aggregation {model_file_paths[0]}: {e}")
            return None # Cannot proceed if first model fails

        #Aggregate all models
        for model_path in model_file_paths:
            try:
                client_state_dict = torch.load(model_path, map_location=device)
                for name, param in client_state_dict.items():
                    if name in aggregated_state_dict:
                        aggregated_state_dict[name] += param / num_models
                    else:
                        #shouldn't happen if all models share structure
                        print(f"Warning: Parameter {name} not found in initial model structure during rewind. Skipping.")
            except Exception as e:
                print(f"Error loading or aggregating model {model_path} during rewind: {e}. Skipping this model.")
                pass # Continue with other models

        return aggregated_state_dict

    def evaluate_model(self, fl_round=None, client_results=None):
        """Evaluate the global model on the test set (delegates to evaluation.evaluate_model)."""
        return evaluate_model(self, fl_round, client_results)
