"""Sklearn-based federated learning orchestrator implementation."""

import os
import argparse
import json
import time
import fl_module

from fl_client_sklearn import SklearnFederatedClient
from fl_server_sklearn import SklearnFederatedServer
from mock_etcd.etcd_loader import MockEtcdLoader


class SklearnFederatedOrchestrator:
    """Orchestrator for coordinating sklearn-based federated learning."""

    def __init__(self, config_path="mock_etcd/configuration.json", model_type="random_forest"):
        """Initialize the sklearn federated learning orchestrator.

        Args:
            config_path: Path to the configuration file
            model_type: Type of sklearn model ('random_forest' or 'xgboost')
        """
        # Store config path for passing to server
        self.config_path = config_path
        
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
        self.model_type = model_type

        # Setup data if needed
        self._setup_data_if_needed()

        # Create model_storage directory
        if self.results_dir:
            structure = self.results_config.get("structure", {})
            model_storage_dir = structure.get("model_storage_dir", "model_storage_sklearn")
            os.makedirs(os.path.join(self.results_dir, model_storage_dir), exist_ok=True)

        #Initialize server
        self.server = self._init_server()
        
        #Initialize experiment metadata
        self.server.init_experiment(
            fl_rounds=self.fl_rounds,
            client_ids=self.client_ids_in_experiment,
            iid=self.exp_config.get("iid", False) if self.experiment_type == "mnist" else None
        )

        # Initialize clients
        self.clients = self._init_clients()
        
        # Set clients in server for unlearning
        self.server.set_clients(self.clients)

        print("Initialized sklearn-based federated learning orchestrator")
        print(f"  - {len(self.client_ids_in_experiment)} clients for {self.experiment_type} experiment")
        print(f"  - Model type: {self.model_type}")
        print(f"  - Results directory: {self.results_dir}")

    def _setup_data_if_needed(self):
        """Setup data if needed via adapter (plug-and-play, same as PyTorch orchestrator)."""
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
        """Initialize the sklearn federated learning server.

        Returns:
            SklearnFederatedServer: Initialized server
        """
        # Get model params from config if available
        model_params = {}
        if hasattr(self.config, 'config') and 'unlearning' in self.config.config:
            model_params = self.config.config.get('unlearning', {}).get('model_params', {})
        
        server = SklearnFederatedServer(
            experiment_type=self.experiment_type,
            model_type=self.model_type,
            test_dir=self.config.get_test_dir(),
            test_units=self.data_config.get("test_units"),
            results_dir=self.results_dir,
            verbose_plots=self.results_config.get("verbose_plots", False),
            model_params=model_params,
            config_path=self.config_path  #Pass config path for unlearning config loading
        )

        #Load test data for evaluation
        if self.experiment_type == "n_cmapss":
            server.load_test_data(sample_size=self.data_config.get("test_sample_size", 500))
        elif self.experiment_type == "mnist":
            server.load_test_data()

        return server

    def _init_clients(self):
        """Initialize federated learning clients.

        Returns:
            dict: Dictionary mapping client IDs to SklearnFederatedClient instances
        """
        clients = {}
        
        # Get model params from config if available
        model_params = {}
        if hasattr(self.config, 'config') and 'unlearning' in self.config.config:
            model_params = self.config.config.get('unlearning', {}).get('model_params', {})

        for client_id in self.client_ids_in_experiment:
            data_dir = self.config.get_train_dir()
            
            clients[client_id] = SklearnFederatedClient(
                client_id=client_id,
                experiment_type=self.experiment_type,
                model_type=self.model_type,
                data_dir=data_dir,
                results_dir=self.results_dir,
                model_params=model_params
            )

            # Load client data
            clients[client_id].load_data(sample_size=self.data_config.get("client_sample_size", 1000))

        # Set clients in server for unlearning (will be called after server init)
        return clients

    def run_federated_learning(self):
        """Execute sklearn-based federated learning.

        This is a simplified version without disagreement resolution for now.
        """
        fl_start_time = time.time()

        print(f"Starting sklearn federated learning with {self.fl_rounds} rounds...")
        print(f"Model type: {self.model_type}")

        #Initialize and save the initial global model
        self.server.initialize_model(round_num=0)

        #Initial evaluation of the global model (round 0)
        self.server.evaluate_model(fl_round=0)
        print("Initial model evaluation completed")

        # Main federated learning loop
        for fl_round in range(1, self.fl_rounds + 1):
            round_start_time = time.time()

            print(f"\n--- Sklearn Federated Learning Round {fl_round}/{self.fl_rounds} ---")

            # 1. Server analyzes disagreements and prepares track-specific models
            print("Analyzing disagreements and preparing track-specific models...")
            
            # Server creates directories and prepares models with disagreement resolution
            if fl_round == 1:
                #For the first round, create initial tracks from global model
                training_model_dir, track_init_time = self.server.prepare_training_model(fl_round, use_initial=True)
                print("Created initial track models from global_model_initial for round 1")
            else:
                #For subsequent rounds, update tracks based on disagreement evolution
                training_model_dir, track_init_time = self.server.prepare_training_model(fl_round, use_initial=False)
                print(f"Updated track models based on disagreement changes from round {fl_round-1}")
            
            structure = self.server._get_structure_config()
            round_dir = os.path.join(
                self.results_dir,
                structure["round_template"].format(round=fl_round)
            )

            # 2. Clients participate in disagreement-aware multi-track training
            print("Starting disagreement-aware multi-track client training...")
            client_training_start_time = time.time()
            client_training_times = {}
            
            for client_id in self.client_ids_in_experiment:
                if client_id not in self.clients:
                    print(f"Warning: Client {client_id} configured in experiment but not initialized. Skipping.")
                    continue
                
                client = self.clients[client_id]
                print(f"Client {client_id}: Loading track models and training with disagreement resolution...")
                
                # Time individual client training
                client_start_time = time.time()
                
                # Client loads primary track model and any background track models
                client.load_track_models_for_round(fl_round)
                
                #Client trains on primary track + participates in background tracks
                training_results = client.train_with_disagreement_resolution(round_num=fl_round)
                
                #Client saves all trained models (primary + background) to filesystem
                client.save_trained_track_models(fl_round)
                
                # Record individual client training time
                client_training_time = time.time() - client_start_time
                client_training_times[client_id] = {
                    "training_time_seconds": client_training_time,
                    "total_training_time_from_results": training_results.get("training_time", {}).get("total_seconds", 0) if training_results else 0
                }
                print(f"Client {client_id} completed training in {client_training_time:.4f} seconds")

            total_client_training_time = time.time() - client_training_start_time

            # 3. Server aggregates models
            print("Performing model aggregation...")
            clients_dir = os.path.join(round_dir, structure["clients_dir"])
            self.server.aggregate_models(fl_round, clients_dir)

            # Save aggregated model
            #aggregate_models() already saves the baseline_global ensemble via save_track_models()
            #and updates server.global_model. We only need to save again if it wasn't saved.
            aggregated_model_dir = os.path.join(round_dir, structure["global_model_aggregated"])
            aggregated_model_path = os.path.join(aggregated_model_dir, "model.pkl")
            
            # Check if it was already saved by save_track_models
            if not os.path.exists(aggregated_model_path):
                os.makedirs(aggregated_model_dir, exist_ok=True)
                # server.global_model should already be the ensemble after aggregate_models()
                self.server.save_model(aggregated_model_dir)
                print(f"Saved aggregated model to {aggregated_model_dir}")
            else:
                print(f"Aggregated model already saved by track aggregation to {aggregated_model_dir}")
                # Verify it's an ensemble
                import pickle
                from fl_server_sklearn.aggregation import _WeightedVotingClassifier
                with open(aggregated_model_path, 'rb') as f:
                    saved_model = pickle.load(f)
                if isinstance(saved_model, _WeightedVotingClassifier):
                    print(f"  Saved model is an ensemble with {len(saved_model.estimators)} estimators")
                else:
                    print(f"  Warning: Saved model is {type(saved_model).__name__}, not an ensemble!")
                    #Re-save the correct ensemble
                    with open(aggregated_model_path, 'wb') as f:
                        pickle.dump(self.server.global_model, f)
                    print(f"  Re-saved correct ensemble model")

            #4. Server evaluates global model
            print("Evaluating global model...")
            evaluation_start_time = time.time()
            self.server.evaluate_model(fl_round=fl_round)
            evaluation_time = time.time() - evaluation_start_time

            # Calculate round time
            total_round_time = time.time() - round_start_time

            print(f"Round {fl_round} completed:")
            print(f"  Client training: {total_client_training_time:.4f}s")
            print(f"  Evaluation: {evaluation_time:.4f}s")
            print(f"  Total round time: {total_round_time:.4f}s")

        # Calculate total running time
        total_running_time = time.time() - fl_start_time

        print("\nSklearn federated learning completed!")
        print(f"Total running time: {total_running_time:.2f} seconds")


def main():
    """Run the sklearn orchestrator as a standalone application."""
    parser = argparse.ArgumentParser(description="Sklearn Federated Learning Orchestrator")
    parser.add_argument("--config", type=str, default="mock_etcd/configuration.json",
                        help="Path to configuration file")
    parser.add_argument("--model_type", type=str, choices=["random_forest", "xgboost"],
                        default="random_forest", help="Type of sklearn model to use")

    args = parser.parse_args()

    # Create and run orchestrator
    orchestrator = SklearnFederatedOrchestrator(
        config_path=args.config,
        model_type=args.model_type
    )

    #Run federated learning
    orchestrator.run_federated_learning()


if __name__ == "__main__":
    main()
