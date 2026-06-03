"""Sklearn-based federated learning client implementation."""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Optional

#Import machine_unlearning_tool for model creation
import sys
#Path: fl_client_sklearn/client.py -> fl-disagreement-resolution -> Thesis -> machine_unlearning_tool
# So we need to go up 3 levels from fl_client_sklearn
current_file = os.path.abspath(__file__)  # /.../Thesis/fl-disagreement-resolution/fl_client_sklearn/client.py
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # /.../Thesis
#Add Thesis directory to path so we can import machine_unlearning_tool
sys.path.insert(0, base_dir)
from machine_unlearning_tool import create_model

import fl_module
from fl_client_sklearn.training import train_sklearn_model


class SklearnFederatedClient:
    """Client-side implementation for sklearn-based federated learning."""

    def __init__(
        self,
        client_id: int,
        experiment_type: str,
        model_type: str = "random_forest",  #"random_forest" or "xgboost"
        data_dir: str = None,
        results_dir: Optional[str] = None,
        model_params: Optional[dict] = None
    ):
        """Initialize the sklearn federated learning client.

        Args:
            client_id: Client ID
            experiment_type: Type of experiment ('n_cmapss' or 'mnist')
            model_type: Type of sklearn model ('random_forest' or 'xgboost')
            data_dir: Directory containing client data
            results_dir: Directory for storing models and results
            model_params: Parameters for sklearn model
        """
        self.client_id = client_id
        self.experiment_type = experiment_type
        self.model_type = model_type
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.model_params = model_params or {}
        
        # Initialize model parameters based on experiment type
        if experiment_type == "n_cmapss":
            self.seq_len = 50
            self.n_features = 20
            self.input_dim = self.seq_len * self.n_features
        elif experiment_type == "mnist":
            self.input_dim = 784  # 28x28 pixels
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        
        # Initialize model (will be set when loading global model)
        self.model = None
        self.sklearn_model_type = model_type  #Store for classification/regression handling
        
        #Create results directory
        if results_dir:
            self.output_dir = os.path.join(results_dir, "output", "clients_sklearn", f"client_{client_id}")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            os.makedirs("output/client_sklearn_results", exist_ok=True)
    
    def load_data(self, sample_size=1000):
        """Load and preprocess client data as numpy arrays.
        
        Args:
            sample_size: Maximum number of samples to load
        """
        if self.experiment_type == "n_cmapss":
            # Load data for this client
            samples, labels = fl_module.load_ncmapss_client_data(
                self.client_id,
                self.data_dir,
                sample_size=sample_size
            )
            
            # Preprocess data
            samples_normalized, _ = fl_module.preprocess_ncmapss_data(samples)
            
            # Convert to numpy arrays and flatten sequences for sklearn
            n_samples, seq_len, n_features = samples_normalized.shape
            self.X_train = samples_normalized.reshape(n_samples, -1)  #[N, seq_len * n_features]
            self.y_train = labels
            
            print(f"Client {self.client_id} loaded {len(self.X_train)} samples (flattened to {self.X_train.shape[1]} features)")
            
        elif self.experiment_type == "mnist":
            #Load MNIST data for this client
            images, labels = fl_module.load_mnist_client_data(
                self.client_id,
                train_dir=self.data_dir,
                sample_size=sample_size
            )
            
            # Flatten images for sklearn
            self.X_train = images.reshape(len(images), -1)  # [N, 784]
            self.y_train = labels
        else:
            raise NotImplementedError(f"{self.experiment_type} data loading not implemented yet")
    
    def create_model_dir(self, round_num, structure=None):
        """Create client model directory for a specific round.

        Args:
            round_num: The round number
            structure: Dictionary with directory structure information

        Returns:
            str: Path to the client model directory
        """
        if not self.results_dir or not structure:
            return None

        # Create the client directory for this round
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)

        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)

        return client_dir
    
    def load_model(self, model_dir):
        """Load sklearn model from a directory.

        Args:
            model_dir: Directory containing the model
        """
        model_path = os.path.join(model_dir, "model.pkl")
        
        if not os.path.exists(model_path):
            #If no model exists, create a new one
            #Handle classification vs regression
            if self.experiment_type == "mnist":
                # Use classifier for MNIST
                if self.model_type == "random_forest":
                    from sklearn.ensemble import RandomForestClassifier
                    self.model = RandomForestClassifier(**self.model_params)
                elif self.model_type == "xgboost":
                    from xgboost import XGBClassifier
                    self.model = XGBClassifier(**self.model_params)
                else:
                    raise ValueError(f"Unknown model type for classification: {self.model_type}")
            else:
                # Use regressor for regression tasks
                self.model = create_model(
                    model_type=self.model_type,
                    input_size=self.input_dim,
                    **self.model_params
                )
            print(f"Client {self.client_id}: No existing model found, created new {self.model_type} model")
        else:
            # Load sklearn model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            #Check if it's an ensemble - if so, extract first estimator or create new model
            from fl_server_sklearn.aggregation import _WeightedVotingClassifier, _WeightedVotingRegressor
            if isinstance(loaded_model, (_WeightedVotingClassifier, _WeightedVotingRegressor)):
                #Extract first estimator from ensemble (or create new one)
                if loaded_model.estimators:
                    # Use first estimator as base, but create a fresh copy for training
                    first_name, first_est = loaded_model.estimators[0]
                    # Create new model with same type and params
                    if self.experiment_type == "mnist":
                        if self.model_type == "random_forest":
                            from sklearn.ensemble import RandomForestClassifier
                            self.model = RandomForestClassifier(**self.model_params)
                        elif self.model_type == "xgboost":
                            from xgboost import XGBClassifier
                            self.model = XGBClassifier(**self.model_params)
                    else:
                        self.model = create_model(
                            model_type=self.model_type,
                            input_size=self.input_dim,
                            **self.model_params
                        )
                    print(f"Client {self.client_id}: Extracted model from ensemble, created fresh {self.model_type} for training")
                else:
                    raise ValueError("Empty ensemble, cannot extract model")
            else:
                # Regular model, use directly
                self.model = loaded_model
            
            print(f"Client {self.client_id} loaded sklearn model from {model_dir}")
    
    def save_model(self, model_dir):
        """Save sklearn model to a directory.

        Args:
            model_dir: Directory to save the model to
        """
        os.makedirs(model_dir, exist_ok=True)

        #Save sklearn model using pickle
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        metadata = {
            "client_id": self.client_id,
            "experiment_type": self.experiment_type,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Client {self.client_id} saved sklearn model to {model_dir}")
    
    def load_track_models_for_round(self, round_num):
        """Load track-aware models for disagreement resolution in a specific round.
        
        This method loads the primary track model for this client and any background
        track models the client should participate in based on active disagreements.
        
        Args:
            round_num: The current round number
        
        Returns:
            bool: Whether the primary model was successfully loaded
        """
        if not self.results_dir:
            return False
        
        print(f"\n=== CLIENT {self.client_id} MODEL LOADING FOR ROUND {round_num} ===")
        
        structure = self._get_structure_config()
        
        #Get the round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        
        # Check if there are tracks for this round
        tracks_dir = os.path.join(round_dir, "tracks")
        primary_track_loaded = False
        
        # Clear any existing background tracks
        self.background_tracks = []
        
        if os.path.exists(tracks_dir):
            print(f"Found tracks directory at: {tracks_dir}")
            
            # Look for track metadata
            metadata_path = os.path.join(tracks_dir, "track_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        track_metadata = json.load(f)
                    
                    track_names = list(track_metadata.get("tracks", {}).keys())
                    print(f"Found {len(track_names)} tracks: {track_names}")
                    
                    #Find this client's primary track
                    primary_track = track_metadata.get("client_tracks", {}).get(str(self.client_id))
                    
                    if primary_track:
                        print(f"Client {self.client_id} is assigned to primary track: '{primary_track}'")
                        
                        #Load the model from this track
                        track_dir = os.path.join(tracks_dir, primary_track)
                        if os.path.exists(track_dir):
                            model_path = os.path.join(track_dir, "model.pkl")
                            if os.path.exists(model_path):
                                # Load the sklearn model
                                with open(model_path, 'rb') as f:
                                    loaded_model = pickle.load(f)
                                
                                # Handle ensemble - extract first estimator or use directly
                                from fl_server_sklearn.aggregation import _WeightedVotingClassifier, _WeightedVotingRegressor
                                if isinstance(loaded_model, (_WeightedVotingClassifier, _WeightedVotingRegressor)):
                                    if loaded_model.estimators:
                                        first_name, first_est = loaded_model.estimators[0]
                                        # Create fresh model for training
                                        if self.experiment_type == "mnist":
                                            if self.model_type == "random_forest":
                                                from sklearn.ensemble import RandomForestClassifier
                                                self.model = RandomForestClassifier(**self.model_params)
                                            elif self.model_type == "xgboost":
                                                from xgboost import XGBClassifier
                                                self.model = XGBClassifier(**self.model_params)
                                        else:
                                            self.model = create_model(
                                                model_type=self.model_type,
                                                input_size=self.input_dim,
                                                **self.model_params
                                            )
                                        print(f"Client {self.client_id}: Extracted model from ensemble for primary track")
                                    else:
                                        raise ValueError("Empty ensemble in track")
                                else:
                                    #Regular model, use directly
                                    self.model = loaded_model
                                
                                print(f"Successfully loaded primary track model from {track_dir}")
                                primary_track_loaded = True
                                
                                #Now check for background tracks this client should train on
                                participation_tracks = []
                                for track_name, track_clients in track_metadata.get("tracks", {}).items():
                                    track_clients_int = [int(c) if isinstance(c, str) else c for c in track_clients]
                                    if track_name != primary_track and self.client_id in track_clients_int:
                                        participation_tracks.append(track_name)
                                
                                if participation_tracks:
                                    print(f"Client {self.client_id} will also train on background tracks: {participation_tracks}")
                                    self.background_tracks = []
                                    
                                    for bg_track in participation_tracks:
                                        bg_track_dir = os.path.join(tracks_dir, bg_track)
                                        if os.path.exists(bg_track_dir):
                                            bg_model_path = os.path.join(bg_track_dir, "model.pkl")
                                            if os.path.exists(bg_model_path):
                                                # Load background model
                                                with open(bg_model_path, 'rb') as f:
                                                    bg_loaded = pickle.load(f)
                                                
                                                # Handle ensemble
                                                if isinstance(bg_loaded, (_WeightedVotingClassifier, _WeightedVotingRegressor)):
                                                    if bg_loaded.estimators:
                                                        first_name, first_est = bg_loaded.estimators[0]
                                                        # Create fresh model for training
                                                        if self.experiment_type == "mnist":
                                                            if self.model_type == "random_forest":
                                                                from sklearn.ensemble import RandomForestClassifier
                                                                bg_model = RandomForestClassifier(**self.model_params)
                                                            elif self.model_type == "xgboost":
                                                                from xgboost import XGBClassifier
                                                                bg_model = XGBClassifier(**self.model_params)
                                                        else:
                                                            bg_model = create_model(
                                                                model_type=self.model_type,
                                                                input_size=self.input_dim,
                                                                **self.model_params
                                                            )
                                                    else:
                                                        raise ValueError("Empty ensemble in background track")
                                                else:
                                                    bg_model = bg_loaded
                                                
                                                self.background_tracks.append({
                                                    "name": bg_track,
                                                    "model": bg_model,
                                                    "dir": bg_track_dir
                                                })
                                                print(f"Successfully loaded background track model '{bg_track}' from {bg_track_dir}")
                                    print(f"Client {self.client_id} has no background tracks to train on")
                            else:
                                print(f"Warning: Track model file not found at {model_path}")
                        else:
                            print(f"Warning: Track directory not found at {track_dir}")
                    else:
                        print(f"Client {self.client_id} has no assigned primary track, will use global model")
                except Exception as e:
                    print(f"Error loading track models: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"No track metadata found at {metadata_path}")
        else:
            print(f"No tracks directory found at {tracks_dir}, will use standard global model")
        
        #If no tracks found or error occurred, load the standard global model
        if not primary_track_loaded:
            global_model_dir = os.path.join(round_dir, structure["global_model"])
            if os.path.exists(global_model_dir):
                self.load_model(global_model_dir)
                self.background_tracks = []  #No background tracks in standard mode
                primary_track_loaded = True
            else:
                print(f"Warning: Global model directory not found at {global_model_dir}")
                # Try to load from previous round or initial
                if round_num > 1:
                    prev_round_dir = os.path.join(
                        self.results_dir,
                        structure["round_template"].format(round=round_num - 1)
                    )
                    prev_global_dir = os.path.join(prev_round_dir, structure["global_model_aggregated"])
                    if os.path.exists(prev_global_dir):
                        self.load_model(prev_global_dir)
                        primary_track_loaded = True
                if not primary_track_loaded:
                    # Last resort: load from initial model
                    initial_dir = os.path.join(self.results_dir, structure["global_model_initial"])
                    if os.path.exists(initial_dir):
                        self.load_model(initial_dir)
                        primary_track_loaded = True
        
        if self.model is None:
            print(f"Warning: Model still not loaded for client {self.client_id}, creating new model")
            if self.experiment_type == "mnist":
                if self.model_type == "random_forest":
                    from sklearn.ensemble import RandomForestClassifier
                    self.model = RandomForestClassifier(**self.model_params)
                elif self.model_type == "xgboost":
                    from xgboost import XGBClassifier
                    self.model = XGBClassifier(**self.model_params)
            else:
                self.model = create_model(
                    model_type=self.model_type,
                    input_size=self.input_dim,
                    **self.model_params
                )
        
        print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
        return primary_track_loaded
    
    def train_with_disagreement_resolution(self, epochs=None, round_num=None):
        """Train models with disagreement resolution (primary + background tracks).
        
        Args:
            epochs: Number of epochs (not used for sklearn, kept for compatibility)
            round_num: The current round number
        
        Returns:
            dict: Dictionary containing training results from primary model
        """
        print(f"\n=== CLIENT {self.client_id} TRAINING FOR ROUND {round_num} ===")
        
        # Train primary model
        print("Training primary model...")
        training_results = self.train()
        
        if round_num is not None:
            training_results["round"] = round_num
        
        #Train background models if any
        if hasattr(self, 'background_tracks') and self.background_tracks:
            print(f"Client {self.client_id} has {len(self.background_tracks)} background tracks to train")
            
            for bg_track in self.background_tracks:
                print(f"Training on background track: '{bg_track['name']}'")
                
                #Save the current primary model
                primary_model = self.model
                
                # Set the model to the background track model
                self.model = bg_track['model']
                
                # Train on this background track
                bg_results = self.train()
                bg_track['trained'] = True
                bg_track['results'] = bg_results
                
                print(f"Background model '{bg_track['name']}' training complete")
                
                # Restore primary model
                self.model = primary_model
        else:
            print(f"Client {self.client_id} has no background tracks to train")
        
        print(f"=== END CLIENT {self.client_id} TRAINING ===\n")
        return training_results
    
    def save_trained_track_models(self, round_num):
        """Save all trained track models (primary + background) for a specific round.
        
        Args:
            round_num: The current round number
        
        Returns:
            str: Path to the saved primary model directory
        """
        if not self.results_dir:
            return None
        
        print(f"\n=== CLIENT {self.client_id} MODEL SAVING FOR ROUND {round_num} ===")
        
        structure = self._get_structure_config()
        
        #Create the client directory for this round
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )
        
        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)
        
        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)
        
        #Save the primary model
        self.save_model(client_dir)
        print(f"Saved primary model to {client_dir}")
        
        # Save background models if any were trained
        bg_models_saved = 0
        if hasattr(self, 'background_tracks') and self.background_tracks:
            for bg_track in self.background_tracks:
                if bg_track.get('trained', False):
                    print(f"Saving trained background model for track: '{bg_track['name']}'")
                    
                    # Create a special directory for this background model
                    bg_dir = os.path.join(client_dir, f"background_{bg_track['name']}")
                    os.makedirs(bg_dir, exist_ok=True)
                    
                    # Save sklearn model
                    model_path = os.path.join(bg_dir, "model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(bg_track['model'], f)
                    
                    #Save model metadata
                    metadata = {
                        "client_id": self.client_id,
                        "experiment_type": self.experiment_type,
                        "track_name": bg_track['name'],
                        "is_background": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    metadata_path = os.path.join(bg_dir, "metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    bg_models_saved += 1
                    print(f"Saved background model for track '{bg_track['name']}' to {bg_dir}")
        
        if bg_models_saved > 0:
            print(f"Saved {bg_models_saved} background track model(s)")
        else:
            print("No background track models to save")
        
        print(f"=== END CLIENT {self.client_id} MODEL SAVING ===\n")
        return client_dir
    
    def _get_structure_config(self):
        """Get directory structure configuration."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "mock_etcd/configuration.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        structure = config["results"]["structure"].copy()
                        #Adjust for sklearn
                        structure["model_storage_dir"] = "model_storage_sklearn"
                        structure["global_model_initial"] = structure["global_model_initial"].replace("model_storage", "model_storage_sklearn")
                        structure["round_template"] = structure["round_template"].replace("model_storage", "model_storage_sklearn")
                        return structure
        except Exception:
            pass
        
        # Default structure
        return {
            "model_storage_dir": "model_storage_sklearn",
            "global_model_initial": "model_storage_sklearn/global_model_initial",
            "round_template": "model_storage_sklearn/round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "global_model_aggregated": "global_model_aggregated",
            "client_prefix": "client_"
        }
    
    def train(self):
        """Train the sklearn model on client data.
        
        Returns:
            dict: Dictionary containing training results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Train sklearn model (no epochs needed, just fit)
        training_results = train_sklearn_model(
            self.model,
            self.X_train,
            self.y_train,
            experiment_type=self.experiment_type
        )
        
        print(f"Client {self.client_id} completed training")
        return training_results
