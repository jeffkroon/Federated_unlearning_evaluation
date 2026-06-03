"""Main federated learning client implementation."""

import os
import torch
import json
import numpy as np
from datetime import datetime

from fl_module.models import create_model
import fl_module
from fl_client.utils import save_training_results
from fl_client.training import train_model

class FederatedClient:
    """Client-side implementation for federated learning."""

    def __init__(
        self,
        client_id,
        experiment_type,
        data_dir,
        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        device=None,
        results_dir=None,
        config_path=None
    ):
        self.client_id = client_id
        self.experiment_type = experiment_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.results_dir = results_dir
        self.config_path = config_path  # Store config path for reading model dimensions

        #Initialize model via adapter and registry (plug-and-play)
        from fl_module.registry import DatasetAdapterRegistry
        from fl_module.model_registry import ModelRegistry

        model_params = {}
        if config_path and os.path.exists(config_path):
            try:
                from mock_etcd.etcd_loader import MockEtcdLoader
                config = MockEtcdLoader(config_path)
                model_params = config.config.get("unlearning", {}).get("model_params", {})
            except Exception:
                pass

        adapter = DatasetAdapterRegistry.get_adapter(experiment_type)
        if adapter is not None:
            self.seq_len = adapter.get_sequence_length()
            self.input_dim = adapter.get_input_dim()
            self.output_dim = adapter.get_output_dim()
            if self.input_dim is None:
                self.input_dim = model_params.get("input_dim", 20)
            if self.output_dim is None:
                self.output_dim = model_params.get("output_dim", 2)
        else:
            self.seq_len = 50 if experiment_type == "n_cmapss" else 1
            self.input_dim = model_params.get("input_dim", 20)
            self.output_dim = model_params.get("output_dim", 2)

        if experiment_type == "n_cmapss":
            self.n_features = 20
            self.hidden_dim = 32
            kwargs = dict(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        elif experiment_type in ("mnist", "cifar10"):
            kwargs = {}
        else:
            if adapter is None and experiment_type == "tabular":
                self.input_dim, self.output_dim = self._get_tabular_model_dims()
            kwargs = dict(input_dim=self.input_dim, output_dim=self.output_dim)

        model_type = experiment_type
        if model_type.startswith("custom") and ModelRegistry.get_factory(model_type) is None:
            model_type = "tabular"
        self.model = create_model(model_type, **kwargs).to(self.device)

        #Create results directory
        if results_dir:
            self.output_dir = os.path.join(results_dir, "output", "clients", f"client_{client_id}")
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            os.makedirs("output/client_results", exist_ok=True)

    def create_model_dir(self, round_num, structure=None):
        """Create this client's model directory for a round."""
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

    def load_data(self, sample_size=1000):
        """Load and preprocess client data (prefer adapter when registered)."""
        from fl_module.registry import DatasetAdapterRegistry
        from torch.utils.data import DataLoader, TensorDataset

        adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
        if adapter is not None:
            samples, labels = adapter.load_client_data(
                client_id=self.client_id,
                data_dir=self.data_dir,
                sample_size=sample_size
            )
            X_tensor = torch.from_numpy(samples).float()
            if adapter.is_classification():
                y_tensor = torch.from_numpy(labels.astype(np.int64)).long()
            else:
                y_tensor = torch.from_numpy(labels.astype(np.float32)).float()
            n_samples = len(X_tensor)
            n_train = max(1, int(0.8 * n_samples))
            indices = torch.randperm(n_samples)
            train_dataset = TensorDataset(X_tensor[indices[:n_train]], y_tensor[indices[:n_train]])
            valid_dataset = TensorDataset(X_tensor[indices[n_train:]], y_tensor[indices[n_train:]])
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
            print(f"Client {self.client_id} loaded {len(samples)} samples (adapter)")
            return

        if self.experiment_type == "n_cmapss":
            # Load data for this client
            samples, labels = fl_module.load_ncmapss_client_data(
                self.client_id,
                self.data_dir,
                sample_size=sample_size
            )

            # Preprocess data
            samples_normalized, _ = fl_module.preprocess_ncmapss_data(samples)

            #Create dataloaders
            self.train_loader, self.valid_loader = fl_module.create_ncmapss_client_dataloaders(
                samples_normalized,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(samples)} samples")
        elif self.experiment_type == "mnist":
            #Load MNIST data for this client
            images, labels = fl_module.load_mnist_client_data(
                self.client_id,
                train_dir=self.data_dir,
                sample_size=sample_size
            )

            # Create dataloaders
            self.train_loader, self.valid_loader = fl_module.create_mnist_client_dataloaders(
                images,
                labels,
                batch_size=self.batch_size
            )
        elif self.experiment_type == "cifar10":
            images, labels = fl_module.load_cifar10_client_data(
                self.client_id,
                train_dir=self.data_dir,
                sample_size=sample_size
            )
            self.train_loader, self.valid_loader = fl_module.create_cifar10_client_dataloaders(
                images,
                labels,
                batch_size=self.batch_size
            )
        elif self.experiment_type == "tabular":
            # Load tabular data for this client
            features, labels = fl_module.load_tabular_client_data(
                self.client_id,
                self.data_dir,
                sample_size=sample_size
            )

            # Create dataloaders
            self.train_loader, self.valid_loader = fl_module.create_tabular_client_dataloaders(
                features,
                labels,
                batch_size=self.batch_size
            )

            print(f"Client {self.client_id} loaded {len(features)} tabular samples")
        elif self.experiment_type.startswith("custom"):
            #Custom dataset: load data using adapter
            try:
                from fl_module.registry import DatasetAdapterRegistry
                from torch.utils.data import DataLoader, TensorDataset
                
                adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
                if adapter is None:
                    raise ValueError(f"No adapter registered for experiment_type='{self.experiment_type}'. "
                                   f"Make sure to call register_custom_dataset() first.")
                
                #Load client data
                samples, labels = adapter.load_client_data(
                    client_id=self.client_id,
                    data_dir=self.data_dir,
                    sample_size=sample_size
                )
                
                # Determine if classification or regression
                # int labels = classification, float = regression
                unique_labels = np.unique(labels)
                # Classification if: few unique values, integer type, or can be cast to int
                try:
                    labels_as_int = labels.astype(int)
                    is_classification = (len(unique_labels) <= 10 and 
                                       np.allclose(labels, labels_as_int))
                except:
                    is_classification = len(unique_labels) <= 10
                
                #Convert to tensors
                X_tensor = torch.from_numpy(samples).float()
                if is_classification:
                    y_tensor = torch.from_numpy(labels.astype(int)).long()
                else:
                    y_tensor = torch.from_numpy(labels).float()
                
                #Split into train/validation (80/20)
                n_samples = len(X_tensor)
                n_train = int(0.8 * n_samples)
                indices = torch.randperm(n_samples)
                train_indices = indices[:n_train]
                valid_indices = indices[n_train:]
                
                train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
                valid_dataset = TensorDataset(X_tensor[valid_indices], y_tensor[valid_indices])
                
                self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
                
                print(f"Client {self.client_id} loaded {len(samples)} custom samples")
            except ImportError:
                raise ImportError("Custom dataset support requires fl_module.custom.adapter. "
                                "Make sure machine_unlearning_tool is available.")
        else:
            # For other experiments
            raise NotImplementedError(f"{self.experiment_type} data loading not implemented yet")

    def save_model(self, model_dir):
        """Save the model state dict and a small metadata file."""
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        metadata = {
            "client_id": self.client_id,
            "experiment_type": self.experiment_type,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Client {self.client_id} saved model to {model_dir}")

    def load_model(self, model_dir):
        """Load the model state dict from a directory."""
        model_path = os.path.join(model_dir, "model.pt")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Client {self.client_id} loaded model from {model_dir}")

    def load_track_models_for_round(self, round_num):
        """Load this client's primary track model (and any background tracks) for a round.

        Falls back to the global model if no track is assigned. Returns whether a
        primary model was loaded.
        """
        if not self.results_dir:
            return False

        print(f"\n=== CLIENT {self.client_id} MODEL LOADING FOR ROUND {round_num} ===")

        # Get directory structure from configuration
        structure = self._get_structure_config()

        # Get the round directory
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        #Check if there are tracks for this round
        tracks_dir = os.path.join(round_dir, "tracks")
        primary_track_loaded = False

        #Clear any existing background tracks
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

                    # Find this client's primary track
                    primary_track = track_metadata.get("client_tracks", {}).get(str(self.client_id))

                    if primary_track:
                        print(f"Client {self.client_id} is assigned to primary track: '{primary_track}'")

                        # Look for track metadata to get details about this track
                        track_metadata_path = os.path.join(tracks_dir, primary_track, "metadata.json")
                        if os.path.exists(track_metadata_path):
                            try:
                                with open(track_metadata_path, 'r') as f:
                                    primary_track_metadata = json.load(f)
                                track_info = f"Primary track '{primary_track}' details:"
                                track_info += f"\n        Round: {primary_track_metadata.get('round', 'N/A')}"
                                track_info += f"\n        Clients: {primary_track_metadata.get('client_ids', [])}"
                                track_info += f"\n        Rewound this round: {primary_track_metadata.get('rewound_this_round', False)}"
                                finetuning_status = primary_track_metadata.get('finetuning_status', {})
                                if finetuning_status:
                                    track_info += f"\n        Finetuning status: {finetuning_status}"
                                    #Check if this client is in finetuning
                                    client_ft_status = finetuning_status.get(str(self.client_id))
                                    if client_ft_status:
                                        track_info += f"\n        This client's finetuning: {client_ft_status}"
                                else:
                                    track_info += "\n        No clients currently finetuning"
                                print(track_info)
                            except Exception as e:
                                print(f"Error reading track metadata: {e}")

                        #Load the model from this track
                        track_dir = os.path.join(tracks_dir, primary_track)
                        if os.path.exists(track_dir):
                            model_path = os.path.join(track_dir, "model.pt")
                            if os.path.exists(model_path):
                                # Load the model
                                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                                print(f"Successfully loaded primary track model from {track_dir}")

                                # Check if this is a continuation from a previous round's track
                                if os.path.exists(track_metadata_path):
                                    previous_round = primary_track_metadata.get("previous_round")
                                    if previous_round is not None:
                                        print(f"This model continues from the same track in round {previous_round}")
                                    else:
                                        print(f"This is a new track model created for round {round_num}")

                                primary_track_loaded = True

                                # background tracks this client should train on
                                participation_tracks = []
                                for track_name, track_clients in track_metadata.get("tracks", {}).items():
                                    #Convert track_clients to integers for comparison if they're strings
                                    track_clients_int = [int(c) if isinstance(c, str) else c for c in track_clients]
                                    if track_name != primary_track and self.client_id in track_clients_int:
                                        participation_tracks.append(track_name)

                                if participation_tracks:
                                    print(f"Client {self.client_id} will also train on background tracks: {participation_tracks}")
                                    self.background_tracks = []

                                    for bg_track in participation_tracks:
                                        bg_track_dir = os.path.join(tracks_dir, bg_track)
                                        if os.path.exists(bg_track_dir):
                                            bg_model_path = os.path.join(bg_track_dir, "model.pt")
                                            if os.path.exists(bg_model_path):
                                                #Create a separate model for this background track
                                                if self.experiment_type == "n_cmapss":
                                                    bg_model = create_model(
                                                        self.experiment_type,
                                                        input_dim=self.input_dim,
                                                        hidden_dim=self.hidden_dim,
                                                        output_dim=self.output_dim
                                                    ).to(self.device)
                                                elif self.experiment_type in ("tabular", "adult"):
                                                    bg_model = create_model(
                                                        self.experiment_type,
                                                        input_dim=self.input_dim,
                                                        output_dim=self.output_dim
                                                    ).to(self.device)
                                                else:
                                                    bg_model = create_model(self.experiment_type).to(self.device)

                                                # Load the model weights
                                                bg_model.load_state_dict(torch.load(bg_model_path, map_location=self.device))
                                                self.background_tracks.append({
                                                    "name": bg_track,
                                                    "model": bg_model,
                                                    "dir": bg_track_dir
                                                })
                                                print(f"Successfully loaded background track model '{bg_track}' from {bg_track_dir}")
                                else:
                                    print(f"Client {self.client_id} has no background tracks to train on")
                            else:
                                print(f"Warning: Model file not found at {model_path}")
                        else:
                            print(f"Warning: Track directory not found at {track_dir}")
                    else:
                        print(f"Client {self.client_id} has no assigned primary track, will use global model")
                except Exception as e:
                    print(f"Error loading track models: {e}")
            else:
                print(f"No track metadata found at {metadata_path}")
        else:
            print(f"No tracks directory found at {tracks_dir}, will use standard global model")

        # fall back to the standard global model if there are no tracks
        if not primary_track_loaded:
            global_model_dir = os.path.join(round_dir, structure["global_model"])
            if os.path.exists(global_model_dir):
                model_path = os.path.join(global_model_dir, "model.pt")
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Successfully loaded standard global model from {global_model_dir}")
                    self.background_tracks = []  # No background tracks in standard mode
                    print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
                    return True
                else:
                    print(f"Warning: Model file not found at {model_path}")
            else:
                print(f"Warning: Global model directory not found at {global_model_dir}")

            print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
            return False

        print(f"=== END CLIENT {self.client_id} MODEL LOADING ===\n")
        return True

    def train_with_disagreement_resolution(self, epochs=None, round_num=None):
        """Train the primary track model plus any background tracks; return primary results."""
        epochs = epochs or self.epochs

        print(f"\n=== CLIENT {self.client_id} TRAINING FOR ROUND {round_num} ===")

        #Train primary model
        print("Training primary model...")
        training_results = train_model(self, epochs)

        #Add round number to the training results
        if round_num is not None:
            training_results["round"] = round_num

        save_training_results(self, training_results, round_num)

        accuracy = training_results.get('accuracy', 'N/A')
        accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else accuracy
        print(f"Primary model training complete. Accuracy: {accuracy_str}")

        # Train background models if any
        if hasattr(self, 'background_tracks') and self.background_tracks:
            print(f"Client {self.client_id} has {len(self.background_tracks)} background tracks to train")

            for bg_track in self.background_tracks:
                print(f"Training on background track: '{bg_track['name']}'")

                # Save the current primary model state
                primary_state = self.model.state_dict()

                # Set the model to the background track model
                self.model = bg_track['model']

                #Train on this background track
                bg_results = train_model(self, epochs)
                bg_track['trained'] = True

                bg_accuracy = bg_results.get('accuracy', 'N/A')
                bg_accuracy_str = f"{bg_accuracy:.4f}" if isinstance(bg_accuracy, (float, int)) else bg_accuracy
                print(f"Background model '{bg_track['name']}' training complete. Accuracy: {bg_accuracy_str}")

                #Restore primary model
                self.model.load_state_dict(primary_state)
        else:
            print(f"Client {self.client_id} has no background tracks to train")

        print(f"=== END CLIENT {self.client_id} TRAINING ===\n")
        return training_results



    def save_trained_track_models(self, round_num):
        """Save the trained primary and background track models for this round."""
        if not self.results_dir:
            return None

        print(f"\n=== CLIENT {self.client_id} MODEL SAVING FOR ROUND {round_num} ===")

        structure = self._get_structure_config()

        # Create the client directory for this round
        round_dir = os.path.join(
            self.results_dir,
            structure["round_template"].format(round=round_num)
        )

        clients_dir = os.path.join(round_dir, structure["clients_dir"])
        os.makedirs(clients_dir, exist_ok=True)

        client_dir = os.path.join(clients_dir, f"{structure['client_prefix']}{self.client_id}")
        os.makedirs(client_dir, exist_ok=True)

        # Save the primary model
        self.save_model(client_dir)
        print(f"Saved primary model to {client_dir}")

        # Save background models if any were trained
        bg_models_saved = 0
        if hasattr(self, 'background_tracks') and self.background_tracks:
            for bg_track in self.background_tracks:
                if bg_track.get('trained', False):
                    print(f"Saving trained background model for track: '{bg_track['name']}'")

                    #Create a special directory for this background model
                    bg_dir = os.path.join(client_dir, f"background_{bg_track['name']}")
                    os.makedirs(bg_dir, exist_ok=True)

                    #Save model state dict
                    model_path = os.path.join(bg_dir, "model.pt")
                    torch.save(bg_track['model'].state_dict(), model_path)

                    # Save model metadata
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

        print(f"Saved {bg_models_saved} background models")
        print(f"=== END CLIENT {self.client_id} MODEL SAVING ===\n")
        return client_dir

    def get_model_parameters(self):
        """Model parameters to send to the server."""
        return self.model.get_parameters()

    def set_model_parameters(self, parameters):
        """Update the model with parameters from the server."""
        self.model.set_parameters(parameters)
        print(f"Client {self.client_id} updated model with parameters from server")

    def _get_tabular_model_dims(self):
        """Read (input_dim, output_dim) for the tabular model from config, else defaults."""
        import os
        import json

        # prefer the config_path passed by the orchestrator
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    if "unlearning" in config and "model_params" in config["unlearning"]:
                        model_params = config["unlearning"]["model_params"]
                        input_dim = model_params.get("input_dim", 20)
                        output_dim = model_params.get("output_dim", 2)
                        return input_dim, output_dim
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}. Falling back to default location.")
        
        # Fallback to default config location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_config_path = os.path.join(base_dir, "mock_etcd/configuration.json")
        
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, 'r') as f:
                    config = json.load(f)
                    if "unlearning" in config and "model_params" in config["unlearning"]:
                        model_params = config["unlearning"]["model_params"]
                        input_dim = model_params.get("input_dim", 20)
                        output_dim = model_params.get("output_dim", 2)
                        return input_dim, output_dim
            except:
                pass
        
        #Final fallback to defaults
        return 20, 2

    def _get_structure_config(self):
        """Return the results directory layout from config, or sensible defaults."""
        default_structure = {
            "model_storage_dir": "model_storage",
            "round_template": "model_storage/round_{round}",
            "clients_dir": "clients",
            "global_model": "global_model_for_training",
            "client_prefix": "client_"
        }

        #Try to load from configuration file
        config_path = os.path.join(os.path.dirname(self.results_dir), "mock_etcd/configuration.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        return config["results"]["structure"]
        except Exception as e:
            print(f"Error loading configuration: {e}")

        return default_structure
