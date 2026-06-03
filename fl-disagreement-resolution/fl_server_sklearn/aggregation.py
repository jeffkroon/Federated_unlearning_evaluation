"""Sklearn model aggregation for federated learning."""

import os
import pickle
import glob
import json
import time
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import VotingRegressor, VotingClassifier

from fl_server_sklearn.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks,
)


def aggregate_sklearn_models(
    client_models: Dict[int, Any],
    aggregation_weights: Dict[int, float],
    model_type: str = "random_forest",
    is_classification: bool = False
) -> Any:
    """Aggregate sklearn models using ensemble method.
    
    For sklearn models, we use VotingRegressor/VotingClassifier to create an ensemble.
    This is fundamentally different from PyTorch's parameter averaging.
    
    Args:
        client_models: Dictionary mapping client IDs to sklearn models
        aggregation_weights: Dictionary mapping client IDs to weights
        model_type: Type of model ('random_forest' or 'xgboost')
        is_classification: Whether this is a classification task (True) or regression (False)
    
    Returns:
        Aggregated sklearn model (VotingRegressor or VotingClassifier ensemble)
    """
    if not client_models:
        raise ValueError("No client models provided for aggregation")
    
    # Create list of (name, model) tuples
    estimators = []
    weights = []
    
    for client_id, model in client_models.items():
        estimators.append((f"client_{client_id}", model))
        weight = aggregation_weights.get(client_id, 1.0 / len(client_models))
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    
    # Create ensemble based on task type
    #VotingClassifier/VotingRegressor requires fit(), but our models are already fitted
    #So we create a wrapper that combines predictions directly
    if is_classification:
        # Use VotingClassifier for classification
        # We need to fit it, but since estimators are already fitted, we can use a dummy fit
        # Actually, VotingClassifier needs to be fitted even if estimators are pre-fitted
        #Use a wrapper class instead
        ensemble = _WeightedVotingClassifier(estimators, weights)
        print(f"Created WeightedVotingClassifier ensemble with {len(estimators)} models")
    else:
        #Use VotingRegressor for regression
        ensemble = _WeightedVotingRegressor(estimators, weights)
        print(f"Created WeightedVotingRegressor ensemble with {len(estimators)} models")
    
    print(f"  Weights: {[f'{w:.3f}' for w in weights]}")
    
    return ensemble


class _WeightedVotingClassifier:
    """Wrapper for weighted voting classifier that works with pre-fitted models."""
    
    def __init__(self, estimators, weights):
        self.estimators = estimators
        self.weights = weights
    
    def predict(self, X):
        """Predict using weighted voting."""
        import numpy as np
        from collections import Counter
        
        # Get predictions from all estimators
        all_predictions = []
        for name, est in self.estimators:
            pred = est.predict(X)
            all_predictions.append(pred)
        
        # Weighted voting: for each sample, count weighted votes per class
        n_samples = len(X)
        n_classes = len(set(all_predictions[0]))
        weighted_votes = np.zeros((n_samples, n_classes))
        
        for i, (name, est) in enumerate(self.estimators):
            pred = all_predictions[i]
            weight = self.weights[i]
            for j, class_pred in enumerate(pred):
                weighted_votes[j, int(class_pred)] += weight
        
        # Return class with highest weighted vote
        return np.argmax(weighted_votes, axis=1)


class _WeightedVotingRegressor:
    """Wrapper for weighted voting regressor that works with pre-fitted models."""
    
    def __init__(self, estimators, weights):
        self.estimators = estimators
        self.weights = weights
    
    def predict(self, X):
        """Predict using weighted average."""
        import numpy as np
        
        #Get predictions from all estimators
        predictions = []
        for name, est in self.estimators:
            pred = est.predict(X)
            predictions.append(pred)
        
        #Weighted average
        predictions = np.array(predictions)  # [n_estimators, n_samples]
        weights = np.array(self.weights).reshape(-1, 1)  # [n_estimators, 1]
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        return weighted_pred


def aggregate_models_from_files(server, clients_dir, aggregation_weights=None):
    """Aggregate sklearn models from client files with disagreement resolution.
    
    Args:
        server: SklearnFederatedServer instance
        clients_dir: Directory containing client model directories
        aggregation_weights: Optional dictionary mapping client IDs to weights
    
    Returns:
        Aggregated sklearn model (ensemble)
    """
    aggregation_start_time = time.time()
    
    server.round += 1
    print(f"Starting file-based aggregation for round {server.round}")
    
    # Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]
    
    print(f"Found {len(client_dirs)} clients: {client_ids}")
    
    #If no weights provided, use equal weighting
    if aggregation_weights is None:
        n_clients = len(client_dirs)
        aggregation_weights = {client_id: 1.0 / n_clients for client_id in client_ids}
    
    disagreement_start_time = time.time()
    
    etcd_dir = "mock_etcd"
    disagreements = load_disagreements(etcd_dir)
    active_disagreements = get_active_disagreements(disagreements, server.round)
    
    disagreement_loading_time = time.time() - disagreement_start_time
    
    timing_metrics = {
        "disagreement_loading_time_seconds": disagreement_loading_time,
        "resolution_time_seconds": 0.0,
        "aggregation_time_seconds": 0.0,
        "track_saving_time_seconds": 0.0,
        "total_aggregation_time_seconds": 0.0
    }
    
    #If there are active disagreements, use robust approach with model tracks
    if active_disagreements:
        print(f"Active disagreements found for round {server.round}: {active_disagreements}")
        
        # Time the track creation
        track_creation_start_time = time.time()
        track_info = create_model_tracks(active_disagreements, client_ids)
        
        # Extract resolution time from track_info if available
        if "timing_metrics" in track_info:
            timing_metrics["resolution_time_seconds"] = track_info["timing_metrics"]["resolution_time_seconds"]
        else:
            timing_metrics["resolution_time_seconds"] = time.time() - track_creation_start_time
        
        # Time the actual aggregation
        track_aggregation_start_time = time.time()
        result = aggregate_with_tracks(server, clients_dir, track_info, aggregation_weights)
        timing_metrics["aggregation_time_seconds"] = time.time() - track_aggregation_start_time
        
    else:
        #If no disagreements, use standard aggregation
        print(f"No active disagreements for round {server.round}, using standard aggregation")
        
        #Time the standard aggregation
        standard_aggregation_start_time = time.time()
        result = aggregate_standard(server, clients_dir, aggregation_weights)
        timing_metrics["aggregation_time_seconds"] = time.time() - standard_aggregation_start_time
    
    total_time = time.time() - aggregation_start_time
    timing_metrics["total_aggregation_time_seconds"] = total_time
    
    # Store timing metrics in server for tracking
    if not hasattr(server, 'aggregation_timing_history'):
        server.aggregation_timing_history = []
    
    round_timing = {
        "round": server.round,
        "has_disagreements": bool(active_disagreements),
        "num_clients": len(client_dirs),
        **timing_metrics
    }
    server.aggregation_timing_history.append(round_timing)
    
    # Add timing metrics to training history for plotting
    for metric_name, metric_value in timing_metrics.items():
        history_key = f"aggregation_{metric_name}"
        if history_key not in server.training_history:
            server.training_history[history_key] = []
        server.training_history[history_key].append(metric_value)
    
    print(f"\n=== AGGREGATION TIMING SUMMARY FOR ROUND {server.round} ===")
    print(f"  Disagreement loading: {timing_metrics['disagreement_loading_time_seconds']:.4f}s")
    print(f"  Resolution time: {timing_metrics['resolution_time_seconds']:.4f}s")
    print(f"  Aggregation time: {timing_metrics['aggregation_time_seconds']:.4f}s")
    print(f"  Track saving time: {timing_metrics['track_saving_time_seconds']:.4f}s")
    print(f"  Total time: {timing_metrics['total_aggregation_time_seconds']:.4f}s")
    print(f"  Has disagreements: {bool(active_disagreements)}")
    print(f"=== END TIMING SUMMARY ===\n")
    
    return result


def aggregate_standard(server, clients_dir, aggregation_weights):
    """Standard sklearn model aggregation without tracks.
    
    Args:
        server: SklearnFederatedServer instance
        clients_dir: Directory containing client model directories
        aggregation_weights: Dictionary mapping client IDs to weights
    
    Returns:
        Aggregated sklearn model (ensemble)
    """
    # Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]
    
    #Load all client models
    client_models = {}
    for client_dir, client_id in zip(client_dirs, client_ids):
        model_path = os.path.join(client_dir, "model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                client_models[client_id] = pickle.load(f)
            print(f"Loaded model from client {client_id}")
        else:
            print(f"Warning: Model file not found for client {client_id}")
    
    if not client_models:
        raise ValueError("No client models found for aggregation")
    
    #Aggregate using ensemble method
    is_classification = (server.experiment_type == "mnist")
    aggregated_model = aggregate_sklearn_models(
        client_models,
        aggregation_weights,
        model_type=server.model_type,
        is_classification=is_classification
    )
    
    # Update server's global model
    server.global_model = aggregated_model
    print(f"Updated global model with ensemble from {len(client_models)} clients")
    
    return aggregated_model


def aggregate_with_tracks(server, clients_dir, track_info, aggregation_weights):
    """Aggregate sklearn models using robust approach with model tracks.
    
    Args:
        server: SklearnFederatedServer instance
        clients_dir: Directory containing client model directories
        track_info: Dictionary containing track information
        aggregation_weights: Dictionary mapping client IDs to weights
    
    Returns:
        Aggregated sklearn model (ensemble) for the global track
    """
    # Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]
    
    print(f"\n=== AGGREGATION FOR ROUND {server.round} ===")
    print(f"Found {len(client_dirs)} clients: {client_ids}")
    
    # Load all client models (both primary and background)
    client_models_dict = {}  #client_id -> model
    background_models_dict = {}  #track_name -> {client_id -> model}
    
    for client_dir, client_id in zip(client_dirs, client_ids):
        # Load primary client model
        model_path = os.path.join(client_dir, "model.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                client_models_dict[client_id] = pickle.load(f)
            print(f"Loaded primary model from client {client_id}")
        else:
            print(f"Warning: Primary model file not found for client {client_id}")
        
        # Check for background models
        background_dirs = glob.glob(os.path.join(client_dir, "background_*"))
        for bg_dir in background_dirs:
            bg_track_name = os.path.basename(bg_dir).replace("background_", "", 1)
            bg_model_path = os.path.join(bg_dir, "model.pkl")
            
            if os.path.exists(bg_model_path):
                with open(bg_model_path, 'rb') as f:
                    bg_model = pickle.load(f)
                
                # Initialize the dictionary for this track if needed
                if bg_track_name not in background_models_dict:
                    background_models_dict[bg_track_name] = {}
                
                #Store the background model
                background_models_dict[bg_track_name][client_id] = bg_model
                print(f"Loaded background model from client {client_id} for track {bg_track_name}")
    
    #Dictionary to store track aggregations
    track_models = {}
    tracks = track_info.get("tracks", {})
    
    print(f"\nAggregating {len(tracks)} tracks:")
    
    # Count the number of custom tracks (excluding global)
    custom_tracks = [t for t in tracks.keys() if t != "global"]
    has_custom_tracks = len(custom_tracks) > 0
    
    # Get initial model as fallback
    initial_model = None
    try:
        structure = get_structure_config(server)
        initial_model_dir = os.path.join(
            server.results_dir,
            structure["global_model_initial"]
        )
        initial_model_path = os.path.join(initial_model_dir, "model.pkl")
        if os.path.exists(initial_model_path):
            with open(initial_model_path, 'rb') as f:
                initial_model = pickle.load(f)
            print("Loaded initial model as fallback")
    except Exception as e:
        print(f"Warning: Could not load initial model as fallback: {e}")
    
    # First, create a baseline global model with standard aggregation for reference
    #This is a reference/baseline model that aggregates ALL clients using standard aggregation,
    #regardless of disagreements. It serves as a comparison point and is never modified by unlearning.
    baseline_global_models = {}
    for client_id, model in client_models_dict.items():
        weight = aggregation_weights.get(client_id, 1.0 / len(client_dirs))
        baseline_global_models[client_id] = (model, weight)
    
    is_classification = (server.experiment_type == "mnist")
    if baseline_global_models:
        baseline_global_ensemble = aggregate_sklearn_models(
            {cid: m for cid, (m, w) in baseline_global_models.items()},
            {cid: w for cid, (m, w) in baseline_global_models.items()},
            model_type=server.model_type,
            is_classification=is_classification
        )
        track_models["baseline_global"] = baseline_global_ensemble
        print(f"Created baseline global model (reference) with ensemble from all {len(baseline_global_models)} clients")
    
    # Aggregate each track
    for track_name, track_clients in tracks.items():
        # Skip the default track if we have disagreements
        if track_name == "default" and has_custom_tracks:
            print("Skipping default track to ensure track separation in disagreement scenario")
            continue
        
        print(f"\nAggregating track: '{track_name}' with clients: {sorted(track_clients)}")
        
        # Skip tracks with no clients
        if not track_clients:
            print(f"Skipping empty track: {track_name}")
            continue
        
        #Collect models for this track
        track_client_models = {}
        track_weights = {}
        
        #First, aggregate primary client models for this track
        for client_id in track_clients:
            # Get client's primary track
            client_primary_track = track_info.get("client_tracks", {}).get(str(client_id))
            
            # Only include this client's model if this is its primary track
            if client_primary_track == track_name:
                if client_id not in client_models_dict:
                    print(f"    Warning: Model for client {client_id} (primary for this track) not found. Skipping.")
                    continue
                
                model = client_models_dict[client_id]
                weight = aggregation_weights.get(client_id, 1.0 / len(track_clients))
                
                track_client_models[client_id] = model
                track_weights[client_id] = weight
                print(f"  Including primary model from client {client_id} with weight {weight:.4f}")
        
        # Now add background models
        if track_name in background_models_dict:
            for client_id, bg_model in background_models_dict[track_name].items():
                #Skip client if it's already included as primary
                if track_info.get("client_tracks", {}).get(str(client_id)) == track_name:
                    continue
                
                weight = aggregation_weights.get(client_id, 1.0 / len(track_clients))
                track_client_models[client_id] = bg_model
                track_weights[client_id] = weight
                print(f"  Including background model from client {client_id} with weight {weight:.4f}")
        
        #Aggregate this track
        if track_client_models:
            track_ensemble = aggregate_sklearn_models(
                track_client_models,
                track_weights,
                model_type=server.model_type,
                is_classification=is_classification
            )
            track_models[track_name] = track_ensemble
            print(f"  Track '{track_name}' aggregation complete with {len(track_client_models)} models")
        else:
            # Fallback to baseline global if no models
            print(f"  Warning: Warning: No models for track '{track_name}', using baseline global as fallback")
            track_models[track_name] = track_models.get("baseline_global", initial_model)
    
    # Update server's global model with baseline global BEFORE saving
    # server.global_model is a model object used for evaluation, not a specific track
    #This ensures server.save_model() in orchestrator saves the correct ensemble
    if "baseline_global" in track_models:
        server.global_model = track_models["baseline_global"]
        print("\nUpdated server's global_model object with baseline global ensemble (reference)")
    
    #Save track models to disk (this also saves baseline_global to aggregated dir)
    save_track_models(server, track_models, track_info)
    
    print(f"=== END AGGREGATION FOR ROUND {server.round} ===\n")
    
    # Return the global track model
    # "global" track is a track for clients without disagreements (may have unlearning applied)
    # This is different from "baseline_global" which is the reference model with all clients
    if "global" in track_models:
        return track_models["global"]
    elif "baseline_global" in track_models:
        return track_models["baseline_global"]
    else:
        return list(track_models.values())[0] if track_models else None


def save_track_models(server, track_models, track_info):
    """Save all track models to disk.
    
    Args:
        server: SklearnFederatedServer instance
        track_models: Dictionary of track models (ensembles)
        track_info: Dictionary of track information
    """
    track_saving_start_time = time.time()
    
    structure = get_structure_config(server)
    
    #Always save the baseline global model (reference) to the global model aggregated location
    #This is the reference model that aggregates all clients using standard aggregation
    global_model_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=server.round),
        structure["global_model_aggregated"]
    )
    os.makedirs(global_model_dir, exist_ok=True)
    
    if "baseline_global" in track_models:
        model_path = os.path.join(global_model_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(track_models["baseline_global"], f)
        print(f"Saved baseline global model (reference) to {global_model_dir}")
    
    # Check if there are any active disagreements - don't create tracks directory if not
    if not track_info.get("tracks", {}) or (len(track_info.get("tracks", {})) == 1 and "global" in track_info.get("tracks", {})):
        print(f"No active disagreements for round {server.round}, skipping track creation")
        
        track_saving_time = time.time() - track_saving_start_time
        # Update the timing metrics in the aggregation history if available
        if hasattr(server, 'aggregation_timing_history') and server.aggregation_timing_history:
            server.aggregation_timing_history[-1]["track_saving_time_seconds"] = track_saving_time
        if "aggregation_track_saving_time_seconds" in server.training_history:
            server.training_history["aggregation_track_saving_time_seconds"][-1] = track_saving_time
        
        return
    
    # Create "tracks" directory for this round
    round_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=server.round)
    )
    tracks_dir = os.path.join(round_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    
    #Save track metadata
    track_metadata = {
        "round": server.round,
        "tracks": {k: list(v) for k, v in track_info.get("tracks", {}).items()},
        "client_tracks": track_info.get("client_tracks", {})
    }
    
    metadata_path = os.path.join(tracks_dir, "track_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(track_metadata, f, indent=2)
    
    #Save each track model
    for track_name, model in track_models.items():
        # Skip baseline_global; it is already saved as the global model aggregated
        if track_name == "baseline_global":
            continue
        
        track_dir = os.path.join(tracks_dir, track_name)
        os.makedirs(track_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(track_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save track-specific metadata
        track_specific_metadata = {
            "track_name": track_name,
            "round": server.round,
            "client_ids": list(track_info.get("tracks", {}).get(track_name, []))
        }
        
        track_metadata_path = os.path.join(track_dir, "metadata.json")
        with open(track_metadata_path, "w") as f:
            json.dump(track_specific_metadata, f, indent=2)
        
        print(f"Saved track model: {track_name}")
    
    track_saving_time = time.time() - track_saving_start_time
    print(f"Track saving completed in {track_saving_time:.4f} seconds")
    
    #Update the timing metrics in the aggregation history if available
    if hasattr(server, 'aggregation_timing_history') and server.aggregation_timing_history:
        server.aggregation_timing_history[-1]["track_saving_time_seconds"] = track_saving_time
    if "aggregation_track_saving_time_seconds" in server.training_history:
        server.training_history["aggregation_track_saving_time_seconds"][-1] = track_saving_time


def get_structure_config(server):
    """Get directory structure configuration.
    
    Args:
        server: SklearnFederatedServer instance
    
    Returns:
        dict: Directory structure configuration
    """
    #Default structure configuration
    default_structure = {
        "model_storage_dir": "model_storage_sklearn",
        "global_model_initial": "model_storage_sklearn/global_model_initial",
        "round_template": "model_storage_sklearn/round_{round}",
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
                    # Override model_storage_dir for sklearn
                    structure = config["results"]["structure"].copy()
                    structure["model_storage_dir"] = "model_storage_sklearn"
                    structure["global_model_initial"] = structure["global_model_initial"].replace("model_storage", "model_storage_sklearn")
                    structure["round_template"] = structure["round_template"].replace("model_storage", "model_storage_sklearn")
                    return structure
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    return default_structure
