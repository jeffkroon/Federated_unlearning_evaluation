"""Model aggregation functionality for federated learning."""

import os
import torch
import glob
import json
import time

from fl_module import create_model
from fl_server.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks,
)


def _make_temp_model(server):
    """Build an empty model of the right type for loading client parameters."""
    from fl_module.model_registry import ModelRegistry
    model_type = server.experiment_type
    if isinstance(model_type, str) and model_type.startswith("custom") and ModelRegistry.get_factory(model_type) is None:
        model_type = "tabular"
    if server.experiment_type == "n_cmapss":
        return create_model(model_type, input_dim=server.input_dim, hidden_dim=server.hidden_dim, output_dim=server.output_dim).to(server.device)
    if server.experiment_type in ("tabular",) or (isinstance(server.experiment_type, str) and server.experiment_type.startswith("custom")):
        return create_model(model_type, input_dim=server.input_dim, output_dim=server.output_dim).to(server.device)
    return create_model(model_type).to(server.device)


def aggregate_models_from_files(server, clients_dir, aggregation_weights=None):
    """FedAvg over the round's client models, using tracks when disagreements are active."""
    aggregation_start_time = time.time()

    server.round += 1
    print(f"Starting file-based aggregation for round {server.round}")

    #Find all client directories
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

    print(f"Found {len(client_dirs)} clients: {client_ids}")

    # If no weights provided, use equal weighting
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

    # active disagreements: use model tracks
    if active_disagreements:
        print(f"Active disagreements found for round {server.round}: {active_disagreements}")

        # Time the track creation
        track_creation_start_time = time.time()
        track_info = create_model_tracks(active_disagreements, client_ids)

        #Extract resolution time from track_info if available
        if "timing_metrics" in track_info:
            timing_metrics["resolution_time_seconds"] = track_info["timing_metrics"]["resolution_time_seconds"]
        else:
            timing_metrics["resolution_time_seconds"] = time.time() - track_creation_start_time

        #Time the actual aggregation
        track_aggregation_start_time = time.time()
        result = aggregate_with_tracks(server, clients_dir, track_info, aggregation_weights)
        timing_metrics["aggregation_time_seconds"] = time.time() - track_aggregation_start_time

    else:
        # If no disagreements, use standard aggregation
        print(f"No active disagreements for round {server.round}, using standard aggregation")

        # Time the standard aggregation
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

    #Add timing metrics to training history for plotting
    for metric_name, metric_value in timing_metrics.items():
        history_key = f"aggregation_{metric_name}"
        if history_key not in server.training_history:
            server.training_history[history_key] = []
        server.training_history[history_key].append(metric_value)

    print(f"\nAggregation timing summary for round {server.round}:")
    print(f"  Disagreement loading: {timing_metrics['disagreement_loading_time_seconds']:.4f}s")
    print(f"  Resolution time: {timing_metrics['resolution_time_seconds']:.4f}s")
    print(f"  Aggregation time: {timing_metrics['aggregation_time_seconds']:.4f}s")
    print(f"  Track saving time: {timing_metrics['track_saving_time_seconds']:.4f}s")
    print(f"  Total time: {timing_metrics['total_aggregation_time_seconds']:.4f}s")
    print(f"  Has disagreements: {bool(active_disagreements)}")
    print()

    return result

def aggregate_standard(server, clients_dir, aggregation_weights):
    """Plain weighted FedAvg over all client models (no tracks)."""
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

    #temp model for loading client models (plug-and-play)
    temp_model = _make_temp_model(server)

    # Initialize new global parameters with zeros
    global_parameters = [torch.zeros_like(param) for param in server.global_model.parameters()]

    # Load global finetuning status (if applicable)
    current_global_finetuning_status = {}
    lifting_mechanism = server.disagreement_settings.get("lifting_mechanism", "shallow")
    finetune_total_rounds = server.disagreement_settings.get("deep_lifting_finetune_rounds", 3)

    if lifting_mechanism == "deep_incr_finetune" and server.round > 0: # server.round is current round
        print(f"  Deep incremental finetuning active for round {server.round}:")
        structure = get_structure_config(server)
        current_round_dir = os.path.join(
            server.results_dir,
            structure["round_template"].format(round=server.round)
        )
        global_finetune_status_path = os.path.join(current_round_dir, "global_finetuning_status.json")

        if os.path.exists(global_finetune_status_path):
            try:
                with open(global_finetune_status_path, 'r') as f_fs:
                    current_global_finetuning_status = json.load(f_fs)
                if current_global_finetuning_status:
                    print(f"    Clients in finetuning: {current_global_finetuning_status}")
                else:
                    print("    No clients currently finetuning")
            except Exception as e:
                print(f"    Warning: Could not load global finetuning status for aggregation: {e}")
        else:
            print(f"    No finetuning status file found for round {server.round}")

    total_effective_weight = 0.0 #For normalization if weights are adjusted
    models_aggregated_count = 0

    #Load each client model and aggregate parameters
    for client_dir, client_id in zip(client_dirs, client_ids):
        # Load client model
        model_path = os.path.join(client_dir, "model.pt")
        temp_model.load_state_dict(torch.load(model_path, map_location=server.device))

        client_parameters = temp_model.get_parameters()
        weight = aggregation_weights.get(client_id, 1.0 / len(client_dirs))

        # Apply incremental finetuning weight adjustment (global)
        finetune_multiplier = 1.0
        client_id_str = str(client_id)
        if lifting_mechanism == "deep_incr_finetune" and client_id_str in current_global_finetuning_status:
            progress = current_global_finetuning_status[client_id_str]
            if finetune_total_rounds > 0:
                finetune_multiplier = min(float(progress) / finetune_total_rounds, 1.0)
                print(f"    Client {client_id_str}: Finetuning round {progress}/{finetune_total_rounds} -> Weight multiplier: {finetune_multiplier:.3f} (original: {weight:.3f})")

        adjusted_weight = finetune_multiplier * weight

        total_effective_weight += adjusted_weight
        models_aggregated_count +=1

        # Add weighted parameters to global parameters
        for i, param in enumerate(client_parameters):
            global_parameters[i] += param * adjusted_weight

    #Normalize global parameters if weights were adjusted
    if models_aggregated_count > 0 and total_effective_weight > 0:
        if abs(total_effective_weight - 1.0) > 1e-6: #If not already normalized by weights
            print(f"  Normalizing global model by total effective weight: {total_effective_weight:.4f}")
            for i in range(len(global_parameters)):
                global_parameters[i] /= total_effective_weight

    # Update global model with aggregated parameters
    server.global_model.set_parameters(global_parameters)
    print(f"Updated global model with parameters from {len(client_dirs)} clients")

    return global_parameters

def aggregate_with_tracks(server, clients_dir, track_info, aggregation_weights):
    """Aggregate a separate model per track (primary + background clients), plus a baseline global."""
    client_dirs = glob.glob(os.path.join(clients_dir, "client_*"))
    client_ids = [int(os.path.basename(d).split("_")[1]) for d in client_dirs]

    print(f"\nAggregation for round {server.round}.")
    print(f"Found {len(client_dirs)} clients: {client_ids}")

    # Initialize temporary model (plug-and-play)
    temp_model = _make_temp_model(server)

    client_parameters_dict = {}
    background_parameters_dict = {}

    # Load all client models (both primary and background)
    for client_dir, client_id in zip(client_dirs, client_ids):
        #Load primary client model
        model_path = os.path.join(client_dir, "model.pt")
        if os.path.exists(model_path):
            temp_model.load_state_dict(torch.load(model_path, map_location=server.device))
            #Get client parameters
            client_parameters_dict[client_id] = [p.clone() for p in temp_model.get_parameters()]
            print(f"Loaded primary model from client {client_id}")
        else:
            print(f"Warning: Primary model file not found for client {client_id}")

        # Check for background models
        background_dirs = glob.glob(os.path.join(client_dir, "background_*"))
        for bg_dir in background_dirs:
            bg_track_name = os.path.basename(bg_dir).replace("background_", "", 1)
            bg_model_path = os.path.join(bg_dir, "model.pt")

            if os.path.exists(bg_model_path):
                # Load the background model
                temp_model.load_state_dict(torch.load(bg_model_path, map_location=server.device))

                # Initialize the dictionary for this track if needed
                if bg_track_name not in background_parameters_dict:
                    background_parameters_dict[bg_track_name] = {}

                #Store the background parameters
                background_parameters_dict[bg_track_name][client_id] = [p.clone() for p in temp_model.get_parameters()]
                print(f"Loaded background model from client {client_id} for track {bg_track_name}")

    #Dictionary to store track aggregations
    track_parameters = {}
    tracks = track_info.get("tracks", {})

    print(f"\nAggregating {len(tracks)} tracks:")

    # Count the number of custom tracks (excluding global)
    custom_tracks = [t for t in tracks.keys() if t != "global"]
    has_custom_tracks = len(custom_tracks) > 0

    # Get initial model parameters as a fallback
    try:
        initial_model_dir = os.path.join(
            server.results_dir,
            "model_storage/global_model_initial"
        )
        if os.path.exists(os.path.join(initial_model_dir, "model.pt")):
            temp_model.load_state_dict(torch.load(os.path.join(initial_model_dir, "model.pt"), map_location=server.device))
            print("Loaded initial model parameters as fallback")
    except Exception as e:
        print(f"Warning: Could not load initial model as fallback: {e}")

    # baseline global model (reference, never unlearned)
    baseline_global_params = [torch.zeros_like(param) for param in server.global_model.parameters()]
    total_global_weight = 0.0

    for client_id, client_params in client_parameters_dict.items():
        #Use standard weight for baseline global model
        weight = aggregation_weights.get(client_id, 1.0 / len(client_dirs))
        total_global_weight += weight

        #Add weighted parameters to baseline global parameters
        for i, param in enumerate(client_params):
            baseline_global_params[i] += param * weight

    # Normalize baseline global parameters
    if total_global_weight > 0:
        for i in range(len(baseline_global_params)):
            baseline_global_params[i] /= total_global_weight

    print(f"Created baseline global model (reference) with parameters from all {len(client_parameters_dict)} clients")

    # Store baseline global model even if it's not part of the tracks
    track_parameters["baseline_global"] = baseline_global_params

    # Aggregate each track
    for track_name, track_clients in tracks.items():
        #skip default track when disagreements exist (tracks stay separate)
        if track_name == "default" and has_custom_tracks:
            print("Skipping default track to ensure track separation in disagreement scenario")
            continue

        print(f"\nAggregating track: '{track_name}' with clients: {sorted(track_clients)}")

        #Skip tracks with no clients
        if not track_clients:
            print(f"Skipping empty track: {track_name}")
            continue

        # Initialize parameters for this track
        track_parameters[track_name] = [torch.zeros_like(param) for param in server.global_model.parameters()]

        # Sum of primary weights for normalization
        primary_weight = 0.0
        primary_clients_aggregated = []

        for client_id in track_clients:
            # Get client's primary track
            client_primary_track = track_info.get("client_tracks", {}).get(str(client_id))

            #Only include this client's model if this is its primary track
            if client_primary_track == track_name:
                if client_id not in client_parameters_dict:
                    print(f"    Warning: Model for client {client_id} (primary for this track) not found in client_parameters_dict. Skipping.")
                    continue

                client_params = client_parameters_dict[client_id]
                weight = aggregation_weights.get(client_id, 1.0 / len(track_clients)) #Default weight

                # Apply incremental finetuning weight adjustment
                finetune_multiplier = 1.0
                if server.disagreement_settings.get("lifting_mechanism") == "deep_incr_finetune":
                    finetune_status_path = os.path.join(
                        clients_dir, # round_X_clients_dir
                        "..", # up to round_X dir
                        "tracks",
                        track_name,
                        "finetuning_status.json"
                    )
                    finetune_status_path = os.path.normpath(finetune_status_path)  #Normalize the path

                    if os.path.exists(finetune_status_path):
                        try:
                            with open(finetune_status_path, 'r') as f_fs:
                                track_finetune_status = json.load(f_fs)
                            client_id_str = str(client_id)
                            if client_id_str in track_finetune_status:
                                progress = track_finetune_status[client_id_str]
                                total_rounds = server.disagreement_settings.get("deep_lifting_finetune_rounds", 3)
                                if total_rounds > 0:
                                    finetune_multiplier = min(float(progress) / total_rounds, 1.0) #Cap at 1.0
                                    print(f"    Client {client_id_str} in track '{track_name}': Finetuning round {progress}/{total_rounds} -> Weight multiplier: {finetune_multiplier:.3f} (original weight: {weight:.3f})")
                            else:
                                print(f"    Client {client_id_str} in track '{track_name}': Not in finetuning status -> Using normal weight: {weight:.3f}")
                        except Exception as e:
                            print(f"    Warning: Could not load finetuning status for track '{track_name}': {e}")
                    else:
                        print(f"    Finetuning status file not found at: {finetune_status_path}")

                adjusted_weight = finetune_multiplier * weight

                primary_weight += adjusted_weight
                primary_clients_aggregated.append(client_id)
                print(f"  Including primary model from client {client_id} with adjusted weight {adjusted_weight:.4f} (original: {weight:.4f})")

                # Add weighted parameters
                for i, param in enumerate(client_params):
                    track_parameters[track_name][i] += param * adjusted_weight

        print(f"  Primary models aggregated: {sorted(primary_clients_aggregated)} with total weight {primary_weight:.4f}")

        background_clients_aggregated = []
        background_weight = 0.0

        if track_name in background_parameters_dict:
            for client_id, bg_params in background_parameters_dict[track_name].items():
                # Skip client if it's already included as primary
                if track_info.get("client_tracks", {}).get(str(client_id)) == track_name:
                    continue

                # Use an equal weight for background participation
                weight = aggregation_weights.get(client_id, 1.0 / len(track_clients))

                #incremental finetuning adjustment for background client
                finetune_multiplier = 1.0
                if server.disagreement_settings.get("lifting_mechanism") == "deep_incr_finetune":
                    finetune_status_path = os.path.join(
                        clients_dir, #clients_dir is actually round_X_clients_dir
                        "..", # up to round_X dir
                        "tracks",
                        track_name,
                        "finetuning_status.json"
                    )
                    finetune_status_path = os.path.normpath(finetune_status_path)  # Normalize the path

                    if os.path.exists(finetune_status_path):
                        try:
                            with open(finetune_status_path, 'r') as f_fs:
                                track_finetune_status = json.load(f_fs)
                            client_id_str = str(client_id)
                            if client_id_str in track_finetune_status:
                                progress = track_finetune_status[client_id_str]
                                total_rounds = server.disagreement_settings.get("deep_lifting_finetune_rounds", 3)
                                if total_rounds > 0:
                                    finetune_multiplier = min(float(progress) / total_rounds, 1.0) # Cap at 1.0
                                    print(f"    Background client {client_id_str} in track '{track_name}': Finetuning round {progress}/{total_rounds} -> Weight multiplier: {finetune_multiplier:.3f} (original weight: {weight:.3f})")
                        except Exception as e:
                            print(f"    Warning: Could not load finetuning status for background client {client_id_str} in track '{track_name}': {e}")

                adjusted_weight = finetune_multiplier * weight

                background_weight += adjusted_weight
                background_clients_aggregated.append(client_id)

                print(f"  Including background model from client {client_id} with adjusted weight {adjusted_weight:.4f} (original: {weight:.4f})")

                #Add weighted parameters
                for i, param in enumerate(bg_params):
                    track_parameters[track_name][i] += param * adjusted_weight

            print(f"  Background models aggregated: {sorted(background_clients_aggregated)} with total weight {background_weight:.4f}")

        #Adjust normalization to account for background models
        total_weight = primary_weight + background_weight

        # Safeguard against empty or extremely imbalanced tracks
        if total_weight < 0.01:
            print(f"  Warning: Very low total weight ({total_weight:.4f}) for track '{track_name}'")

            # fall back to baseline global model (more reliable than initial)
            print(f"  Using baseline global model parameters as fallback for track '{track_name}'")
            track_parameters[track_name] = [p.clone() for p in baseline_global_params]
        elif total_weight > 0:
            for i in range(len(track_parameters[track_name])):
                track_parameters[track_name][i] /= total_weight
            print(f"  Track '{track_name}' normalized with combined weight {total_weight:.4f}")

        # Check for invalid values in the aggregated parameters
        has_invalid = False
        for param in track_parameters[track_name]:
            if torch.isnan(param).any() or torch.isinf(param).any():
                has_invalid = True
                break

        if has_invalid:
            print(f"  Warning: Invalid values detected in track '{track_name}' parameters")
            #fall back to baseline global model (more reliable than initial)
            print(f"  Using baseline global model parameters as fallback for track '{track_name}'")
            track_parameters[track_name] = [p.clone() for p in baseline_global_params]

        print(f"  Track '{track_name}' aggregation complete.")

    #Save track models to disk
    save_track_models(server, track_parameters, track_info)

    # update global_model with baseline params for eval
    # server.global_model is for evaluation, not a specific track
    server.global_model.set_parameters(track_parameters["baseline_global"])
    print("\nUpdated server's global_model object with parameters from baseline global model (reference)")
    print(f"Aggregation for round {server.round} complete.\n")

    # also return selected track params for track-specific logic
    selected_track = None

    #prefer the "global" track if present
    #"global" track is for clients without disagreements (may have unlearning)
    # Different from "baseline_global" (reference model with all clients)
    if "global" in track_parameters:
        selected_track = "global"
    elif len(track_parameters) > 1:  # If we have more than just the baseline_global
        # Use the first regular track (not baseline_global)
        for track_name in track_parameters:
            if track_name != "baseline_global":
                selected_track = track_name
                break

    if selected_track:
        return track_parameters[selected_track]
    else:
        #Fallback to baseline global parameters
        return track_parameters["baseline_global"]

def save_track_models(server, track_parameters, track_info):
    """Write the baseline global model and each track model (+ metadata) to disk."""
    track_saving_start_time = time.time()

    structure = get_structure_config(server)

    #Get clean model for applying parameters (plug-and-play)
    temp_model = _make_temp_model(server)

    # Save baseline global model (reference) to aggregated location
    global_model_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=server.round),
        structure["global_model_aggregated"]
    )
    os.makedirs(os.path.dirname(global_model_dir), exist_ok=True)

    if "baseline_global" in track_parameters:
        temp_model.set_parameters(track_parameters["baseline_global"])
        model_path = os.path.join(global_model_dir, "model.pt")
        torch.save(temp_model.state_dict(), model_path)
        print(f"Saved baseline global model (reference) to {global_model_dir}")

    # only create tracks dir when disagreements active
    if not track_info.get("tracks", {}) or (len(track_info.get("tracks", {})) == 1 and "global" in track_info.get("tracks", {})):
        print(f"No active disagreements for round {server.round}, skipping track creation")

        track_saving_time = time.time() - track_saving_start_time
        # Update the timing metrics in the aggregation history if available
        if hasattr(server, 'aggregation_timing_history') and server.aggregation_timing_history:
            server.aggregation_timing_history[-1]["track_saving_time_seconds"] = track_saving_time
        if "aggregation_track_saving_time_seconds" in server.training_history:
            server.training_history["aggregation_track_saving_time_seconds"][-1] = track_saving_time

        return

    #Create "tracks" directory for this round
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

    # Save each track model
    for track_name, parameters in track_parameters.items():
        # Skip baseline_global (already saved as global model aggregated)
        if track_name == "baseline_global":
            continue

        track_dir = os.path.join(tracks_dir, track_name)
        os.makedirs(track_dir, exist_ok=True)

        # Apply parameters to temp model
        temp_model.set_parameters(parameters)

        #Save model
        model_path = os.path.join(track_dir, "model.pt")
        torch.save(temp_model.state_dict(), model_path)

        #Save track-specific metadata
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

    # Update the timing metrics in the aggregation history if available
    if hasattr(server, 'aggregation_timing_history') and server.aggregation_timing_history:
        server.aggregation_timing_history[-1]["track_saving_time_seconds"] = track_saving_time

    # Update the training history timing as well
    if "aggregation_track_saving_time_seconds" in server.training_history:
        server.training_history["aggregation_track_saving_time_seconds"][-1] = track_saving_time

def get_structure_config(server):
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
    config_path = os.path.join(os.path.dirname(server.results_dir), "mock_etcd/configuration.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "results" in config and "structure" in config["results"]:
                    return config["results"]["structure"]
    except Exception as e:
        print(f"Error loading configuration: {e}")

    return default_structure
