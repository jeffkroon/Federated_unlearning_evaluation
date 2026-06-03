"""Utility functions for federated learning client."""

import os
import json
import torch
import numpy as np
from datetime import datetime

def save_training_results(client, results, round_num=None):
    """Save a client's training results (plus model/config info) as JSON."""
    if client.results_dir:
        output_dir = os.path.join(client.results_dir, "output", "clients", f"client_{client.client_id}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        # legacy fallback location
        output_dir = "output/client_results"
        os.makedirs(output_dir, exist_ok=True)

    enhanced_results = results.copy()

    enhanced_results.update({
        "client_id": client.client_id,
        "experiment_type": client.experiment_type,
        "batch_size": client.batch_size,
        "learning_rate": client.learning_rate,
        "device": str(client.device),
        "timestamp": datetime.now().isoformat(),
        "round": round_num,
    })

    if hasattr(client, 'train_loader') and client.train_loader:
        enhanced_results["dataset_size"] = {
            "train": len(client.train_loader.dataset) if hasattr(client.train_loader, 'dataset') else None,
            "validation": len(client.valid_loader.dataset) if hasattr(client.valid_loader, 'dataset') else None
        }

    model_summary = {}
    total_params = 0
    trainable_params = 0

    for name, param in client.model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        layer_name = name.split('.')[0] if '.' in name else name
        if layer_name not in model_summary:
            model_summary[layer_name] = 0
        model_summary[layer_name] += param_count

    enhanced_results["model_info"] = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layer_summary": model_summary
    }

    # extra per-task metrics
    if client.experiment_type == "mnist":
        if "final_train_accuracy" in results and "final_valid_accuracy" in results:
            enhanced_results["improvement"] = {
                "accuracy_gain": results["final_valid_accuracy"] - results["valid_accuracies"][0]
                                if len(results["valid_accuracies"]) > 0 else 0,
                "loss_reduction": results["valid_losses"][0] - results["final_valid_loss"]
                               if len(results["valid_losses"]) > 0 else 0
            }

    elif client.experiment_type == "n_cmapss":
        if len(results["valid_losses"]) > 0:
            enhanced_results["improvement"] = {
                "loss_reduction": results["valid_losses"][0] - results["final_valid_loss"]
                                if len(results["valid_losses"]) > 0 else 0,
                "loss_reduction_percent": ((results["valid_losses"][0] - results["final_valid_loss"]) /
                                         results["valid_losses"][0]) * 100
                                         if len(results["valid_losses"]) > 0 and results["valid_losses"][0] != 0 else 0
            }

    if round_num is not None:
        results_path = os.path.join(output_dir, f"training_results_round_{round_num}.json")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(output_dir, f"training_results_{timestamp}.json")

    enhanced_results_serializable = make_json_serializable(enhanced_results)

    with open(results_path, "w") as f:
        json.dump(enhanced_results_serializable, f, indent=2)
    print(f"Client {client.client_id} saved training results to {results_path}")

def get_structure_config(results_dir):
    """Return the results directory layout from config, or sensible defaults."""
    default_structure = {
        "round_template": "round_{round}",
        "clients_dir": "clients",
        "global_model": "global_model_for_training",
        "client_prefix": "client_"
    }

    config_path = os.path.join(os.path.dirname(results_dir), "mock_etcd/configuration.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "results" in config and "structure" in config["results"]:
                    return config["results"]["structure"]
    except Exception as e:
        print(f"Error loading configuration: {e}")

    return default_structure

def make_json_serializable(obj):
    """Recursively convert numpy/torch types to plain Python so json can dump them."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif hasattr(obj, 'dtype') and np.isscalar(obj):  # any other numpy scalar
        return obj.item()
    else:
        return obj
