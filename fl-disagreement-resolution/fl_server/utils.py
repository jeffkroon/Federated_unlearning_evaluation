"""Utility functions for federated learning server."""

import os
import json
import numpy as np

def make_json_serializable(obj):
    """Recursively convert numpy types to plain Python so json can dump them."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif hasattr(obj, 'dtype') and np.isscalar(obj):  #Catch any other NumPy scalar types
        return obj.item()
    else:
        return obj

def read_client_results_from_files(results_dir, client_ids, round_num):
    """Read each client's latest training-results JSON for a round."""
    if not results_dir or not client_ids:
        return {}

    client_results = {}

    for client_id in client_ids:
        try:
            client_output_dir = os.path.join(
                results_dir,
                "output",
                "clients",
                f"client_{client_id}"
            )

            if os.path.exists(client_output_dir):
                result_files = [f for f in os.listdir(client_output_dir) if f.startswith("training_results_")]

                if result_files:
                    result_files.sort(reverse=True)  #newest first
                    latest_result_file = os.path.join(client_output_dir, result_files[0])

                    with open(latest_result_file, 'r') as f:
                        client_results[client_id] = json.load(f)
                else:
                    print(f"No training results found for client {client_id}")
            else:
                print(f"Output directory for client {client_id} not found")
        except Exception as e:
            print(f"Error reading results for client {client_id}: {e}")

    return client_results
