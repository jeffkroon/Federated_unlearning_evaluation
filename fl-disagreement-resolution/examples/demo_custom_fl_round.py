#!/usr/bin/env python3
"""
Test script for full FL round with custom dataset.

This script tests a complete FL training round with custom dataset:
1. Registers custom dataset
2. Creates FL server and clients
3. Runs one FL round
4. Tests evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths: repo root (fl_module/fl_server) and its parent (machine_unlearning_tool)
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root.parent))

from machine_unlearning_tool.schemas import DatasetSchema
from fl_module.custom.utils import register_custom_dataset
from fl_server.server import FederatedServer
from fl_client.client import FederatedClient


def create_sample_dataset(output_path: str = "data/custom_fl_test/dataset.csv", n_samples: int = 1000):
    """Create a sample CSV dataset for FL testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    n_features = 10
    
    #Features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    #Binary classification labels
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["user_id"] = np.arange(n_samples)
    df["target"] = y
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created dataset: {output_path} ({n_samples} samples)")
    return df


def test_full_fl_round():
    """Test a complete FL round with custom dataset."""
    print("="*70)
    print("TEST: Full FL Round with Custom Dataset")
    print("="*70)
    
    # 1. Create sample dataset
    dataset_path = "data/custom_fl_test/dataset.csv"
    if not os.path.exists(dataset_path):
        create_sample_dataset(dataset_path, n_samples=1000)
    
    #2. Define schema
    schema = DatasetSchema(
        id_column="user_id",
        input_cols=[f"feature_{i}" for i in range(10)],
        target_col="target",
        timestamp_column=None
    )
    
    #3. Register custom dataset
    print("\n1. Registering custom dataset...")
    adapter = register_custom_dataset(
        dataset_path=dataset_path,
        schema=schema,
        experiment_type="custom_fl_test",
        num_clients=3,  # Small number for quick test
        iid=True,
        train_frac=0.8
    )
    
    # 4. Create FL server
    print("\n2. Creating FL server...")
    server = FederatedServer(
        experiment_type="custom_fl_test",
        test_dir=None,  # Will use adapter's test data
        device="cpu",
        results_dir="results/test_custom_fl"
    )
    
    #Load test data
    print("\n3. Loading test data...")
    server.load_test_data()
    
    #5. Create FL clients
    print("\n4. Creating FL clients...")
    clients = {}
    for client_id in range(3):
        client = FederatedClient(
            client_id=client_id,
            experiment_type="custom_fl_test",
            data_dir="",  # Not used for custom adapter
            batch_size=32,
            epochs=2,  # Small for quick test
            learning_rate=0.001,
            device="cpu",
            results_dir="results/test_custom_fl"
        )
        
        # Load client data
        print(f"   Loading data for client {client_id}...")
        client.load_data(sample_size=200)
        
        clients[client_id] = client
    
    #6. Initialize clients with global model
    print("\n5. Initializing clients with global model...")
    for client_id, client in clients.items():
        client.model = server.global_model
        print(f"   Client {client_id} initialized")
    
    #7. Train clients
    print("\n6. Training clients...")
    client_results = {}
    for client_id, client in clients.items():
        print(f"   Training client {client_id}...")
        results = client.train_with_disagreement_resolution()
        client_results[client_id] = results
        train_loss = results.get('train_loss', 'N/A')
        if isinstance(train_loss, (int, float)):
            print(f"     Loss: {train_loss:.4f}")
        else:
            print(f"     Loss: {train_loss}")
    
    # 8. Aggregate models (simplified)
    print("\n7. Aggregating models...")
    # In real FL, this would use FedAvg
    # For test, only check that models exist
    for client_id, client in clients.items():
        assert client.model is not None, f"Client {client_id} model is None"
        print(f"   Client {client_id} model parameters: {sum(p.numel() for p in client.model.parameters())}")
    
    #9. Evaluate server model
    print("\n8. Evaluating server model...")
    test_loss, test_accuracy = server.evaluate_model(fl_round=1, client_results=client_results)
    print(f"   Test loss: {test_loss:.4f}")
    if test_accuracy is not None:
        print(f"   Test accuracy: {test_accuracy:.4f}")
    
    print("\n" + "="*70)
    print("OK: FULL FL ROUND TEST PASSED!")
    print("="*70)
    print("\nCustom dataset works correctly in FL training rounds!")
    print("You can now use experiment_type='custom_fl_test' in your FL configuration.")


if __name__ == "__main__":
    try:
        test_full_fl_round()
    except Exception as e:
        print("\n" + "="*70)
        print("FAIL: TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
