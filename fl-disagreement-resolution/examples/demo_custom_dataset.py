#!/usr/bin/env python3
"""
Test script for custom dataset adapter.

This script demonstrates how to:
1. Register a custom dataset
2. Use it with the FL framework
3. Test data loading and partitioning
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
from fl_module.custom.utils import (
    register_custom_dataset,
    load_custom_test_data,
    create_custom_test_dataloader
)
from fl_module.registry import DatasetAdapterRegistry
# Full integration testing requires FL server/client setup


def create_sample_dataset(output_path: str = "data/custom_test/dataset.csv", n_samples: int = 1000):
    """Create a sample CSV dataset for testing.
    
    Args:
        output_path: Path to save the CSV file
        n_samples: Number of samples to generate
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    #Generate synthetic data
    np.random.seed(42)
    n_features = 10
    
    #Features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Binary classification labels
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create DataFrame
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["user_id"] = np.arange(n_samples)  # ID column for unlearning
    df["target"] = y  #Target column
    
    #Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Created sample dataset: {output_path}")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {len(np.unique(y))}")
    
    return df


def test_adapter_basic():
    """Test basic adapter functionality."""
    print("\n" + "="*70)
    print("TEST 1: Basic Adapter Functionality")
    print("="*70)
    
    # Create sample dataset
    dataset_path = "data/custom_test/dataset.csv"
    if not os.path.exists(dataset_path):
        create_sample_dataset(dataset_path, n_samples=1000)
    
    # Define schema
    schema = DatasetSchema(
        id_column="user_id",
        input_cols=[f"feature_{i}" for i in range(10)],
        target_col="target",
        timestamp_column=None
    )
    
    # Register custom dataset
    adapter = register_custom_dataset(
        dataset_path=dataset_path,
        schema=schema,
        experiment_type="custom_test",
        num_clients=6,
        iid=True,
        train_frac=0.8
    )
    
    #Test loading client data
    print("\nTesting client data loading...")
    for client_id in range(6):
        X, y = adapter.load_client_data(client_id, data_dir="", sample_size=1000)
        print(f"  Client {client_id}: {len(X)} samples, {X.shape[1]} features")
        print(f"    Label distribution: {np.bincount(y.astype(int))}")
    
    #Test test data loading
    print("\nTesting test data loading...")
    X_test, y_test = adapter.load_test_data()
    print(f"  Test set: {len(X_test)} samples")
    print(f"    Label distribution: {np.bincount(y_test.astype(int))}")
    
    print("\nOK: Basic adapter test passed!")


def test_collect_all_client_data():
    """Test collect_all_client_data with custom adapter."""
    print("\n" + "="*70)
    print("TEST 2: collect_all_client_data Integration")
    print("="*70)
    
    adapter = DatasetAdapterRegistry.get_adapter("custom_test")
    if adapter is None:
        print("Warning:  Adapter not found. Running basic test first...")
        test_adapter_basic()
        adapter = DatasetAdapterRegistry.get_adapter("custom_test")
    
    # Create mock clients (we don't need real FL clients for this test)
    # We'll simulate what collect_all_client_data does
    print("\nSimulating client data collection...")
    
    all_samples = []
    all_labels = []
    all_client_ids = []
    
    for client_id in range(6):
        samples, labels = adapter.load_client_data(client_id, data_dir="", sample_size=1000)
        seq_len = adapter.get_sequence_length()
        
        # Flatten if needed
        samples_flat = adapter.flatten_samples(samples)
        labels_flat = labels
        client_ids_flat = np.repeat([client_id], len(samples_flat))
        
        all_samples.append(samples_flat)
        all_labels.append(labels_flat)
        all_client_ids.append(client_ids_flat)
    
    #Concatenate
    X = np.concatenate(all_samples, axis=0)
    y = np.concatenate(all_labels, axis=0)
    client_ids = np.concatenate(all_client_ids, axis=0)
    
    #Convert to unlearning format
    unlearning_data = adapter.to_unlearning_format(X, y, client_ids)
    
    print(f"\nCollected data:")
    print(f"  Total samples: {len(unlearning_data['df'])}")
    print(f"  Features: {len(unlearning_data['input_cols'])}")
    print(f"  Clients: {len(np.unique(client_ids))}")
    print(f"  DataFrame shape: {unlearning_data['df'].shape}")
    print(f"  Columns: {list(unlearning_data['df'].columns[:5])}...")
    
    print("\nOK: collect_all_client_data integration test passed!")


def test_test_dataloader():
    """Test creating test DataLoader."""
    print("\n" + "="*70)
    print("TEST 3: Test DataLoader Creation")
    print("="*70)
    
    # Load test data
    X_test, y_test = load_custom_test_data("custom_test")
    
    # Create DataLoader
    test_loader = create_custom_test_dataloader(
        X_test,
        y_test,
        batch_size=32,
        is_classification=True
    )
    
    print(f"\nCreated test DataLoader:")
    print(f"  Batches: {len(test_loader)}")
    print(f"  Batch size: {test_loader.batch_size}")
    
    # Test iteration
    batch_X, batch_y = next(iter(test_loader))
    print(f"  Sample batch shape: X={batch_X.shape}, y={batch_y.shape}")
    print(f"  Sample batch dtype: X={batch_X.dtype}, y={batch_y.dtype}")
    
    print("\nOK: Test DataLoader creation test passed!")


def test_non_iid_partitioning():
    """Test non-IID partitioning."""
    print("\n" + "="*70)
    print("TEST 4: Non-IID Partitioning")
    print("="*70)
    
    #Create dataset
    dataset_path = "data/custom_test/dataset_non_iid.csv"
    if not os.path.exists(dataset_path):
        create_sample_dataset(dataset_path, n_samples=1000)
    
    #Define schema
    schema = DatasetSchema(
        id_column="user_id",
        input_cols=[f"feature_{i}" for i in range(10)],
        target_col="target",
        timestamp_column=None
    )
    
    # Register with non-IID partitioning
    adapter = register_custom_dataset(
        dataset_path=dataset_path,
        schema=schema,
        experiment_type="custom_test_non_iid",
        num_clients=6,
        iid=False,  # non-IID
        train_frac=0.8
    )
    
    print("\nNon-IID client distributions:")
    for client_id in range(6):
        X, y = adapter.load_client_data(client_id, data_dir="", sample_size=1000)
        label_dist = np.bincount(y.astype(int))
        print(f"  Client {client_id}: {len(X)} samples")
        print(f"    Label distribution: {label_dist}")
        print(f"    Dominant class: {np.argmax(label_dist)} ({np.max(label_dist)/len(y)*100:.1f}%)")
    
    print("\nOK: Non-IID partitioning test passed!")


def main():
    """Run all tests."""
    print("="*70)
    print("CUSTOM DATASET ADAPTER TEST SUITE")
    print("="*70)
    
    try:
        test_adapter_basic()
        test_collect_all_client_data()
        test_test_dataloader()
        test_non_iid_partitioning()
        
        print("\n" + "="*70)
        print("OK: ALL TESTS PASSED!")
        print("="*70)
        print("\nThe custom dataset adapter is working correctly.")
        print("You can now use experiment_type='custom_test' in your FL configuration.")
        
    except Exception as e:
        print("\n" + "="*70)
        print("FAIL: TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
