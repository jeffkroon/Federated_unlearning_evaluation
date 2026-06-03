"""Utility functions for N-CMAPSS dataset."""

import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from fl_module.n_cmapss.dataset import NCMAPSSDataset

# which engine unit belongs to which client
UNIT_TO_CLIENT = {
    2: 0,
    5: 1,
    10: 2,
    16: 3,
    18: 4,
    20: 5
}

def load_client_data(client_id, train_dir, sample_size=1000):
    """Load N-CMAPSS samples/labels for one client (capped at sample_size)."""
    unit = None
    for u, c in UNIT_TO_CLIENT.items():
        if c == client_id:
            unit = u
            break

    if unit is None:
        raise ValueError(f"No unit found for client {client_id}")

    npz_file = f"Unit{unit}_win50_str1_smp10.npz"
    data_path = os.path.join(train_dir, f"client_{client_id}", npz_file)
    print(f"Loading data from {data_path}")

    data = np.load(data_path)

    #stored as (window, n_features, n_samples) -> want (n_samples, window, n_features)
    samples = data['sample'].transpose(2, 0, 1)
    labels = data['label']

    if len(samples) > sample_size:
        print(f"Sampling {sample_size} instances from {len(samples)} for client {client_id}")
        indices = np.random.choice(len(samples), sample_size, replace=False)
        samples = samples[indices]
        labels = labels[indices]
    else:
        print(f"Using all {len(samples)} instances for client {client_id}")

    return samples, labels

def load_test_data(test_dir, test_units, sample_size=500):
    """Load and stack N-CMAPSS test data for the given units."""
    test_samples = []
    test_labels = []

    for unit in test_units:
        npz_file = f"Unit{unit}_win50_str1_smp10.npz"
        data_path = os.path.join(test_dir, npz_file)
        print(f"Loading test data from {data_path}")

        data = np.load(data_path)

        unit_samples = data['sample'].transpose(2, 0, 1)
        unit_labels = data['label']

        if len(unit_samples) > sample_size:
            print(f"Sampling {sample_size} instances from {len(unit_samples)} for test unit {unit}")
            indices = np.random.choice(len(unit_samples), sample_size, replace=False)
            unit_samples = unit_samples[indices]
            unit_labels = unit_labels[indices]
        else:
            print(f"Using all {len(unit_samples)} instances for test unit {unit}")

        test_samples.append(unit_samples)
        test_labels.append(unit_labels)

    test_samples = np.vstack(test_samples)
    test_labels = np.concatenate(test_labels)

    return test_samples, test_labels

def preprocess_data(train_samples, test_samples=None):
    """Standardize samples per feature; also transforms test data if given."""
    n_train_samples, seq_len, n_features = train_samples.shape

    #flatten over the time axis to fit one scaler per feature
    train_flat = train_samples.reshape(-1, n_features)

    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_flat)
    train_normalized = train_flat.reshape(n_train_samples, seq_len, n_features)

    if test_samples is not None:
        n_test_samples = test_samples.shape[0]
        test_flat = test_samples.reshape(-1, n_features)
        test_flat = scaler.transform(test_flat)
        test_normalized = test_flat.reshape(n_test_samples, seq_len, n_features)
        return train_normalized, test_normalized, scaler

    return train_normalized, scaler

def create_client_dataloaders(train_samples, train_labels, batch_size=64, valid_split=0.2):
    """Split client data into train/val dataloaders."""
    split_idx = int(len(train_samples) * (1 - valid_split))

    train_data = train_samples[:split_idx]
    train_labels_split = train_labels[:split_idx]
    valid_data = train_samples[split_idx:]
    valid_labels = train_labels[split_idx:]

    train_dataset = NCMAPSSDataset(train_data, train_labels_split)
    valid_dataset = NCMAPSSDataset(valid_data, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader

def create_test_dataloader(test_samples, test_labels, batch_size=64):
    test_dataset = NCMAPSSDataset(test_samples, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader
