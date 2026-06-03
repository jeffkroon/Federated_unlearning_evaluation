"""Utility functions for tabular dataset."""

import os
import random
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from fl_module.tabular.dataset import TabularDataset

def generate_tabular_data(
    n_samples=10000,
    n_features=20,
    n_classes=2,
    task='classification',
    random_state=42
):
    """Generate synthetic tabular data with sklearn (classification or regression)."""
    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_classes=n_classes,
            random_state=random_state
        )
    else:  # regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            noise=10.0,
            random_state=random_state
        )

    return X.astype(np.float32), y.astype(np.float32)

def distribute_tabular_to_clients(
    X, y, num_clients=6, samples_per_client=1000, iid=False, data_dir='data/tabular'
):
    """Split tabular data across clients and save per client (IID shuffle or skewed non-IID)."""
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)

    client_data = [[] for _ in range(num_clients)]
    data = list(zip(X, y))

    if iid:
        random.shuffle(data)
        for i, (features, label) in enumerate(data):
            if i < num_clients * samples_per_client:
                client_data[i % num_clients].append((features, label))
    else:
        # non-IID: regression/many-class gets value ranges, few-class gets skewed classes
        if len(np.unique(y)) > 10:  #regression or many classes, so split by sorted value
            sorted_indices = np.argsort(y)
            sorted_data = [data[i] for i in sorted_indices]

            for client_id in range(num_clients):
                start_idx = client_id * len(sorted_data) // num_clients
                end_idx = min((client_id + 1) * len(sorted_data) // num_clients, len(sorted_data))
                client_data[client_id] = sorted_data[start_idx:end_idx][:samples_per_client]
        else:
            class_data = {}
            for features, label in data:
                label_int = int(label)
                if label_int not in class_data:
                    class_data[label_int] = []
                class_data[label_int].append((features, label))

            classes = list(class_data.keys())
            for client_id in range(num_clients):
                primary_classes = [classes[client_id % len(classes)]]
                secondary_classes = [c for c in classes if c not in primary_classes]

                #70% from primary class, 30% from the rest
                primary_samples = int(samples_per_client * 0.7)
                for cls in primary_classes:
                    if len(class_data[cls]) > 0:
                        samples_from_class = min(primary_samples, len(class_data[cls]))
                        client_data[client_id].extend(class_data[cls][:samples_from_class])
                        class_data[cls] = class_data[cls][samples_from_class:]

                remaining = samples_per_client - len(client_data[client_id])
                for cls in secondary_classes:
                    if remaining <= 0:
                        break
                    if len(class_data[cls]) > 0:
                        samples_from_class = min(remaining, len(class_data[cls]))
                        client_data[client_id].extend(class_data[cls][:samples_from_class])
                        class_data[cls] = class_data[cls][samples_from_class:]
                        remaining -= samples_from_class

    for client_id in range(num_clients):
        client_dir = os.path.join(data_dir, 'train', f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)

        if len(client_data[client_id]) == 0:
            print(f"Warning: Client {client_id} has no data!")
            continue

        features_list = [item[0] for item in client_data[client_id]]
        labels_list = [item[1] for item in client_data[client_id]]

        data_path = os.path.join(client_dir, 'tabular_data.npz')
        np.savez(
            data_path,
            features=np.array(features_list),
            labels=np.array(labels_list)
        )

        labels_array = np.array(labels_list)
        if len(np.unique(labels_array)) <= 10:  # classification
            unique, counts = np.unique(labels_array, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"Client {client_id} has {len(labels_list)} samples. Class distribution: {class_dist}")
        else:  # regression
            print(f"Client {client_id} has {len(labels_list)} samples. Label range: [{labels_array.min():.2f}, {labels_array.max():.2f}]")

    print(f"Tabular data distributed to {num_clients} clients")

def prepare_tabular_test_data(X_test, y_test, data_dir='data/tabular'):
    """Save the tabular test set as a single .npz."""
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

    test_data_path = os.path.join(data_dir, 'test', 'tabular_test.npz')
    np.savez(
        test_data_path,
        features=X_test.astype(np.float32),
        labels=y_test.astype(np.float32)
    )

    print(f"Tabular test data prepared with {len(y_test)} samples")

def setup_federated_data(
    num_clients=6,
    samples_per_client=1000,
    n_features=20,
    n_classes=2,
    task='classification',
    iid=False,
    data_dir='data/tabular'
):
    """Generate synthetic tabular data and distribute it to clients (skips if present)."""
    train_client_data_exists = all(
        os.path.exists(os.path.join(data_dir, 'train', f'client_{i}', 'tabular_data.npz'))
        for i in range(num_clients)
    )
    test_data_exists = os.path.exists(os.path.join(data_dir, 'test', 'tabular_test.npz'))

    if train_client_data_exists and test_data_exists:
        print(f"Tabular data already exists for {num_clients} clients.")
        return

    print(f"Generating synthetic tabular data ({task})...")
    X, y = generate_tabular_data(
        n_samples=num_clients * samples_per_client * 2,  # extra for the test split
        n_features=n_features,
        n_classes=n_classes,
        task=task
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    distribute_tabular_to_clients(
        X_train,
        y_train,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        iid=iid,
        data_dir=data_dir
    )

    prepare_tabular_test_data(X_test, y_test, data_dir)

    print(f"Tabular data setup completed for federated learning with {num_clients} clients")

def load_client_data(client_id, train_dir='data/tabular/train', sample_size=None):
    """Load tabular data for one client (optionally subsampled)."""
    data_path = os.path.join(train_dir, f'client_{client_id}', 'tabular_data.npz')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Tabular data not found at {data_path}")

    print(f"Loading tabular data from {data_path}")

    data = np.load(data_path)
    features = data['features']
    labels = data['labels']

    if sample_size and len(features) > sample_size:
        print(f"Sampling {sample_size} instances from {len(features)} for client {client_id}")
        indices = np.random.choice(len(features), sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]

    print(f"Client {client_id} loaded {len(features)} tabular samples")
    return features, labels

def load_test_data(test_dir='data/tabular/test'):
    """Load the tabular test set."""
    data_path = os.path.join(test_dir, 'tabular_test.npz')
    print(f"Loading tabular test data from {data_path}")

    data = np.load(data_path)
    features = data['features']
    labels = data['labels']

    print(f"Loaded {len(features)} tabular test samples")
    return features, labels

def create_client_dataloaders(features, labels, batch_size=64, valid_split=0.2, is_classification=True):
    """Split a client's data into train/val dataloaders."""
    split_idx = int(len(features) * (1 - valid_split))

    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    valid_features = features[split_idx:]
    valid_labels = labels[split_idx:]

    train_dataset = TabularDataset(train_features, train_labels, is_classification=is_classification)
    valid_dataset = TabularDataset(valid_features, valid_labels, is_classification=is_classification)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader

def create_test_dataloader(features, labels, batch_size=64, is_classification=True):
    test_dataset = TabularDataset(features, labels, is_classification=is_classification)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader
