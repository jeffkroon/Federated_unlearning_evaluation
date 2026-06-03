"""Utility functions for MNIST dataset."""

import os
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl_module.mnist.dataset import MNISTDataset
from fl_module.partitioning import dirichlet_partition

def download_mnist_dataset(data_dir='data/mnist'):
    """Download MNIST via torchvision and return (train, test) datasets."""
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    print(f"MNIST dataset downloaded to {data_dir}")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset

def distribute_mnist_to_clients(train_dataset, num_clients=6, samples_per_client=1000, iid=False, data_dir='data/mnist', alpha=0.5):
    """Split MNIST across clients and save per client.

    IID: random shuffle, equal slices. Non-IID: Dirichlet(alpha) label skew over the
    full dataset (every sample used, matching the IID data volume).
    """
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)

    all_data = []
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        all_data.append((img, label))

    client_data = [[] for _ in range(num_clients)]

    if iid:
        random.shuffle(all_data)
        for i, item in enumerate(all_data):
            if i < num_clients * samples_per_client:
                client_data[i % num_clients].append(item)
    else:
        # non-IID: Dirichlet label skew across ALL samples (no data wasted)
        labels = [int(label) for _, label in all_data]
        parts = dirichlet_partition(labels, num_clients, alpha=alpha)
        for client_id, idx in enumerate(parts):
            client_data[client_id] = [all_data[i] for i in idx]

    for client_id in range(num_clients):
        client_dir = os.path.join(data_dir, 'train', f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)

        images = []
        labels = []
        for img, label in client_data[client_id]:
            images.append(img.numpy())
            labels.append(label)

        data_path = os.path.join(client_dir, 'mnist_data.npz')
        np.savez(
            data_path,
            images=np.array(images),
            labels=np.array(labels)
        )

        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Client {client_id} has {len(labels)} samples. Class distribution: {class_dist}")

    print(f"MNIST data distributed to {num_clients} clients")

def prepare_mnist_test_data(test_dataset, data_dir='data/mnist'):
    """Save the MNIST test set as a single .npz."""
    os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)

    images = []
    labels = []
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        images.append(img.numpy())
        labels.append(label)

    test_data_path = os.path.join(data_dir, 'test', 'mnist_test.npz')
    np.savez(
        test_data_path,
        images=np.array(images),
        labels=np.array(labels)
    )

    print(f"MNIST test data prepared with {len(labels)} samples")

def setup_federated_data(num_clients=6, samples_per_client=1000, iid=False, data_dir='data/mnist'):
    """Download MNIST and distribute it to clients (skips if already present)."""
    train_client_data_exists = all(os.path.exists(os.path.join(data_dir, 'train', f'client_{i}', 'mnist_data.npz'))
                                 for i in range(num_clients))
    test_data_exists = os.path.exists(os.path.join(data_dir, 'test', 'mnist_test.npz'))

    if train_client_data_exists and test_data_exists:
        print(f"MNIST data already exists for {num_clients} clients.")
        client_samples = []
        for i in range(num_clients):
            data_path = os.path.join(data_dir, 'train', f'client_{i}', 'mnist_data.npz')
            data = np.load(data_path)
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"Client {i} has {len(labels)} samples. Class distribution: {class_dist}")
            client_samples.append(len(labels))

        print(f"Test data exists at {os.path.join(data_dir, 'test', 'mnist_test.npz')}")
        print(f"Using existing MNIST data with distribution type: {'IID' if iid else 'Non-IID'}")
        print(f"Clients have an average of {sum(client_samples) / len(client_samples):.0f} samples each")
        return

    train_dataset, test_dataset = download_mnist_dataset(data_dir)

    distribute_mnist_to_clients(
        train_dataset,
        num_clients=num_clients,
        samples_per_client=samples_per_client,
        iid=iid,
        data_dir=data_dir
    )

    prepare_mnist_test_data(test_dataset, data_dir)

    print(f"MNIST data setup completed for federated learning with {num_clients} clients")

def load_client_data(client_id, train_dir='data/mnist/train', sample_size=None):
    """Load MNIST images/labels for one client (optionally subsampled)."""
    data_path = os.path.join(train_dir, f'client_{client_id}', 'mnist_data.npz')
    print(f"Loading MNIST data from {data_path}")

    data = np.load(data_path)
    images = data['images']
    labels = data['labels']

    if sample_size and len(images) > sample_size:
        print(f"Sampling {sample_size} instances from {len(images)} for client {client_id}")
        indices = np.random.choice(len(images), sample_size, replace=False)
        images = images[indices]
        labels = labels[indices]

    print(f"Client {client_id} loaded {len(images)} MNIST samples")
    return images, labels

def load_test_data(test_dir='data/mnist/test'):
    """Load the MNIST test set."""
    data_path = os.path.join(test_dir, 'mnist_test.npz')
    print(f"Loading MNIST test data from {data_path}")

    data = np.load(data_path)
    images = data['images']
    labels = data['labels']

    print(f"Loaded {len(images)} MNIST test samples")
    return images, labels

def create_client_dataloaders(images, labels, batch_size=64, valid_split=0.2):
    """Split a client's data into train/val dataloaders."""
    split_idx = int(len(images) * (1 - valid_split))

    train_images = images[:split_idx]
    train_labels = labels[:split_idx]
    valid_images = images[split_idx:]
    valid_labels = labels[split_idx:]

    train_dataset = MNISTDataset(train_images, train_labels)
    valid_dataset = MNISTDataset(valid_images, valid_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader

def create_test_dataloader(images, labels, batch_size=64):
    test_dataset = MNISTDataset(images, labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader
