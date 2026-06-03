"""Utility functions for CIFAR-10 dataset."""

import os
import random
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fl_module.cifar10.dataset import CIFAR10Dataset


def download_cifar10_dataset(data_dir: str = "data/cifar10"):
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    print(f"CIFAR-10 downloaded to {data_dir}")
    return train_dataset, test_dataset


def _split_data_non_iid(train_dataset, num_clients, samples_per_client):
    """Simple non-IID split: over-sample two classes per client."""
    class_data = {cls: [] for cls in range(10)}
    for img, label in train_dataset:
        class_data[label].append((img, label))

    client_data = [[] for _ in range(num_clients)]
    primary_pairs = [(2 * i) % 10 for i in range(num_clients)]

    for client_id in range(num_clients):
        primary = {primary_pairs[client_id], (primary_pairs[client_id] + 1) % 10}
        secondary = set(range(10)) - primary

        # 70% primary, 30% secondary
        primary_samples = int(samples_per_client * 0.7)
        per_class_primary = max(1, primary_samples // len(primary))
        for cls in primary:
            take = min(per_class_primary, len(class_data[cls]))
            client_data[client_id].extend(class_data[cls][:take])
            class_data[cls] = class_data[cls][take:]

        remaining = samples_per_client - len(client_data[client_id])
        if remaining > 0:
            sec_list = list(secondary)
            random.shuffle(sec_list)
            per_cls = max(1, remaining // len(sec_list))
            for cls in sec_list:
                if remaining <= 0:
                    break
                take = min(per_cls, len(class_data[cls]))
                client_data[client_id].extend(class_data[cls][:take])
                class_data[cls] = class_data[cls][take:]
                remaining -= take
    return client_data


def distribute_cifar10_to_clients(train_dataset, num_clients=6, samples_per_client=5000, iid=False, data_dir="data/cifar10"):
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)

    if iid:
        all_data = list(train_dataset)
        random.shuffle(all_data)
        client_data = [[] for _ in range(num_clients)]
        for i, item in enumerate(all_data[: num_clients * samples_per_client]):
            client_data[i % num_clients].append(item)
    else:
        client_data = _split_data_non_iid(train_dataset, num_clients, samples_per_client)

    for client_id, samples in enumerate(client_data):
        imgs = []
        labels = []
        for img, label in samples:
            imgs.append(img.numpy())
            labels.append(label)
        client_dir = os.path.join(data_dir, "train", f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        np.savez(
            os.path.join(client_dir, "cifar10_data.npz"),
            images=np.array(imgs),
            labels=np.array(labels),
        )
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {client_id}: {len(labels)} samples, class dist={dict(zip(unique, counts))}")


def prepare_cifar10_test_data(test_dataset, data_dir="data/cifar10"):
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
    imgs, labels = [], []
    for img, label in test_dataset:
        imgs.append(img.numpy())
        labels.append(label)
    np.savez(
        os.path.join(data_dir, "test", "cifar10_test.npz"),
        images=np.array(imgs),
        labels=np.array(labels),
    )
    print(f"CIFAR-10 test saved: {len(labels)} samples")


def setup_federated_data(num_clients=6, samples_per_client=5000, iid=False, data_dir="data/cifar10"):
    train_exists = all(os.path.exists(os.path.join(data_dir, "train", f"client_{i}", "cifar10_data.npz"))
                      for i in range(num_clients))
    test_exists = os.path.exists(os.path.join(data_dir, "test", "cifar10_test.npz"))
    if train_exists and test_exists:
        print(f"CIFAR-10 data already exists for {num_clients} clients.")
        return

    train_ds, test_ds = download_cifar10_dataset(data_dir)
    distribute_cifar10_to_clients(train_ds, num_clients=num_clients, samples_per_client=samples_per_client, iid=iid, data_dir=data_dir)
    prepare_cifar10_test_data(test_ds, data_dir=data_dir)
    print("CIFAR-10 federated data setup complete.")


def load_client_data(client_id, train_dir="data/cifar10/train", sample_size=None) -> Tuple[np.ndarray, np.ndarray]:
    data_path = os.path.join(train_dir, f"client_{client_id}", "cifar10_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CIFAR-10 data not found at {data_path}")

    data = np.load(data_path)
    images, labels = data["images"], data["labels"]

    if sample_size and len(images) > sample_size:
        idx = np.random.choice(len(images), sample_size, replace=False)
        images, labels = images[idx], labels[idx]

    print(f"Loaded CIFAR-10 client {client_id}: {len(images)} samples")
    return images, labels


def load_test_data(test_dir="data/cifar10/test"):
    data_path = os.path.join(test_dir, "cifar10_test.npz")
    data = np.load(data_path)
    print(f"Loaded CIFAR-10 test: {len(data['images'])} samples")
    return data["images"], data["labels"]


def create_client_dataloaders(images, labels, batch_size=128, valid_split=0.2):
    split_idx = int(len(images) * (1 - valid_split))
    train_imgs, val_imgs = images[:split_idx], images[split_idx:]
    train_lbls, val_lbls = labels[:split_idx], labels[split_idx:]
    train_ds = CIFAR10Dataset(train_imgs, train_lbls)
    val_ds = CIFAR10Dataset(val_imgs, val_lbls)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def create_test_dataloader(images, labels, batch_size=128):
    test_ds = CIFAR10Dataset(images, labels)
    return DataLoader(test_ds, batch_size=batch_size)
