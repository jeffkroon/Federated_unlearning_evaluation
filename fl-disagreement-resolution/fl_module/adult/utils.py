"""Data utilities for the Adult Income (UCI) dataset.

Downloads via sklearn fetch_openml, preprocesses (one-hot categoricals,
standard-scales numerics), and distributes to federated clients.
"""

import os
import numpy as np
import pandas as pd
from typing import List

from fl_module.partitioning import dirichlet_partition


NUMERICAL_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]


def _load_and_preprocess() -> tuple:
    """Download Adult and return (X float32, y 0/1) after scaling + one-hot."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df: pd.DataFrame = adult.data.copy()
    target = adult.target.copy()

    # Drop rows with missing values
    df = df.replace("?", np.nan)
    mask = df.notna().all(axis=1)
    df = df[mask]
    target = target[mask]

    # Encode target: '>50K' -> 1, '<=50K' -> 0
    y = (target.str.strip().str.replace(".", "", regex=False) == ">50K").astype(np.int64).values

    #Numerical: standard scale
    num_data = df[NUMERICAL_COLS].astype(np.float32)
    scaler = StandardScaler()
    num_data = scaler.fit_transform(num_data)

    #Categorical: one-hot encode
    cat_data = pd.get_dummies(df[CATEGORICAL_COLS]).astype(np.float32).values

    X = np.concatenate([num_data, cat_data], axis=1).astype(np.float32)
    return X, y


def setup_federated_data(
    num_clients: int = 5,
    samples_per_client: int = 1000,
    data_dir: str = "data/adult",
    iid: bool = True,
    force: bool = False,
    alpha: float = 0.5,
) -> None:
    """Download, preprocess, and distribute Adult data to federated clients."""

    train_ok = all(
        os.path.exists(os.path.join(data_dir, "train", f"client_{i}", "adult_data.npz"))
        for i in range(num_clients)
    )
    test_ok = os.path.exists(os.path.join(data_dir, "test", "adult_test.npz"))

    if train_ok and test_ok and not force:
        print("Adult data already distributed. Skipping setup.")
        return

    print("Downloading and preprocessing Adult Income dataset...")
    X, y = _load_and_preprocess()

    # Train / test split (80 / 20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test set
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
    np.savez(
        os.path.join(data_dir, "test", "adult_test.npz"),
        features=X_test.astype(np.float32),
        labels=y_test.astype(np.float32),
    )
    print(f"Test set: {len(y_test)} samples saved.")

    # Distribute training data to clients
    if iid:
        indices = np.random.permutation(len(X_train))
        client_index_lists = [
            indices[cid * samples_per_client : (cid + 1) * samples_per_client]
            for cid in range(num_clients)
        ]
    else:
        #Non-IID: Dirichlet(alpha) label skew over ALL samples (no data wasted,
        #no single-class clients), same data volume as the IID split.
        client_index_lists = dirichlet_partition(y_train, num_clients, alpha=alpha)

    for client_id in range(num_clients):
        client_indices = client_index_lists[client_id]

        X_c = X_train[client_indices]
        y_c = y_train[client_indices]

        client_dir = os.path.join(data_dir, "train", f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        np.savez(
            os.path.join(client_dir, "adult_data.npz"),
            features=X_c.astype(np.float32),
            labels=y_c.astype(np.float32),
        )
        unique, counts = np.unique(y_c, return_counts=True)
        print(f"Client {client_id}: {len(y_c)} samples, class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    print(f"Adult data distributed to {num_clients} clients.")


def load_client_data(client_id: int, train_dir: str = "data/adult/train", sample_size: int = None):
    """Load Adult data for one client."""
    path = os.path.join(train_dir, f"client_{client_id}", "adult_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adult client data not found: {path}")

    data = np.load(path)
    X, y = data["features"], data["labels"]

    if sample_size and len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X, y = X[idx], y[idx]

    print(f"Client {client_id} loaded {len(X)} Adult samples")
    return X, y


def load_test_data(test_dir: str = "data/adult/test"):
    """Load Adult test data."""
    path = os.path.join(test_dir, "adult_test.npz")
    data = np.load(path)
    return data["features"], data["labels"]
