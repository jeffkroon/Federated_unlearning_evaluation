from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model_utils import is_pytorch_model, is_sklearn_model


def create_loader(dataset: Dataset, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_model(
    net: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    is_classification: bool = False,
) -> Tuple[nn.Module, dict]:
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        net.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = net(xb)
            if is_classification:
                if yb.dtype != torch.long:  # CrossEntropy needs long targets
                    yb = yb.long()
                loss = criterion(preds, yb)
            else:
                # match preds/targets shapes for MSE
                if preds.dim() > 1:
                    preds = preds.squeeze()
                if yb.dim() > 1:
                    yb = yb.squeeze()
                loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        train_loss = running / max(1, count)
        history["train_loss"].append(train_loss)
        val_loss = None
        if val_loader is not None:
            net.eval()
            running_v = 0.0
            count_v = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = net(xb)
                    if is_classification:
                        if yb.dtype != torch.long:
                            yb = yb.long()
                        loss = criterion(preds, yb)
                    else:
                        if preds.dim() > 1:
                            preds = preds.squeeze()
                        if yb.dim() > 1:
                            yb = yb.squeeze()
                        loss = criterion(preds, yb)
                    running_v += float(loss.detach().cpu()) * len(xb)
                    count_v += len(xb)
            val_loss = running_v / max(1, count_v)
            history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


def train_with_soft_labels(
    net: nn.Module,
    train_loader: DataLoader,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    alpha: float = 1.0,
    is_classification: bool = False,
    temperature: float = 3.0,
) -> nn.Module:
    if is_classification:
        criterion = nn.KLDivLoss(reduction='batchmean')
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(epochs):
        for xb, y_soft in train_loader:
            xb = xb.to(device)
            y_soft = y_soft.to(device)
            optimizer.zero_grad()
            preds = net(xb)

            if is_classification:
                #temperature-softened KL distillation
                y_soft = softmax(y_soft / temperature)
                preds_log = log_softmax(preds / temperature)
                loss = alpha * criterion(preds_log, y_soft)
            else:
                loss = alpha * criterion(preds, y_soft)

            loss.backward()
            optimizer.step()
    net.eval()
    return net


def train_sklearn_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
):
    """Fit an sklearn/XGBoost model (flattens sequence input first)."""
    #[N, T, D] -> [N, T*D]
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    model.fit(X_flat, y)
    return model


def train_model_universal(
    model,
    train_loader=None,
    X_train=None,
    y_train=None,
    val_loader=None,
    device=None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    is_classification: bool = False,
    **kwargs
):
    """Train a PyTorch or sklearn model, picking the right path automatically."""
    if is_pytorch_model(model):
        if train_loader is None:
            raise ValueError("PyTorch models require train_loader")
        return train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay, patience, is_classification=is_classification)
    elif is_sklearn_model(model):
        if X_train is None or y_train is None:
            raise ValueError("sklearn models require X_train and y_train arrays")
        trained = train_sklearn_model(model, X_train, y_train)
        return trained, {}  # sklearn has no training history
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

