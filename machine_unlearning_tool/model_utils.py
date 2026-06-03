from typing import List, Optional, Union

import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class LSTMForecast(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: [B, T, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]  #[B, H]
        y = self.head(last).squeeze(-1)  # [B]
        return y


class MLPForecast(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = input_size
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> flatten to [B, T*D]
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        y = self.net(x).squeeze(-1)  # [B]
        return y


def create_lstm_model(
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    model = LSTMForecast(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def create_mlp_model(
    input_size: int,
    hidden_sizes: List[int] = None,
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if hidden_sizes is None:
        hidden_sizes = [64, 32]
    model = MLPForecast(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(device)
    return model


def is_pytorch_model(model) -> bool:
    return isinstance(model, nn.Module)


def is_sklearn_model(model) -> bool:
    from sklearn.base import BaseEstimator
    return isinstance(model, BaseEstimator)


def create_model(
    model_type: str,
    input_size: int = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Union[nn.Module, RandomForestRegressor, XGBRegressor]:
    """Build a model by type: lstm, mlp, random_forest or xgboost."""
    if model_type == "lstm":
        if input_size is None:
            raise ValueError("input_size required for LSTM models")
        return create_lstm_model(input_size=input_size, device=device, **kwargs)
    elif model_type == "mlp":
        if input_size is None:
            raise ValueError("input_size required for MLP models")
        return create_mlp_model(input_size=input_size, device=device, **kwargs)
    elif model_type == "random_forest":
        return RandomForestRegressor(**kwargs)
    elif model_type == "xgboost":
        return XGBRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose: lstm, mlp, random_forest, xgboost")


