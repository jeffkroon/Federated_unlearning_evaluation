"""Model definitions for federated learning experiments."""

import torch
import torch.nn as nn
from .model_registry import ModelRegistry


class BaseModel(nn.Module):
    """Base model with get/set parameter helpers used by the FL framework."""

    def __init__(self):
        super(BaseModel, self).__init__()

    def get_parameters(self):
        return [param.data.clone() for param in self.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.clone()


class RULPredictor(BaseModel):
    """MLP for N-CMAPSS Remaining Useful Life regression."""

    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(RULPredictor, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTClassifier(BaseModel):
    """CNN for MNIST image classification."""

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CIFAR10Classifier(BaseModel):
    """Small CNN for CIFAR-10 image classification (legacy baseline)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def _group_norm(num_channels: int, num_groups: int = 2) -> nn.GroupNorm:
    """GroupNorm with a small fixed group count, FedAvg-safe alternative to BatchNorm.

    Reference: Hsieh et al. 2019 "The Non-IID Data Quagmire", Wu & He 2018 "Group Normalization".
    BatchNorm running statistics break under FedAvg parameter averaging; GroupNorm is stateless.
    """
    groups = min(num_groups, num_channels)
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels)


class _ResNetBasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNet (He et al. 2016), GroupNorm variant."""

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = _group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                _group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class CIFAR10ResNet20(BaseModel):
    """ResNet-20 with GroupNorm for CIFAR-10 federated learning.

    Architecture: He et al. 2016 ("Deep Residual Learning"), 6n+2 layers with n=3.
    Stages: 16 -> 32 -> 64 filters. ~270K parameters.
    Normalization: GroupNorm replaces BatchNorm to remain stable under FedAvg
    (Hsieh et al. 2019 "Non-IID Quagmire"; Li et al. 2021 "FedBN").
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = _group_norm(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_ResNetBasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class TabularClassifier(BaseModel):
    """MLP for tabular data (classification or regression)."""

    def __init__(self, input_dim, hidden_dims=None, output_dim=1, dropout=0.2):
        super(TabularClassifier, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def _factory_n_cmapss(**kwargs):
    return RULPredictor(**kwargs)


def _factory_mnist(**kwargs):
    return MNISTClassifier()


def _factory_cifar10(**kwargs):
    return CIFAR10ResNet20(num_classes=kwargs.get("output_dim", 10))


def _factory_tabular(**kwargs):
    input_dim = kwargs.get("input_dim", 20)
    output_dim = kwargs.get("output_dim", 2)
    if output_dim > 50:
        default_hidden_dims = [256, 128, 64]
    else:
        default_hidden_dims = [128, 64]
    hidden_dims = kwargs.get("hidden_dims", default_hidden_dims)
    dropout = kwargs.get("dropout", 0.2)
    return TabularClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout
    )


#Register built-in experiment types
ModelRegistry.register("n_cmapss", _factory_n_cmapss)
ModelRegistry.register("mnist", _factory_mnist)
ModelRegistry.register("cifar10", _factory_cifar10)
ModelRegistry.register("tabular", _factory_tabular)


def create_model(experiment_type, **kwargs):
    """Create a model for the given experiment type via ModelRegistry."""
    return ModelRegistry.create_model(experiment_type, **kwargs)
