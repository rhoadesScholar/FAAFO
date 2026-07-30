# %%
"""Small models used by the training experiments."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MLP", "SmallCNN", "build_model"]


class MLP(nn.Module):
    """A plain multi-layer perceptron classifier."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


class SmallCNN(nn.Module):
    """A compact CNN for 1x28x28 images (MNIST / FashionMNIST)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(dataset: str, in_dim: int, n_classes: int) -> nn.Module:
    """Pick a sensible model for the dataset."""
    if dataset in ("mnist", "fashionmnist"):
        return SmallCNN(n_classes=n_classes)
    return MLP(in_dim=in_dim, n_classes=n_classes)
