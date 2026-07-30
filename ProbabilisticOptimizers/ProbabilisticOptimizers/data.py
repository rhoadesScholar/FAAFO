# %%
"""Datasets for the training experiments.

Two options, both fast on a laptop (Apple-silicon MPS, CUDA, or CPU):

* ``"synthetic"`` -- a fully **offline** teacher-student classification task: a
  fixed random MLP "teacher" labels Gaussian inputs, and we train a student to
  imitate it.  This always runs (no download) and is a genuine, non-convex
  optimisation problem.
* ``"mnist"`` / ``"fashionmnist"`` -- the classic image benchmark via
  ``torchvision`` (downloaded on first use).  Optionally subset for speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["DataBundle", "get_data"]


@dataclass
class DataBundle:
    train: DataLoader
    test: DataLoader
    in_dim: int
    n_classes: int
    name: str


def _make_synthetic(
    n_train: int,
    n_test: int,
    in_dim: int,
    n_classes: int,
    teacher_hidden: int,
    seed: int,
    label_noise: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    # Fixed random 2-layer teacher; labels are its argmax (a hard decision
    # boundary the student must recover). Weights scaled for a spread of logits.
    w1, b1 = randn(in_dim, teacher_hidden) * (in_dim ** -0.5), randn(teacher_hidden)
    w2, b2 = randn(teacher_hidden, n_classes) * (teacher_hidden ** -0.5), randn(n_classes)

    def teacher(x):
        h = torch.tanh(x @ w1 + b1)
        return h @ w2 + b2

    n = n_train + n_test
    x = randn(n, in_dim)
    y = teacher(x).argmax(dim=1)
    xtr, ytr, xte, yte = x[:n_train], y[:n_train].clone(), x[n_train:], y[n_train:]

    # Corrupt a fraction of *training* labels only. A small train set plus label
    # noise creates a real train/val gap, so regularisation-like effects (e.g.
    # weight resampling) have room to help or hurt generalisation.
    if label_noise > 0:
        k = int(label_noise * n_train)
        idx = torch.randperm(n_train, generator=g)[:k]
        ytr[idx] = torch.randint(0, n_classes, (k,), generator=g)
    return xtr, ytr, xte, yte


def get_data(
    dataset: str = "synthetic",
    batch_size: int = 128,
    seed: int = 0,
    root: str = "./data",
    subset: Optional[int] = None,
    n_train: int = 1500,
    n_test: int = 3000,
    in_dim: int = 30,
    n_classes: int = 5,
    teacher_hidden: int = 128,
    label_noise: float = 0.2,
) -> DataBundle:
    """Build train/test dataloaders for the requested dataset."""
    if dataset == "synthetic":
        xtr, ytr, xte, yte = _make_synthetic(
            n_train, n_test, in_dim, n_classes, teacher_hidden, seed, label_noise
        )
        train = DataLoader(
            TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True
        )
        test = DataLoader(TensorDataset(xte, yte), batch_size=512, shuffle=False)
        return DataBundle(train, test, in_dim, n_classes, "synthetic")

    if dataset in ("mnist", "fashionmnist"):
        # Lazy import: torchvision is only needed for the image datasets.
        from torchvision import datasets, transforms

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train_ds = cls(root, train=True, download=True, transform=tfm)
        test_ds = cls(root, train=False, download=True, transform=tfm)
        if subset is not None:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(train_ds), generator=g)[:subset]
            train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
        train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test = DataLoader(test_ds, batch_size=512, shuffle=False)
        return DataBundle(train, test, 28 * 28, 10, dataset)

    raise ValueError(f"unknown dataset {dataset!r}")
