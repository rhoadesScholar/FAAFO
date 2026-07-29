# %%
"""A real training run for the probabilistic optimizers.

Trains a small model (MLP on a synthetic teacher-student task, or a CNN on
MNIST) and compares a vanilla Adam baseline against several probabilistic-Adam
configurations that differ in their **gate** (high / none / inverted) and in how
the **mutation budget** is chosen (fixed vs. adaptive).

Runs on Apple-silicon (MPS), CUDA, or CPU -- device is auto-detected.

Single run::

    python -m ProbabilisticOptimizers.train --config gate_none --epochs 5

See :mod:`ProbabilisticOptimizers.compare` to sweep every config over seeds and
plot the comparison.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .data import get_data
from .models import build_model
from .mutation_counts import FractionOverGate, GradientScaled
from .mutations import NormalMutator
from .optimizer import ProbabilisticOptimizer

__all__ = ["get_device", "OPTIMIZER_CONFIGS", "build_optimizer", "train_one", "RunResult"]


def get_device(pref: str = "auto") -> torch.device:
    """Auto-detect the best available device (MPS > CUDA > CPU)."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -- The configurations under test --------------------------------------------
# Each value is either None (plain Adam) or a dict of ProbabilisticOptimizer
# kwargs. A default relative Gaussian perturbation mutator is used unless a
# config overrides it. Gates use quantile thresholds so they are scale-free:
# the median split (0.5) gives high/low gates the same eligible-set size, so any
# difference is purely about *which* weights get mutated.
def _default_mutator() -> NormalMutator:
    # Perturb each selected weight by ~30% of its magnitude (scale-free), a
    # strong-enough dose to act as a regulariser.
    return NormalMutator(std=0.3, relative=True, additive=True)


# Two controlled contrasts share this file:
#   (A) GATE ablation  -- gate_high / gate_none / gate_low, all using the same
#       count strategy (a fraction of the eligible entries) and strength, so any
#       difference is purely *which* weights get mutated.
#   (B) COUNT ablation -- how the per-step budget is chosen, all with gate=none:
#       gate_none (fraction of eligible) vs count_fixed (tiny constant) vs
#       count_meangrad (scaled by mean gradient magnitude).
_FRAC = FractionOverGate(fraction=0.02)  # ~2% of eligible entries per layer

OPTIMIZER_CONFIGS: Dict[str, Optional[dict]] = {
    "adam": None,
    # (A) High-gradient gate (the original design): the upper-gradient half.
    "gate_high": dict(
        gate="high", threshold_mode="quantile", threshold=0.5,
        num_mutations=_FRAC, temperature=1.0, weight_by="grad",
    ),
    # (A/B) No gate: any weight can mutate, gradient-weighted.
    "gate_none": dict(
        gate="none", num_mutations=_FRAC, temperature=1.0, weight_by="grad",
    ),
    # (A) Inverted gate: the lower-gradient (stuck) half, favouring the smallest
    # gradients.
    "gate_low": dict(
        gate="low", threshold_mode="quantile", threshold=0.5,
        num_mutations=_FRAC, temperature=1.0, weight_by="neg_grad",
    ),
    # (B) Fixed tiny budget: a few mutations per layer regardless of size.
    "count_fixed": dict(
        gate="none", num_mutations=4.0, temperature=1.0, weight_by="grad",
    ),
    # (B) Adaptive budget scaled by the mean gradient magnitude: more mutation
    # while gradients are large (early), less as they decay.
    "count_meangrad": dict(
        gate="none",
        num_mutations=GradientScaled(scale=1500.0, stat="mean", max_count=80.0),
        temperature=1.0, weight_by="grad",
    ),
}


def build_optimizer(
    name: str,
    params,
    lr: float,
    generator: Optional[torch.Generator] = None,
):
    """Construct the (base or probabilistic) optimizer for a named config."""
    if name not in OPTIMIZER_CONFIGS:
        raise ValueError(f"unknown config {name!r}; choices: {list(OPTIMIZER_CONFIGS)}")
    cfg = OPTIMIZER_CONFIGS[name]
    base = torch.optim.Adam(params, lr=lr)
    if cfg is None:
        return base
    cfg = dict(cfg)
    mutator = cfg.pop("mutator", None) or _default_mutator()
    return ProbabilisticOptimizer(base, mutator=mutator, generator=generator, **cfg)


@dataclass
class RunResult:
    config: str
    seed: int
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    mutated_per_step: float = 0.0
    final_val_acc: float = 0.0
    best_val_acc: float = 0.0
    wall_time: float = 0.0


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total, correct, loss_sum, n = 0, 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += float(loss_fn(out, y)) * y.size(0)
        correct += int((out.argmax(1) == y).sum())
        total += y.size(0)
        n += y.size(0)
    return loss_sum / n, correct / total


def train_one(
    config: str,
    dataset: str = "synthetic",
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 0,
    device: Optional[torch.device] = None,
    subset: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> RunResult:
    """Train one model/optimizer config and return its metrics."""
    device = device or get_device()
    torch.manual_seed(seed)
    data = get_data(dataset, batch_size=batch_size, seed=seed, subset=subset)
    model = build_model(dataset, data.in_dim, data.n_classes).to(device)
    # CPU generator for reproducible sampling on CPU; on MPS/CUDA the optimizer
    # detects the device mismatch and falls back to the (seeded) global RNG.
    gen = torch.Generator().manual_seed(seed + 999)
    opt = build_optimizer(config, model.parameters(), lr, generator=gen)
    loss_fn = nn.CrossEntropyLoss()

    res = RunResult(config=config, seed=seed)
    t0 = time.time()
    total_mut, total_steps = 0, 0
    for epoch in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
            if isinstance(opt, ProbabilisticOptimizer):
                total_mut += opt.last_num_mutated
                total_steps += 1
        vl, va = evaluate(model, data.test, device, loss_fn)
        res.train_loss.append(ep_loss / nb)
        res.val_loss.append(vl)
        res.val_acc.append(va)
        if log:
            log(f"[{config}] epoch {epoch+1}/{epochs} "
                f"train_loss={ep_loss/nb:.4f} val_loss={vl:.4f} val_acc={va:.4f}")

    res.wall_time = time.time() - t0
    res.mutated_per_step = (total_mut / total_steps) if total_steps else 0.0
    res.final_val_acc = res.val_acc[-1]
    res.best_val_acc = max(res.val_acc)
    return res


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Train one probabilistic-optimizer config.")
    p.add_argument("--config", default="gate_high", choices=list(OPTIMIZER_CONFIGS))
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "mnist", "fashionmnist"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", type=int, default=None,
                   help="Optional MNIST training-subset size (for speed).")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = get_device(args.device)
    print(f"Device: {device}")
    res = train_one(
        args.config, dataset=args.dataset, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed, device=device,
        subset=args.subset, log=print,
    )
    print(f"\nconfig={res.config}  final_val_acc={res.final_val_acc:.4f}  "
          f"best_val_acc={res.best_val_acc:.4f}  "
          f"mutated/step={res.mutated_per_step:.1f}  time={res.wall_time:.1f}s")


if __name__ == "__main__":
    main()
