# %%
"""Adaptive strategies for the *expected number of mutations per layer*.

:class:`~.optimizer.ProbabilisticOptimizer` accepts ``num_mutations`` as either a
constant or a callable ``fn(grad, eligible) -> float``.  This module provides a
few ready-made callables so the per-step mutation budget can *react* to the
gradient statistics instead of being fixed:

* :class:`Fixed` -- a constant (equivalent to passing a float).
* :class:`FractionOverGate` -- a fraction of the entries that pass the gate, so
  the budget grows when many parameters are eligible and shrinks as the layer
  settles.
* :class:`GradientScaled` -- a multiple of the mean or max ``|grad|`` (over the
  eligible entries or the whole layer), so the budget tracks gradient "energy".

Each callable returns a non-negative float; the optimizer clamps the resulting
per-entry Bernoulli probabilities to ``[0, 1]``, so overshooting simply
saturates.  All accept ``min_count`` / ``max_count`` bounds.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = ["MutationCount", "Fixed", "FractionOverGate", "GradientScaled"]


class MutationCount:
    """Base class for adaptive mutation-count callables."""

    def __init__(self, min_count: float = 0.0, max_count: Optional[float] = None):
        self.min_count = float(min_count)
        self.max_count = None if max_count is None else float(max_count)

    def _clamp(self, k: float) -> float:
        k = max(k, self.min_count)
        if self.max_count is not None:
            k = min(k, self.max_count)
        return k

    def __call__(self, grad: Tensor, eligible: Tensor) -> float:  # pragma: no cover
        raise NotImplementedError


class Fixed(MutationCount):
    """Constant expected count (a callable form of passing a float)."""

    def __init__(self, count: float):
        super().__init__()
        self.count = float(count)

    def __call__(self, grad, eligible):
        return self.count

    def __repr__(self):
        return f"Fixed({self.count})"


class FractionOverGate(MutationCount):
    """Expected count = ``fraction`` x (number of entries passing the gate).

    Mutates a roughly fixed *proportion* of the currently-eligible parameters, so
    the budget adapts to how many entries the gate lets through at each step.
    """

    def __init__(
        self,
        fraction: float = 0.01,
        min_count: float = 0.0,
        max_count: Optional[float] = None,
    ):
        super().__init__(min_count, max_count)
        self.fraction = float(fraction)

    def __call__(self, grad, eligible):
        n = float(eligible.sum().item())
        return self._clamp(self.fraction * n)

    def __repr__(self):
        return f"FractionOverGate(fraction={self.fraction})"


class GradientScaled(MutationCount):
    """Expected count = ``scale`` x stat(``|grad|``).

    Ties the mutation budget to gradient magnitude, so the optimizer gambles more
    while gradients are large (early / contested training) and less as they
    decay.  ``stat`` selects ``"mean"`` or ``"max"``; ``region`` selects whether
    the statistic is taken over the eligible entries or the whole layer.
    """

    def __init__(
        self,
        scale: float = 1.0,
        stat: str = "mean",
        region: str = "all",
        min_count: float = 0.0,
        max_count: Optional[float] = None,
    ):
        super().__init__(min_count, max_count)
        if stat not in ("mean", "max"):
            raise ValueError("stat must be 'mean' or 'max'")
        if region not in ("all", "eligible"):
            raise ValueError("region must be 'all' or 'eligible'")
        self.scale = float(scale)
        self.stat = stat
        self.region = region

    def __call__(self, grad, eligible):
        mag = grad.detach().abs().reshape(-1)
        if self.region == "eligible":
            sel = mag[eligible.reshape(-1)]
            if sel.numel() == 0:
                return self.min_count
            mag = sel
        val = mag.mean() if self.stat == "mean" else mag.max()
        return self._clamp(self.scale * float(val.item()))

    def __repr__(self):
        return (
            f"GradientScaled(scale={self.scale}, stat={self.stat!r}, "
            f"region={self.region!r})"
        )
