# %%
"""Adaptive per-loss weighting for multi-loss training.

Each loss ``L_i`` gets a weight computed from its own statistics, with no
gradient through the weight::

    ref_i  = mean |L_i| over the first ``warmup`` steps     (its natural scale)
    ema_i  = EMA of |L_i| over past steps (decay ``beta``)  (where it is now)
    r_i    = ema_i / ref_i                                  (progress ratio; 1 = no progress)
    w_i    = priority_i / ref_i * (r_i / mean_j r_j) ** alpha

``alpha`` sets how the weights react to progress:

* ``alpha = -1``: ``w_i ~ 1 / ema_i``. Every loss contributes equally *now*.
  Losses that shrink fast (easy tasks) gain weight over time.
* ``alpha = 0``: weights fixed at ``1 / ref_i``. Removes units/scale only.
* ``alpha > 0``: losses that lag behind (high ``r_i``) gain weight, the
  GradNorm-style "inverse training rate" idea. Counters easy-task upweighting.

Design notes:

* Weights use only *past* batches. A weight computed from the current batch is
  correlated with that batch's gradient and biases training.
* Slow weights balance losses and track drift. They do not flatten per-batch
  spikes; use the optional gradient clipping in ``MultiLossOptimizer`` for that.
* Losses may be negative (e.g. NLL); statistics use ``|L_i|``.
* Under DDP the loss values are averaged across ranks first, so every rank
  computes identical weights.
* Caveat for ``alpha > 0``: a loss with a large irreducible floor (label noise)
  never "progresses", so it keeps gaining weight. Cap with ``max_ratio``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


class LossBalancer:
    """Turn a dict of scalar losses into one weighted total.

    Args:
        names: loss names, fixed for the run.
        alpha: progress exponent (see module docstring). Default 1.0.
        beta: EMA decay for the current loss level, in [0, 1). Higher is
            slower and less noisy.
        warmup: number of initial steps averaged to get each loss's reference
            scale. During warmup, weights are ``priority / running mean``.
        priorities: optional ``{name: multiplier}``, default 1.0 each. The one
            deliberate knob: how much each loss matters to you.
        max_ratio: clamp on the relative-progress factor ``(r_i / mean r) **
            alpha`` to ``[1/max_ratio, max_ratio]``. Limits runaway weights.
        eps: numerical floor.
        enabled: if False, behaves as a plain sum weighted by ``priorities``
            (an A/B baseline with the same code path).
    """

    def __init__(
        self,
        names: Iterable[str],
        alpha: float = 1.0,
        beta: float = 0.99,
        warmup: int = 20,
        priorities: Optional[Mapping[str, float]] = None,
        max_ratio: float = 10.0,
        eps: float = 1e-12,
        enabled: bool = True,
    ):
        self.names = list(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("loss names must be unique")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        if max_ratio < 1.0:
            raise ValueError("max_ratio must be >= 1")
        priorities = dict(priorities or {})
        unknown = set(priorities) - set(self.names)
        if unknown:
            raise ValueError(f"priorities for unknown losses: {sorted(unknown)}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.warmup = int(warmup)
        self.max_ratio = float(max_ratio)
        self.eps = float(eps)
        self.enabled = enabled
        self.priorities = {n: float(priorities.get(n, 1.0)) for n in self.names}
        self.ref: Dict[str, float] = {}  # running mean during warmup, then frozen
        self.ema: Dict[str, float] = {}
        self.steps = 0

    # -- core -----------------------------------------------------------------

    def weights(self) -> Dict[str, float]:
        """Weights for the next step, from the statistics so far."""
        if not self.enabled or not self.ref:
            return dict(self.priorities)
        base = {n: self.priorities[n] / max(self.ref[n], self.eps) for n in self.names}
        if self.steps < self.warmup or self.alpha == 0.0:
            return base
        r = {n: max(self.ema[n], self.eps) / max(self.ref[n], self.eps) for n in self.names}
        r_mean = sum(r.values()) / len(r)
        lo, hi = 1.0 / self.max_ratio, self.max_ratio
        return {
            n: base[n] * min(max((r[n] / r_mean) ** self.alpha, lo), hi)
            for n in self.names
        }

    def __call__(self, losses: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Return ``(total_loss, stats)``. Updates the statistics afterwards."""
        missing = set(self.names) - set(losses)
        extra = set(losses) - set(self.names)
        if missing or extra:
            raise KeyError(
                f"loss names mismatch; missing={sorted(missing)} extra={sorted(extra)}"
            )

        mags = self._magnitudes(losses)
        if self.enabled and not self.ref:
            # First step: no history, so this batch sets the initial scale.
            self.ref, self.ema = dict(mags), dict(mags)

        w = self.weights()
        total = sum(w[n] * losses[n] for n in self.names)

        if self.enabled:
            self._update(mags)
        self.steps += 1

        stats = {f"loss/{n}": float(losses[n].detach()) for n in self.names}
        stats.update({f"weight/{n}": w[n] for n in self.names})
        return total, stats

    def _update(self, mags: Mapping[str, float]) -> None:
        k = self.steps  # number of past samples already in ref
        for n in self.names:
            if k < self.warmup and k > 0:
                self.ref[n] += (mags[n] - self.ref[n]) / (k + 1)
            if k > 0:
                self.ema[n] = self.beta * self.ema[n] + (1 - self.beta) * mags[n]

    def _magnitudes(self, losses: Mapping[str, Tensor]) -> Dict[str, float]:
        vals = torch.stack(
            [losses[n].detach().float().abs().reshape(()) for n in self.names]
        )
        if dist.is_available() and dist.is_initialized():
            vals = vals.clone()
            dist.all_reduce(vals, op=dist.ReduceOp.SUM)
            vals /= dist.get_world_size()
        return dict(zip(self.names, vals.tolist()))

    # -- checkpointing --------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "ref": dict(self.ref),
            "ema": dict(self.ema),
            "steps": self.steps,
            "priorities": dict(self.priorities),
            "alpha": self.alpha,
            "beta": self.beta,
            "warmup": self.warmup,
            "max_ratio": self.max_ratio,
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.ref = dict(state["ref"])
        self.ema = dict(state["ema"])
        self.steps = int(state["steps"])
        self.priorities = dict(state["priorities"])
        self.alpha = float(state["alpha"])
        self.beta = float(state["beta"])
        self.warmup = int(state["warmup"])
        self.max_ratio = float(state["max_ratio"])


class MultiLossOptimizer:
    """Wrap any optimizer: balance losses, backward, optional clip, step.

    Example::

        opt = MultiLossOptimizer(
            torch.optim.Adam(model.parameters(), lr=1e-3),
            names=["recon", "kl"],
            clip_grad_norm=None,   # optional, off by default
        )
        stats = opt.step({"recon": recon_loss, "kl": kl_loss})

    Extra keyword arguments go to :class:`LossBalancer`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        names: Iterable[str],
        clip_grad_norm: Optional[float] = None,
        balance: bool = True,
        **balancer_kwargs,
    ):
        self.optimizer = optimizer
        self.balancer = LossBalancer(names, enabled=balance, **balancer_kwargs)
        self.clip_grad_norm = clip_grad_norm

    def _params(self):
        for group in self.optimizer.param_groups:
            yield from group["params"]

    def step(self, losses: Mapping[str, Tensor]) -> Dict[str, float]:
        """Zero grads, combine ``losses``, backprop, optionally clip, step."""
        self.optimizer.zero_grad(set_to_none=True)
        total, stats = self.balancer(losses)
        total.backward()
        if self.clip_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(list(self._params()), self.clip_grad_norm)
            stats["grad_norm"] = float(norm)
        self.optimizer.step()
        stats["loss/total"] = float(total.detach())
        return stats

    def state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "balancer": self.balancer.state_dict(),
            "clip_grad_norm": self.clip_grad_norm,
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.balancer.load_state_dict(state["balancer"])
        self.clip_grad_norm = state["clip_grad_norm"]
