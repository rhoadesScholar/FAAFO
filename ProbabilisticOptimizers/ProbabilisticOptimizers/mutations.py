# %%
"""Mutation functions ("mutators") for :mod:`ProbabilisticOptimizers`.

A *mutator* decides what new value a selected parameter entry takes when it is
resampled.  Every mutator receives the *current* values of the selected entries
and the *gradients* at those entries, and returns a tensor of replacement (or
additive) values of the same shape.

Two broad flavours are provided, mirroring the two options in the study:

* **Distribution draws** -- :class:`NormalMutator`, :class:`UniformMutator`.
  New values are sampled i.i.d. from a fixed distribution.
* **Deterministic chaotic functions** -- :class:`ChaoticMutator` (the logistic
  map).  New values are a deterministic-but-chaotic function of the current
  value, so the "randomness" comes from sensitive dependence on initial
  conditions rather than from a PRNG.

Any callable with the signature ``fn(values, grads, generator) -> Tensor`` can be
used as a mutator; :class:`CallableMutator` adapts a plain function into the
:class:`Mutator` interface (with the standard ``additive`` handling).
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor

__all__ = [
    "Mutator",
    "NormalMutator",
    "UniformMutator",
    "ChaoticMutator",
    "CallableMutator",
]


class Mutator(torch.nn.Module):
    """Base class for mutators.

    Subclasses implement :meth:`sample`, which returns the *proposed* values for
    the selected entries.  The base class then either replaces the current
    values with the proposal (``additive=False``) or adds the proposal to them
    (``additive=True``), optionally scaled by ``strength``.

    Args:
        additive: If ``True``, the sampled value is added to the current
            parameter value (a local perturbation).  If ``False`` (default) the
            current value is discarded and replaced by the sample (a true
            resample / jump).
        strength: Scalar multiplier applied to the sampled value before it is
            used.  Handy for annealing the mutation magnitude over training.
    """

    def __init__(self, additive: bool = False, strength: float = 1.0):
        super().__init__()
        self.additive = additive
        self.strength = float(strength)

    def sample(
        self, values: Tensor, grads: Tensor, generator: Optional[torch.Generator]
    ) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self,
        values: Tensor,
        grads: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        proposal = self.strength * self.sample(values, grads, generator)
        return values + proposal if self.additive else proposal

    # Allow use as a plain callable (mutator(values, grads)).
    __call__ = forward


class NormalMutator(Mutator):
    """Draw replacement values from ``Normal(mean, std)``.

    With ``additive=True`` this becomes Gaussian perturbation of the current
    value; with ``relative=True`` the standard deviation is scaled by the
    magnitude of each current value so that large weights move more than small
    ones.
    """

    def __init__(
        self,
        std: float = 1.0,
        mean: float = 0.0,
        relative: bool = False,
        additive: bool = False,
        strength: float = 1.0,
    ):
        super().__init__(additive=additive, strength=strength)
        self.std = float(std)
        self.mean = float(mean)
        self.relative = relative

    def sample(self, values, grads, generator):
        noise = torch.randn(
            values.shape, generator=generator, device=values.device, dtype=values.dtype
        )
        scale = self.std * (values.abs() if self.relative else 1.0)
        return self.mean + scale * noise


class UniformMutator(Mutator):
    """Draw replacement values from ``Uniform(low, high)``."""

    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        additive: bool = False,
        strength: float = 1.0,
    ):
        super().__init__(additive=additive, strength=strength)
        self.low = float(low)
        self.high = float(high)

    def sample(self, values, grads, generator):
        u = torch.rand(
            values.shape, generator=generator, device=values.device, dtype=values.dtype
        )
        return self.low + (self.high - self.low) * u


class ChaoticMutator(Mutator):
    """Deterministic chaotic mutation via the logistic map.

    The logistic map ``x <- r * x * (1 - x)`` is chaotic for ``r`` in roughly
    ``(3.57, 4]``.  Each selected parameter is squashed into ``(0, 1)`` with a
    sigmoid to seed the map, iterated ``iterations`` times, and mapped back onto
    ``(-scale, scale)``.  Because the map has sensitive dependence on its seed,
    nearby parameters diverge into very different values -- deterministic yet
    effectively unpredictable, and requiring no PRNG.

    Args:
        r: Logistic-map rate.  ``3.99`` sits deep in the chaotic regime.
        iterations: Number of map iterations per mutation.  More iterations
            amplify sensitivity to the seed.
        scale: Half-width of the output range when ``additive=False``.
        seed_gain: Multiplier applied before the seeding sigmoid; larger values
            spread seeds toward the edges of ``(0, 1)``.
        additive: If ``True``, the chaotic value is added to the current
            parameter (a chaotic perturbation) rather than replacing it.
    """

    def __init__(
        self,
        r: float = 3.99,
        iterations: int = 5,
        scale: float = 1.0,
        seed_gain: float = 1.0,
        additive: bool = False,
        strength: float = 1.0,
    ):
        super().__init__(additive=additive, strength=strength)
        if not 0.0 < r <= 4.0:
            raise ValueError("logistic-map rate r must be in (0, 4]")
        self.r = float(r)
        self.iterations = int(iterations)
        self.scale = float(scale)
        self.seed_gain = float(seed_gain)

    def sample(self, values, grads, generator):
        # Seed the map in (0, 1) from a mix of the current value and its
        # gradient so that two entries with identical weights but different
        # gradients still diverge.
        seed = self.seed_gain * (values + grads)
        x = torch.sigmoid(seed).clamp(1e-6, 1 - 1e-6)
        for _ in range(self.iterations):
            x = self.r * x * (1.0 - x)
        return self.scale * (2.0 * x - 1.0)


class CallableMutator(Mutator):
    """Adapt an arbitrary callable into the :class:`Mutator` interface.

    The wrapped function is called as ``fn(values, grads, generator)`` and may
    ignore any argument it does not need.  ``additive``/``strength`` handling is
    provided by the base class.
    """

    def __init__(
        self,
        fn: Callable[..., Tensor],
        additive: bool = False,
        strength: float = 1.0,
    ):
        super().__init__(additive=additive, strength=strength)
        self.fn = fn

    def sample(self, values, grads, generator):
        try:
            return self.fn(values, grads, generator)
        except TypeError:
            # Support fn(values) and fn(values, grads) signatures too.
            try:
                return self.fn(values, grads)
            except TypeError:
                return self.fn(values)
