# %%
"""A family of *probabilistic* optimizers.

:class:`ProbabilisticOptimizer` wraps **any** off-the-shelf
:class:`torch.optim.Optimizer` (Adam, SGD, ...) and injects a post-step hook
that randomly resamples ("mutates") individual parameter entries.

The recipe, per parameter tensor (i.e. *per layer*):

1. Take the just-computed gradient ``g`` and its per-entry magnitude ``|g|``.
2. Only entries whose ``|g|`` exceeds ``threshold`` are *eligible* to mutate --
   the intuition being that high-gradient entries are the contested,
   still-moving ones worth gambling on.
3. Turn the eligible magnitudes into a probability distribution with a
   temperature-scaled **softmax over the layer**.  Entries with larger
   gradients get proportionally more probability mass.
4. Convert those probabilities into independent Bernoulli mutation decisions so
   that, in expectation, ``num_mutations`` entries mutate per layer, biased
   toward the high-gradient ones.
5. Replace (or perturb) the chosen entries using a :class:`~.mutations.Mutator`
   -- a distribution draw, a deterministic chaotic map, or any callable.

Because the wrapper delegates ``step``/``zero_grad``/state to the base
optimizer, it is a drop-in replacement: swap ``Adam(...)`` for
``ProbabilisticOptimizer(Adam(...), mutator=...)`` and keep the rest of your
training loop untouched.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Type

import torch
from torch import Tensor
from torch.optim import Optimizer

from .mutations import Mutator, NormalMutator

__all__ = ["ProbabilisticOptimizer", "make_probabilistic"]


class ProbabilisticOptimizer(Optimizer):
    """Wrap a base optimizer with gradient-weighted probabilistic mutation.

    Args:
        base_optimizer: An already-constructed :class:`torch.optim.Optimizer`
            whose ``step`` performs the ordinary update.
        mutator: A :class:`~.mutations.Mutator` (or plain callable
            ``fn(values, grads, generator) -> Tensor``) producing replacement
            values for the selected entries.  Defaults to a unit
            :class:`~.mutations.NormalMutator`.
        threshold: Gradient-magnitude gate.  Only entries with ``|grad| >
            threshold`` are eligible to mutate.  ``0.0`` makes every entry with
            a non-zero gradient eligible.
        num_mutations: Expected number of entries mutated per layer per step.
            The per-entry softmax probabilities are scaled by this value (and
            clamped to ``[0, 1]``) to form the Bernoulli mutation probabilities.
        temperature: Softmax temperature over gradient magnitudes.  ``-> 0``
            concentrates mutations on the single largest-gradient entry;
            ``large`` spreads them uniformly across eligible entries.
        mutation_prob: Probability of running the mutation hook on any given
            step (a coarse global "how often do we gamble at all" knob).
        per_step_hook: Optional callable ``fn(optimizer)`` invoked after each
            mutation pass -- e.g. to anneal ``num_mutations`` or ``threshold``.
        generator: Optional :class:`torch.Generator` for reproducible sampling.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        mutator: Optional[Callable[..., Tensor]] = None,
        threshold: float = 0.0,
        num_mutations: float = 1.0,
        temperature: float = 1.0,
        mutation_prob: float = 1.0,
        per_step_hook: Optional[Callable[["ProbabilisticOptimizer"], None]] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError("base_optimizer must be a torch.optim.Optimizer instance")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if num_mutations < 0:
            raise ValueError("num_mutations must be non-negative")

        self.base_optimizer = base_optimizer
        self.mutator = mutator if mutator is not None else NormalMutator()
        self.threshold = float(threshold)
        self.num_mutations = float(num_mutations)
        self.temperature = float(temperature)
        self.mutation_prob = float(mutation_prob)
        self.per_step_hook = per_step_hook
        self.generator = generator

        # Running statistics, handy for the "find out" part of the study.
        self.last_num_mutated: int = 0
        self.total_mutated: int = 0
        self.num_steps: int = 0

    # -- Optimizer plumbing: share state with the wrapped optimizer ----------
    @property
    def param_groups(self):  # type: ignore[override]
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value

    @property
    def state(self):  # type: ignore[override]
        return self.base_optimizer.state

    @state.setter
    def state(self, value):
        self.base_optimizer.state = value

    @property
    def defaults(self):  # type: ignore[override]
        return self.base_optimizer.defaults

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):  # type: ignore[override]
        return self.base_optimizer.add_param_group(param_group)

    def state_dict(self):  # type: ignore[override]
        return {
            "base": self.base_optimizer.state_dict(),
            "probabilistic": {
                "threshold": self.threshold,
                "num_mutations": self.num_mutations,
                "temperature": self.temperature,
                "mutation_prob": self.mutation_prob,
                "total_mutated": self.total_mutated,
                "num_steps": self.num_steps,
            },
        }

    def load_state_dict(self, state_dict):  # type: ignore[override]
        self.base_optimizer.load_state_dict(state_dict["base"])
        for k, v in state_dict.get("probabilistic", {}).items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f"Probabilistic({self.base_optimizer.__class__.__name__}, "
            f"mutator={self.mutator.__class__.__name__}, "
            f"threshold={self.threshold}, num_mutations={self.num_mutations}, "
            f"temperature={self.temperature})"
        )

    # -- The interesting part ------------------------------------------------
    def _rand_like(self, x: Tensor) -> Tensor:
        return torch.rand(
            x.shape, generator=self.generator, device=x.device, dtype=x.dtype
        )

    @torch.no_grad()
    def mutation_probabilities(self, grad: Tensor) -> Tensor:
        """Per-entry Bernoulli mutation probability for one gradient tensor.

        This is the softmax-of-gradient-magnitudes rule, gated by ``threshold``
        and scaled to an expected ``num_mutations`` entries per layer.  Exposed
        publicly so the behaviour can be inspected/plotted directly.
        """
        mag = grad.detach().abs().reshape(-1)
        eligible = mag > self.threshold
        probs = torch.zeros_like(mag)
        if not bool(eligible.any()):
            return probs.reshape(grad.shape)

        # Softmax over eligible entries only; ineligible entries get -inf logits
        # so they receive zero probability mass and never dilute the others.
        logits = mag / self.temperature
        logits = logits.masked_fill(~eligible, float("-inf"))
        soft = torch.softmax(logits, dim=0)

        # Scale so the expected count of mutations is ~num_mutations, clamped to
        # valid Bernoulli probabilities.
        bern = (soft * self.num_mutations).clamp_(0.0, 1.0)
        return bern.reshape(grad.shape)

    @torch.no_grad()
    def mutate(self) -> int:
        """Run one mutation pass over all parameters. Returns entries mutated."""
        num_mutated = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                bern = self.mutation_probabilities(grad)
                mask = self._rand_like(bern) < bern
                k = int(mask.sum())
                if k == 0:
                    continue

                flat_p = p.data.reshape(-1)
                flat_g = grad.detach().reshape(-1)
                flat_mask = mask.reshape(-1)
                selected_vals = flat_p[flat_mask]
                selected_grads = flat_g[flat_mask]
                new_vals = self.mutator(
                    selected_vals, selected_grads, self.generator
                ).to(dtype=flat_p.dtype)
                flat_p[flat_mask] = new_vals
                num_mutated += k
        return num_mutated

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):  # type: ignore[override]
        # Run the ordinary update first (with grads still attached), then gamble.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.base_optimizer.step()

        self.num_steps += 1
        self.last_num_mutated = 0
        if self.num_mutations > 0 and (
            self.mutation_prob >= 1.0
            or float(torch.rand((), generator=self.generator)) < self.mutation_prob
        ):
            self.last_num_mutated = self.mutate()
            self.total_mutated += self.last_num_mutated

        if self.per_step_hook is not None:
            self.per_step_hook(self)
        return loss


def make_probabilistic(optimizer_cls: Type[Optimizer]) -> Type:
    """Build a probabilistic variant of an optimizer *class*.

    Returns a callable that constructs the base optimizer and wraps it, so the
    result is usable exactly like the original class::

        ProbAdam = make_probabilistic(torch.optim.Adam)
        opt = ProbAdam(model.parameters(), lr=1e-3,
                       mutator=ChaoticMutator(), num_mutations=2.0)

    Mutation keyword arguments (``mutator``, ``threshold``, ``num_mutations``,
    ``temperature``, ``mutation_prob``, ``per_step_hook``, ``generator``) are
    peeled off and forwarded to :class:`ProbabilisticOptimizer`; everything else
    goes to the base optimizer's constructor.
    """
    prob_keys = {
        "mutator",
        "threshold",
        "num_mutations",
        "temperature",
        "mutation_prob",
        "per_step_hook",
        "generator",
    }

    def factory(params: Iterable, **kwargs: Any) -> ProbabilisticOptimizer:
        prob_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in prob_keys}
        base = optimizer_cls(params, **kwargs)
        return ProbabilisticOptimizer(base, **prob_kwargs)

    factory.__name__ = f"Probabilistic{optimizer_cls.__name__}"
    factory.__qualname__ = factory.__name__
    factory.__doc__ = f"Probabilistic variant of {optimizer_cls.__name__}."
    return factory
