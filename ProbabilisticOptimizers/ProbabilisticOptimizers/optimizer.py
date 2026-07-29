# %%
"""A family of *probabilistic* optimizers.

:class:`ProbabilisticOptimizer` wraps **any** off-the-shelf
:class:`torch.optim.Optimizer` (Adam, SGD, ...) and injects a post-step hook
that randomly resamples ("mutates") individual parameter entries.

The recipe, per parameter tensor (i.e. *per layer*):

1. Take the just-computed gradient ``g`` and its per-entry magnitude ``|g|``.
2. Decide which entries are *eligible* to mutate via the ``gate``:

   * ``"high"`` (default) -- ``|g| > threshold``: the contested, still-moving
     entries.  This makes mutation a *descent-time* exploration trick.
   * ``"low"`` -- ``|g| < threshold``: the stuck / near-dead entries.  The
     inverted hypothesis -- resample what has stopped learning.
   * ``"none"`` -- every entry with a gradient is eligible (drop the gate).

3. Turn the eligible magnitudes into a probability distribution with a
   temperature-scaled **softmax over the layer** (``weight_by="grad"`` favours
   large gradients; ``"neg_grad"`` favours small ones -- the natural pairing for
   the ``"low"`` gate).
4. Convert those probabilities into independent Bernoulli mutation decisions so
   that, in expectation, ``num_mutations`` entries mutate per layer.  ``num_mutations``
   may be a constant *or* a callable ``fn(grad, eligible) -> float`` that adapts
   the count to the gradient statistics (see :mod:`.mutation_counts`).
5. Replace (or perturb) the chosen entries using a :class:`~.mutations.Mutator`
   -- a distribution draw, a deterministic chaotic map, or any callable.

Because the wrapper delegates ``step``/``zero_grad``/state to the base
optimizer, it is a drop-in replacement: swap ``Adam(...)`` for
``ProbabilisticOptimizer(Adam(...), mutator=...)`` and keep the rest of your
training loop untouched.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Type, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from .mutations import NormalMutator

__all__ = ["ProbabilisticOptimizer", "make_probabilistic"]

GATES = ("high", "low", "none")
WEIGHTINGS = ("grad", "neg_grad", "uniform")
THRESHOLD_MODES = ("abs", "quantile")

# A mutation-count spec is either a constant or fn(grad, eligible_mask) -> float.
CountSpec = Union[float, Callable[[Tensor, Tensor], float]]


class ProbabilisticOptimizer(Optimizer):
    """Wrap a base optimizer with gradient-weighted probabilistic mutation.

    Args:
        base_optimizer: An already-constructed :class:`torch.optim.Optimizer`
            whose ``step`` performs the ordinary update.
        mutator: A :class:`~.mutations.Mutator` (or plain callable
            ``fn(values, grads, generator) -> Tensor``) producing replacement
            values for the selected entries.  Defaults to a unit
            :class:`~.mutations.NormalMutator`.
        threshold: Gradient-magnitude gate boundary (see ``gate`` and
            ``threshold_mode``).
        threshold_mode: How ``threshold`` is interpreted.  ``"abs"`` (default)
            treats it as an absolute ``|grad|`` value.  ``"quantile"`` treats it
            as a fraction in ``[0, 1]`` and uses the per-layer quantile of
            ``|grad|`` as the boundary -- scale-free across layers and over
            training (e.g. ``gate="high", threshold_mode="quantile",
            threshold=0.5`` mutates the upper-gradient half of each layer).
        num_mutations: Expected number of entries mutated per layer per step.
            Either a constant, or a callable ``fn(grad, eligible) -> float`` for
            adaptive counts (e.g. a fraction of the entries over the gate, or a
            multiple of the mean/max gradient -- see :mod:`.mutation_counts`).
        temperature: Softmax temperature over gradient magnitudes.  ``-> 0``
            concentrates mutations on the extreme-gradient entry; ``large``
            spreads them uniformly across eligible entries.
        gate: Eligibility rule, one of ``"high"``, ``"low"``, ``"none"``.
        weight_by: Softmax weighting, ``"grad"`` (mass on large ``|g|``),
            ``"neg_grad"`` (mass on small ``|g|``), or ``"uniform"`` (ignore the
            gradient -- pick eligible entries with equal probability; the control
            for whether gradient weighting matters).
        mutation_prob: Probability of running the mutation hook on any given
            step (a coarse global "how often do we gamble at all" knob).
        per_step_hook: Optional callable ``fn(optimizer)`` invoked after each
            mutation pass -- e.g. to anneal ``num_mutations`` or ``threshold``.
        generator: Optional :class:`torch.Generator` for reproducible sampling.
            Automatically ignored for parameters on a different device (e.g. a
            CPU generator with MPS/CUDA tensors), falling back to the global RNG.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        mutator: Optional[Callable[..., Tensor]] = None,
        threshold: float = 0.0,
        threshold_mode: str = "abs",
        num_mutations: CountSpec = 1.0,
        temperature: float = 1.0,
        gate: str = "high",
        weight_by: str = "grad",
        mutation_prob: float = 1.0,
        per_step_hook: Optional[Callable[["ProbabilisticOptimizer"], None]] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError("base_optimizer must be a torch.optim.Optimizer instance")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        if threshold_mode not in THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {THRESHOLD_MODES}, got {threshold_mode!r}"
            )
        if weight_by not in WEIGHTINGS:
            raise ValueError(f"weight_by must be one of {WEIGHTINGS}, got {weight_by!r}")
        if not callable(num_mutations) and num_mutations < 0:
            raise ValueError("num_mutations must be non-negative")

        self.base_optimizer = base_optimizer
        self.mutator = mutator if mutator is not None else NormalMutator()
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.num_mutations = num_mutations if callable(num_mutations) else float(num_mutations)
        self.temperature = float(temperature)
        self.gate = gate
        self.weight_by = weight_by
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
        prob = {
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "temperature": self.temperature,
            "gate": self.gate,
            "weight_by": self.weight_by,
            "mutation_prob": self.mutation_prob,
            "total_mutated": self.total_mutated,
            "num_steps": self.num_steps,
        }
        if not callable(self.num_mutations):
            prob["num_mutations"] = self.num_mutations
        return {"base": self.base_optimizer.state_dict(), "probabilistic": prob}

    def load_state_dict(self, state_dict):  # type: ignore[override]
        self.base_optimizer.load_state_dict(state_dict["base"])
        for k, v in state_dict.get("probabilistic", {}).items():
            setattr(self, k, v)

    def __repr__(self):
        nm = "adaptive" if callable(self.num_mutations) else self.num_mutations
        return (
            f"Probabilistic({self.base_optimizer.__class__.__name__}, "
            f"mutator={self.mutator.__class__.__name__}, gate={self.gate!r}, "
            f"weight_by={self.weight_by!r}, threshold={self.threshold}, "
            f"num_mutations={nm}, temperature={self.temperature})"
        )

    # -- The interesting part ------------------------------------------------
    def _gen_for(self, x: Tensor) -> Optional[torch.Generator]:
        """Return ``self.generator`` only if it matches ``x``'s device.

        A CPU generator cannot seed sampling for an MPS/CUDA tensor, so in that
        case we fall back to the global RNG (``None``) to keep runs portable.
        """
        g = self.generator
        if g is not None and g.device != x.device:
            return None
        return g

    def _rand_like(self, x: Tensor, generator: Optional[torch.Generator]) -> Tensor:
        return torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    def _boundary(self, mag: Tensor) -> float:
        if self.threshold_mode == "abs":
            return self.threshold
        q = min(max(self.threshold, 0.0), 1.0)  # quantile fraction
        try:
            return float(torch.quantile(mag, q))
        except Exception:
            # torch.quantile is unsupported on some backends (e.g. MPS); fall
            # back to a CPU computation.
            return float(torch.quantile(mag.detach().cpu().float(), q))

    def _eligible(self, mag: Tensor) -> Tensor:
        if self.gate == "none":
            return torch.ones_like(mag, dtype=torch.bool)
        boundary = self._boundary(mag)
        return mag > boundary if self.gate == "high" else mag < boundary

    def _expected_count(self, grad: Tensor, eligible: Tensor) -> float:
        nm = self.num_mutations
        return float(nm(grad, eligible)) if callable(nm) else float(nm)

    @torch.no_grad()
    def mutation_probabilities(self, grad: Tensor) -> Tensor:
        """Per-entry Bernoulli mutation probability for one gradient tensor.

        This is the (optionally sign-flipped) softmax-of-gradient-magnitudes
        rule, gated by ``gate``/``threshold`` and scaled to an expected
        ``num_mutations`` entries per layer.  Exposed publicly so the behaviour
        can be inspected/plotted directly.
        """
        mag = grad.detach().abs().reshape(-1)
        eligible = self._eligible(mag)
        probs = torch.zeros_like(mag)
        if not bool(eligible.any()):
            return probs.reshape(grad.shape)

        # Softmax over eligible entries only; ineligible entries get -inf logits
        # so they receive zero probability mass and never dilute the others.
        # "uniform" ignores the gradient (equal logits) -- the control for
        # whether gradient weighting matters at all.
        if self.weight_by == "uniform":
            signed = torch.zeros_like(mag)
        else:
            signed = mag if self.weight_by == "grad" else -mag
        logits = signed / self.temperature
        logits = logits.masked_fill(~eligible, float("-inf"))
        soft = torch.softmax(logits, dim=0)

        # Scale so the expected count of mutations is ~num_mutations, clamped to
        # valid Bernoulli probabilities.
        k = self._expected_count(grad, eligible)
        bern = (soft * k).clamp_(0.0, 1.0)
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
                gen = self._gen_for(p)
                bern = self.mutation_probabilities(grad)
                mask = self._rand_like(bern, gen) < bern
                k = int(mask.sum())
                if k == 0:
                    continue

                flat_p = p.data.reshape(-1)
                flat_g = grad.detach().reshape(-1)
                flat_mask = mask.reshape(-1)
                selected_vals = flat_p[flat_mask]
                selected_grads = flat_g[flat_mask]
                new_vals = self.mutator(selected_vals, selected_grads, gen).to(
                    dtype=flat_p.dtype
                )
                flat_p[flat_mask] = new_vals
                num_mutated += k
        return num_mutated

    def _should_mutate(self) -> bool:
        nm = self.num_mutations
        if not callable(nm) and nm <= 0:
            return False
        if self.mutation_prob >= 1.0:
            return True
        return float(torch.rand((), generator=self.generator)) < self.mutation_prob

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
        if self._should_mutate():
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
    ``temperature``, ``gate``, ``weight_by``, ``mutation_prob``,
    ``per_step_hook``, ``generator``) are peeled off and forwarded to
    :class:`ProbabilisticOptimizer`; everything else goes to the base
    optimizer's constructor.
    """
    prob_keys = {
        "mutator",
        "threshold",
        "threshold_mode",
        "num_mutations",
        "temperature",
        "gate",
        "weight_by",
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
