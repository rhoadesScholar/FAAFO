"""ProbabilisticOptimizers.

A family of optimizers that wrap any off-the-shelf ``torch.optim.Optimizer`` and
inject a hook that probabilistically resamples high-gradient parameters, per
layer, using a softmax over gradient magnitudes.

Example::

    import torch
    from ProbabilisticOptimizers import ProbabilisticOptimizer, ChaoticMutator

    base = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = ProbabilisticOptimizer(
        base,
        mutator=ChaoticMutator(additive=True, strength=1e-2),
        threshold=1e-3,
        num_mutations=2.0,
        temperature=1.0,
    )
    # ... use `opt` exactly like a normal optimizer.
"""
from .mutation_counts import (
    Fixed,
    FractionOverGate,
    GradientScaled,
    MutationCount,
)
from .mutations import (
    CallableMutator,
    ChaoticMutator,
    Mutator,
    NormalMutator,
    UniformMutator,
)
from .optimizer import ProbabilisticOptimizer, make_probabilistic

__all__ = [
    "ProbabilisticOptimizer",
    "make_probabilistic",
    "Mutator",
    "NormalMutator",
    "UniformMutator",
    "ChaoticMutator",
    "CallableMutator",
    "MutationCount",
    "Fixed",
    "FractionOverGate",
    "GradientScaled",
]

__version__ = "1.0.0"
