# Probabilistic Optimizers

This repository contains the code for the study "Probabilistic Optimizers:
gradient-weighted parameter resampling as a drop-in exploration hook" by Jeff
Rhoades (@rhoadesScholar), submitted to the FAAFO Consortium of Rhoades.

## Idea

Take **any** off-the-shelf optimizer (Adam, SGD, ...) and, after each ordinary
update, inject a hook that randomly **resamples ("mutates") individual
parameters, per layer**. The recipe, applied to each parameter tensor:

1. Look at the per-entry gradient magnitude `|g|`.
2. Only entries whose `|g|` exceeds a **threshold** are *eligible* to mutate —
   the still-moving, contested entries.
3. Turn the eligible magnitudes into probabilities with a temperature-scaled
   **softmax over the layer**, so high-gradient entries get more probability
   mass.
4. Draw independent Bernoulli mutation decisions from those probabilities,
   scaled so that ~`num_mutations` entries mutate per layer in expectation.
5. Replace (or perturb) the chosen entries with a **mutator**: a draw from a
   distribution (Gaussian, uniform), a **deterministic chaotic function** (the
   logistic map), or any callable you supply.

The result is a *family* of optimizers — one per (base optimizer × mutator ×
schedule) combination — that behave like the wrapped optimizer but with a tunable
amount of gradient-weighted stochastic exploration folded in.

```python
import torch
from ProbabilisticOptimizers import ProbabilisticOptimizer, ChaoticMutator

base = torch.optim.Adam(model.parameters(), lr=1e-3)
opt = ProbabilisticOptimizer(
    base,
    mutator=ChaoticMutator(additive=True, strength=1e-2),
    threshold=1e-3,      # only mutate high-gradient entries
    num_mutations=2.0,   # ~2 mutated entries per layer per step (in expectation)
    temperature=1.0,     # softmax temperature over |grad|
)
# ...then use `opt` exactly like a normal optimizer.
```

There is also a class factory for the "swap `Adam` for `ProbabilisticAdam`" style:

```python
from ProbabilisticOptimizers import make_probabilistic
ProbAdam = make_probabilistic(torch.optim.Adam)
opt = ProbAdam(model.parameters(), lr=1e-3, num_mutations=2.0)
```

## What the mutation rule actually does

A subtle but important consequence of gating on **high** gradients: at a local
minimum the gradients are ~0 everywhere, so *nothing is eligible* and no mutation
happens. This method therefore does **not** kick a settled model out of a
minimum. Instead it injects exploration **during active descent** — high
gradient magnitude behaves like a high "temperature", much as in simulated
annealing — and biases that exploration toward the entries that are currently
moving the most. Best-so-far tracking then keeps whatever good basin the noisy
trajectory happens to pass through.

## Mutators

| Mutator | Behaviour |
| --- | --- |
| `NormalMutator(std, mean, relative, additive)` | Draw from `Normal(mean, std)`; `relative` scales `std` by `|weight|`. |
| `UniformMutator(low, high, additive)` | Draw from `Uniform(low, high)`. |
| `ChaoticMutator(r, iterations, scale, additive)` | Deterministic logistic-map chaos, seeded from `weight + grad`. No PRNG. |
| `CallableMutator(fn, additive)` | Wrap any `fn(values, grads, generator)`. |

All mutators support `additive=True` (perturb the current value) vs. the default
`additive=False` (fully replace it), and a `strength` multiplier that can be
annealed over training.

## Findings

Benchmark: minimise the **Rastrigin** function (a grid of deep local minima
around a single global minimum at 0) from 60 random starts, wrapping Adam
(`lr=0.05`, 600 steps) with `threshold=0.5`, `num_mutations=3`, `temperature=2`,
and mutation magnitude annealed to 2% over training. We report the best-so-far
loss reached (lower is better).

![Rastrigin benchmark](ProbabilisticOptimizers/rastrigin_benchmark.png)

| Method | median final | mean final | escape rate (<1.0) |
| --- | ---: | ---: | ---: |
| Adam (baseline) | 25.4 | 24.4 | 3.3% |
| Prob-Adam + Gaussian | 20.5 (**−19%**) | 23.5 | 0% |
| Prob-Adam + Uniform | 17.5 (**−31%**) | 23.0 | 0% |
| Prob-Adam + Chaotic | 47.1 (**+86%**) | 48.3 | 0% |

The result is **mixed, and instructive**:

* **Gradient-weighted Gaussian/uniform resampling improves the typical case.**
  It cuts the *median* final loss by ~20–30% relative to vanilla Adam: the
  gradient-biased noise nudges the descent trajectory into modestly better local
  basins more often than not.
* **It does not improve the rate of finding the global optimum** (escape rate
  stays ~0). This follows directly from the design: because only *high-gradient*
  entries are eligible to mutate, at a settled local minimum — where all
  gradients are ~0 — nothing is eligible, so the hook cannot kick a converged
  model out of a basin. The exploration all happens *during* descent, not after
  it stalls. (Adam's lone 3.3% is 2/60 lucky starts, i.e. noise.)
* **The deterministic chaotic mutator hurts** (+86% median loss). Its
  structured, geometry-blind jumps are a poor match for this landscape; simple
  distribution draws beat it handily.

Takeaway for the "find out" ledger: framing mutation eligibility on *high*
gradient magnitude makes this a **descent-time exploration** trick (a
gradient-aware cousin of injecting annealed noise) rather than a
minimum-escaping one. If escaping stalled minima is the goal, the eligibility
rule should be inverted or the gate dropped — a natural next stunt.

## Setup

```bash
micromamba env create -n probopt python==3.11 -f requirements.txt -c pytorch -c nvidia -y
micromamba activate probopt
pip install -e .
```

## Reproduce

```bash
python -m ProbabilisticOptimizers.experiment   # Rastrigin escape benchmark -> rastrigin_benchmark.png
python -m ProbabilisticOptimizers.tests        # sanity tests
```
