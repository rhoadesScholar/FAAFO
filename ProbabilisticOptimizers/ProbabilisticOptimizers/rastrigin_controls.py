# %%
"""Was the Rastrigin "improvement" real, or an artifact of best-so-far?

The headline Rastrigin number (see :mod:`.experiment`) reported the **best-so-far**
loss, evaluated at the *mutated* point each step.  That metric is suspect:
injecting noise and keeping the luckiest point you ever pass through is a
"best-of-N-samples" estimator -- more noise means more distinct points visited,
so best-so-far drops almost mechanically, whether or not the noise is useful.
Tellingly, when the *final settled* loss was reported instead, the mutated runs
looked **worse**.

This script disentangles the effect with matched controls, all on the same
Rastrigin task and seeds:

* ``adam``           -- plain Adam.  Best-so-far over its own (smooth) trajectory.
* ``adam_probe``     -- plain Adam trajectory, but at each step we also evaluate a
                        *jittered shadow* of the weights (a uniform-position
                        perturbation, matched in count and magnitude) and let it
                        count toward best-so-far.  Pure best-of-N: extra lucky
                        probes, zero real exploration.
* ``mutate_grad``    -- the actual method: gradient-weighted mutation, committed.
* ``mutate_uniform`` -- identical machinery but positions chosen *uniformly at
                        random* (``weight_by="uniform"``).  Isolates whether the
                        gradient weighting does anything.

For each we report both **best-so-far** (the flattering metric) and **final**
(the settled, deployable iterate).

Run: ``python -m ProbabilisticOptimizers.rastrigin_controls``
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .mutations import NormalMutator
from .optimizer import ProbabilisticOptimizer

HERE = os.path.dirname(os.path.abspath(__file__))
A = 10.0

# Shared mutation dose (constant -- no annealing, so grad/uniform/probe are
# perfectly matched and the only difference is *where* the noise lands).
STD, NUM_MUT, THRESH, TEMP = 1.0, 3.0, 0.5, 2.0


def rastrigin(x):
    return A * x.numel() + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))


def _annealer(total_steps, final_frac=0.02):
    """Exponentially anneal the mutator strength, matching :mod:`.experiment`."""
    decay = final_frac ** (1.0 / max(total_steps - 1, 1))
    state = {}

    def hook(opt):
        if "s0" not in state:
            state["s0"] = getattr(opt.mutator, "strength", 1.0)
        opt.mutator.strength = max(opt.mutator.strength * decay, state["s0"] * final_frac)

    return hook


def _prob_opt(weight_by, x, gen, lr=0.05, hook=None):
    base = torch.optim.Adam([x], lr=lr)
    return ProbabilisticOptimizer(
        base, mutator=NormalMutator(std=STD, additive=True),
        threshold=THRESH, num_mutations=NUM_MUT, temperature=TEMP,
        gate="high", weight_by=weight_by, generator=gen, per_step_hook=hook,
    )


def _apply_mutation_clone(x, grad, prob_opt, gen):
    """Return a mutated *copy* of x (x itself is untouched) using prob_opt's rule."""
    xf = x.detach().clone().reshape(-1)
    bern = prob_opt.mutation_probabilities(grad).reshape(-1)
    mask = torch.rand(xf.shape, generator=gen) < bern
    if bool(mask.any()):
        xf[mask] = prob_opt.mutator(xf[mask], grad.reshape(-1)[mask], gen)
    return xf.reshape(x.shape)


def optimize(method, x0, steps, lr, seed):
    gen = torch.Generator().manual_seed(seed + 10_000)
    x = torch.nn.Parameter(x0.clone())

    committed_mut = method in ("mutate_grad", "mutate_uniform")
    weight_by = "grad" if method == "mutate_grad" else "uniform"

    # Mutation magnitude is annealed for every method (matching .experiment), so
    # the committed methods actually settle and the probe stays magnitude-matched.
    anneal = _annealer(steps)
    if committed_mut:
        opt = _prob_opt(weight_by, x, gen, lr=lr, hook=anneal)
        probe = None
    else:
        opt = torch.optim.Adam([x], lr=lr)
        # A uniform-mutation rule used only to build shadow probes for adam_probe;
        # annealed by hand each step to stay matched to the committed methods.
        probe = _prob_opt("uniform", x, gen, lr=lr) if method == "adam_probe" else None

    best = float("inf")
    curve = np.empty(steps)
    for t in range(steps):
        opt.zero_grad()
        loss = rastrigin(x)
        loss.backward()
        grad = x.grad.detach().clone()
        opt.step()  # for mutate_*, this also commits the (annealed) mutation to x

        visited = float(rastrigin(x).detach())  # committed iterate this step
        if method == "adam_probe":
            xs = _apply_mutation_clone(x, grad, probe, gen)
            visited = min(visited, float(rastrigin(xs).detach()))
            anneal(probe)  # keep the probe magnitude in lockstep
        best = min(best, visited)
        curve[t] = best

    with torch.no_grad():
        final = float(rastrigin(x).detach())
    return curve, best, final


def run(dims=3, n_seeds=100, steps=600, lr=0.05):
    methods = ["adam", "adam_probe", "mutate_grad", "mutate_uniform"]
    curves, bests, finals = {}, {}, {}
    for m in methods:
        cc = np.empty((n_seeds, steps))
        bb = np.empty(n_seeds)
        ff = np.empty(n_seeds)
        for s in range(n_seeds):
            g = torch.Generator().manual_seed(s)
            x0 = (torch.rand(dims, generator=g) * 2 - 1) * 5.12
            c, b, f = optimize(m, x0, steps, lr, s)
            cc[s], bb[s], ff[s] = c, b, f
        curves[m], bests[m], finals[m] = cc, bb, ff

    labels = {
        "adam": "Adam",
        "adam_probe": "Adam + jitter-probe (best-of-N control)",
        "mutate_grad": "Mutation (gradient-weighted)",
        "mutate_uniform": "Mutation (uniform position)",
    }
    print(f"\nRastrigin controls  (dims={dims}, seeds={n_seeds}, steps={steps})")
    print("-" * 78)
    print(f"{'method':<42}{'best-so-far':>16}{'final':>16}")
    print("-" * 78)
    for m in methods:
        print(f"{labels[m]:<42}{np.median(bests[m]):>16.2f}{np.median(finals[m]):>16.2f}")
    print("-" * 78)
    print("(medians over seeds; 'best-so-far' = the flattering metric, "
          "'final' = settled iterate)")

    _plot(methods, labels, curves, bests, finals)
    return dict(bests=bests, finals=finals)


def _plot(methods, labels, curves, bests, finals):
    colors = {
        "adam": "#7f7f7f",
        "adam_probe": "#000000",
        "mutate_grad": "#1f77b4",
        "mutate_uniform": "#ff7f0e",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    steps = curves["adam"].shape[1]
    x = np.arange(steps)
    for m in methods:
        med = np.median(curves[m], axis=0)
        ls = "--" if m == "adam_probe" else "-"
        ax1.plot(x, med, label=labels[m], color=colors[m], linewidth=2, linestyle=ls)
    ax1.set_yscale("log")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Best-so-far Rastrigin loss (median)")
    ax1.set_title("Best-so-far: mutation vs. a pure best-of-N control")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    xpos = np.arange(len(methods))
    w = 0.38
    best_med = [np.median(bests[m]) for m in methods]
    final_med = [np.median(finals[m]) for m in methods]
    ax2.bar(xpos - w / 2, best_med, w, label="best-so-far", color="#4c72b0")
    ax2.bar(xpos + w / 2, final_med, w, label="final (settled)", color="#c44e52")
    ax2.axhline(np.median(finals["adam"]), color="#7f7f7f", linestyle="--",
                linewidth=1, alpha=0.8, label="Adam final")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(["Adam", "Adam\n+probe", "mut\ngrad", "mut\nunif"], fontsize=9)
    ax2.set_ylabel("Median Rastrigin loss")
    ax2.set_title("Best-so-far flatters; final tells the truth")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle("Rastrigin: is the mutation gain real or a best-so-far artifact?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "rastrigin_controls.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
