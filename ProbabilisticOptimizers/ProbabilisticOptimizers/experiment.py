# %%
"""Does gradient-weighted probabilistic mutation help escape local minima?

Testbed: the **Rastrigin** function, a standard highly-multimodal benchmark with
a single global minimum at the origin (value 0) surrounded by a grid of deep
local minima.  We treat the input coordinates as a single parameter tensor (one
"layer") and let each optimizer minimise it from many random starting points.

We compare a vanilla Adam baseline against several probabilistic Adam variants
that differ only in their mutator (Gaussian perturbation, chaotic logistic map,
uniform jump).  For each we record how often the optimizer gets within a small
distance of the global minimum ("escape rate") and the median convergence curve.

Run: ``python -m ProbabilisticOptimizers.experiment``
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .mutations import ChaoticMutator, NormalMutator, UniformMutator
from .optimizer import ProbabilisticOptimizer

HERE = os.path.dirname(os.path.abspath(__file__))
A = 10.0  # Rastrigin amplitude


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Rastrigin function; global minimum 0 at x = 0."""
    n = x.numel()
    return A * n + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))


def _annealer(total_steps, final_frac=0.02):
    """Return a per-step hook that exponentially anneals mutation magnitude.

    Exploration is loud early (to hop between Rastrigin basins) and quiet late
    (to settle into whichever basin it found).  We decay both the expected
    number of mutations and the mutator's ``strength`` toward ``final_frac`` of
    their initial values over training.
    """
    decay = final_frac ** (1.0 / max(total_steps - 1, 1))
    state = {"n0": None, "s0": None}

    def hook(opt):
        if state["n0"] is None:
            state["n0"] = opt.num_mutations
            state["s0"] = getattr(opt.mutator, "strength", 1.0)
        opt.num_mutations = max(opt.num_mutations * decay, state["n0"] * final_frac)
        if hasattr(opt.mutator, "strength"):
            opt.mutator.strength = max(
                opt.mutator.strength * decay, state["s0"] * final_frac
            )

    return hook


def make_optimizer(name, params, lr, generator, steps=600):
    """Return (optimizer, label) for a named configuration."""
    base = torch.optim.Adam(params, lr=lr)
    if name == "adam":
        return base, "Adam (baseline)"

    common = dict(
        threshold=0.5,
        num_mutations=3.0,
        temperature=2.0,
        generator=generator,
        per_step_hook=_annealer(steps),
    )
    if name == "normal":
        return (
            ProbabilisticOptimizer(
                base, mutator=NormalMutator(std=1.0, additive=True), **common
            ),
            "Prob-Adam + Gaussian",
        )
    if name == "chaotic":
        return (
            ProbabilisticOptimizer(
                base,
                mutator=ChaoticMutator(r=3.99, iterations=5, scale=1.5, additive=True),
                **common,
            ),
            "Prob-Adam + Chaotic",
        )
    if name == "uniform":
        return (
            ProbabilisticOptimizer(
                base, mutator=UniformMutator(low=-1.5, high=1.5, additive=True), **common
            ),
            "Prob-Adam + Uniform",
        )
    raise ValueError(name)


def optimize(name, x0, steps, lr, seed):
    """Minimise Rastrigin from x0; return (best-so-far curve, best final loss)."""
    gen = torch.Generator().manual_seed(seed + 10_000)
    x = torch.nn.Parameter(x0.clone())
    opt, _ = make_optimizer(name, [x], lr, gen, steps=steps)
    curve = np.empty(steps, dtype=np.float64)
    best = float("inf")
    for t in range(steps):
        opt.zero_grad()
        loss = rastrigin(x)
        loss.backward()
        opt.step()
        # Evaluate the *updated* point; track the best solution ever visited,
        # which is the natural readout for a stochastic global optimizer.
        with torch.no_grad():
            cur = float(rastrigin(x).detach())
        best = min(best, cur)
        curve[t] = best
    return curve, best


def run(dims=3, n_seeds=60, steps=600, lr=0.05, success_tol=1.0):
    methods = ["adam", "normal", "chaotic", "uniform"]
    labels, curves, finals = {}, {}, {}

    for name in methods:
        all_curves = np.empty((n_seeds, steps))
        all_finals = np.empty(n_seeds)
        for s in range(n_seeds):
            # Same start points across methods for a fair comparison.
            g = torch.Generator().manual_seed(s)
            x0 = (torch.rand(dims, generator=g) * 2 - 1) * 5.12
            c, f = optimize(name, x0, steps, lr, s)
            all_curves[s] = c
            all_finals[s] = f
        _, label = make_optimizer(name, [torch.zeros(1, requires_grad=True)], lr, None)
        labels[name] = label
        curves[name] = all_curves
        finals[name] = all_finals

    # -- Report ------------------------------------------------------------
    print(f"\nRastrigin escape benchmark  (dims={dims}, seeds={n_seeds}, "
          f"steps={steps}, lr={lr}, success<{success_tol})")
    print("-" * 68)
    print(f"{'method':<24}{'median':>10}{'mean':>10}{'best':>9}{'escape%':>10}")
    print("-" * 68)
    summary = {}
    for name in methods:
        f = finals[name]
        escape = 100.0 * np.mean(f < success_tol)
        summary[name] = (np.median(f), f.min(), escape, float(f.mean()))
        print(f"{labels[name]:<24}{np.median(f):>10.2f}{f.mean():>10.2f}"
              f"{f.min():>9.2f}{escape:>9.1f}%")
    print("-" * 68)

    _plot(methods, labels, curves, summary, steps, success_tol)
    return summary


def _plot(methods, labels, curves, summary, steps, success_tol):
    colors = {
        "adam": "#7f7f7f",
        "normal": "#1f77b4",
        "chaotic": "#d62728",
        "uniform": "#2ca02c",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(steps)
    for name in methods:
        c = curves[name]
        med = np.median(c, axis=0)
        q1, q3 = np.percentile(c, [25, 75], axis=0)
        ax1.plot(x, med, label=labels[name], color=colors[name], linewidth=2)
        ax1.fill_between(x, q1, q3, color=colors[name], alpha=0.12)
    ax1.set_yscale("log")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Rastrigin loss (median, IQR band)")
    ax1.set_title("Convergence")
    ax1.legend()
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    names = methods
    short = [labels[n].replace("Prob-Adam + ", "").replace(" (baseline)", "") for n in names]
    medians = [summary[n][0] for n in names]
    baseline = summary["adam"][0]
    bars = ax2.bar(short, medians, color=[colors[n] for n in names])
    ax2.axhline(baseline, color=colors["adam"], linestyle="--", linewidth=1,
                alpha=0.7, label="Adam median")
    ax2.set_ylabel("Median final Rastrigin loss (lower is better)")
    ax2.set_title("Typical-case solution quality")
    for b, m in zip(bars, medians):
        delta = 100.0 * (m - baseline) / baseline
        tag = f"{m:.1f}\n({delta:+.0f}%)"
        ax2.text(b.get_x() + b.get_width() / 2, m + 0.5, tag,
                 ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(medians) * 1.25)
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle("Probabilistic mutation vs. vanilla Adam on Rastrigin",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, "rastrigin_benchmark.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
