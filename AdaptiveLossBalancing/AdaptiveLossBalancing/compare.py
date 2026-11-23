# %%
"""Benchmark: raw sum vs. tuned fixed weights vs. adaptive balancing.

Synthetic 3-task problem with a shared MLP trunk and very different loss scales:

* A: regression, target scale ~1 (easy)
* B: regression, target scale ~100 (hardest); optional 3% heavy-tailed label
  outliers that create per-batch loss spikes and an irreducible loss floor
* C: 4-way classification, cross-entropy

Score per task (lower is better): normalised test MSE (MSE / target variance)
for A and B, error rate for C. ``mean`` averages the three.

    python -m AdaptiveLossBalancing.compare                 # with outliers
    python -m AdaptiveLossBalancing.compare --no-outliers

Writes ``results_<setting>.json`` and ``results_<setting>.png`` next to this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .balancer import MultiLossOptimizer

HERE = os.path.dirname(os.path.abspath(__file__))
D, BATCH, LR, N_TRAIN = 16, 64, 1e-3, 20000
NAMES = ["A", "B", "C"]

# label -> kwargs for MultiLossOptimizer
METHODS = {
    "raw sum": dict(balance=False),
    "alpha=-1 (equalise now)": dict(alpha=-1.0),
    "alpha=0 (initial scale)": dict(alpha=0.0),
    "alpha=0.5": dict(alpha=0.5),
    "alpha=1 (default)": dict(alpha=1.0),
    "alpha=2": dict(alpha=2.0),
    "alpha=1 + clip 1.0": dict(alpha=1.0, clip_grad_norm=1.0),
    "per-batch (beta=0, alpha=-1)": dict(alpha=-1.0, beta=0.0),
}
GRID = list(itertools.product([1e-5, 1e-4, 1e-3, 1e-2], [0.3, 1.0, 3.0]))  # (w_B, w_C)


def make_data(seed: int, outliers: bool):
    g = torch.Generator().manual_seed(1000 + seed)
    W = torch.randn(D, 3, generator=g)

    def sample(n, noisy):
        x = torch.randn(n, D, generator=g)
        h = x @ W
        ya = torch.sin(h[:, 0]) + 0.5 * h[:, 1]
        yb = 100 * (torch.tanh(h[:, 1] * h[:, 2]) + 0.3 * h[:, 0])
        yc = (h[:, 0] > 0).long() * 2 + (h[:, 2] > 0).long()
        if noisy:
            mask = torch.rand(n, generator=g) < 0.03
            yb = yb + mask * 3000 * torch.randn(n, generator=g)
        return x, ya, yb, yc

    return sample(N_TRAIN, outliers), sample(2000, False), sample(2000, False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.a, self.b, self.c = nn.Linear(64, 1), nn.Linear(64, 1), nn.Linear(64, 4)

    def forward(self, x):
        z = self.trunk(x)
        return self.a(z).squeeze(-1), self.b(z).squeeze(-1), self.c(z)


def task_losses(model, batch):
    x, ya, yb, yc = batch
    pa, pb, pc = model(x)
    return {"A": F.mse_loss(pa, ya), "B": F.mse_loss(pb, yb), "C": F.cross_entropy(pc, yc)}


@torch.no_grad()
def score(model, data):
    x, ya, yb, yc = data
    pa, pb, pc = model(x)
    s = {
        "A": float(F.mse_loss(pa, ya) / ya.var()),
        "B": float(F.mse_loss(pb, yb) / yb.var()),
        "C": float((pc.argmax(-1) != yc).float().mean()),
    }
    s["mean"] = statistics.mean(s[n] for n in NAMES)
    return s


def run(seed: int, outliers: bool, steps: int, **opt_kwargs):
    """Train once; return (val_scores, test_scores)."""
    train, val, test = make_data(seed, outliers)
    torch.manual_seed(seed)
    model = Net()
    opt = MultiLossOptimizer(torch.optim.Adam(model.parameters(), LR), NAMES, **opt_kwargs)
    for _ in range(steps):
        idx = torch.randint(0, N_TRAIN, (BATCH,))
        opt.step(task_losses(model, [t[idx] for t in train]))
    return score(model, val), score(model, test)


def summarise(rs):
    return {
        k: {"mean": statistics.mean(r[k] for r in rs), "sd": statistics.stdev(r[k] for r in rs)}
        for k in NAMES + ["mean"]
    }


def plot(results, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results)
    means = [results[m]["scores"]["mean"]["mean"] for m in labels]
    sds = [results[m]["scores"]["mean"]["sd"] for m in labels]
    colors = ["#7f7f7f" if m == "raw sum" else "#2a9d8f" if "tuned" in m else "#457b9d"
              for m in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], means[::-1], xerr=sds[::-1], color=colors[::-1])
    ax.set_xlabel("mean task score (lower is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--no-outliers", action="store_true")
    args = p.parse_args()
    outliers = not args.no_outliers
    setting = "outliers" if outliers else "clean"
    seeds = range(args.seeds)
    t0 = time.time()

    results = {}
    for label, kw in METHODS.items():
        tests = [run(s, outliers, args.steps, **kw)[1] for s in seeds]
        results[label] = {"runs_per_seed": 1, "scores": summarise(tests)}

    tuned, picks = [], []
    for s in seeds:
        runs = [(wb, wc, run(s, outliers, args.steps, balance=False,
                             priorities={"B": wb, "C": wc})) for wb, wc in GRID]
        wb, wc, (_, test) = min(runs, key=lambda r: r[2][0]["mean"])  # select on val
        tuned.append(test)
        picks.append({"B": wb, "C": wc})
    label = f"tuned fixed ({len(GRID)}-run grid)"
    results[label] = {"runs_per_seed": len(GRID), "scores": summarise(tuned), "picks": picks}

    print(f"\nsetting={setting}, {args.seeds} seeds, {args.steps} steps. "
          "Lower is better; mean ± sd over seeds.\n")
    print(f"{'method':30s} {'runs':>4s} " + " ".join(f"{k:>13s}" for k in NAMES + ["mean"]))
    for label, r in results.items():
        cells = [f"{r['scores'][k]['mean']:.3f}±{r['scores'][k]['sd']:.3f}" for k in NAMES + ["mean"]]
        print(f"{label:30s} {r['runs_per_seed']:>4d} " + " ".join(f"{c:>13s}" for c in cells))
    print(f"\n(wall time {time.time() - t0:.0f}s)")

    with open(os.path.join(HERE, f"results_{setting}.json"), "w") as f:
        json.dump({"setting": setting, "seeds": args.seeds, "steps": args.steps,
                   "results": results}, f, indent=2)
    plot(results, os.path.join(HERE, f"results_{setting}.png"),
         f"3-task benchmark ({setting}), {args.seeds} seeds")


if __name__ == "__main__":
    main()
