# %%
"""Compare every optimizer config over multiple seeds and plot the results.

This is the headline neural-network experiment: does gating on high vs. low vs.
no gradient (and adaptive mutation budgets) change training?

    python -m ProbabilisticOptimizers.compare --dataset synthetic --seeds 5 --epochs 8
    python -m ProbabilisticOptimizers.compare --dataset mnist --seeds 3 --epochs 3 --subset 8000

Writes ``training_comparison.png`` and ``training_comparison.json`` next to this
file.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import OPTIMIZER_CONFIGS, RunResult, get_device, train_one

HERE = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "adam": "Adam (baseline)",
    "gate_high": "gate=high (orig.)",
    "gate_none": "gate=none (dropped)",
    "gate_low": "gate=low (inverted)",
    "count_fixed": "count=fixed(4)",
    "count_meangrad": "count=mean-grad",
}
COLORS = {
    "adam": "#7f7f7f",
    "gate_high": "#1f77b4",
    "gate_none": "#9467bd",
    "gate_low": "#d62728",
    "count_fixed": "#2ca02c",
    "count_meangrad": "#ff7f0e",
}


def run(dataset="synthetic", seeds=5, epochs=8, lr=1e-3, batch_size=128,
        subset=None, configs=None, device=None):
    device = device or get_device()
    configs = configs or list(OPTIMIZER_CONFIGS)
    print(f"Device: {device} | dataset={dataset} | seeds={seeds} | epochs={epochs}")

    results: Dict[str, List[RunResult]] = {c: [] for c in configs}
    for cfg in configs:
        for s in range(seeds):
            res = train_one(cfg, dataset=dataset, epochs=epochs, lr=lr,
                            batch_size=batch_size, seed=s, device=device,
                            subset=subset)
            results[cfg].append(res)
            print(f"  {cfg:<22} seed={s}  final_acc={res.final_val_acc:.4f}  "
                  f"best={res.best_val_acc:.4f}  mut/step={res.mutated_per_step:.1f}")

    _report(results, configs)
    _plot(results, configs, dataset)
    _dump(results, configs, dataset, dict(seeds=seeds, epochs=epochs, lr=lr))
    return results


def _agg(results, cfg, key):
    return np.array([getattr(r, key) for r in results[cfg]], dtype=float)


def _report(results, configs):
    # Paired delta vs Adam (same seeds) cancels the large across-seed variance
    # in task difficulty, so small but consistent effects become visible.
    base_by_seed = None
    if "adam" in configs:
        base_by_seed = {r.seed: r.final_val_acc for r in results["adam"]}

    print("\n" + "=" * 82)
    print(f"{'config':<24}{'final acc':>14}{'best acc':>10}"
          f"{'Δ vs Adam (paired)':>22}{'mut/step':>12}")
    print("-" * 82)
    for cfg in configs:
        fa = _agg(results, cfg, "final_val_acc")
        ba = _agg(results, cfg, "best_val_acc")
        mut = _agg(results, cfg, "mutated_per_step").mean()
        if base_by_seed is not None and cfg != "adam":
            deltas = np.array([r.final_val_acc - base_by_seed[r.seed]
                               for r in results[cfg]])
            dcol = f"{100*deltas.mean():>+8.2f} ± {100*deltas.std():<6.2f}pts"
        else:
            dcol = ""
        print(f"{LABELS.get(cfg, cfg):<24}{fa.mean():>8.4f}±{fa.std():<5.4f}"
              f"{ba.mean():>10.4f}{dcol:>22}{mut:>12.1f}")
    print("=" * 82)


def _plot(results, configs, dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: validation-accuracy curves (mean ± std band across seeds).
    for cfg in configs:
        curves = np.stack([np.array(r.val_acc) for r in results[cfg]])
        x = np.arange(1, curves.shape[1] + 1)
        mean, std = curves.mean(0), curves.std(0)
        c = COLORS.get(cfg, None)
        ax1.plot(x, mean, label=LABELS.get(cfg, cfg), color=c, linewidth=2)
        ax1.fill_between(x, mean - std, mean + std, color=c, alpha=0.12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("Validation accuracy over training")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Right: paired delta vs Adam (same seeds), in accuracy points. This cancels
    # the large across-seed variance so consistent effects stand out.
    others = [c for c in configs if c != "adam"]
    base_by_seed = {r.seed: r.final_val_acc for r in results["adam"]} \
        if "adam" in configs else None
    if base_by_seed is not None and others:
        dmeans, dstds = [], []
        for c in others:
            d = np.array([r.final_val_acc - base_by_seed[r.seed] for r in results[c]])
            dmeans.append(100 * d.mean())
            dstds.append(100 * d.std())
        short = [LABELS.get(c, c).replace("gate=", "").replace("count=", "cnt=")
                 for c in others]
        bars = ax2.bar(short, dmeans, yerr=dstds, capsize=4,
                       color=[COLORS.get(c) for c in others])
        ax2.axhline(0, color=COLORS["adam"], linestyle="--", linewidth=1, alpha=0.8)
        ax2.set_ylabel("Δ final val-acc vs Adam (points, paired)")
        ax2.set_title("Effect vs baseline (same seeds)")
        ax2.tick_params(axis="x", labelsize=8, rotation=15)
        for b, m in zip(bars, dmeans):
            va = "bottom" if m >= 0 else "top"
            ax2.text(b.get_x() + b.get_width() / 2, m, f"{m:+.2f}",
                     ha="center", va=va, fontsize=8)
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    fig.suptitle(f"Probabilistic optimizers on {dataset}: gate & count ablation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, f"training_comparison_{dataset}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved plot to {out}")


def _dump(results, configs, dataset, meta):
    payload = {"dataset": dataset, "meta": meta, "results": {}}
    for cfg in configs:
        payload["results"][cfg] = {
            "final_val_acc_mean": float(_agg(results, cfg, "final_val_acc").mean()),
            "final_val_acc_std": float(_agg(results, cfg, "final_val_acc").std()),
            "best_val_acc_mean": float(_agg(results, cfg, "best_val_acc").mean()),
            "mutated_per_step": float(_agg(results, cfg, "mutated_per_step").mean()),
            "val_acc_curves": [r.val_acc for r in results[cfg]],
        }
    out = os.path.join(HERE, f"training_comparison_{dataset}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare probabilistic-optimizer configs.")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "mnist", "fashionmnist"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--configs", nargs="*", default=None,
                   help="Subset of configs to run (default: all).")
    args = p.parse_args(argv)
    run(dataset=args.dataset, seeds=args.seeds, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, subset=args.subset,
        configs=args.configs, device=get_device(args.device))


if __name__ == "__main__":
    main()
