"""AdaptiveLossBalancing.

Adaptive per-loss weighting for multi-loss training, wrapping any
``torch.optim.Optimizer``. Each loss is normalised by its own scale, and an
exponent ``alpha`` steers weight toward losses that are lagging behind.

Example::

    import torch
    from AdaptiveLossBalancing import MultiLossOptimizer

    opt = MultiLossOptimizer(
        torch.optim.Adam(model.parameters(), lr=1e-3),
        names=["recon", "kl"],
        alpha=1.0,             # >0: favour lagging losses; -1: equalise now; 0: initial scale
        clip_grad_norm=None,   # optional gradient clipping, off by default
    )
    stats = opt.step({"recon": recon_loss, "kl": kl_loss})
"""
from .balancer import LossBalancer, MultiLossOptimizer

__all__ = ["LossBalancer", "MultiLossOptimizer"]

__version__ = "1.0.0"
