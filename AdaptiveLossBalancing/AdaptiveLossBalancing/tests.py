# %%
"""Sanity tests for :mod:`AdaptiveLossBalancing`.

Run with ``pytest AdaptiveLossBalancing/tests.py`` from the project folder.
"""
import pytest
import torch

from .balancer import LossBalancer, MultiLossOptimizer


def t(x):
    return torch.tensor(float(x), requires_grad=True)


def feed(b, seq):
    """Feed a sequence of {name: value} dicts; return stats from the last step."""
    for vals in seq:
        _, s = b({k: t(v) for k, v in vals.items()})
    return s


def test_first_step_equalises_contributions():
    b = LossBalancer(["a", "b"])
    total, s = b({"a": t(2.0), "b": t(2000.0)})
    assert s["weight/a"] == pytest.approx(0.5)
    assert s["weight/b"] == pytest.approx(1 / 2000)
    assert float(total.detach()) == pytest.approx(2.0)  # each contributes ~1


def test_weight_uses_past_not_current_batch():
    b = LossBalancer(["a"], warmup=5)
    b({"a": t(1.0)})
    _, s = b({"a": t(100.0)})  # spike: this step's weight must come from history
    assert s["weight/a"] == pytest.approx(1.0)


def test_warmup_reference_is_mean_then_frozen():
    b = LossBalancer(["a"], warmup=3, alpha=0.0)
    feed(b, [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}, {"a": 50.0}, {"a": 50.0}])
    assert b.ref["a"] == pytest.approx(2.0)
    assert b.weights()["a"] == pytest.approx(0.5)  # alpha=0: fixed at 1/ref


def _two_task_run(alpha):
    """'easy' drops 100x, 'hard' only 2x; return final weight ratio easy/hard."""
    b = LossBalancer(["easy", "hard"], alpha=alpha, beta=0.5, warmup=2, max_ratio=1e9)
    feed(b, [{"easy": 1.0, "hard": 1.0}] * 2 + [{"easy": 0.01, "hard": 0.5}] * 30)
    w = b.weights()
    return w["easy"] / w["hard"]


def test_alpha_controls_who_gains_weight():
    assert _two_task_run(-1.0) == pytest.approx(50.0, rel=1e-3)  # easy task upweighted
    assert _two_task_run(0.0) == pytest.approx(1.0)               # unchanged
    assert _two_task_run(1.0) == pytest.approx(1 / 50.0, rel=1e-3)  # lagging task upweighted


def test_max_ratio_caps_progress_factor():
    b = LossBalancer(["easy", "hard"], alpha=1.0, beta=0.5, warmup=2, max_ratio=2.0)
    feed(b, [{"easy": 1.0, "hard": 1.0}] * 2 + [{"easy": 0.01, "hard": 0.5}] * 30)
    w = b.weights()
    # r_mean = 0.255; hard factor 0.5/0.255 = 1.96 (uncapped), easy 0.039 -> capped at 0.5
    assert w["hard"] / w["easy"] == pytest.approx((0.5 / 0.255) / 0.5, rel=1e-3)


def test_priorities_and_negative_losses():
    b = LossBalancer(["nll", "mse"], priorities={"nll": 3.0})
    _, s = b({"nll": t(-4.0), "mse": t(0.5)})
    assert s["weight/nll"] == pytest.approx(3.0 / 4.0)
    assert s["weight/mse"] == pytest.approx(2.0)


def test_disabled_is_plain_weighted_sum():
    b = LossBalancer(["a", "b"], priorities={"b": 0.1}, enabled=False)
    total, _ = b({"a": t(5.0), "b": t(10.0)})
    assert float(total.detach()) == pytest.approx(6.0)


def test_no_grad_through_weights():
    x = t(4.0)
    b = LossBalancer(["a"])
    total, _ = b({"a": x * x})
    total.backward()
    assert float(x.grad) == pytest.approx(2 * 4.0 / 16.0)  # w = 1/16 treated as constant


def test_bad_arguments_raise():
    b = LossBalancer(["a", "b"])
    with pytest.raises(KeyError):
        b({"a": t(1.0)})
    with pytest.raises(ValueError):
        LossBalancer(["a"], priorities={"z": 1.0})
    with pytest.raises(ValueError):
        LossBalancer(["a", "a"])
    with pytest.raises(ValueError):
        LossBalancer(["a"], beta=1.0)


def test_optimizer_clip_and_state_roundtrip():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    opt = MultiLossOptimizer(torch.optim.Adam(model.parameters(), 1e-2), ["a", "b"],
                             clip_grad_norm=0.5, alpha=1.0)
    for _ in range(3):
        y = model(torch.randn(8, 3))
        stats = opt.step({"a": y.pow(2).mean(), "b": 1000 * y.abs().mean()})
    assert "grad_norm" in stats

    opt2 = MultiLossOptimizer(torch.optim.Adam(model.parameters(), 1e-2), ["a", "b"])
    opt2.load_state_dict(opt.state_dict())
    assert opt2.balancer.weights() == opt.balancer.weights()
    assert opt2.clip_grad_norm == 0.5


def test_clipping_off_by_default():
    model = torch.nn.Linear(2, 1)
    opt = MultiLossOptimizer(torch.optim.SGD(model.parameters(), 0.1), ["a"])
    stats = opt.step({"a": model(torch.ones(1, 2)).sum() ** 2})
    assert "grad_norm" not in stats
