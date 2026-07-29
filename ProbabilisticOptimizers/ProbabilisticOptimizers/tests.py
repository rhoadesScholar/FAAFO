# %%
"""Sanity tests for :mod:`ProbabilisticOptimizers`.

Run with ``pytest`` or directly as ``python -m ProbabilisticOptimizers.tests``.
"""
import torch

from .mutations import (
    ChaoticMutator,
    NormalMutator,
    UniformMutator,
    CallableMutator,
)
from .optimizer import ProbabilisticOptimizer, make_probabilistic


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_softmax_probabilities_favor_large_gradients():
    base = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    opt = ProbabilisticOptimizer(base, threshold=0.0, num_mutations=1.0)
    grad = torch.tensor([0.1, 10.0, 0.1])
    probs = opt.mutation_probabilities(grad)
    assert probs.shape == grad.shape
    # The large-gradient entry should be the most likely to mutate.
    assert probs.argmax().item() == 1
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_threshold_gates_eligibility():
    base = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    opt = ProbabilisticOptimizer(base, threshold=1.0, num_mutations=5.0)
    grad = torch.tensor([0.5, 2.0, 0.9, 3.0])
    probs = opt.mutation_probabilities(grad)
    # Sub-threshold entries get exactly zero probability.
    assert probs[0].item() == 0.0
    assert probs[2].item() == 0.0
    assert probs[1].item() > 0.0
    assert probs[3].item() > 0.0


def test_expected_mutation_count_scales_with_num_mutations():
    # With uniform gradients, softmax is uniform, so expected mutations ~ num_mutations.
    n = 1000
    base = torch.optim.SGD([torch.zeros(n, requires_grad=True)], lr=0.1)
    opt = ProbabilisticOptimizer(base, threshold=0.0, num_mutations=10.0)
    grad = torch.ones(n)
    probs = opt.mutation_probabilities(grad)
    assert abs(probs.sum().item() - 10.0) < 1e-3


def test_step_mutates_and_delegates_update():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(200))
    base = torch.optim.SGD([p], lr=1.0)
    opt = ProbabilisticOptimizer(
        base,
        mutator=NormalMutator(std=1.0),
        threshold=0.0,
        num_mutations=50.0,
        generator=_gen(1),
    )
    p.grad = torch.ones(200)  # constant, above threshold
    before = p.detach().clone()
    opt.step()
    # SGD moved everything by -lr*grad = -1; some entries then got resampled.
    assert opt.last_num_mutated > 0
    changed_from_sgd = (p.detach() != before).sum().item()
    assert changed_from_sgd == 200


def test_no_grad_no_mutation():
    p = torch.nn.Parameter(torch.zeros(10))
    base = torch.optim.SGD([p], lr=1.0)
    opt = ProbabilisticOptimizer(base, num_mutations=10.0)
    opt.step()  # p.grad is None
    assert opt.last_num_mutated == 0


def test_num_mutations_zero_is_plain_optimizer():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(50))
    base = torch.optim.SGD([p], lr=0.5)
    opt = ProbabilisticOptimizer(base, num_mutations=0.0)
    p.grad = torch.ones(50)
    opt.step()
    assert opt.last_num_mutated == 0
    assert torch.allclose(p.detach(), torch.full((50,), -0.5))


def test_reproducible_with_generator():
    def run():
        torch.manual_seed(7)
        p = torch.nn.Parameter(torch.zeros(500))
        base = torch.optim.SGD([p], lr=0.1)
        opt = ProbabilisticOptimizer(
            base, mutator=NormalMutator(std=1.0),
            num_mutations=20.0, generator=_gen(123),
        )
        p.grad = torch.linspace(0.1, 5.0, 500)
        opt.step()
        return p.detach().clone(), opt.last_num_mutated

    a_vals, a_n = run()
    b_vals, b_n = run()
    assert a_n == b_n
    assert torch.equal(a_vals, b_vals)


def test_chaotic_mutator_deterministic_and_bounded():
    m = ChaoticMutator(r=3.99, iterations=5, scale=2.0)
    vals = torch.linspace(-3, 3, 64)
    grads = torch.ones(64)
    out1 = m(vals, grads)
    out2 = m(vals, grads)
    assert torch.equal(out1, out2)  # deterministic
    assert torch.all(out1.abs() <= 2.0 + 1e-5)  # bounded by scale


def test_uniform_mutator_range():
    m = UniformMutator(low=-0.5, high=0.5)
    vals = torch.zeros(1000)
    out = m(vals, torch.ones(1000), _gen(0))
    assert out.min() >= -0.5 and out.max() <= 0.5


def test_callable_mutator():
    m = CallableMutator(lambda v, g, gen: torch.full_like(v, 42.0))
    out = m(torch.zeros(5), torch.ones(5))
    assert torch.all(out == 42.0)


def test_additive_vs_replace():
    vals = torch.full((100,), 5.0)
    replace = NormalMutator(std=0.0, mean=1.0, additive=False)(vals, torch.ones(100))
    perturb = NormalMutator(std=0.0, mean=1.0, additive=True)(vals, torch.ones(100))
    assert torch.allclose(replace, torch.ones(100))       # value discarded
    assert torch.allclose(perturb, torch.full((100,), 6.0))  # value + 1


def test_make_probabilistic_factory():
    ProbAdam = make_probabilistic(torch.optim.Adam)
    p = torch.nn.Parameter(torch.zeros(10))
    opt = ProbAdam([p], lr=1e-2, num_mutations=1.0, mutator=NormalMutator())
    assert isinstance(opt, ProbabilisticOptimizer)
    assert isinstance(opt.base_optimizer, torch.optim.Adam)
    p.grad = torch.ones(10)
    opt.step()  # should not raise


def test_state_dict_roundtrip():
    p = torch.nn.Parameter(torch.zeros(10))
    base = torch.optim.Adam([p], lr=1e-2)
    opt = ProbabilisticOptimizer(base, threshold=0.5, num_mutations=3.0)
    p.grad = torch.ones(10)
    opt.step()
    sd = opt.state_dict()

    p2 = torch.nn.Parameter(torch.zeros(10))
    base2 = torch.optim.Adam([p2], lr=1e-2)
    opt2 = ProbabilisticOptimizer(base2)
    opt2.load_state_dict(sd)
    assert opt2.threshold == 0.5
    assert opt2.num_mutations == 3.0


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _main()
