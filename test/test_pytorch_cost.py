import copy

import pytest

from decent_bench.costs import PyTorchCost
from decent_bench.utils.types import SupportedDevices

torch = pytest.importorskip("torch")


class _TinyMLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _make_dataset(n_samples: int = 19, input_size: int = 4, n_classes: int = 3):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(11)
    x = torch.randn((n_samples, input_size), generator=gen, dtype=torch.float32)
    y = torch.randint(0, n_classes, (n_samples,), generator=gen, dtype=torch.long)
    return [(x[i], y[i]) for i in range(n_samples)]


def _make_cost(
    dataset,
    *,
    max_batch_size: int,
    batch_size: int = 8,
    input_size: int = 4,
    hidden_size: int = 10,
    output_size: int = 3,
) -> PyTorchCost:
    torch.manual_seed(7)
    model = _TinyMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    return PyTorchCost(
        dataset=dataset,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=SupportedDevices.CPU,
    )


def test_per_sample_gradients_match_individual_gradients() -> None:
    dataset = _make_dataset()
    cost = _make_cost(dataset, max_batch_size=3, batch_size=10)

    indices = [5, 1, 7, 2, 0, 3, 4, 6]
    x = cost._get_model_parameters().detach().clone()  # noqa: SLF001

    per_sample = cost.gradient(x, indices=indices, reduction=None)
    batch_used_after_per_sample = list(cost.batch_used)
    expected = torch.stack([cost.gradient(x, indices=[i]) for i in indices], dim=0)

    assert batch_used_after_per_sample == indices
    assert per_sample.shape == expected.shape
    assert torch.allclose(per_sample, expected, rtol=5e-5, atol=1e-6)


def test_mean_gradient_matches_mean_of_per_sample_gradients() -> None:
    dataset = _make_dataset()
    cost = _make_cost(dataset, max_batch_size=4, batch_size=9)

    indices = list(range(13))
    x = cost._get_model_parameters().detach().clone()  # noqa: SLF001

    grad_mean = cost.gradient(x, indices=indices)
    grad_per_sample = cost.gradient(x, indices=indices, reduction=None)

    assert grad_per_sample.shape[0] == len(indices)
    assert torch.allclose(grad_mean, grad_per_sample.mean(dim=0), rtol=5e-5, atol=1e-6)


def test_max_batch_size_does_not_change_function_or_gradients() -> None:
    dataset = _make_dataset(n_samples=23)
    base_cost = _make_cost(dataset, max_batch_size=23, batch_size=11)
    chunked_cost = _make_cost(dataset, max_batch_size=4, batch_size=11)

    x = base_cost._get_model_parameters().detach().clone()  # noqa: SLF001
    indices = [0, 4, 2, 10, 9, 3, 8, 7, 6, 1, 5, 11, 12, 13, 14]

    f_base = base_cost.function(x, indices=indices)
    f_chunked = chunked_cost.function(x, indices=indices)
    assert f_base == pytest.approx(f_chunked, rel=1e-7, abs=1e-8)

    g_base = base_cost.gradient(x, indices=indices)
    g_chunked = chunked_cost.gradient(x, indices=indices)
    assert torch.allclose(g_base, g_chunked, rtol=5e-5, atol=1e-6)

    g_ind_base = base_cost.gradient(x, indices=indices, reduction=None)
    g_ind_chunked = chunked_cost.gradient(x, indices=indices, reduction=None)
    assert torch.allclose(g_ind_base, g_ind_chunked, rtol=5e-5, atol=1e-6)


def test_max_batch_size_does_not_change_predict_outputs() -> None:
    dataset = _make_dataset(n_samples=15)
    cost_a = _make_cost(dataset, max_batch_size=15, batch_size=8)
    cost_b = _make_cost(dataset, max_batch_size=2, batch_size=8)

    x = cost_a._get_model_parameters().detach().clone()  # noqa: SLF001
    data = [sample[0] for sample in dataset]

    pred_a = cost_a.predict(x, data)
    pred_b = cost_b.predict(x, data)

    assert len(pred_a) == len(pred_b)
    for pa, pb in zip(pred_a, pred_b, strict=True):
        assert pa == pytest.approx(pb, rel=1e-6, abs=1e-7)


def test_chunked_and_unchunked_costs_match_with_identical_model_snapshot() -> None:
    dataset = _make_dataset(n_samples=21)

    torch.manual_seed(21)
    model = _TinyMLP(input_size=4, hidden_size=12, output_size=3)
    model_copy = copy.deepcopy(model)

    cost_full = PyTorchCost(
        dataset=dataset,
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        batch_size=10,
        max_batch_size=21,
        device=SupportedDevices.CPU,
    )
    cost_chunked = PyTorchCost(
        dataset=dataset,
        model=model_copy,
        loss_fn=torch.nn.CrossEntropyLoss(),
        batch_size=10,
        max_batch_size=3,
        device=SupportedDevices.CPU,
    )

    x = cost_full._get_model_parameters().detach().clone()  # noqa: SLF001
    indices = list(range(21))

    assert torch.allclose(
        cost_full.gradient(x, indices=indices),
        cost_chunked.gradient(x, indices=indices),
        rtol=5e-5,
        atol=1e-6,
    )
