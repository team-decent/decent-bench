import math

import numpy as np
import pytest

from decent_bench.costs import (
    LinearRegressionCost,
    LogisticRegressionCost,
    PyTorchCost,
    QuadraticCost,
    ZeroCost,
)
from decent_array import Array, interoperability as iop
from decent_array.types import Devices, Frameworks

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


def test_linear_regression_cost_matches_closed_form_values() -> None:
    dataset = [
        (Array(np.array([1.0, 0.0])), Array(np.array([1.0]))),
        (Array(np.array([0.0, 1.0])), Array(np.array([2.0]))),
    ]
    cost = LinearRegressionCost(dataset=dataset, batch_size="all")
    x = Array(np.array([3.0, 4.0]))

    assert cost.n_samples == 2
    assert cost.batch_size == 2
    assert cost.function(x, indices="all") == pytest.approx(2.0)
    np.testing.assert_allclose(iop.to_numpy(cost.gradient(x, indices="all")), np.array([1.0, 1.0]))
    np.testing.assert_allclose(
        iop.to_numpy(cost.gradient(x, indices="all", reduction=None)),
        np.array([[2.0, 0.0], [0.0, 2.0]]),
    )
    np.testing.assert_allclose(iop.to_numpy(cost.hessian(x, indices="all")), 0.5 * np.eye(2))
    np.testing.assert_allclose(iop.to_numpy(cost.proximal(x, penalty=2.0)), np.array([2.0, 3.0]))
    assert cost.batch_used == [0, 1]


def test_linear_regression_cost_validates_constructor_and_indices() -> None:
    with pytest.raises(ValueError, match="Dataset features must be vectors"):
        LinearRegressionCost(dataset=[(Array(np.array([[1.0, 2.0]])), Array(np.array([1.0])))])

    with pytest.raises(TypeError, match="Dataset targets must be single dimensional values"):
        LinearRegressionCost(dataset=[(Array(np.array([1.0, 2.0])), Array(np.array([[1.0]])))])

    with pytest.raises(ValueError, match="Batch size must be positive"):
        LinearRegressionCost(dataset=[(Array(np.array([1.0])), Array(np.array([1.0])))], batch_size=0)

    cost = LinearRegressionCost(dataset=[(Array(np.array([1.0])), Array(np.array([1.0])))], batch_size="all")
    with pytest.raises(ValueError, match="Invalid indices string"):
        cost.function(Array(np.array([0.0])), indices="invalid")


def test_logistic_regression_cost_matches_closed_form_values() -> None:
    dataset = [
        (Array(np.array([1.0, 0.0])), Array(np.array([0.0]))),
        (Array(np.array([0.0, 1.0])), Array(np.array([1.0]))),
    ]
    cost = LogisticRegressionCost(dataset=dataset, batch_size="all")
    x = Array(np.zeros(2))

    assert cost.function(x, indices="all") == pytest.approx(math.log(2.0))
    np.testing.assert_allclose(iop.to_numpy(cost.gradient(x, indices="all")), np.array([0.25, -0.25]))
    np.testing.assert_allclose(iop.to_numpy(cost.hessian(x, indices="all")), 0.125 * np.eye(2))


def test_logistic_regression_proximal_preserves_batch_size_and_returns_finite_result() -> None:
    dataset = [
        (Array(np.array([1.0, 0.0])), Array(np.array([0.0]))),
        (Array(np.array([0.0, 1.0])), Array(np.array([1.0]))),
        (Array(np.array([1.0, 1.0])), Array(np.array([1.0]))),
    ]
    cost = LogisticRegressionCost(dataset=dataset, batch_size=2)
    x = Array(np.array([0.5, -0.25]))

    prox = cost.proximal(x, penalty=0.5)

    assert prox.shape == x.shape
    assert np.all(np.isfinite(iop.to_numpy(prox)))
    assert cost.batch_size == 2


def test_logistic_regression_validates_labels_and_indices() -> None:
    with pytest.raises(ValueError, match="exactly two classes"):
        LogisticRegressionCost(
            dataset=[
                (Array(np.array([1.0])), Array(np.array([0.0]))),
                (Array(np.array([2.0])), Array(np.array([1.0]))),
                (Array(np.array([3.0])), Array(np.array([2.0]))),
            ]
        )

    cost = LogisticRegressionCost(
        dataset=[
            (Array(np.array([1.0, 0.0])), Array(np.array([0.0]))),
            (Array(np.array([0.0, 1.0])), Array(np.array([1.0]))),
        ],
        batch_size="all",
    )
    with pytest.raises(ValueError, match="Invalid indices string"):
        cost.gradient(Array(np.zeros(2)), indices="invalid")


def test_quadratic_cost_matches_direct_formula() -> None:
    A = np.array([[2.0, 1.0], [3.0, 4.0]])
    b = np.array([1.0, -2.0])
    x = Array(np.array([1.0, -1.0]))
    cost = QuadraticCost(A=Array(A), b=Array(b), c=3.0)

    assert cost.function(x) == pytest.approx(7.0)
    np.testing.assert_allclose(iop.to_numpy(cost.gradient(x)), A @ iop.to_numpy(x) + b)
    np.testing.assert_allclose(iop.to_numpy(cost.hessian(x)), A)
    expected_proximal = np.linalg.solve(0.5 * A + np.eye(2), np.array([1.0, -1.0]) - 0.5 * b)
    np.testing.assert_allclose(iop.to_numpy(cost.proximal(x, penalty=0.5)), expected_proximal)
    eigvals = np.linalg.eigvalsh(A)
    assert cost.m_smooth == pytest.approx(float(np.max(np.abs(eigvals))))
    assert np.isnan(cost.m_cvx)


@pytest.mark.parametrize(
    ("A", "b", "match"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "Matrix A must be 2D"),
        (np.array([[1.0, 2.0, 3.0]]), np.array([1.0]), "Matrix A must be square"),
        (np.eye(2), np.array([[1.0], [2.0]]), "Vector b must be 1D"),
        (np.eye(2), np.array([1.0, 2.0, 3.0]), "Dimension mismatch"),
    ],
)
def test_quadratic_cost_validates_constructor_inputs(A: np.ndarray, b: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        QuadraticCost(A=A, b=b)


def test_zero_cost_returns_zero_values() -> None:
    cost = ZeroCost(shape=(2,))
    x = Array(np.array([1.5, -2.5]))

    assert cost.function(x) == 0.0
    np.testing.assert_allclose(iop.to_numpy(cost.gradient(x)), np.zeros(2))
    np.testing.assert_allclose(iop.to_numpy(cost.hessian(x)), np.zeros((2, 2)))
    np.testing.assert_allclose(iop.to_numpy(cost.proximal(x, penalty=1.0)), iop.to_numpy(x))
    assert cost.m_smooth == 0.0
    assert cost.m_cvx == 0.0


def test_zero_cost_validates_shape_and_penalty() -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        ZeroCost(shape=(2, -1))

    cost = ZeroCost(shape=(2,))
    with pytest.raises(ValueError, match="Mismatching domain shapes"):
        cost.gradient(Array(np.array([1.0, 2.0, 3.0])))

    with pytest.raises(ValueError, match="penalty parameter penalty must be positive"):
        cost.proximal(Array(np.array([1.0, 2.0])), penalty=0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def _make_pytorch_cost() -> PyTorchCost:
    dataset = [
        (torch.tensor([1.0, 0.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32)),
        (torch.tensor([0.0, 1.0], dtype=torch.float32), torch.tensor([-1.0], dtype=torch.float32)),
    ]
    model = torch.nn.Linear(2, 1, bias=False)
    return PyTorchCost(
        dataset=dataset,
        model=model,
        loss_fn=torch.nn.MSELoss(),
        batch_size="all",
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.backend_framework(Frameworks.PYTORCH)
def test_pytorch_cost_function_and_gradient_match_direct_torch_computation() -> None:
    from decent_bench.benchmark._configure import get_config

    cost = _make_pytorch_cost()
    x = torch.tensor([0.5, -1.0], dtype=torch.float32)

    expected_model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        expected_model.weight.copy_(x.reshape(1, -1))
    inputs = torch.stack([sample_x for sample_x, _ in cost.dataset])
    targets = torch.stack([sample_y for _, sample_y in cost.dataset])
    loss = torch.nn.MSELoss()(expected_model(inputs), targets)
    loss.backward()
    expected_gradient = expected_model.weight.grad.flatten()

    assert get_config().framework == Frameworks.PYTORCH
    assert get_config().device == Devices.CPU
    assert cost.function(x, indices="all") == pytest.approx(float(loss.item()))
    gradient = cost.gradient(x, indices="all")
    assert isinstance(gradient, torch.Tensor)
    assert gradient.device.type == "cpu"
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.backend_framework(Frameworks.PYTORCH)
def test_pytorch_cost_per_sample_gradient_and_error_paths() -> None:
    cost = _make_pytorch_cost()
    x = torch.tensor([0.5, -1.0], dtype=torch.float32)

    gradient = cost.gradient(x, indices="all")
    per_sample_gradient = cost.gradient(x, indices="all", reduction=None)
    assert per_sample_gradient.shape == (2, 2)
    torch.testing.assert_close(per_sample_gradient.mean(dim=0), gradient)

    with pytest.raises(ValueError, match="does not match total model parameters"):
        cost.function(torch.tensor([1.0], dtype=torch.float32), indices="all")

    with pytest.raises(ValueError, match="Invalid indices string"):
        cost.gradient(x, indices="invalid")

    with pytest.raises(NotImplementedError, match="Hessian computation is not implemented"):
        cost.hessian(x)

    with pytest.raises(NotImplementedError, match="Proximal operator is not implemented"):
        cost.proximal(x, penalty=1.0)
