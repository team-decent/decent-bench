from collections.abc import Callable

import numpy as np
import pytest

from decent_array import Array
from decent_array import interoperability as iop
from decent_array.types import Frameworks
from decent_bench.costs import (
    BaseRegularizerCost,
    Cost,
    EmpiricalRegularizedCost,
    EmpiricalRiskCost,
    L1RegularizerCost,
    L2RegularizerCost,
    LinearRegressionCost,
    LogisticRegressionCost,
    PyTorchCost,
    SumCost,
)


def _linear_cost() -> LinearRegressionCost:
    dataset = [
        (Array(np.array([1.0, 0.0])), Array(np.array([1.0]))),
        (Array(np.array([0.0, 1.0])), Array(np.array([-1.0]))),
        (Array(np.array([1.0, 1.0])), Array(np.array([0.5]))),
    ]
    return LinearRegressionCost(dataset=dataset, batch_size="all")


def _other_linear_cost() -> LinearRegressionCost:
    dataset = [
        (Array(np.array([2.0, 0.0])), Array(np.array([0.0]))),
        (Array(np.array([0.0, 2.0])), Array(np.array([1.0]))),
        (Array(np.array([1.0, -1.0])), Array(np.array([-0.5]))),
    ]
    return LinearRegressionCost(dataset=dataset, batch_size="all")


def _logistic_cost() -> LogisticRegressionCost:
    dataset = [
        (Array(np.array([1.0, 0.0])), Array(np.array([0.0]))),
        (Array(np.array([0.0, 1.0])), Array(np.array([1.0]))),
        (Array(np.array([1.0, 1.0])), Array(np.array([1.0]))),
    ]
    return LogisticRegressionCost(dataset=dataset, batch_size="all")


def _assert_same_values(actual: Cost, expected: Cost, x: Array, *, indices: str = "all") -> None:
    assert actual.function(x, indices=indices) == pytest.approx(expected.function(x, indices=indices))
    np.testing.assert_allclose(
        iop.to_numpy(actual.gradient(x, indices=indices)),
        iop.to_numpy(expected.gradient(x, indices=indices)),
    )
    np.testing.assert_allclose(
        iop.to_numpy(actual.hessian(x, indices=indices)),
        iop.to_numpy(expected.hessian(x, indices=indices)),
    )


def test_regularizer_composition_preserves_regularizer_semantics() -> None:
    x = Array(np.array([1.5, -0.5]))
    l1 = L1RegularizerCost(shape=x.shape)
    l2 = L2RegularizerCost(shape=x.shape)

    combined = l1 + l2

    assert isinstance(combined, BaseRegularizerCost)
    assert combined.function(x) == pytest.approx(l1.function(x) + l2.function(x))
    np.testing.assert_allclose(
        iop.to_numpy(combined.gradient(x)),
        iop.to_numpy(l1.gradient(x)) + iop.to_numpy(l2.gradient(x)),
    )
    np.testing.assert_allclose(
        iop.to_numpy(combined.hessian(x)),
        iop.to_numpy(l1.hessian(x)) + iop.to_numpy(l2.hessian(x)),
    )
    with pytest.raises(NotImplementedError, match="Composite regularizers"):
        combined.proximal(x, penalty=0.5)


@pytest.mark.parametrize(
    ("operation", "factor"),
    [
        (lambda cost, value: value * cost, 2.0),
        (lambda cost, value: cost / value, 0.5),
        (lambda cost, value: -cost, -1.0),
    ],
    ids=["multiply", "divide", "negate"],
)
def test_empirical_regularized_scaling_preserves_empirical_behavior(
    operation: Callable[[Cost, float], Cost], factor: float
) -> None:
    risk = _linear_cost()
    regularizer = L2RegularizerCost(shape=risk.shape)
    x = Array(np.array([0.25, -0.75]))
    data = [sample[0] for sample in risk.dataset]
    objective = risk + regularizer

    scaled = operation(objective, 2.0)
    expected = factor * objective

    assert isinstance(scaled, EmpiricalRiskCost)
    assert scaled.dataset is risk.dataset
    assert scaled.batch_size == risk.batch_size
    np.testing.assert_allclose(iop.to_numpy(scaled.predict(x, data)), iop.to_numpy(risk.predict(x, data)))
    _assert_same_values(scaled, expected, x)


def test_empirical_regularization_preserves_mean_and_per_sample_gradients() -> None:
    risk = _linear_cost()
    regularizer = L2RegularizerCost(shape=risk.shape)
    x = Array(np.array([0.25, -0.75]))
    objective = risk + regularizer

    assert isinstance(objective, EmpiricalRegularizedCost)
    assert objective.dataset is risk.dataset
    assert objective.batch_size == risk.batch_size
    assert objective.function(x, indices="all") == pytest.approx(
        risk.function(x, indices="all") + regularizer.function(x)
    )
    np.testing.assert_allclose(
        iop.to_numpy(objective.gradient(x, indices="all")),
        iop.to_numpy(risk.gradient(x, indices="all")) + iop.to_numpy(regularizer.gradient(x)),
    )

    per_sample = iop.to_numpy(objective.gradient(x, indices="all", reduction=None))
    np.testing.assert_allclose(
        per_sample.mean(axis=0),
        iop.to_numpy(objective.gradient(x, indices="all", reduction="mean")),
    )


@pytest.mark.parametrize(
    ("left_factory", "right_factory", "operator"),
    [
        (_linear_cost, _other_linear_cost, "add"),
        (_linear_cost, _other_linear_cost, "subtract"),
        (_linear_cost, _logistic_cost, "add"),
        (_linear_cost, _logistic_cost, "subtract"),
    ],
    ids=["same-type-add", "same-type-subtract", "different-type-add", "different-type-subtract"],
)
def test_incompatible_empirical_compositions_fall_back_to_sum_cost(
    left_factory: Callable[[], EmpiricalRiskCost],
    right_factory: Callable[[], EmpiricalRiskCost],
    operator: str,
) -> None:
    left = left_factory()
    right = right_factory()
    x = Array(np.array([0.25, -0.75]))

    combined = left + right if operator == "add" else left - right
    sign = 1.0 if operator == "add" else -1.0

    assert isinstance(combined, SumCost)
    assert combined.function(x, indices="all") == pytest.approx(
        left.function(x, indices="all") + sign * right.function(x, indices="all")
    )
    np.testing.assert_allclose(
        iop.to_numpy(combined.gradient(x, indices="all")),
        iop.to_numpy(left.gradient(x, indices="all")) + sign * iop.to_numpy(right.gradient(x, indices="all")),
    )
    np.testing.assert_allclose(
        iop.to_numpy(combined.hessian(x, indices="all")),
        iop.to_numpy(left.hessian(x, indices="all")) + sign * iop.to_numpy(right.hessian(x, indices="all")),
    )


@pytest.mark.backend_framework(Frameworks.PYTORCH)
def test_pytorch_cost_composes_with_configured_regularizer() -> None:
    torch = pytest.importorskip("torch")
    dataset = [
        (torch.tensor([1.0, 0.0]), torch.tensor([1.0])),
        (torch.tensor([0.0, 1.0]), torch.tensor([-1.0])),
        (torch.tensor([1.0, 1.0]), torch.tensor([0.5])),
    ]
    cost = PyTorchCost(
        dataset=dataset,
        model=torch.nn.Linear(2, 1, bias=False),
        loss_fn=torch.nn.MSELoss(),
        batch_size=2,
    )
    regularizer = L2RegularizerCost(shape=cost.shape)
    objective = cost + regularizer
    x = torch.tensor([0.25, -0.75])

    assert isinstance(objective, EmpiricalRegularizedCost)
    assert objective.shape == cost.shape
    assert objective.batch_size == cost.batch_size
    torch.testing.assert_close(
        objective.gradient(x, indices="all"),
        cost.gradient(x, indices="all") + regularizer.gradient(x),
    )
