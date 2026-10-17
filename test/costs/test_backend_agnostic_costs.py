"""Numerical contract tests for costs implemented with decent-array interoperability operations."""

import math

import numpy as np
import pytest

from decent_array import Array, interoperability as iop
from decent_array.types import Devices, Frameworks
from decent_bench.costs import L2RegularizerCost, LinearRegressionCost, LogisticRegressionCost, QuadraticCost


def _assert_array_close(actual: Array, expected: np.ndarray) -> None:
    np.testing.assert_allclose(iop.to_numpy(actual), expected, rtol=5e-4, atol=5e-5)


def _array(values: object) -> Array:
    """Create portable single-precision inputs, including for PyTorch MPS."""
    return Array(np.asarray(values, dtype=np.float32))


def test_linear_regression_cost_contract(backend: tuple[Frameworks, Devices]) -> None:  # noqa: ARG001
    dataset = [
        (_array([1.0, 0.0]), _array([1.0])),
        (_array([0.0, 1.0]), _array([2.0])),
    ]
    cost = LinearRegressionCost(dataset, batch_size="all")
    x = _array([3.0, 4.0])

    assert cost.function(x, indices="all") == pytest.approx(2.0, rel=5e-4, abs=5e-5)
    _assert_array_close(cost.gradient(x, indices="all"), np.array([1.0, 1.0]))
    _assert_array_close(cost.hessian(x, indices="all"), 0.5 * np.eye(2))
    _assert_array_close(cost.proximal(x, penalty=2.0), np.array([2.0, 3.0]))


def test_logistic_regression_cost_contract(backend: tuple[Frameworks, Devices]) -> None:  # noqa: ARG001
    dataset = [
        (_array([1.0, 0.0]), _array([0.0])),
        (_array([0.0, 1.0]), _array([1.0])),
    ]
    cost = LogisticRegressionCost(dataset, batch_size="all")
    x = _array(np.zeros(2))

    assert cost.function(x, indices="all") == pytest.approx(math.log(2.0), rel=5e-4, abs=5e-5)
    _assert_array_close(cost.gradient(x, indices="all"), np.array([0.25, -0.25]))
    _assert_array_close(cost.hessian(x, indices="all"), 0.125 * np.eye(2))


def test_l2_regularizer_contract(backend: tuple[Frameworks, Devices]) -> None:  # noqa: ARG001
    cost = L2RegularizerCost(shape=(2,))
    x = _array([2.0, -1.0])

    assert cost.function(x) == pytest.approx(2.5, rel=5e-4, abs=5e-5)
    _assert_array_close(cost.gradient(x), np.array([2.0, -1.0]))
    _assert_array_close(cost.hessian(x), np.eye(2))
    _assert_array_close(cost.proximal(x, penalty=0.5), np.array([4.0 / 3.0, -2.0 / 3.0]))


def test_quadratic_cost_contract(backend: tuple[Frameworks, Devices]) -> None:  # noqa: ARG001
    matrix = np.array([[2.0, 0.25], [0.25, 1.0]])
    linear = np.array([0.5, -1.0])
    x_np = np.array([0.3, -0.2])
    cost = QuadraticCost(A=_array(matrix), b=_array(linear), c=0.3)
    x = _array(x_np)

    expected_function = 0.5 * x_np @ matrix @ x_np + linear @ x_np + 0.3
    expected_proximal = np.linalg.solve(0.5 * matrix + np.eye(2), x_np - 0.5 * linear)
    assert cost.function(x) == pytest.approx(expected_function, rel=5e-4, abs=5e-5)
    _assert_array_close(cost.gradient(x), matrix @ x_np + linear)
    _assert_array_close(cost.hessian(x), matrix)
    _assert_array_close(cost.proximal(x, penalty=0.5), expected_proximal)
