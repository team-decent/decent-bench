import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.costs import QuadraticCost, SumCost


def _simple_quadratic(A_scale: float, b_scale: float, c: float = 0.0) -> QuadraticCost:
    A = np.eye(2) * A_scale
    b = np.ones(2) * b_scale
    return QuadraticCost(A=A, b=b, c=c)


def _assert_quadratic_matches(
    actual: QuadraticCost,
    expected: QuadraticCost,
    x: np.ndarray,
    rho: float,
) -> None:
    assert actual.function(x) == pytest.approx(expected.function(x))
    np.testing.assert_allclose(iop.to_numpy(actual.gradient(x)), iop.to_numpy(expected.gradient(x)))
    np.testing.assert_allclose(iop.to_numpy(actual.hessian(x)), iop.to_numpy(expected.hessian(x)))
    np.testing.assert_allclose(iop.to_numpy(actual.proximal(x, rho)), iop.to_numpy(expected.proximal(x, rho)))


def test_cost_scalar_multiplication_and_reverse_multiplication() -> None:
    cost = _simple_quadratic(A_scale=2.0, b_scale=1.0, c=3.0)
    x = np.array([1.0, -2.0])

    scaled_left = 2.5 * cost
    scaled_right = cost * 2.5

    assert scaled_left.function(x) == pytest.approx(2.5 * cost.function(x))
    assert scaled_right.function(x) == pytest.approx(2.5 * cost.function(x))
    np.testing.assert_allclose(iop.to_numpy(scaled_left.gradient(x)), 2.5 * iop.to_numpy(cost.gradient(x)))
    np.testing.assert_allclose(iop.to_numpy(scaled_left.hessian(x)), 2.5 * iop.to_numpy(cost.hessian(x)))


def test_cost_scalar_division() -> None:
    cost = _simple_quadratic(A_scale=4.0, b_scale=2.0, c=1.0)
    x = np.array([0.5, -0.25])
    divided = cost / 2.0

    assert divided.function(x) == pytest.approx(0.5 * cost.function(x))
    np.testing.assert_allclose(iop.to_numpy(divided.gradient(x)), 0.5 * iop.to_numpy(cost.gradient(x)))
    np.testing.assert_allclose(iop.to_numpy(divided.hessian(x)), 0.5 * iop.to_numpy(cost.hessian(x)))


def test_quadratic_addition_preserves_type_and_exact_behavior() -> None:
    cost_a = _simple_quadratic(A_scale=1.0, b_scale=1.0, c=1.0)
    cost_b = _simple_quadratic(A_scale=3.0, b_scale=2.0, c=-1.0)
    x = np.array([1.0, 2.0])
    rho = 0.4

    added = cost_a + cost_b
    expected = QuadraticCost(A=cost_a.A + cost_b.A, b=cost_a.b + cost_b.b, c=cost_a.c + cost_b.c)

    assert isinstance(added, QuadraticCost)
    _assert_quadratic_matches(added, expected, x, rho)


def test_quadratic_subtraction_preserves_type_and_exact_behavior() -> None:
    cost_a = _simple_quadratic(A_scale=1.0, b_scale=1.0, c=1.0)
    cost_b = _simple_quadratic(A_scale=3.0, b_scale=2.0, c=-1.0)
    x = np.array([1.0, 2.0])
    rho = 0.4

    subtracted = cost_a - cost_b
    expected = QuadraticCost(A=cost_a.A - cost_b.A, b=cost_a.b - cost_b.b, c=cost_a.c - cost_b.c)

    assert isinstance(subtracted, QuadraticCost)
    _assert_quadratic_matches(subtracted, expected, x, rho)


def test_cost_radd_supports_sum() -> None:
    cost_a = _simple_quadratic(A_scale=1.0, b_scale=1.0, c=0.0)
    cost_b = _simple_quadratic(A_scale=2.0, b_scale=0.0, c=1.0)
    x = np.array([1.0, 2.0])

    summed = sum([cost_a, cost_b])
    expected_value = cost_a.function(x) + cost_b.function(x)

    assert summed.function(x) == pytest.approx(expected_value)
    assert (0 + cost_a).function(x) == pytest.approx(cost_a.function(x))


def test_cost_scalar_ops_reject_invalid_inputs() -> None:
    cost = _simple_quadratic(A_scale=1.0, b_scale=1.0)

    with pytest.raises(TypeError):
        _ = cost * "2"  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = cost / "2"  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = "2" / cost  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = cost * True  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = cost - 1.0  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = 1.0 + cost  # type: ignore[operator]
    with pytest.raises(ZeroDivisionError):
        _ = cost / 0.0
    with pytest.raises(TypeError):
        _ = 0.0 / cost
