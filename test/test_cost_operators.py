import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.costs import Cost, L1RegularizerCost, L2RegularizerCost, QuadraticCost, SumCost


def _simple_quadratic(A_scale: float, b_scale: float, c: float = 0.0) -> QuadraticCost:
    A = np.eye(2) * A_scale
    b = np.ones(2) * b_scale
    return QuadraticCost(A=A, b=b, c=c)


class _SimpleCost(Cost):
    def __init__(self, scale: float):
        self.scale = scale

    @property
    def shape(self) -> tuple[int, ...]:
        return (2,)

    @property
    def framework(self) -> str:
        return "numpy"

    @property
    def device(self) -> str | None:
        return "cpu"

    @property
    def m_smooth(self) -> float:
        return self.scale

    @property
    def m_cvx(self) -> float:
        return 0.0

    def function(self, x: np.ndarray) -> float:
        return float(self.scale * np.sum(x * x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * self.scale * x

    def hessian(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return 2.0 * self.scale * np.eye(2)

    def proximal(self, x: np.ndarray, rho: float) -> np.ndarray:
        return x / (1.0 + 2.0 * rho * self.scale)


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


def test_custom_cost_inherits_generic_addition_fallback() -> None:
    cost_a = _SimpleCost(scale=1.0)
    cost_b = _SimpleCost(scale=2.0)
    x = np.array([1.5, -0.5])

    added = cost_a + cost_b

    assert isinstance(added, SumCost)
    assert added.function(x) == pytest.approx(cost_a.function(x) + cost_b.function(x))
    np.testing.assert_allclose(iop.to_numpy(added.gradient(x)), cost_a.gradient(x) + cost_b.gradient(x))
    np.testing.assert_allclose(iop.to_numpy(added.hessian(x)), cost_a.hessian(x) + cost_b.hessian(x))


def test_sum_cost_proximal_matches_exact_quadratic_proximal_and_not_sum_of_term_proximals() -> None:
    cost_a = _simple_quadratic(A_scale=1.0, b_scale=1.0, c=1.0)
    cost_b = _simple_quadratic(A_scale=3.0, b_scale=2.0, c=-1.0)
    summed = SumCost([cost_a, cost_b])
    expected = QuadraticCost(A=cost_a.A + cost_b.A, b=cost_a.b + cost_b.b, c=cost_a.c + cost_b.c)
    x = np.array([1.0, -0.5])
    rho = 0.3

    actual = iop.to_numpy(summed.proximal(x, rho))
    exact = iop.to_numpy(expected.proximal(x, rho))
    old_approximation = iop.to_numpy(cost_a.proximal(x, rho)) + iop.to_numpy(cost_b.proximal(x, rho))

    np.testing.assert_allclose(actual, exact, atol=1e-8, rtol=1e-7)
    assert not np.allclose(actual, old_approximation, atol=1e-8, rtol=1e-7)


def test_sum_cost_proximal_raises_for_unsupported_nonsmooth_sum() -> None:
    reg_l1 = L1RegularizerCost(shape=(2,))
    reg_l2 = L2RegularizerCost(shape=(2,))
    summed = SumCost([reg_l1, reg_l2])
    x = np.array([1.0, -2.0])

    with pytest.raises(
        NotImplementedError,
        match="SumCost.proximal uses centralized_algorithms.proximal_solver",
    ):
        summed.proximal(x, rho=0.5)


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
