import copy

import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.costs import (
    BaseRegularizerCost,
    EmpiricalRiskCost,
    L1RegularizerCost,
    L2RegularizerCost,
    LinearRegressionCost,
    SumCost,
)


def _simple_regularizers() -> tuple[L1RegularizerCost, L2RegularizerCost]:
    shape = (2,)
    return L1RegularizerCost(shape=shape), L2RegularizerCost(shape=shape)


def _simple_linear_regression_cost() -> LinearRegressionCost:
    dataset = [
        (np.array([1.0, 0.0]), np.array([1.0])),
        (np.array([0.0, 1.0]), np.array([-1.0])),
        (np.array([1.0, 1.0]), np.array([0.5])),
    ]
    return LinearRegressionCost(dataset=dataset, batch_size="all")


def _assert_cost_matches_expression(
    actual_function: float,
    expected_function: float,
    actual_gradient: np.ndarray,
    expected_gradient: np.ndarray,
    actual_hessian: np.ndarray,
    expected_hessian: np.ndarray,
) -> None:
    assert actual_function == pytest.approx(expected_function)
    np.testing.assert_allclose(actual_gradient, expected_gradient)
    np.testing.assert_allclose(actual_hessian, expected_hessian)


def test_regularizer_addition_preserves_regularizer_aware_type() -> None:
    reg_l1, reg_l2 = _simple_regularizers()
    x = np.array([1.5, -0.5])

    combined = reg_l1 + reg_l2

    assert isinstance(combined, BaseRegularizerCost)
    assert combined.function(x) == pytest.approx(reg_l1.function(x) + reg_l2.function(x))
    np.testing.assert_allclose(
        iop.to_numpy(combined.gradient(x)),
        iop.to_numpy(reg_l1.gradient(x)) + iop.to_numpy(reg_l2.gradient(x)),
    )


def test_regularizer_scalar_multiplication_preserves_regularizer_aware_type() -> None:
    _, reg_l2 = _simple_regularizers()
    x = np.array([2.0, -1.0])

    scaled = 3.0 * reg_l2

    assert isinstance(scaled, BaseRegularizerCost)
    assert scaled.function(x) == pytest.approx(3.0 * reg_l2.function(x))
    np.testing.assert_allclose(iop.to_numpy(scaled.gradient(x)), 3.0 * iop.to_numpy(reg_l2.gradient(x)))


def test_regularizer_scalar_division_preserves_regularizer_aware_type() -> None:
    _, reg_l2 = _simple_regularizers()
    x = np.array([2.0, -1.0])

    scaled = reg_l2 / 4.0

    assert isinstance(scaled, BaseRegularizerCost)
    assert scaled.function(x) == pytest.approx(reg_l2.function(x) / 4.0)
    np.testing.assert_allclose(iop.to_numpy(scaled.gradient(x)), iop.to_numpy(reg_l2.gradient(x)) / 4.0)


def test_empirical_risk_scalar_multiplication_preserves_empirical_risk_aware_type() -> None:
    risk = _simple_linear_regression_cost()
    x = np.array([0.25, -0.75])

    scaled = 2.0 * risk

    assert isinstance(scaled, EmpiricalRiskCost)
    assert scaled.function(x, indices="all") == pytest.approx(2.0 * risk.function(x, indices="all"))
    np.testing.assert_allclose(
        iop.to_numpy(scaled.gradient(x, indices="all")),
        2.0 * iop.to_numpy(risk.gradient(x, indices="all")),
    )


def test_empirical_risk_addition_with_regularizer_preserves_empirical_risk_aware_type() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk + reg_l2

    assert isinstance(objective, EmpiricalRiskCost)
    assert objective.function(x, indices="all") == pytest.approx(risk.function(x, indices="all") + reg_l2.function(x))
    np.testing.assert_allclose(
        iop.to_numpy(objective.gradient(x, indices="all")),
        iop.to_numpy(risk.gradient(x, indices="all")) + iop.to_numpy(reg_l2.gradient(x)),
    )


def test_regularizer_negation_preserves_regularizer_aware_type() -> None:
    reg_l1, _ = _simple_regularizers()
    x = np.array([1.5, -0.5])

    negated = -reg_l1

    assert isinstance(negated, BaseRegularizerCost)
    assert negated.function(x) == pytest.approx(-reg_l1.function(x))
    np.testing.assert_allclose(iop.to_numpy(negated.gradient(x)), -iop.to_numpy(reg_l1.gradient(x)))


def test_empirical_risk_subtraction_with_regularizer_preserves_empirical_risk_aware_type() -> None:
    risk = _simple_linear_regression_cost()
    reg_l1, _ = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk - reg_l1

    assert isinstance(objective, EmpiricalRiskCost)
    assert objective.function(x, indices="all") == pytest.approx(risk.function(x, indices="all") - reg_l1.function(x))
    np.testing.assert_allclose(
        iop.to_numpy(objective.gradient(x, indices="all")),
        iop.to_numpy(risk.gradient(x, indices="all")) - iop.to_numpy(reg_l1.gradient(x)),
    )


def test_composition_wrappers_keep_references_to_wrapped_costs() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()

    summed = SumCost([risk, reg_l2])
    scaled_empirical = 2.0 * risk
    scaled_regularizer = 3.0 * reg_l2
    regularized = risk + reg_l2

    assert summed.costs[0] is risk
    assert summed.costs[1] is reg_l2
    assert scaled_empirical.cost is risk
    assert scaled_regularizer._terms[0][0] is reg_l2
    assert regularized.empirical_cost is risk
    assert regularized.regularizer is reg_l2

    risk._batch_size = 1

    assert scaled_empirical.batch_size == 1
    assert regularized.batch_size == 1


def test_deepcopy_of_composed_costs_is_independent() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()

    scaled_empirical = 2.0 * risk
    regularized = risk + reg_l2

    scaled_empirical_copy = copy.deepcopy(scaled_empirical)
    regularized_copy = copy.deepcopy(regularized)

    assert scaled_empirical_copy.cost is not risk
    assert scaled_empirical_copy.cost is not scaled_empirical.cost
    assert regularized_copy.empirical_cost is not risk
    assert regularized_copy.empirical_cost is not regularized.empirical_cost
    assert regularized_copy.regularizer is not reg_l2
    assert regularized_copy.regularizer is not regularized.regularizer

    risk._batch_size = 1

    assert scaled_empirical.batch_size == 1
    assert scaled_empirical_copy.batch_size == 3
    assert regularized.batch_size == 1
    assert regularized_copy.batch_size == 3


def test_regularizer_addition_matches_manual_expression() -> None:
    reg_l1, reg_l2 = _simple_regularizers()
    x = np.array([1.5, -0.5])

    combined = reg_l1 + reg_l2

    _assert_cost_matches_expression(
        actual_function=combined.function(x),
        expected_function=reg_l1.function(x) + reg_l2.function(x),
        actual_gradient=iop.to_numpy(combined.gradient(x)),
        expected_gradient=iop.to_numpy(reg_l1.gradient(x)) + iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(combined.hessian(x)),
        expected_hessian=iop.to_numpy(reg_l1.hessian(x)) + iop.to_numpy(reg_l2.hessian(x)),
    )


def test_regularizer_scalar_multiplication_matches_manual_expression_and_proximal() -> None:
    _, reg_l2 = _simple_regularizers()
    x = np.array([2.0, -1.0])
    rho = 0.3

    scaled = 3.0 * reg_l2

    _assert_cost_matches_expression(
        actual_function=scaled.function(x),
        expected_function=3.0 * reg_l2.function(x),
        actual_gradient=iop.to_numpy(scaled.gradient(x)),
        expected_gradient=3.0 * iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(scaled.hessian(x)),
        expected_hessian=3.0 * iop.to_numpy(reg_l2.hessian(x)),
    )
    np.testing.assert_allclose(iop.to_numpy(scaled.proximal(x, rho)), iop.to_numpy(reg_l2.proximal(x, 3.0 * rho)))


def test_regularizer_scalar_division_matches_manual_expression_and_proximal() -> None:
    _, reg_l2 = _simple_regularizers()
    x = np.array([2.0, -1.0])
    rho = 0.6

    scaled = reg_l2 / 4.0

    _assert_cost_matches_expression(
        actual_function=scaled.function(x),
        expected_function=reg_l2.function(x) / 4.0,
        actual_gradient=iop.to_numpy(scaled.gradient(x)),
        expected_gradient=iop.to_numpy(reg_l2.gradient(x)) / 4.0,
        actual_hessian=iop.to_numpy(scaled.hessian(x)),
        expected_hessian=iop.to_numpy(reg_l2.hessian(x)) / 4.0,
    )
    np.testing.assert_allclose(iop.to_numpy(scaled.proximal(x, rho)), iop.to_numpy(reg_l2.proximal(x, rho / 4.0)))


def test_regularizer_negation_matches_manual_expression() -> None:
    reg_l1, _ = _simple_regularizers()
    x = np.array([1.5, -0.5])

    negated = -reg_l1

    _assert_cost_matches_expression(
        actual_function=negated.function(x),
        expected_function=-reg_l1.function(x),
        actual_gradient=iop.to_numpy(negated.gradient(x)),
        expected_gradient=-iop.to_numpy(reg_l1.gradient(x)),
        actual_hessian=iop.to_numpy(negated.hessian(x)),
        expected_hessian=-iop.to_numpy(reg_l1.hessian(x)),
    )


def test_regularizer_subtraction_matches_manual_expression() -> None:
    reg_l1, reg_l2 = _simple_regularizers()
    x = np.array([1.5, -0.5])

    combined = reg_l1 - reg_l2

    _assert_cost_matches_expression(
        actual_function=combined.function(x),
        expected_function=reg_l1.function(x) - reg_l2.function(x),
        actual_gradient=iop.to_numpy(combined.gradient(x)),
        expected_gradient=iop.to_numpy(reg_l1.gradient(x)) - iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(combined.hessian(x)),
        expected_hessian=iop.to_numpy(reg_l1.hessian(x)) - iop.to_numpy(reg_l2.hessian(x)),
    )


def test_empirical_risk_scalar_multiplication_matches_manual_expression_and_proximal() -> None:
    risk = _simple_linear_regression_cost()
    x = np.array([0.25, -0.75])
    rho = 0.4

    scaled = 2.0 * risk

    _assert_cost_matches_expression(
        actual_function=scaled.function(x, indices="all"),
        expected_function=2.0 * risk.function(x, indices="all"),
        actual_gradient=iop.to_numpy(scaled.gradient(x, indices="all")),
        expected_gradient=2.0 * iop.to_numpy(risk.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(scaled.hessian(x, indices="all")),
        expected_hessian=2.0 * iop.to_numpy(risk.hessian(x, indices="all")),
    )
    np.testing.assert_allclose(
        iop.to_numpy(scaled.proximal(x, rho)),
        iop.to_numpy(risk.proximal(x, 2.0 * rho)),
    )


def test_empirical_risk_plus_regularizer_matches_manual_expression() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk + reg_l2

    _assert_cost_matches_expression(
        actual_function=objective.function(x, indices="all"),
        expected_function=risk.function(x, indices="all") + reg_l2.function(x),
        actual_gradient=iop.to_numpy(objective.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(risk.gradient(x, indices="all")) + iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(objective.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(risk.hessian(x, indices="all")) + iop.to_numpy(reg_l2.hessian(x)),
    )


def test_empirical_regularized_gradient_mean_matches_manual_expression() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk + reg_l2

    np.testing.assert_allclose(
        iop.to_numpy(objective.gradient(x, indices="all", reduction="mean")),
        iop.to_numpy(risk.gradient(x, indices="all", reduction="mean")) + iop.to_numpy(reg_l2.gradient(x)),
    )


def test_empirical_regularized_gradient_none_broadcasts_regularizer_and_recovers_mean() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk + reg_l2

    actual = iop.to_numpy(objective.gradient(x, indices="all", reduction=None))
    empirical_per_sample = iop.to_numpy(risk.gradient(x, indices="all", reduction=None))
    regularizer_gradient = iop.to_numpy(reg_l2.gradient(x))
    expected = empirical_per_sample + np.stack([regularizer_gradient] * risk.n_samples)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.mean(axis=0), iop.to_numpy(objective.gradient(x, indices="all", reduction="mean")))


def test_empirical_risk_minus_regularizer_matches_manual_expression() -> None:
    risk = _simple_linear_regression_cost()
    reg_l1, _ = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk - reg_l1

    _assert_cost_matches_expression(
        actual_function=objective.function(x, indices="all"),
        expected_function=risk.function(x, indices="all") - reg_l1.function(x),
        actual_gradient=iop.to_numpy(objective.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(risk.gradient(x, indices="all")) - iop.to_numpy(reg_l1.gradient(x)),
        actual_hessian=iop.to_numpy(objective.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(risk.hessian(x, indices="all")) - iop.to_numpy(reg_l1.hessian(x)),
    )


def test_empirical_risk_plus_scaled_regularizer_matches_lambda_expression() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])
    lambda_ = 1.75

    objective = risk + (lambda_ * reg_l2)

    _assert_cost_matches_expression(
        actual_function=objective.function(x, indices="all"),
        expected_function=risk.function(x, indices="all") + lambda_ * reg_l2.function(x),
        actual_gradient=iop.to_numpy(objective.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(risk.gradient(x, indices="all")) + lambda_ * iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(objective.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(risk.hessian(x, indices="all")) + lambda_ * iop.to_numpy(reg_l2.hessian(x)),
    )

    np.testing.assert_allclose(
        iop.to_numpy(objective.gradient(x, indices="all", reduction="mean")),
        iop.to_numpy(risk.gradient(x, indices="all", reduction="mean")) + lambda_ * iop.to_numpy(reg_l2.gradient(x)),
    )

    actual_per_sample = iop.to_numpy(objective.gradient(x, indices="all", reduction=None))
    expected_per_sample = iop.to_numpy(risk.gradient(x, indices="all", reduction=None)) + np.stack(
        [lambda_ * iop.to_numpy(reg_l2.gradient(x))] * risk.n_samples
    )
    np.testing.assert_allclose(actual_per_sample, expected_per_sample)
    np.testing.assert_allclose(actual_per_sample.mean(axis=0), iop.to_numpy(objective.gradient(x, indices="all")))


def test_composite_regularizer_proximal_is_unsupported_for_regularizer_sums() -> None:
    reg_l1, reg_l2 = _simple_regularizers()
    x = np.array([1.0, -2.0])

    with pytest.raises(NotImplementedError, match="Composite regularizers do not implement a generic proximal operator"):
        (reg_l1 + reg_l2).proximal(x, rho=0.5)


def test_empirical_regularized_cost_proximal_is_explicitly_unsupported() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])

    with pytest.raises(NotImplementedError, match="EmpiricalRegularizedCost does not implement a generic proximal"):
        (risk + reg_l2).proximal(x, rho=0.5)
