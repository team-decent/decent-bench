import copy

import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.costs import (
    BaseRegularizerCost,
    EmpiricalRegularizedCost,
    EmpiricalRiskCost,
    L1RegularizerCost,
    L2RegularizerCost,
    LinearRegressionCost,
    LogisticRegressionCost,
    QuadraticCost,
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


def _second_linear_regression_cost() -> LinearRegressionCost:
    dataset = [
        (np.array([2.0, 0.0]), np.array([0.0])),
        (np.array([0.0, 2.0]), np.array([1.0])),
        (np.array([1.0, -1.0]), np.array([-0.5])),
    ]
    return LinearRegressionCost(dataset=dataset, batch_size="all")


def _simple_logistic_regression_cost() -> LogisticRegressionCost:
    dataset = [
        (np.array([1.0, 0.0]), np.array([0.0])),
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([1.0, 1.0]), np.array([1.0])),
    ]
    return LogisticRegressionCost(dataset=dataset, batch_size="all")


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


def test_same_type_empirical_addition_preserves_concrete_type_and_metadata() -> None:
    risk_a = _simple_linear_regression_cost()
    risk_b = _second_linear_regression_cost()
    x = np.array([0.25, -0.75])
    prediction_data = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    combined = risk_a + risk_b
    total_samples = risk_a.n_samples + risk_b.n_samples

    assert isinstance(combined, LinearRegressionCost)
    assert combined.n_samples == total_samples
    assert combined.batch_size == combined.n_samples
    np.testing.assert_allclose(iop.to_numpy(combined.predict(x, prediction_data)), iop.to_numpy(risk_a.predict(x, prediction_data)))
    _assert_cost_matches_expression(
        actual_function=combined.function(x, indices="all"),
        expected_function=(
            risk_a.n_samples * risk_a.function(x, indices="all") + risk_b.n_samples * risk_b.function(x, indices="all")
        )
        / total_samples,
        actual_gradient=iop.to_numpy(combined.gradient(x, indices="all")),
        expected_gradient=(
            risk_a.n_samples * iop.to_numpy(risk_a.gradient(x, indices="all"))
            + risk_b.n_samples * iop.to_numpy(risk_b.gradient(x, indices="all"))
        )
        / total_samples,
        actual_hessian=iop.to_numpy(combined.hessian(x, indices="all")),
        expected_hessian=(
            risk_a.n_samples * iop.to_numpy(risk_a.hessian(x, indices="all"))
            + risk_b.n_samples * iop.to_numpy(risk_b.hessian(x, indices="all"))
        )
        / total_samples,
    )


def test_same_type_empirical_subtraction_falls_back_to_sumcost_with_correct_numerics() -> None:
    risk_a = _simple_linear_regression_cost()
    risk_b = _second_linear_regression_cost()
    x = np.array([0.25, -0.75])

    combined = risk_a - risk_b

    assert isinstance(combined, SumCost)
    _assert_cost_matches_expression(
        actual_function=combined.function(x, indices="all"),
        expected_function=risk_a.function(x, indices="all") - risk_b.function(x, indices="all"),
        actual_gradient=iop.to_numpy(combined.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(risk_a.gradient(x, indices="all")) - iop.to_numpy(risk_b.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(combined.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(risk_a.hessian(x, indices="all")) - iop.to_numpy(risk_b.hessian(x, indices="all")),
    )


def test_different_empirical_types_fall_back_to_sumcost_for_add_and_subtract() -> None:
    linear = _simple_linear_regression_cost()
    logistic = _simple_logistic_regression_cost()
    x = np.array([0.25, -0.75])

    added = linear + logistic
    subtracted = linear - logistic

    assert isinstance(added, SumCost)
    assert isinstance(subtracted, SumCost)
    _assert_cost_matches_expression(
        actual_function=added.function(x, indices="all"),
        expected_function=linear.function(x, indices="all") + logistic.function(x, indices="all"),
        actual_gradient=iop.to_numpy(added.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(linear.gradient(x, indices="all")) + iop.to_numpy(logistic.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(added.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(linear.hessian(x, indices="all")) + iop.to_numpy(logistic.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=subtracted.function(x, indices="all"),
        expected_function=linear.function(x, indices="all") - logistic.function(x, indices="all"),
        actual_gradient=iop.to_numpy(subtracted.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(linear.gradient(x, indices="all")) - iop.to_numpy(logistic.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(subtracted.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(linear.hessian(x, indices="all")) - iop.to_numpy(logistic.hessian(x, indices="all")),
    )


def test_scaled_empirical_with_compound_regularizer_preserves_empirical_behavior() -> None:
    risk = _simple_linear_regression_cost()
    reg_l1, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])
    prediction_data = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    compound = reg_l1 + reg_l2

    objective = (2.0 * risk) + compound
    objective_minus = (2.0 * risk) - compound

    assert isinstance(objective, EmpiricalRegularizedCost)
    assert isinstance(objective_minus, EmpiricalRegularizedCost)
    assert objective.dataset is risk.dataset
    assert objective.batch_size == risk.batch_size
    np.testing.assert_allclose(iop.to_numpy(objective.predict(x, prediction_data)), iop.to_numpy(risk.predict(x, prediction_data)))
    _assert_cost_matches_expression(
        actual_function=objective.function(x, indices="all"),
        expected_function=2.0 * risk.function(x, indices="all") + compound.function(x),
        actual_gradient=iop.to_numpy(objective.gradient(x, indices="all")),
        expected_gradient=2.0 * iop.to_numpy(risk.gradient(x, indices="all")) + iop.to_numpy(compound.gradient(x)),
        actual_hessian=iop.to_numpy(objective.hessian(x, indices="all")),
        expected_hessian=2.0 * iop.to_numpy(risk.hessian(x, indices="all")) + iop.to_numpy(compound.hessian(x)),
    )
    _assert_cost_matches_expression(
        actual_function=objective_minus.function(x, indices="all"),
        expected_function=2.0 * risk.function(x, indices="all") - compound.function(x),
        actual_gradient=iop.to_numpy(objective_minus.gradient(x, indices="all")),
        expected_gradient=2.0 * iop.to_numpy(risk.gradient(x, indices="all")) - iop.to_numpy(compound.gradient(x)),
        actual_hessian=iop.to_numpy(objective_minus.hessian(x, indices="all")),
        expected_hessian=2.0 * iop.to_numpy(risk.hessian(x, indices="all")) - iop.to_numpy(compound.hessian(x)),
    )


def test_scaled_empirical_falls_back_to_sumcost_for_empirical_and_generic_addition() -> None:
    risk_a = _simple_linear_regression_cost()
    risk_b = _second_linear_regression_cost()
    x = np.array([0.25, -0.75])
    generic = QuadraticCost(A=np.eye(2), b=np.zeros(2))

    added_empirical = (2.0 * risk_a) + risk_b
    subtracted_scaled = (2.0 * risk_a) - (3.0 * risk_b)
    added_generic = (2.0 * risk_a) + generic

    assert isinstance(added_empirical, SumCost)
    assert isinstance(subtracted_scaled, SumCost)
    assert isinstance(added_generic, SumCost)
    _assert_cost_matches_expression(
        actual_function=added_empirical.function(x, indices="all"),
        expected_function=2.0 * risk_a.function(x, indices="all") + risk_b.function(x, indices="all"),
        actual_gradient=iop.to_numpy(added_empirical.gradient(x, indices="all")),
        expected_gradient=2.0 * iop.to_numpy(risk_a.gradient(x, indices="all")) + iop.to_numpy(risk_b.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(added_empirical.hessian(x, indices="all")),
        expected_hessian=2.0 * iop.to_numpy(risk_a.hessian(x, indices="all")) + iop.to_numpy(risk_b.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=subtracted_scaled.function(x, indices="all"),
        expected_function=2.0 * risk_a.function(x, indices="all") - 3.0 * risk_b.function(x, indices="all"),
        actual_gradient=iop.to_numpy(subtracted_scaled.gradient(x, indices="all")),
        expected_gradient=2.0 * iop.to_numpy(risk_a.gradient(x, indices="all")) - 3.0 * iop.to_numpy(risk_b.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(subtracted_scaled.hessian(x, indices="all")),
        expected_hessian=2.0 * iop.to_numpy(risk_a.hessian(x, indices="all")) - 3.0 * iop.to_numpy(risk_b.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=added_generic.function(x),
        expected_function=2.0 * risk_a.function(x) + generic.function(x),
        actual_gradient=iop.to_numpy(added_generic.gradient(x)),
        expected_gradient=2.0 * iop.to_numpy(risk_a.gradient(x)) + iop.to_numpy(generic.gradient(x)),
        actual_hessian=iop.to_numpy(added_generic.hessian(x)),
        expected_hessian=2.0 * iop.to_numpy(risk_a.hessian(x)) + iop.to_numpy(generic.hessian(x)),
    )


def test_regularized_empirical_with_more_regularizers_preserves_empirical_behavior() -> None:
    risk = _simple_linear_regression_cost()
    reg_l1, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])
    prediction_data = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    objective = risk + reg_l1
    objective_plus = objective + reg_l2
    objective_minus = objective - (reg_l1 + reg_l2)

    assert isinstance(objective_plus, EmpiricalRegularizedCost)
    assert isinstance(objective_minus, EmpiricalRegularizedCost)
    assert objective_plus.dataset is risk.dataset
    assert objective_plus.batch_size == risk.batch_size
    np.testing.assert_allclose(iop.to_numpy(objective_plus.predict(x, prediction_data)), iop.to_numpy(risk.predict(x, prediction_data)))
    _assert_cost_matches_expression(
        actual_function=objective_plus.function(x, indices="all"),
        expected_function=risk.function(x, indices="all") + reg_l1.function(x) + reg_l2.function(x),
        actual_gradient=iop.to_numpy(objective_plus.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(risk.gradient(x, indices="all")) + iop.to_numpy(reg_l1.gradient(x)) + iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(objective_plus.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(risk.hessian(x, indices="all")) + iop.to_numpy(reg_l1.hessian(x)) + iop.to_numpy(reg_l2.hessian(x)),
    )
    _assert_cost_matches_expression(
        actual_function=objective_minus.function(x, indices="all"),
        expected_function=risk.function(x, indices="all") - reg_l2.function(x),
        actual_gradient=iop.to_numpy(objective_minus.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(risk.gradient(x, indices="all")) - iop.to_numpy(reg_l2.gradient(x)),
        actual_hessian=iop.to_numpy(objective_minus.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(risk.hessian(x, indices="all")) - iop.to_numpy(reg_l2.hessian(x)),
    )


def test_regularized_empirical_falls_back_to_sumcost_for_non_regularizer_arithmetic() -> None:
    risk_a = _simple_linear_regression_cost()
    risk_b = _second_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])

    objective = risk_a + reg_l2
    other_regularized = risk_b + reg_l2
    added_empirical = objective + risk_b
    added_scaled = objective + (2.0 * risk_b)
    added_regularized = objective + other_regularized

    assert isinstance(added_empirical, SumCost)
    assert isinstance(added_scaled, SumCost)
    assert isinstance(added_regularized, SumCost)
    _assert_cost_matches_expression(
        actual_function=added_empirical.function(x, indices="all"),
        expected_function=objective.function(x, indices="all") + risk_b.function(x, indices="all"),
        actual_gradient=iop.to_numpy(added_empirical.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(objective.gradient(x, indices="all")) + iop.to_numpy(risk_b.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(added_empirical.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(objective.hessian(x, indices="all")) + iop.to_numpy(risk_b.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=added_scaled.function(x, indices="all"),
        expected_function=objective.function(x, indices="all") + 2.0 * risk_b.function(x, indices="all"),
        actual_gradient=iop.to_numpy(added_scaled.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(objective.gradient(x, indices="all")) + 2.0 * iop.to_numpy(risk_b.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(added_scaled.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(objective.hessian(x, indices="all")) + 2.0 * iop.to_numpy(risk_b.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=added_regularized.function(x, indices="all"),
        expected_function=objective.function(x, indices="all") + other_regularized.function(x, indices="all"),
        actual_gradient=iop.to_numpy(added_regularized.gradient(x, indices="all")),
        expected_gradient=iop.to_numpy(objective.gradient(x, indices="all")) + iop.to_numpy(other_regularized.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(added_regularized.hessian(x, indices="all")),
        expected_hessian=iop.to_numpy(objective.hessian(x, indices="all")) + iop.to_numpy(other_regularized.hessian(x, indices="all")),
    )


def test_scaling_regularized_empirical_returns_empirical_scaled_cost() -> None:
    risk = _simple_linear_regression_cost()
    _, reg_l2 = _simple_regularizers()
    x = np.array([0.25, -0.75])
    prediction_data = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    objective = risk + reg_l2

    scaled = 3.0 * objective
    divided = objective / 2.0
    negated = -objective

    assert isinstance(scaled, EmpiricalRiskCost)
    assert isinstance(divided, EmpiricalRiskCost)
    assert isinstance(negated, EmpiricalRiskCost)
    assert not isinstance(scaled, SumCost)
    assert not isinstance(divided, SumCost)
    assert not isinstance(negated, SumCost)
    assert scaled.dataset is risk.dataset
    assert scaled.batch_size == risk.batch_size
    np.testing.assert_allclose(iop.to_numpy(scaled.predict(x, prediction_data)), iop.to_numpy(risk.predict(x, prediction_data)))
    _assert_cost_matches_expression(
        actual_function=scaled.function(x, indices="all"),
        expected_function=3.0 * objective.function(x, indices="all"),
        actual_gradient=iop.to_numpy(scaled.gradient(x, indices="all")),
        expected_gradient=3.0 * iop.to_numpy(objective.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(scaled.hessian(x, indices="all")),
        expected_hessian=3.0 * iop.to_numpy(objective.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=divided.function(x, indices="all"),
        expected_function=0.5 * objective.function(x, indices="all"),
        actual_gradient=iop.to_numpy(divided.gradient(x, indices="all")),
        expected_gradient=0.5 * iop.to_numpy(objective.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(divided.hessian(x, indices="all")),
        expected_hessian=0.5 * iop.to_numpy(objective.hessian(x, indices="all")),
    )
    _assert_cost_matches_expression(
        actual_function=negated.function(x, indices="all"),
        expected_function=-objective.function(x, indices="all"),
        actual_gradient=iop.to_numpy(negated.gradient(x, indices="all")),
        expected_gradient=-iop.to_numpy(objective.gradient(x, indices="all")),
        actual_hessian=iop.to_numpy(negated.hessian(x, indices="all")),
        expected_hessian=-iop.to_numpy(objective.hessian(x, indices="all")),
    )


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
