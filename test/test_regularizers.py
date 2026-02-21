import numpy as np

from decent_bench.costs import FractionalQuadraticRegularizerCost, L1RegularizerCost, L2RegularizerCost


def test_l1_regularizer_basics() -> None:
    x = np.array([1.0, -2.0, 0.0])
    cost = L1RegularizerCost(shape=x.shape)

    assert cost.function(x) == 3.0
    np.testing.assert_allclose(cost.gradient(x), np.array([1.0, -1.0, 0.0]))
    np.testing.assert_allclose(cost.proximal(x, rho=0.5), np.array([0.5, -1.5, 0.0]))
    np.testing.assert_allclose(cost.hessian(x), np.zeros((3, 3)))
    assert np.isnan(cost.m_smooth)
    assert cost.m_cvx == 0.0


def test_l2_regularizer_basics() -> None:
    x = np.array([1.0, -2.0, 0.0])
    cost = L2RegularizerCost(shape=x.shape)

    assert cost.function(x) == 2.5
    np.testing.assert_allclose(cost.gradient(x), x)
    np.testing.assert_allclose(cost.proximal(x, rho=1.0), x / 2.0)
    np.testing.assert_allclose(cost.hessian(x), np.eye(3))
    assert cost.m_smooth == 1.0
    assert cost.m_cvx == 1.0


def test_fractional_quadratic_regularizer_basics() -> None:
    x = np.array([1.0, 0.0, -2.0])
    cost = FractionalQuadraticRegularizerCost(shape=x.shape)

    assert cost.function(x) == 1.3
    np.testing.assert_allclose(cost.gradient(x), np.array([0.5, 0.0, -0.16]), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        cost.hessian(x),
        np.diag(np.array([-0.5, 2.0, -22.0 / 125.0])),
        rtol=1e-6,
        atol=1e-8,
    )
    assert cost.m_smooth == 2.0
    assert np.isnan(cost.m_cvx)
    prox = cost.proximal(x, rho=0.5)
    assert prox.shape == x.shape
    assert np.all(np.isfinite(prox))


def test_regularizers_ignore_erm_kwargs() -> None:
    x = np.array([1.0, -2.0, 0.0])
    r_l1 = L1RegularizerCost(shape=x.shape)
    r_l2 = L2RegularizerCost(shape=x.shape)

    # Direct calls should ignore ERM-style kwargs.
    np.testing.assert_allclose(r_l1.gradient(x, indices="batch"), np.sign(x))
    np.testing.assert_allclose(r_l2.gradient(x, indices="batch", reduction="mean"), x)

    # Composed objective should also accept ERM kwargs without errors.
    obj = 2.0 * r_l2 + r_l1
    grad = obj.gradient(x, indices="batch")
    assert grad.shape == x.shape
