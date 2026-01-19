from __future__ import annotations

from typing import Any

import decent_bench.utils.interoperability as iop
from decent_bench.costs.base._cost import Cost
from decent_bench.costs.empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class LinearRegressionCost(EmpiricalRiskCost):
    r"""
    Linear regression cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2

    or in the general quadratic form

    .. math::
        f(\mathbf{x})
        = \frac{1}{2} \mathbf{x}^T\mathbf{A}^T\mathbf{Ax}
        - (\mathbf{A}^T \mathbf{b})^T \mathbf{x}
        + \frac{1}{2} \mathbf{b}^T\mathbf{b}
    """

    def __init__(self, A: Array, b: Array, batch_size: int | None = None, batch_seed: int | None = None):  # noqa: N803
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has {iop.shape(A)[0]} rows but b has {iop.shape(b)[0]} elements")

        from decent_bench.costs.empirical_risk import QuadraticCost  # noqa: PLC0415

        self.inner = QuadraticCost(
            iop.dot(iop.transpose(A), A),
            -iop.dot(iop.transpose(A), b),
            float(0.5 * iop.dot(b, b)),
            batch_size=batch_size,
            batch_seed=batch_seed,
        )
        self.A = A
        self.b = b

    @property
    def shape(self) -> tuple[int, ...]:
        return self.inner.shape

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def n_samples(self) -> int:
        return self.inner.n_samples

    @property
    def batch_size(self) -> int:
        return self.inner.batch_size

    @property
    def batch_used(self) -> list[int]:
        return self.inner.batch_used

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.base.Cost.m_smooth>`.
        """
        return self.inner.m_smooth

    @property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \begin{array}{ll}
                \min_i \lambda_i, & \text{if } \min_i \lambda_i > 0, \\
                0, & \text{if } \min_i \lambda_i = 0, \\
                \text{NaN}, & \text{if } \min_i \lambda_i < 0
            \end{array}

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.base.Cost.m_cvx>`.
        """
        return self.inner.m_cvx

    def predict(self, x: Array, data: list[Array]) -> Array:
        """
        Make predictions at x on the given data.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predictions as an array.'

        """
        return self.inner.predict(x, data)

    def function(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> float:  # noqa: ANN401
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        return self.inner.function(x, indices, **kwargs)

    def gradient(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> Array:  # noqa: ANN401
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.inner.gradient(x, indices, **kwargs)

    def hessian(self, x: Array, indices: list[int] | None = None, **kwargs: Any) -> Array:  # noqa: ANN401
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        return self.inner.hessian(x, indices, **kwargs)

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        r"""
        Proximal at x.

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{x} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.base.Cost.proximal>`
        for the general proximal definition.
        """
        return self.inner.proximal(x, rho, **kwargs)

    def __add__(self, other: Cost) -> Cost:
        """Add another cost function."""
        return self.inner + other
