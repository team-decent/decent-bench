from __future__ import annotations

from typing import Unpack

import decent_bench.utils.interoperability as iop
from decent_bench.costs._cost import Cost
from decent_bench.costs._kwarg_types import CostKwargs
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class LinearRegressionCost(Cost):
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

    def __init__(self, A: Array, b: Array):  # noqa: N803
        if iop.shape(A)[0] != iop.shape(b)[0]:
            raise ValueError(f"Dimension mismatch: A has {iop.shape(A)[0]} rows but b has {iop.shape(b)[0]} elements")

        from decent_bench.costs import QuadraticCost  # noqa: PLC0415

        self.inner = QuadraticCost(
            iop.dot(iop.transpose(A), A),
            -iop.dot(iop.transpose(A), b),
            float(0.5 * iop.dot(b, b)),
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
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\mathbf{A}^T \mathbf{A}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
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
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return self.inner.m_cvx

    def function(self, x: Array, **kwargs: Unpack[CostKwargs]) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \| \mathbf{Ax} - \mathbf{b} \|^2
        """
        return self.inner.function(x, **kwargs)

    def gradient(self, x: Array, **kwargs: Unpack[CostKwargs]) -> Array:
        r"""
        Gradient at x.

        .. math:: \mathbf{A}^T\mathbf{Ax} - \mathbf{A}^T \mathbf{b}
        """
        return self.inner.gradient(x, **kwargs)

    def hessian(self, x: Array, **kwargs: Unpack[CostKwargs]) -> Array:
        r"""
        Hessian at x.

        .. math:: \mathbf{A}^T\mathbf{A}
        """
        return self.inner.hessian(x, **kwargs)

    def proximal(self, x: Array, rho: float, **kwargs: Unpack[CostKwargs]) -> Array:
        r"""
        Proximal at x.

        .. math::
            (\rho \mathbf{A}^T \mathbf{A} + \mathbf{I})^{-1} (\mathbf{x} + \rho \mathbf{A}^T\mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        return self.inner.proximal(x, rho, **kwargs)

    def __add__(self, other: Cost) -> Cost:
        """Add another cost function."""
        return self.inner + other
