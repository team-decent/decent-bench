from __future__ import annotations

from functools import cached_property

from decent_array import Array, nan
from decent_array import interoperability as iop

from decent_bench.costs._base._cost import Cost
from decent_bench.costs._decorators import autodecorate_cost_method


class QuadraticCost(Cost):
    r"""
    Quadratic cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
    """

    def __init__(
        self,
        A: Array,  # noqa: N803
        b: Array,
        c: float = 0,
    ):
        self.A = A
        self.b = b

        if self.A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {self.A.shape} but b has length {self.b.shape[0]}")

        self.c = c

    @property
    def shape(self) -> tuple[int, ...]:
        return self.b.shape

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        eigs = iop.eigvalsh(self.A)
        return float(iop.max(iop.absolute(eigs)))

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's convexity constant.

        .. math::
            \begin{array}{ll}
                \min_i \lambda_i, & \text{if } \min_i \lambda_i > 0, \\
                0, & \text{if } \min_i \lambda_i = 0, \\
                \text{NaN}, & \text{if } \min_i \lambda_i < 0
            \end{array}

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        eigs = iop.eigvalsh(self.A)
        l_min = float(iop.min(eigs))
        tol = 1e-12
        if l_min > tol:
            return l_min
        if abs(l_min) <= tol:
            return 0
        return nan

    @autodecorate_cost_method(Cost.function)
    def function(self, x: Array) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
        """
        return float(0.5 * iop.dot(x, self.A @ x) + iop.dot(self.b, x) + self.c)

    @autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array) -> Array:
        r"""
        Gradient at x.

        .. math:: \mathbf{A} \mathbf{x} + \mathbf{b}
        """
        return self.A @ x + self.b

    @autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array) -> Array:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \mathbf{A}
        """
        return iop.copy(self.A)

    @autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, penalty: float) -> Array:
        r"""
        Proximal at x.

        .. math::
            (\frac{\rho}{2} \mathbf{A} + \mathbf{I})^{-1} (\mathbf{x} - \rho \mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>`
        for the general proximal definition.
        """
        lhs = penalty * self.A + iop.eye(self.A.shape[1])
        rhs = x - self.b * penalty

        return iop.solve(lhs, rhs)

    def __add__(self, other: Cost) -> Cost:
        """Add another cost function."""
        self._validate_cost_operation(other)
        if isinstance(other, QuadraticCost):
            return QuadraticCost(
                A=self.A + other.A,
                b=self.b + other.b,
                c=self.c + other.c,
            )

        return super().__add__(other)

    def __sub__(self, other: Cost) -> Cost:
        """
        Subtract another cost function.

        Preserves :class:`QuadraticCost` when subtracting another quadratic cost.
        """
        self._validate_cost_operation(other)
        if isinstance(other, QuadraticCost):
            return self.__add__(
                QuadraticCost(
                    A=-other.A,
                    b=-other.b,
                    c=-other.c,
                )
            )
        return super().__sub__(other)
