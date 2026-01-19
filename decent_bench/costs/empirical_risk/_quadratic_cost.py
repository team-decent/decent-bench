from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import decent_bench.utils.interoperability as iop
from decent_bench.costs.base._sum_cost import SumCost
from decent_bench.costs.base._cost import Cost
from decent_bench.costs.empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class QuadraticCost(EmpiricalRiskCost):
    r"""
    Quadratic cost function.

    .. math:: f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
    """

    def __init__(self, A: Array, b: Array, c: float, batch_size: int | None = None, batch_seed: int | None = None):  # noqa: N803
        self.A: NDArray[float64] = iop.to_numpy(A)
        self.b: NDArray[float64] = iop.to_numpy(b)

        if self.A.ndim != 2:
            raise ValueError("Matrix A must be 2D")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be square")
        if self.b.ndim != 1:
            raise ValueError("Vector b must be 1D")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(f"Dimension mismatch: A has shape {self.A.shape} but b has length {self.b.shape[0]}")

        self.A_sym = 0.5 * (self.A + self.A.T)
        self.c = c
        self._batch_size = batch_size if batch_size is not None else self.A.shape[0]
        self._last_batch_used: list[int] = []
        self._rand = np.random.default_rng(seed=batch_seed)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.b.shape

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def n_samples(self) -> int:
        return self.A.shape[0]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def batch_used(self) -> list[int]:
        return self._last_batch_used

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant.

        .. math::
            \max_{i} \left| \lambda_i \right|

        where :math:`\lambda_i` are the eigenvalues of :math:`\frac{1}{2} (\mathbf{A}+\mathbf{A}^T)`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.base.Cost.m_smooth>`.
        """
        eigs = np.linalg.eigvalsh(self.A_sym)
        return float(np.max(np.abs(eigs)))

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
        :attr:`Cost.m_cvx <decent_bench.costs.base.Cost.m_cvx>`.
        """
        eigs = np.linalg.eigvalsh(self.A_sym)
        l_min = float(np.min(eigs))
        tol = 1e-12
        if l_min > tol:
            return l_min
        if abs(l_min) <= tol:
            return 0
        return np.nan

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: NDArray[float64], data: list[NDArray[float64]]) -> NDArray[float64]:
        """
        Make predictions at x on the given data.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predictions as an array.

        """
        pred_data = np.stack(data)
        return pred_data.dot(x)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: NDArray[float64], indices: list[int] | None = None) -> float:
        r"""
        Evaluate function at x.

        .. math:: \frac{1}{2} \mathbf{x}^T \mathbf{Ax} + \mathbf{b}^T \mathbf{x} + c
        """
        A, b, _ = self._get_batch_data(indices)  # noqa: N806
        return float(0.5 * x.dot(A.dot(x)) + b.dot(x) + self.c)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(self, x: NDArray[float64], indices: list[int] | None = None) -> NDArray[float64]:
        r"""
        Gradient at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)\mathbf{x} + \mathbf{b}
        """
        _, b, A_sym = self._get_batch_data(indices)  # noqa: N806
        return A_sym @ x + b

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: NDArray[float64], indices: list[int] | None = None) -> NDArray[float64]:  # noqa: ARG002
        r"""
        Hessian at x.

        .. math:: \frac{1}{2} (\mathbf{A}+\mathbf{A}^T)
        """
        _, _, A_sym = self._get_batch_data(indices)  # noqa: N806
        ret: NDArray[float64] = A_sym.copy()
        return ret

    @iop.autodecorate_cost_method(EmpiricalRiskCost.proximal)
    def proximal(self, x: NDArray[float64], rho: float, indices: list[int] | None = None) -> NDArray[float64]:
        r"""
        Proximal at x.

        .. math::
            (\frac{\rho}{2} (\mathbf{A} + \mathbf{A}^T) + \mathbf{I})^{-1} (\mathbf{x} - \rho \mathbf{b})

        where :math:`\rho > 0` is the penalty.

        This is a closed form solution, see
        :meth:`Cost.proximal() <decent_bench.costs.base.Cost.proximal>`
        for the general proximal definition.
        """
        A, b, A_sym = self._get_batch_data(indices)  # noqa: N806
        lhs = rho * A_sym + np.eye(A.shape[1])
        rhs = x - b * rho

        return np.asarray(np.linalg.solve(lhs, rhs), dtype=float64)

    def _get_batch_data(self, indices: list[int] | None) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        """Get data for a batch. Returns A, b, and A_sym for the batch."""
        if indices is not None:
            self._last_batch_used = indices
        elif self.batch_size < self.n_samples:
            # Sample a random batch
            self._last_batch_used = self._rand.choice(self.n_samples, size=self._batch_size, replace=False).tolist()
        else:
            # Use full dataset
            self._last_batch_used = list(range(self.n_samples))
            return self.A, self.b, self.A_sym

        A, b = self.A[self._last_batch_used, :], self.b[self._last_batch_used]  # noqa: N806
        return A, b, 0.5 * (self.A + self.A.T)

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, QuadraticCost):
            return QuadraticCost(
                iop.to_array(self.A + other.A, self.framework, self.device),
                iop.to_array(self.b + other.b, self.framework, self.device),
                self.c + other.c,
            )

        from decent_bench.costs.empirical_risk import LinearRegressionCost  # noqa: PLC0415

        if isinstance(other, LinearRegressionCost):
            return self + other.inner
        return SumCost([self, other])
