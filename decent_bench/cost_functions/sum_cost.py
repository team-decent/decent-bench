from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import decent_bench.algorithms.cent_algorithms as ca
from decent_bench.cost_functions.cost_function import CostFunction


class SumCost(CostFunction):
    """The sum of multiple cost functions."""

    def __init__(self, cost_functions: list[CostFunction]):
        if not all(cost_functions[0].domain_shape == cf.domain_shape for cf in cost_functions):
            raise ValueError("All cost functions must have the same domain shape")
        self.cost_functions: list[CostFunction] = []
        for cf in cost_functions:
            if isinstance(cf, SumCost):
                self.cost_functions.extend(cf.cost_functions)
            else:
                self.cost_functions.append(cf)

    @property
    def domain_shape(self) -> tuple[int, ...]:  # noqa: D102
        return self.cost_functions[0].domain_shape

    @property
    def m_smooth(self) -> float:
        r"""
        The cost function's smoothness constant.

        .. math::
            \sum f_{k_\text{m_smooth}}

        provided all :math:`f_{k_\text{m_smooth}}` are finite.
        If any :math:`f_{k_\text{m_smooth}} = \text{NaN}`,
        the result is :math:`\text{NaN}`.

        For the general definition, see
        :attr:`CostFunction.m_smooth <decent_bench.cost_functions.cost_function.CostFunction.m_smooth>`.
        """
        m_smooth_vals = [cf.m_smooth for cf in self.cost_functions]
        return np.nan if any(np.isnan(v) for v in m_smooth_vals) else sum(m_smooth_vals)

    @property
    def m_cvx(self) -> float:
        r"""
        The cost function's convexity constant.

        .. math::
            \sum f_{k_\text{m_cvx}}

        provided all :math:`f_{k_\text{m_cvx}}` are finite.
        If any :math:`f_{k_\text{m_cvx}} = \text{NaN}`,
        the result is :math:`\text{NaN}`.

        For the general definition, see
        :attr:`CostFunction.m_cvx <decent_bench.cost_functions.cost_function.CostFunction.m_cvx>`.
        """
        m_cvx_vals = [cf.m_cvx for cf in self.cost_functions]
        return np.nan if any(np.isnan(v) for v in m_cvx_vals) else sum(m_cvx_vals)

    def evaluate(self, x: NDArray[float64]) -> float:
        """Sum the :meth:`evaluate` of each cost function."""
        return sum(cf.evaluate(x) for cf in self.cost_functions)

    def gradient(self, x: NDArray[float64]) -> NDArray[float64]:
        """Sum the :meth:`gradient` of each cost function."""
        res: NDArray[float64] = np.sum([cf.gradient(x) for cf in self.cost_functions], axis=0)
        return res

    def hessian(self, x: NDArray[float64]) -> NDArray[float64]:
        """Sum the :meth:`hessian` of each cost function."""
        res: NDArray[float64] = np.sum([cf.hessian(x) for cf in self.cost_functions], axis=0)
        return res

    def proximal(self, y: NDArray[float64], rho: float) -> NDArray[float64]:
        """
        Proximal at y solved using an iterative method.

        See
        :meth:`CostFunction.proximal() <decent_bench.cost_functions.cost_function.CostFunction.proximal>`
        for the general proximal definition.
        """
        return ca.proximal_solver(self, y, rho)

    def __add__(self, other: CostFunction) -> SumCost:
        """Add another cost function."""
        return SumCost([self, other])
