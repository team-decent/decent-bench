from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class SumCost(Cost):
    """The sum of multiple cost functions."""

    def __init__(self, costs: list[Cost]):
        if not all(costs[0].shape == cf.shape for cf in costs):
            raise ValueError("All cost functions must have the same domain shape")
        self.costs: list[Cost] = []
        for cf in costs:
            if isinstance(cf, SumCost):
                self.costs.extend(cf.costs)
            else:
                self.costs.append(cf)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.costs[0].shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self.costs[0].framework

    @property
    def device(self) -> SupportedDevices:
        return self.costs[0].device

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant.

        .. math::
            \sum m_{\text{smooth}, k}

        where :math:`m_{\text{smooth}, k}` is the smoothness constant of each individual cost function :math:`f_k`.
        If any :math:`m_{\text{smooth}, k} = \text{NaN}`, the result is :math:`\text{NaN}`.

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        m_smooth_vals = [cf.m_smooth for cf in self.costs]
        return np.nan if any(np.isnan(v) for v in m_smooth_vals) else sum(m_smooth_vals)

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's convexity constant.

        .. math::
            \sum m_{\text{cvx}, k}

        where :math:`m_{\text{cvx}, k}` is the convexity constant of each individual cost function :math:`f_k`.
        If any :math:`m_{\text{cvx}, k} = \text{NaN}`, the result is :math:`\text{NaN}`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        m_cvx_vals = [cf.m_cvx for cf in self.costs]
        return np.nan if any(np.isnan(v) for v in m_cvx_vals) else sum(m_cvx_vals)

    def function(self, x: Array, *args: Any, **kwargs: Any) -> float:  # noqa: ANN401
        """Sum the :meth:`Cost.function <decent_bench.costs.Cost.function>` of each cost function."""
        return sum(cf.function(x, *args, **kwargs) for cf in self.costs)

    def gradient(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        """Sum the :meth:`Cost.gradient <decent_bench.costs.Cost.gradient>` of each cost function."""
        return iop.sum(iop.stack([cf.gradient(x, *args, **kwargs) for cf in self.costs]), dim=0)

    def hessian(self, x: Array, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        """Sum the :meth:`Cost.hessian <decent_bench.costs.Cost.hessian>` of each cost function."""
        return iop.sum(iop.stack([cf.hessian(x, *args, **kwargs) for cf in self.costs]), dim=0)

    def proximal(self, x: Array, rho: float, *args: Any, **kwargs: Any) -> Array:  # noqa: ANN401
        """Sum the :meth:`Cost.proximal() <decent_bench.costs.Cost.proximal>` of each cost function."""
        return iop.sum(iop.stack([cf.proximal(x, rho, *args, **kwargs) for cf in self.costs]), dim=0)

    def __add__(self, other: Cost) -> SumCost:
        """Add another cost function."""
        return SumCost([self, other])
