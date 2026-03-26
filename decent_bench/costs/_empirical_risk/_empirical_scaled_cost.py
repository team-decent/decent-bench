from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from decent_bench.utils.array import Array
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)

from ._empirical_risk_cost import EmpiricalRiskCost


class _EmpiricalScaledCost(EmpiricalRiskCost):
    """
    Private scalar wrapper around an empirical-risk cost.

    ``_EmpiricalScaledCost`` preserves empirical-risk-specific behavior such as :meth:`predict`, dataset access, and
    batch handling under scalar scaling. Scaling changes the objective value, gradient, Hessian, and proximal
    parameterization, but does not change the prediction map of the underlying model at a fixed parameter vector.

    Instances keep references to the wrapped cost object. No implicit copying is performed; use
    :func:`copy.deepcopy` explicitly if independent objects are required.
    """

    def __init__(self, cost: EmpiricalRiskCost, scalar: float):
        self.cost: EmpiricalRiskCost
        self.scalar: float
        if isinstance(cost, _EmpiricalScaledCost):
            self.cost = cost.cost
            self.scalar = scalar * cost.scalar
        else:
            self.cost = cost
            self.scalar = scalar

    @property
    def shape(self) -> tuple[int, ...]:
        return self.cost.shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self.cost.framework

    @property
    def device(self) -> SupportedDevices:
        return self.cost.device

    @property
    def n_samples(self) -> int:
        return self.cost.n_samples

    @property
    def batch_size(self) -> int:
        return self.cost.batch_size

    @property
    def batch_used(self) -> list[int]:
        return self.cost.batch_used

    @property
    def dataset(self) -> Dataset:
        return self.cost.dataset

    @cached_property
    def m_smooth(self) -> float:
        if self.scalar == 0:
            return 0.0
        return float(abs(self.scalar) * self.cost.m_smooth)

    @cached_property
    def m_cvx(self) -> float:
        if self.scalar > 0:
            return float(self.scalar * self.cost.m_cvx)
        if self.scalar == 0:
            return 0.0
        return np.nan

    def predict(self, x: Array, data: list[Array]) -> Array:
        """Predictions are unchanged by scalar scaling of the objective."""
        return self.cost.predict(x, data)

    def function(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        return float(self.scalar * self.cost.function(x, indices=indices, **kwargs))

    def gradient(
        self,
        x: Array,
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
        **kwargs: Any,  # noqa: ANN401
    ) -> Array:
        return self.cost.gradient(x, indices=indices, reduction=reduction, **kwargs) * self.scalar

    def hessian(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> Array:  # noqa: ANN401
        return self.cost.hessian(x, indices=indices, **kwargs) * self.scalar

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        if rho <= 0:
            raise ValueError("The penalty parameter rho must be positive.")
        if self.scalar < 0:
            raise ValueError("The proximal operator is not defined for negative scaling.")
        if self.scalar == 0:
            return x
        return self.cost.proximal(x, rho * self.scalar, **kwargs)

    def _sample_batch_indices(self, indices: EmpiricalRiskIndices = "batch") -> list[int]:
        return self.cost._sample_batch_indices(indices)  # noqa: SLF001

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> Any:  # noqa: ANN401
        return self.cost._get_batch_data(indices)  # noqa: SLF001
