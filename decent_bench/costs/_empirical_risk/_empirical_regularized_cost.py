from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._regularizer_costs import BaseRegularizerCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)

from ._empirical_risk_cost import EmpiricalRiskCost


class EmpiricalRegularizedCost(EmpiricalRiskCost):
    """
    Composite objective of an empirical risk term plus a regularizer.

    This wrapper preserves empirical-risk-specific behavior from the empirical term, including :meth:`predict`,
    dataset access, and batch metadata, while combining function, gradient, and Hessian values with the regularizer.
    When :meth:`gradient` is called with ``reduction=None``, the regularizer gradient is broadcast across the leading
    sample dimension so that averaging over samples recovers the composite mean gradient. A generic proximal is
    intentionally not implemented.
    """

    def __init__(self, empirical_cost: EmpiricalRiskCost, regularizer: BaseRegularizerCost):
        if empirical_cost.shape != regularizer.shape:
            raise ValueError(f"Mismatching domain shapes: {empirical_cost.shape} vs {regularizer.shape}")
        if empirical_cost.framework != regularizer.framework:
            raise ValueError(f"Mismatching frameworks: {empirical_cost.framework} vs {regularizer.framework}")
        if empirical_cost.device != regularizer.device:
            raise ValueError(f"Mismatching devices: {empirical_cost.device} vs {regularizer.device}")

        self.empirical_cost = empirical_cost
        self.regularizer = regularizer

    @property
    def shape(self) -> tuple[int, ...]:
        return self.empirical_cost.shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self.empirical_cost.framework

    @property
    def device(self) -> SupportedDevices:
        return self.empirical_cost.device

    @property
    def n_samples(self) -> int:
        return self.empirical_cost.n_samples

    @property
    def batch_size(self) -> int:
        return self.empirical_cost.batch_size

    @property
    def batch_used(self) -> list[int]:
        return self.empirical_cost.batch_used

    @property
    def dataset(self) -> Dataset:
        return self.empirical_cost.dataset

    @cached_property
    def m_smooth(self) -> float:
        m_smooth_vals = [self.empirical_cost.m_smooth, self.regularizer.m_smooth]
        return np.nan if any(np.isnan(v) for v in m_smooth_vals) else float(sum(m_smooth_vals))

    @cached_property
    def m_cvx(self) -> float:
        m_cvx_vals = [self.empirical_cost.m_cvx, self.regularizer.m_cvx]
        return np.nan if any(np.isnan(v) for v in m_cvx_vals) else float(sum(m_cvx_vals))

    def predict(self, x: Array, data: list[Array]) -> Array:
        """Predictions are determined by the empirical term."""
        return self.empirical_cost.predict(x, data)

    def function(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> float:  # noqa: ANN401
        return self.empirical_cost.function(x, indices=indices, **kwargs) + self.regularizer.function(x)

    def gradient(
        self,
        x: Array,
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
        **kwargs: Any,  # noqa: ANN401
    ) -> Array:
        """
        Gradient of the empirical objective plus regularizer.

        When ``reduction="mean"``, this returns the mean empirical gradient over the selected samples plus the
        regularizer gradient.

        When ``reduction=None``, this returns one gradient per selected sample with the regularizer gradient broadcast
        along the leading sample dimension. Averaging the result over that leading dimension recovers the composite
        gradient returned by ``reduction="mean"``.
        """
        empirical_gradient = self.empirical_cost.gradient(x, indices=indices, reduction=reduction, **kwargs)
        regularizer_gradient = self.regularizer.gradient(x)

        if reduction is None:
            regularizer_gradients = [regularizer_gradient for _ in self.empirical_cost.batch_used]
            return empirical_gradient + iop.stack(regularizer_gradients)

        return empirical_gradient + regularizer_gradient

    def hessian(self, x: Array, indices: EmpiricalRiskIndices = "batch", **kwargs: Any) -> Array:  # noqa: ANN401
        return self.empirical_cost.hessian(x, indices=indices, **kwargs) + self.regularizer.hessian(x)

    def proximal(self, x: Array, rho: float, **kwargs: Any) -> Array:  # noqa: ANN401
        """
        Generic proximal of an empirical cost plus regularizer is intentionally unsupported.

        This wrapper preserves evaluation, gradient, and Hessian structure, but does not imply a closed-form proximal.
        Use a specialized composite cost if one exists, or use
        :func:`decent_bench.centralized_algorithms.proximal_solver` when its assumptions are satisfied.
        """
        raise NotImplementedError(
            "EmpiricalRegularizedCost does not implement a generic proximal operator. Use a specialized proximal if "
            "available, or use centralized_algorithms.proximal_solver when applicable."
        )

    def _sample_batch_indices(self, indices: EmpiricalRiskIndices = "batch") -> list[int]:
        return self.empirical_cost._sample_batch_indices(indices)

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> Any:  # noqa: ANN401
        return self.empirical_cost._get_batch_data(indices)

    def __add__(self, other: Cost) -> Cost:
        if isinstance(other, BaseRegularizerCost):
            return EmpiricalRegularizedCost(self.empirical_cost, self.regularizer + other)
        return super().__add__(other)
