"""
Cost composition rules.

Developer note:
    Generic cost arithmetic falls back to :class:`SumCost` and :class:`ScaledCost`.

    Regularizers preserve their abstraction under ``+``, ``-``, unary ``-``, ``*``, and ``/`` by returning
    :class:`BaseRegularizerCost`-aware composites. Empirical-risk costs preserve their abstraction under scalar
    scaling through a private empirical scaling wrapper, and under ``empirical + regularizer`` through
    :class:`EmpiricalRegularizedCost`. Unsupported mixed compositions still fall back to the generic wrappers.

    :class:`EmpiricalRegularizedCost.gradient` uses broadcast semantics when ``reduction=None``: it returns one
    composite gradient per sample by adding ``regularizer.gradient(x) / m`` to each per-sample empirical gradient,
    where ``m`` is the number of selected samples. Summing over the leading sample dimension recovers the full
    composite gradient.

    Composition wrappers keep references to their underlying cost objects; they do not make implicit shallow or deep
    copies at construction time. Mutating a wrapped cost after composition therefore affects the composite view as
    well. Agent-installed call-counting hooks on reused cost objects are therefore shared too. Use
    :func:`copy.deepcopy` explicitly when an independent copy of a composed objective or independent counting behavior
    is needed.

    Proximal support is intentionally conservative for the specialized wrappers:
    concrete costs may implement specialized proximals, positive scalar scaling preserves proximal support, and a
    single positively scaled regularizer term preserves regularizer proximal support. ``SumCost`` computes the
    proximal of the full summed objective through
    :func:`decent_bench.centralized_algorithms.proximal_solver` when that accelerated-gradient backend is applicable.
    Multi-term regularizer composites and :class:`EmpiricalRegularizedCost` do not provide a generic proximal. Use a
    specialized proximal if one exists, or use
    :func:`decent_bench.centralized_algorithms.proximal_solver` when applicable.
"""

from . import _base as base
from . import _empirical_risk as empirical_risk
from ._base import (
    BaseRegularizerCost,
    Cost,
    FractionalQuadraticRegularizerCost,
    L1RegularizerCost,
    L2RegularizerCost,
    QuadraticCost,
    ScaledCost,
    SumCost,
    ZeroCost,
)
from ._empirical_risk import (
    EmpiricalRegularizedCost,
    EmpiricalRiskCost,
    LinearRegressionCost,
    LogisticRegressionCost,
    PyTorchCost,
)

__all__ = [
    "BaseRegularizerCost",
    "Cost",
    "EmpiricalRegularizedCost",
    "EmpiricalRiskCost",
    "FractionalQuadraticRegularizerCost",
    "L1RegularizerCost",
    "L2RegularizerCost",
    "LinearRegressionCost",
    "LogisticRegressionCost",
    "PyTorchCost",
    "QuadraticCost",
    "ScaledCost",
    "SumCost",
    "ZeroCost",
    "base",
    "empirical_risk",
]
