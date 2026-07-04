"""Custom metrics for the FEMNIST example.

This module shows how to define custom metrics that can be passed to
``benchmark.compute_metrics``. The two metrics here compute balanced accuracy —
the mean per-class recall — which is more informative than plain accuracy on
FEMNIST's naturally imbalanced per-writer class distributions.

See the ``decent_bench.metrics.Metric`` base class documentation for the full
interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn import metrics as sk_metrics

import decent_bench.utils.interoperability as iop
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.metrics import Metric, NetworkMetricsView
from decent_bench.networks import FedNetwork

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


def _split_xy(data: list[Any]) -> tuple[tuple[Any, ...], np.ndarray]:
    x, y = zip(*data, strict=True)
    return tuple(x), np.array(y)


def _predict(cost: EmpiricalRiskCost, x: Any, test_x: tuple[Any, ...]) -> np.ndarray:
    return iop.to_numpy(cost.predict(x, list(test_x)))


class BalancedAccuracy(Metric):
    """
    Balanced accuracy of the agents'/clients' predictions.

    Balanced accuracy is the mean per-class recall, making it more informative
    than plain accuracy for imbalanced class distributions such as those that
    arise naturally in per-writer FEMNIST partitions.

    Only available when ``problem.test_data`` is provided, all agent costs are
    :class:`~decent_bench.costs.EmpiricalRiskCost`, and target labels are integers.
    """

    description: str = "balanced accuracy"

    def is_available(  # noqa: D102
        self,
        problem: BenchmarkProblem,
    ) -> tuple[bool, str | None]:
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "balanced accuracy only applies if all agents have EmpiricalRiskCost"
        _, test_y = _split_xy(problem.test_data)
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"balanced accuracy only applies for integer targets, got dtype {test_y.dtype}"
        return True, None

    def compute(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: BenchmarkProblem,
        iteration: int,
    ) -> list[float]:
        test_x, test_y = _split_xy(problem.test_data)
        results: list[float] = []
        for agent in network.agents():
            preds = _predict(agent.cost, agent.x_history[iteration], test_x)
            if np.any(~np.isfinite(preds)):
                return [np.nan] * len(network.agents())
            results.append(float(sk_metrics.balanced_accuracy_score(test_y, preds)))
        return results


class ServerBalancedAccuracy(Metric):
    """
    Balanced accuracy of the server model's predictions.

    Uses the server's aggregated model weights to predict on the shared test
    set. Only available for :class:`~decent_bench.networks.FedNetwork` with
    ``problem.test_data``, empirical-risk client costs, and integer targets.
    """

    description: str = "server balanced accuracy"

    def is_available(  # noqa: D102
        self,
        problem: BenchmarkProblem,
    ) -> tuple[bool, str | None]:
        if not isinstance(problem.network, FedNetwork):
            return False, "server balanced accuracy only applies to FedNetwork"
        if getattr(problem, "test_data", None) is None:
            return False, "requires problem.test_data"
        if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
            return False, "server balanced accuracy only applies if all clients have EmpiricalRiskCost"
        _, test_y = _split_xy(problem.test_data)
        if test_y.dtype.kind not in {"i", "u"}:
            return False, f"server balanced accuracy only applies for integer targets, got dtype {test_y.dtype}"
        return True, None

    def compute(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: BenchmarkProblem,
        iteration: int,
    ) -> list[float]:
        client_views = network.agents()
        if not client_views:
            raise ValueError("server balanced accuracy requires at least one client")
        cost = client_views[0].cost
        test_x, test_y = _split_xy(problem.test_data)
        preds = _predict(cost, network.server().x_history[iteration], test_x)
        if np.any(~np.isfinite(preds)):
            return [np.nan]
        return [float(sk_metrics.balanced_accuracy_score(test_y, preds))]
