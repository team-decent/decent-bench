import decent_bench.library.core.metrics.metric_utils as utils
from decent_bench.library.core.agent import AgentMetricsView
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem
from decent_bench.library.core.metrics.plot_metrics.plot_metrics_constructs import X, Y


def global_cost_error_per_iteration(agents: list[AgentMetricsView], problem: BenchmarkProblem) -> list[tuple[X, Y]]:
    r"""
    Calculate the global cost error (y-axis) for each iteration (x-axis).

    Global cost error is defined at :func:`~decent_bench.library.core.metrics.metric_utils.global_cost_error_at_iter`.

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This is done to not miscalculate the global cost error which relies on all agents for
    its calculation.
    """
    iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
    return [(i, utils.global_cost_error_at_iter(agents, problem, i)) for i in range(iter_reached_by_all)]


def global_gradient_optimality_per_iteration(agents: list[AgentMetricsView], _: BenchmarkProblem) -> list[tuple[X, Y]]:
    r"""
    Calculate the global gradient optimality (y-axis) for each iteration (x-axis).

    Global gradient optimality is defined at
    :func:`~decent_bench.library.core.metrics.metric_utils.global_gradient_optimality_at_iter`.

    All iterations up to and including the last one reached by all *agents* are taken into account, subsequent
    iterations are disregarded. This avoids the curve volatility that occurs when fewer and fewer agents are included in
    the calculation.
    """
    iter_reached_by_all = min(len(a.x_per_iteration) for a in agents)
    return [(i, utils.global_gradient_optimality_at_iter(agents, i)) for i in range(iter_reached_by_all)]
