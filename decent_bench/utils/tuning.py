"""Utilities for selecting algorithm hyperparameters."""

from collections.abc import Sequence
from itertools import product
from typing import Any, cast

import pandas as pd

from decent_bench.algorithms import Algorithm
from decent_bench.benchmark import BenchmarkProblem, benchmark, compute_metrics
from decent_bench.metrics import Metric, metric_library
from decent_bench.networks import Network


def grid_search(
    benchmark_problem: BenchmarkProblem,
    algorithm_class: type[Algorithm[Network]],
    tunable_params: dict[str, Sequence[Any]],
    fixed_params: dict[str, Any] | None = None,
    *,
    iterations: int = 100,
    metric: Metric = metric_library.GradientNorm(),  # noqa: B008
    n_trials: int = 1,
    max_processes: int | None = 1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Apply grid search to tune the algorithm's hyperparameters.

    Args:
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints.
        algorithm_class: class of the algorithm to be tuned.
        tunable_params: hyperparameters to be tuned with candidate values.
        fixed_params: fixed paramaters/arguments of the algorithm.
        iterations: number of iterations to run.
        metric: performance metric used to rate combinations of hyperparameters.
        n_trials: number of trials to run for each parameter combination, select >1 when the algorithm is stochastic.
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default. If your algorithm is very lightweight you
            may want to set this to 1 to avoid the multiprocessing overhead.

    Returns:
        1) the best hyperparameters combination, selected as the combination minimizing the metric in mean; and 2) the
        table of metric values for each hyperparameters combination that was tested (the table is the output of
        :func:`~decent_bench.benchmark.compute_metrics` with the additional ``hyperparams`` column, which contains
        dictionaries of hyperparameters names and values).

    Raises:
        ValueError: If no tunable parameters are passed, or the metric is unavailable.

    """
    fixed = {} if fixed_params is None else dict(fixed_params)
    overlaps = fixed.keys() & tunable_params.keys()
    if overlaps:
        raise ValueError(f"Parameters cannot be both tunable and fixed: {', '.join(sorted(overlaps))}")
    if "name" in fixed or "name" in tunable_params:
        raise ValueError("`name` is reserved for uniquely naming algorithm configurations")
    if not tunable_params:
        raise ValueError("At least one tunable parameter is required")
    empty_params = [name for name, values in tunable_params.items() if not values]
    if empty_params:
        raise ValueError(f"Tunable parameter values cannot be empty: {', '.join(empty_params)}")

    unavailable_reason = metric.is_available(benchmark_problem)
    if unavailable_reason is not None:
        raise ValueError(f"Metric '{metric.description}' is unavailable: {unavailable_reason}")

    names = list(tunable_params)
    candidates = product(*(tunable_params[name] for name in names))
    algorithms: list[Algorithm[Network]] = []
    hyperparams_by_algorithm: dict[str, dict[str, Any]] = {}
    class_name = algorithm_class.name

    for values in candidates:
        hyperparams = dict(zip(names, values, strict=True))
        suffix = ", ".join(f"{name}={value!r}" for name, value in hyperparams.items())
        unique_name = f"{class_name} ({suffix})"
        algorithm = algorithm_class(**(fixed | hyperparams), name=unique_name)  # type: ignore[call-arg]
        algorithms.append(algorithm)
        hyperparams_by_algorithm[unique_name] = hyperparams

    result = benchmark(
        algorithms=algorithms,
        benchmark_problem=benchmark_problem,
        iterations=iterations,
        n_trials=n_trials,
        max_processes=max_processes,
    )
    metrics = compute_metrics(result, table_metrics=[metric], plot_metrics=[])

    table_results = metrics.table_results.copy()  # type: ignore[union-attr]
    table_results["hyperparams"] = (
        table_results["algorithm"].astype(object).apply(lambda algorithm: hyperparams_by_algorithm[algorithm])
    )
    statistic = "mean" if "mean" in table_results["statistic"].to_numpy() else ""
    mean_rows = table_results.loc[table_results["statistic"] == statistic]
    best_row = mean_rows.nsmallest(1, "mean").iloc[0]
    return cast("dict[str, Any]", best_row["hyperparams"]), table_results
