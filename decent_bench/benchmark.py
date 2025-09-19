import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from logging.handlers import QueueListener
from multiprocessing import Manager
from typing import Literal

from rich.status import Status

from decent_bench.library.core import network
from decent_bench.library.core.benchmark_problem.benchmark_problems import BenchmarkProblem
from decent_bench.library.core.dst_algorithms import DstAlgorithm
from decent_bench.library.core.metrics.plot_metrics import plot
from decent_bench.library.core.metrics.plot_metrics.default_plot_metrics import DEFAULT_PLOT_METRICS
from decent_bench.library.core.metrics.plot_metrics.plot_metrics_constructs import PlotMetric
from decent_bench.library.core.metrics.table_metrics import tabulate
from decent_bench.library.core.metrics.table_metrics.default_table_metrics import DEFAULT_TABLE_METRICS
from decent_bench.library.core.metrics.table_metrics.table_metrics_constructs import TableMetric
from decent_bench.library.core.network import Network
from decent_bench.library.utils import logger
from decent_bench.library.utils.logger import LOGGER
from decent_bench.library.utils.progress_bar import ProgressBarController


def benchmark(
    algorithms: list[DstAlgorithm],
    benchmark_problem: BenchmarkProblem,
    plot_metrics: list[PlotMetric] = DEFAULT_PLOT_METRICS,
    table_metrics: list[TableMetric] = DEFAULT_TABLE_METRICS,
    table_fmt: Literal["grid", "latex"] = "grid",
    *,
    n_trials: int = 30,
    confidence_level: float = 0.95,
    log_level: int = logging.INFO,
    max_processes: int | None = None,
) -> None:
    """
    Benchmark distributed algorithms.

    Args:
        algorithms: algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints
        plot_metrics: metrics to plot after the execution, defaults to
            :const:`~decent_bench.library.core.metrics.plot_metrics.default_plot_metrics.DEFAULT_PLOT_METRICS`
        table_metrics: metrics to tabulate as confidence intervals after the execution, defaults to
            :const:`~decent_bench.library.core.metrics.table_metrics.default_table_metrics.DEFAULT_TABLE_METRICS`
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
        confidence_level: confidence level of the confidence intervals
        log_level: minimum level to log, e.g. :data:`logging.INFO`
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default

    """
    manager = Manager()
    log_listener = logger.start_log_listener(manager, log_level)
    LOGGER.info("Starting benchmark execution")
    with Status("Generating initial network state"):
        nw_init_state = network.create_distributed_network(benchmark_problem)
    LOGGER.info(f"Nr of agents: {len(nw_init_state.get_all_agents())}")
    pb_ctrl = ProgressBarController(manager, algorithms, n_trials)
    resulting_nw_states = _run_trials(algorithms, n_trials, nw_init_state, pb_ctrl, log_listener, max_processes)
    LOGGER.info("All trials complete")
    with Status("Creating table"):
        tabulate.tabulate(resulting_nw_states, benchmark_problem, table_metrics, confidence_level, table_fmt)
    with Status("Creating plot"):
        plot.plot(resulting_nw_states, benchmark_problem, plot_metrics)


def _run_trials(  # noqa: PLR0917
    algorithms: list[DstAlgorithm],
    n_trials: int,
    nw_init_state: Network,
    progress_bar_ctrl: ProgressBarController,
    log_listener: QueueListener,
    max_processes: int | None,
) -> dict[DstAlgorithm, list[Network]]:
    def _start_logger() -> None:
        logger.start_queue_logger(log_listener.queue)

    if max_processes == 1:
        return {alg: [_run_trial(alg, nw_init_state, progress_bar_ctrl) for _ in range(n_trials)] for alg in algorithms}
    with ProcessPoolExecutor(initializer=_start_logger, max_workers=max_processes) as executor:
        LOGGER.info(f"Concurrent processes: {executor._max_workers}")  # type: ignore[attr-defined] # noqa: SLF001
        all_futures = {
            alg: [executor.submit(_run_trial, alg, nw_init_state, progress_bar_ctrl) for _ in range(n_trials)]
            for alg in algorithms
        }
        return {alg: [f.result() for f in as_completed(futures)] for alg, futures in all_futures.items()}


def _run_trial(
    algorithm: DstAlgorithm,
    nw_init_state: Network,
    progress_bar_ctrl: ProgressBarController,
) -> Network:
    progress_bar_ctrl.start_progress_bar(algorithm)
    network = deepcopy(nw_init_state)
    with warnings.catch_warnings(action="error"):
        try:
            algorithm.run(network)
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {algorithm.name}: {type(e).__name__}: {e}")
    progress_bar_ctrl.advance_progress_bar(algorithm)
    return network
