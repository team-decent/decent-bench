import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from logging.handlers import QueueListener
from multiprocessing import Manager, get_context
from multiprocessing.context import BaseContext
from typing import TYPE_CHECKING, Literal

from rich.status import Status

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import plot_metrics as pm
from decent_bench.metrics import table_metrics as tm
from decent_bench.metrics.plot_metrics import DEFAULT_PLOT_METRICS, PlotMetric
from decent_bench.metrics.table_metrics import DEFAULT_TABLE_METRICS, TableMetric
from decent_bench.networks import P2PNetwork, create_distributed_network
from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.progress_bar import ProgressBarController

if TYPE_CHECKING:
    from decent_bench.utils.progress_bar import ProgressBarHandle


def benchmark(
    algorithms: list[Algorithm],
    benchmark_problem: BenchmarkProblem,
    plot_metrics: list[PlotMetric] = DEFAULT_PLOT_METRICS,
    table_metrics: list[TableMetric] = DEFAULT_TABLE_METRICS,
    table_fmt: Literal["grid", "latex"] = "grid",
    *,
    plot_grid: bool = True,
    plot_path: str | None = None,
    computational_cost: pm.ComputationalCost | None = None,
    x_axis_scaling: float = 1e-4,
    n_trials: int = 30,
    confidence_level: float = 0.95,
    log_level: int = logging.INFO,
    max_processes: int | None = None,
    progress_step: int | None = None,
    show_speed: bool = False,
    show_trial: bool = False,
    compare_iterations_and_computational_cost: bool = False,
) -> None:
    """
    Benchmark distributed algorithms.

    Args:
        algorithms: algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints
        plot_metrics: metrics to plot after the execution, defaults to
            :const:`~decent_bench.metrics.plot_metrics.DEFAULT_PLOT_METRICS`
        table_metrics: metrics to tabulate as confidence intervals after the execution, defaults to
            :const:`~decent_bench.metrics.table_metrics.DEFAULT_TABLE_METRICS`
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        plot_grid: whether to show grid lines on the plots
        plot_path: optional file path to save the generated plot as an image file (e.g., "plots.png"). If ``None``,
            the plot will only be displayed
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        x_axis_scaling: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
        confidence_level: confidence level of the confidence intervals
        log_level: minimum level to log, e.g. :data:`logging.INFO`
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default. If your algorithm is very lightweight you
            may want to set this to 1 to avoid the multiprocessing overhead.
        progress_step: if provided, the progress bar will step every `progress_step` iterations.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.
        show_speed: whether to show speed (iterations/second) in the progress bar.
        show_trial: whether to show which trials are currently running in the progress bar.
        compare_iterations_and_computational_cost: whether to plot both metric vs computational cost and
            metric vs iterations. Only used if ``computational_cost`` is provided.

    Note:
        If ``progress_step`` is too small performance may degrade due to the
        overhead of updating the progress bar too often.

        Computational cost can be interpreted as the cost of running the algorithm on a specific hardware setup.
        Therefore the computational cost could be seen as the number of operations performed (similar to FLOPS) but
        weighted by the time or energy it takes to perform them on the specific hardware.

        .. include:: snippets/computational_cost.rst

        If ``computational_cost`` is provided and ``compare_iterations_and_computational_cost`` is ``True``, each metric
        will be plotted twice: once against computational cost and once against iterations.
        Computational cost plots will be shown on the left and iteration plots on the right.

    """
    # Detect if PyTorch costs are being used to determine multiprocessing context
    if max_processes != 1:
        use_spawn = _should_use_spawn_context(benchmark_problem)
        mp_context = get_context("spawn") if use_spawn else None
    else:
        use_spawn = False
        mp_context = None

    manager = Manager() if not use_spawn else get_context("spawn").Manager()
    log_listener = logger.start_log_listener(manager, log_level)
    LOGGER.info("Starting benchmark execution ")
    if use_spawn:
        LOGGER.debug("Using spawn multiprocessing context for PyTorch compatibility")
    with Status("Generating initial network state"):
        nw_init_state = create_distributed_network(benchmark_problem)
    LOGGER.debug(f"Nr of agents: {len(nw_init_state.agents())}")
    prog_ctrl = ProgressBarController(manager, algorithms, n_trials, progress_step, show_speed, show_trial)
    resulting_nw_states = _run_trials(
        algorithms, n_trials, nw_init_state, prog_ctrl, log_listener, max_processes, mp_context
    )
    LOGGER.info("All trials complete")
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]] = {}
    for alg, networks in resulting_nw_states.items():
        resulting_agent_states[alg] = [[AgentMetricsView.from_agent(a) for a in nw.agents()] for nw in networks]
    tm.tabulate(resulting_agent_states, benchmark_problem, table_metrics, confidence_level, table_fmt)
    pm.plot(
        resulting_agent_states,
        benchmark_problem,
        plot_metrics,
        computational_cost,
        x_axis_scaling,
        compare_iterations_and_computational_cost,
        plot_path,
        plot_grid,
    )
    LOGGER.info("Benchmark execution complete, thanks for using decent-bench")
    log_listener.stop()


def _run_trials(  # noqa: PLR0917
    algorithms: list[Algorithm],
    n_trials: int,
    nw_init_state: P2PNetwork,
    progress_bar_ctrl: ProgressBarController,
    log_listener: QueueListener,
    max_processes: int | None,
    mp_context: BaseContext | None = None,
) -> dict[Algorithm, list[P2PNetwork]]:
    progress_bar_handle = progress_bar_ctrl.get_handle()
    if max_processes == 1:
        result = {
            alg: [_run_trial(alg, nw_init_state, progress_bar_handle, trial) for trial in range(n_trials)]
            for alg in algorithms
        }
    else:
        with ProcessPoolExecutor(
            initializer=logger.start_queue_logger,
            initargs=(log_listener.queue,),
            max_workers=max_processes,
            mp_context=mp_context,
        ) as executor:
            LOGGER.debug(f"Concurrent processes: {executor._max_workers}")  # type: ignore[attr-defined] # noqa: SLF001
            all_futures = {
                alg: [
                    executor.submit(_run_trial, alg, nw_init_state, progress_bar_handle, trial)
                    for trial in range(n_trials)
                ]
                for alg in algorithms
            }
            result = {alg: [f.result() for f in as_completed(futures)] for alg, futures in all_futures.items()}

    progress_bar_ctrl.stop()
    return result


def _run_trial(
    algorithm: Algorithm,
    nw_init_state: P2PNetwork,
    progress_bar_handle: "ProgressBarHandle",
    trial: int,
) -> P2PNetwork:
    progress_bar_handle.start_progress_bar(algorithm, trial)
    network = deepcopy(nw_init_state)
    alg = deepcopy(algorithm)

    with warnings.catch_warnings(action="error"):
        try:
            alg.run(network, lambda iteration: progress_bar_handle.advance_progress_bar(algorithm, iteration))
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")
    return network


def _should_use_spawn_context(benchmark_problem: BenchmarkProblem) -> bool:
    """
    Check if any cost function is a PyTorchCost, which requires spawn context.

    Raises:
        RuntimeError: if user chooses not to continue when PyTorchCost is detected.

    """
    try:
        from decent_bench.costs import PyTorchCost  # noqa: PLC0415

        if any(isinstance(cost, PyTorchCost) for cost in benchmark_problem.costs):
            LOGGER.warning(
                "It is not recommended to use use multiprocessing with PyTorchCost, "
                "may cause unexpected behavior. Consider setting max_processes=1 to disable multiprocessing."
            )
            ans = input("Are you sure you want to continue? (y/n): ")
            if ans.lower() != "y":
                raise RuntimeError("Benchmarking aborted by user.")

            return True
    except ImportError:
        return False
    return False
