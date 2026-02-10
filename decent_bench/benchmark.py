import logging
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from logging.handlers import QueueListener
from multiprocessing import Manager, get_context
from multiprocessing.context import BaseContext
from time import sleep
from typing import TYPE_CHECKING, Literal

from rich.status import Status

from decent_bench.agents import AgentMetricsView
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import ComputationalCost, Metric, create_plots, create_tables
from decent_bench.metrics import metric_collection as mc
from decent_bench.networks import P2PNetwork, create_distributed_network
from decent_bench.utils import logger
from decent_bench.utils.checkpoint_manager import CheckpointManager
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.progress_bar import ProgressBarController

if TYPE_CHECKING:
    from decent_bench.utils.progress_bar import ProgressBarHandle


def benchmark(
    algorithms: list[Algorithm],
    benchmark_problem: BenchmarkProblem,
    plot_metrics: list[Metric] | list[list[Metric]] = mc.DEFAULT_PLOT_METRICS,
    table_metrics: list[Metric] = mc.DEFAULT_TABLE_METRICS,
    table_fmt: Literal["grid", "latex"] = "grid",
    *,
    plot_grid: bool = True,
    individual_plots: bool = False,
    computational_cost: ComputationalCost | None = None,
    x_axis_scaling: float = 1e-4,
    n_trials: int = 30,
    confidence_level: float = 0.95,
    log_level: int = logging.INFO,
    max_processes: int | None = 1,
    progress_step: int | None = None,
    show_speed: bool = False,
    show_trial: bool = False,
    compare_iterations_and_computational_cost: bool = False,
    checkpoint_dir: str | None = None,
    checkpoint_step: int | None = None,
    keep_n_checkpoints: int = 3,
) -> None:
    """
    Benchmark decentralized algorithms.

    Args:
        algorithms: algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints
        plot_metrics: metrics to plot after the execution, defaults to
            :const:`~decent_bench.metrics.metric_collection.DEFAULT_PLOT_METRICS`.
            If a list of lists is provided, each inner list will be plotted in a separate figure. Otherwise up to 3
            metrics will be grouped and plotted in their own figure with subplots.
        table_metrics: metrics to tabulate as confidence intervals after the execution, defaults to
            :const:`~decent_bench.metrics.metric_collection.DEFAULT_TABLE_METRICS`
        table_fmt: table format, grid is suitable for the terminal while latex can be copy-pasted into a latex document
        plot_grid: whether to show grid lines on the plots
        individual_plots: whether to plot each metric in a separate figure
        computational_cost: computational cost settings for plot metrics, if ``None`` x-axis will be iterations instead
            of computational cost
        x_axis_scaling: scaling factor for computational cost x-axis, used to convert the cost units into more
            manageable units for plotting. Only used if ``computational_cost`` is provided.
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply
        confidence_level: confidence level for computing confidence intervals of the table metrics, expressed as a value
            between 0 and 1 (e.g., 0.95 for 95% confidence, 0.99 for 99% confidence). Higher values result in
            wider confidence intervals.
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
        checkpoint_dir: directory path to save checkpoints during execution. If provided, progress will be saved
            at regular intervals allowing resumption if interrupted. When starting a new benchmark
            the directory must be empty or non-existent. Plots and tables will be saved to the subfolder *results*.
        checkpoint_step: number of iterations between checkpoints within each trial. Only used if ``checkpoint_dir``
            is provided. If ``None``, only save checkpoint at the end of each trial. For long-running algorithms,
            set this to checkpoint during trial execution (e.g., every 1000 iterations).
        keep_n_checkpoints: maximum number of iteration checkpoints to keep per trial. Older checkpoints are
            automatically deleted to save disk space. Only applies to within-trial checkpoints, not final results.

    Important:
        Multiprocessing with certain frameworks (e.g., PyTorch) can lead to unexpected behavior due to how they handle
        multiprocessing. It is recommended to not use multiprocessing when benchmarking algorithms that utilize such
        frameworks. If you choose to use multiprocessing with such frameworks, please ensure that you understand
        the potential issues and have taken appropriate measures. Decent-Bench will attempt to detect if any
        cost function is a PyTorchCost and warn the user accordingly. Multiprocessing is mostly intended to be used
        with Numpy-based implementations. Feel free to try using multiprocessing by setting ``max_processes`` to a value
        other than ``1`` to see if it works with your specific algorithm and setup. See the documentation for
        ``max_processes`` for available options.

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

    Raises:
        ValueError: If the checkpoint directory is not empty when initializing the CheckpointManager.

    """
    # Detect if PyTorch costs are being used to determine multiprocessing context
    if max_processes != 1:
        use_spawn = _should_use_spawn_context(benchmark_problem)
        mp_context = get_context("spawn") if use_spawn else None
    else:
        use_spawn = False
        mp_context = None

    checkpoint_manager = None
    if checkpoint_dir is not None:
        checkpoint_manager = CheckpointManager(checkpoint_dir, checkpoint_step, keep_n_checkpoints)
        if not checkpoint_manager.is_empty():
            raise ValueError(
                f"Checkpoint directory '{checkpoint_dir}' is not empty. "
                f"Please provide an empty or non-existent directory to save checkpoints."
            )

    manager = Manager() if not use_spawn else get_context("spawn").Manager()
    log_listener = logger.start_log_listener(manager, log_level)
    LOGGER.info("Starting benchmark execution ")
    if use_spawn:
        LOGGER.debug("Using spawn multiprocessing context for PyTorch/JAX compatibility")
    with Status("Generating initial network state"):
        nw_init_state = create_distributed_network(benchmark_problem)
    LOGGER.debug(f"Nr of agents: {len(nw_init_state.agents())}")

    if checkpoint_manager is not None:
        benchmark_metadata = {
            "n_trails": n_trials,
            "checkpoint_step": checkpoint_step,
            "keep_n_checkpoints": keep_n_checkpoints,
        }
        checkpoint_manager.initialize(algorithms, nw_init_state, benchmark_metadata)

    prog_ctrl = ProgressBarController(manager, algorithms, n_trials, progress_step, show_speed, show_trial)
    resulting_nw_states = _run_trials(
        algorithms,
        n_trials,
        nw_init_state,
        prog_ctrl,
        log_listener,
        max_processes,
        mp_context,
        checkpoint_manager,
    )
    LOGGER.info("All trials complete")
    resulting_agent_states: dict[Algorithm, list[list[AgentMetricsView]]] = {}
    for alg, networks in resulting_nw_states.items():
        resulting_agent_states[alg] = [[AgentMetricsView.from_agent(a) for a in nw.agents()] for nw in networks]
    create_tables(
        resulting_agent_states,
        benchmark_problem,
        table_metrics,
        confidence_level,
        table_fmt,
        table_path=checkpoint_manager.get_results_path("table_results.txt") if checkpoint_manager else None,
    )
    create_plots(
        resulting_agent_states,
        benchmark_problem,
        plot_metrics,
        computational_cost=computational_cost,
        x_axis_scaling=x_axis_scaling,
        compare_iterations_and_computational_cost=compare_iterations_and_computational_cost,
        individual_plots=individual_plots,
        plot_path=checkpoint_manager.get_results_path("plots.png") if checkpoint_manager else None,
        plot_grid=plot_grid,
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
    checkpoint_manager: CheckpointManager | None = None,
) -> dict[Algorithm, list[P2PNetwork]]:
    results: dict[Algorithm, list[P2PNetwork]] = defaultdict(list)
    progress_bar_handle = progress_bar_ctrl.get_handle()

    # Filter out completed trials if checkpoint manager is provided, and load their results, otherwise run all trials
    # Used when resuming from a previous benchmark run, so that only incomplete trials are run and completed trial
    # results are loaded from the checkpoint directory
    to_run: dict[Algorithm, list[int]] = defaultdict(list)
    if checkpoint_manager is not None:
        for alg_idx, alg in enumerate(algorithms):
            completed_trials = checkpoint_manager.get_completed_trials(alg_idx, n_trials)
            incompleted_trials = [t for t in range(n_trials) if t not in completed_trials]
            to_run[alg] = incompleted_trials

            # load completed trials
            for trial in completed_trials:
                results[alg].append(checkpoint_manager.load_trial_result(alg_idx, trial))
                LOGGER.debug(f"Loaded completed trial {trial} for algorithm {alg.name} from checkpoint")
    else:
        to_run = {alg: list(range(n_trials)) for alg in algorithms}

    if max_processes == 1:
        partial_result = {
            alg: [
                _run_trial(alg, nw_init_state, progress_bar_handle, trial, alg_idx, checkpoint_manager)
                for trial in to_run[alg]
            ]
            for alg_idx, alg in enumerate(algorithms)
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
                    executor.submit(
                        _run_trial, alg, nw_init_state, progress_bar_handle, trial, alg_idx, checkpoint_manager
                    )
                    for trial in to_run[alg]
                ]
                for alg_idx, alg in enumerate(algorithms)
            }
            partial_result = {alg: [f.result() for f in as_completed(futures)] for alg, futures in all_futures.items()}

    progress_bar_ctrl.stop()
    for alg in partial_result:
        results[alg].extend(partial_result[alg])

    return results


def _run_trial(  # noqa: PLR0917
    algorithm: Algorithm,
    nw_init_state: P2PNetwork,
    progress_bar_handle: "ProgressBarHandle",
    trial: int,
    alg_idx: int,
    checkpoint_manager: CheckpointManager | None = None,
) -> P2PNetwork:
    progress_bar_handle.start_progress_bar(algorithm, trial)

    if checkpoint_manager is not None:
        checkpoint = checkpoint_manager.load_checkpoint(alg_idx, trial)
        if checkpoint is not None:
            alg, network, start_iteration = checkpoint
            LOGGER.debug(
                f"Resuming {algorithm.name} trial {trial} from iteration {start_iteration}/{algorithm.iterations}"
            )
        else:
            start_iteration = 0
            network = deepcopy(nw_init_state)
            alg = deepcopy(algorithm)
    else:
        start_iteration = 0
        network = deepcopy(nw_init_state)
        alg = deepcopy(algorithm)

    def progress_callback(iteration: int) -> None:
        progress_bar_handle.advance_progress_bar(algorithm, iteration)
        if checkpoint_manager is not None and checkpoint_manager.should_checkpoint(algorithm.iterations, iteration):
            checkpoint_manager.save_checkpoint(alg_idx, trial, iteration, alg, network)

    alg_failed = False
    with warnings.catch_warnings(action="error"):
        try:
            alg.run(network, start_iteration, progress_callback)
        except Exception as e:
            alg_failed = True
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")

    if not alg_failed and checkpoint_manager is not None:
        checkpoint_manager.mark_trial_complete(alg_idx, trial, algorithm.iterations - 1, algorithm=alg, network=network)

    return network


def _should_use_spawn_context(benchmark_problem: BenchmarkProblem) -> bool:
    """Check if any cost function is a PyTorchCost, which requires spawn context."""
    try:
        from decent_bench.costs import PyTorchCost  # noqa: PLC0415

        if any(isinstance(cost, PyTorchCost) for cost in benchmark_problem.costs):
            LOGGER.warning(
                "It is not recommended to use use multiprocessing with PyTorchCost, "
                "may cause unexpected behavior. Consider setting max_processes=1 to disable multiprocessing.\n"
                "Execution will continue in 5 seconds, Ctrl+C to abort..."
            )
            sleep(5)  # Sleep to give the user a chance to read the warning

            return True
    except ImportError:
        return False
    return False
