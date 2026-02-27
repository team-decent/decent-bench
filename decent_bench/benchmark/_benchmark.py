import logging
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from json import JSONDecodeError
from logging.handlers import QueueListener
from multiprocessing import Manager, get_context
from time import sleep
from typing import TYPE_CHECKING, Any

from rich.status import Status

from decent_bench.benchmark._benchmark_result import BenchmarkResult
from decent_bench.benchmark_problem import BenchmarkProblem
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.metrics import RuntimeMetricPlotter
from decent_bench.networks import Network, P2PNetwork, create_distributed_network
from decent_bench.utils import logger
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.progress_bar import ProgressBarController

if TYPE_CHECKING:
    import queue
    from multiprocessing.context import SpawnContext
    from multiprocessing.managers import SyncManager

    from decent_bench.metrics import RuntimeMetric
    from decent_bench.utils.checkpoint_manager import CheckpointManager
    from decent_bench.utils.progress_bar import ProgressBarHandle


def resume_benchmark(  # noqa: PLR0912
    checkpoint_manager: "CheckpointManager",
    increase_iterations: int = 0,
    increase_trials: int = 0,
    create_backup: bool = True,
    *,
    max_processes: int | None = 1,
    progress_step: int | None = 100,
    show_speed: bool = False,
    show_trial: bool = False,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
    log_level: int = logging.INFO,
) -> BenchmarkResult:
    """
    Resume a benchmark from an existing checkpoint directory.

    Args:
        checkpoint_manager: CheckpointManager instance to load checkpoints from. Must contain valid checkpoints and
            metadata from a previous benchmark run. Progress will be loaded from the latest checkpoints and the
            benchmark will resume from there.
        increase_iterations: number of iterations to add to each algorithm's existing iteration count. This allows
            you to extend the benchmark run and collect more data points for the metrics. The additional iterations will
            be added on top of the existing iterations defined in the checkpoint metadata for each algorithm. If set to
            0 (default), the benchmark will resume with the same number of iterations as defined in the checkpoint.
        increase_trials: number of additional trials to run for each algorithm. This allows you to increase the
            statistical significance of the benchmark results by collecting more trials. If set to 0 (default), the
            benchmark will resume with the same number of trials as defined in the checkpoint.
        create_backup: whether to create a backup of the existing checkpoint directory before resuming. It is
            recommended to set this to True to avoid accidental data loss, as resuming will modify the checkpoint
            directory by adding new checkpoints and metadata. If True, a backup will be created with the name
            ``{checkpoint_manager.checkpoint_dir}_backup_{timestamp}.zip`` before resuming.
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default. If your algorithm is very lightweight you
            may want to set this to 1 to avoid the multiprocessing overhead.
        progress_step: if provided, the progress bar will step every `progress_step` iterations.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.
        show_speed: whether to show speed (iterations/second) in the progress bar.
        show_trial: whether to show which trials are currently running in the progress bar.
        runtime_metrics: optional list of :class:`~decent_bench.metrics.RuntimeMetric` to compute and plot during
            algorithm execution. Each metric will open a plot window for each trial showing live updates. Useful for
            early stopping if convergence is not happening. Disabled by default. When using multiprocessing
            (``max_processes > 1``), each trial will open its own plot windows in separate processes.
        log_level: minimum level to log, e.g. :data:`logging.INFO`

    Returns:
        BenchmarkResult containing the results of the benchmark.

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

    Raises:
        ValueError: If the checkpoint directory does not exist, is empty, or contains invalid metadata.
        ValueError: If increase_iterations or increase_trials is negative.

    """
    if not checkpoint_manager.checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory '{checkpoint_manager.checkpoint_dir}' does not exist for resume")
    if checkpoint_manager.is_empty():
        raise ValueError(f"Checkpoint directory '{checkpoint_manager.checkpoint_dir}' is empty or invalid for resume")
    if increase_iterations < 0:
        raise ValueError("increase_iterations must be a non-negative integer")
    if increase_trials < 0:
        raise ValueError("increase_trials must be a non-negative integer")

    with Status("Loading benchmark state from checkpoint..."):
        try:
            metadata = checkpoint_manager.load_metadata()
            if metadata is None or "n_trials" not in metadata:
                raise ValueError("Invalid or missing metadata in checkpoint directory")

            algorithms = checkpoint_manager.load_initial_algorithms()
            if algorithms is None:
                raise ValueError("Initial algorithms not found in checkpoint metadata")

            problem = checkpoint_manager.load_benchmark_problem()
            if problem is None:
                raise ValueError("Benchmark problem not found in checkpoint metadata")

            nw_init_state = checkpoint_manager.load_initial_network()
            if nw_init_state is None:
                raise ValueError("Initial network state not found in checkpoint metadata")

            log_listener, manager, mp_context = _init_logging_and_multiprocessing(log_level, max_processes, problem)

            LOGGER.debug(f"Loaded checkpoint: algorithms={algorithms}")
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Invalid checkpoint directory: missing or corrupted metadata - {e}") from e
        except JSONDecodeError as e:
            raise ValueError(f"Invalid checkpoint directory: metadata is not valid JSON - {e}") from e

    if create_backup:
        backup_path = checkpoint_manager.create_backup()
        LOGGER.info(f"Created backup of checkpoint directory at '{backup_path}'")

    LOGGER.info(
        f"Resuming benchmark from checkpoint '{checkpoint_manager.checkpoint_dir}' with {metadata['n_trials']} trials "
        f"and algorithms: {[alg.name for alg in algorithms]}\n"
    )

    total_increase_trials = increase_trials + metadata.get("benchmark_metadata", {}).get("increased_trials", 0)
    n_trials = metadata["n_trials"] + total_increase_trials
    if increase_trials != 0:
        metadata = checkpoint_manager.append_metadata({"increased_trials": total_increase_trials})
        LOGGER.info(
            f"Increasing number of trials for each algorithm by {increase_trials}, "
            f"total increase is {total_increase_trials}"
        )

    total_increase_iterations = increase_iterations + metadata.get("benchmark_metadata", {}).get(
        "increased_iterations", 0
    )
    if increase_iterations != 0:
        for alg_idx, alg in enumerate(algorithms):
            alg.iterations += total_increase_iterations
            # Unmark all trials as incomplete to resume them with increased iterations
            for trial in range(n_trials):
                checkpoint_manager.unmark_trial_complete(alg_idx, trial)
        # If we resume again, we have to increase the iterations on top of the already increased iterations,
        # so we need to keep track of the total increase in the metadata
        metadata = checkpoint_manager.append_metadata({"increased_iterations": total_increase_iterations})
        LOGGER.info(
            f"Increased iterations for all algorithms by {increase_iterations}, "
            f"total increase is {total_increase_iterations}"
        )

    results = _benchmark(
        algorithms=algorithms,
        benchmark_problem=problem,
        nw_init_state=nw_init_state,
        log_listener=log_listener,
        manager=manager,
        mp_context=mp_context,
        n_trials=n_trials,
        max_processes=max_processes,
        progress_step=progress_step,
        show_speed=show_speed,
        show_trial=show_trial,
        checkpoint_manager=checkpoint_manager,
        runtime_metrics=runtime_metrics,
    )
    log_listener.stop()
    return results


def benchmark(
    algorithms: list[Algorithm],
    benchmark_problem: BenchmarkProblem,
    *,
    n_trials: int = 30,
    max_processes: int | None = 1,
    progress_step: int | None = 100,
    show_speed: bool = False,
    show_trial: bool = False,
    checkpoint_manager: "CheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
    log_level: int = logging.INFO,
) -> BenchmarkResult:
    """
    Benchmark decentralized algorithms.

    Args:
        algorithms: algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints.
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply.
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default. If your algorithm is very lightweight you
            may want to set this to 1 to avoid the multiprocessing overhead.
        progress_step: if provided, the progress bar will step every `progress_step` iterations.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.
        show_speed: whether to show speed (iterations/second) in the progress bar.
        show_trial: whether to show which trials are currently running in the progress bar.
        checkpoint_manager: if provided, will be used to save checkpoints during execution.
        runtime_metrics: optional list of :class:`~decent_bench.metrics.RuntimeMetric` to compute and plot during
            algorithm execution. Each metric will open a plot window for each trial showing live updates. Useful for
            early stopping if convergence is not happening. Disabled by default. When using multiprocessing
            (``max_processes > 1``), each trial will open its own plot windows in separate processes.
        log_level: minimum level to log, e.g. :data:`logging.INFO`.

    Returns:
        BenchmarkResult containing the results of the benchmark.

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

    Raises:
        ValueError: If the checkpoint directory is not empty when initializing the CheckpointManager.

    """
    log_listener, manager, mp_context = _init_logging_and_multiprocessing(log_level, max_processes, benchmark_problem)

    nw_init_state = create_distributed_network(benchmark_problem)
    LOGGER.debug("Created initial network state from benchmark problem configuration")

    if checkpoint_manager is not None:
        if not checkpoint_manager.is_empty():
            raise ValueError(
                f"Checkpoint directory '{checkpoint_manager.checkpoint_dir}' is not empty. "
                f"Please provide an empty or non-existent directory to save checkpoints."
            )

        checkpoint_manager.initialize(algorithms, nw_init_state, benchmark_problem, n_trials)
    else:
        LOGGER.info(
            "No checkpoint manager provided, running benchmark without checkpointing. "
            "Progress cannot be resumed if interrupted."
        )

    results = _benchmark(
        algorithms=algorithms,
        benchmark_problem=benchmark_problem,
        nw_init_state=nw_init_state,
        log_listener=log_listener,
        manager=manager,
        mp_context=mp_context,
        n_trials=n_trials,
        max_processes=max_processes,
        progress_step=progress_step,
        show_speed=show_speed,
        show_trial=show_trial,
        checkpoint_manager=checkpoint_manager,
        runtime_metrics=runtime_metrics,
    )
    log_listener.stop()
    return results


def _benchmark(
    algorithms: list[Algorithm],
    benchmark_problem: BenchmarkProblem,
    nw_init_state: Network,
    log_listener: QueueListener,
    manager: "SyncManager",
    *,
    mp_context: "SpawnContext | None" = None,
    n_trials: int = 30,
    max_processes: int | None = 1,
    progress_step: int | None = None,
    show_speed: bool = False,
    show_trial: bool = False,
    checkpoint_manager: "CheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
) -> BenchmarkResult:
    """
    Benchmark decentralized algorithms.

    Args:
        algorithms: algorithms to benchmark
        benchmark_problem: problem to benchmark on, defines the network topology, cost functions, and communication
            constraints.
        nw_init_state: initial state of the network to run the algorithms on.
        log_listener: multiprocessing logging listener to handle log messages from worker processes.
        manager: multiprocessing manager for sharing data between processes.
        mp_context: multiprocessing context to use for creating new processes.
        n_trials: number of times to run each algorithm on the benchmark problem, running more trials improves the
            statistical results, at least 30 trials are recommended for the central limit theorem to apply.
        max_processes: maximum number of processes to use when running trials, multiprocessing improves performance
            but can be inhibiting when debugging or using a profiler, set to 1 to disable multiprocessing or ``None`` to
            use :class:`~concurrent.futures.ProcessPoolExecutor`'s default. If your algorithm is very lightweight you
            may want to set this to 1 to avoid the multiprocessing overhead.
        progress_step: if provided, the progress bar will step every `progress_step` iterations.
            When provided, each algorithm's task total becomes `n_trials * ceil(algorithm.iterations / progress_step)`.
            If `None`, the progress bar uses 1 unit per trial.
        show_speed: whether to show speed (iterations/second) in the progress bar.
        show_trial: whether to show which trials are currently running in the progress bar.
        checkpoint_manager: if provided, will be used to save and load checkpoints during execution.
            If ``None``, no checkpoints will be saved and the benchmark will run from start to finish
            without resumption capability.
        runtime_metrics: optional list of :class:`~decent_bench.metrics.RuntimeMetric` to compute and plot during
            algorithm execution. Each metric will open a plot window for each trial showing live updates. Useful for
            early stopping if convergence is not happening. Disabled by default. When using multiprocessing
            (``max_processes > 1``), each trial will open its own plot windows in separate processes.

    Returns:
        BenchmarkResult containing the results of the benchmark.

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

    """
    LOGGER.info("Starting benchmark execution ")
    LOGGER.debug(f"Nr of agents: {len(nw_init_state.agents())}")
    prog_ctrl = ProgressBarController(manager, algorithms, n_trials, progress_step, show_speed, show_trial)
    resulting_nw_states = _run_trials(
        algorithms,
        n_trials,
        nw_init_state,
        benchmark_problem,
        prog_ctrl,
        log_listener,
        max_processes,
        mp_context,
        checkpoint_manager,
        runtime_metrics,
    )
    LOGGER.info("Benchmark execution complete, thanks for using decent-bench")
    return BenchmarkResult(problem=benchmark_problem, states=resulting_nw_states)


def _init_logging_and_multiprocessing(
    log_level: int,
    max_processes: int | None,
    benchmark_problem: BenchmarkProblem,
) -> tuple[QueueListener, "SyncManager", "SpawnContext | None"]:
    # Detect if PyTorch costs are being used to determine multiprocessing context
    if max_processes != 1:
        use_spawn = _should_use_spawn_context(benchmark_problem)
        mp_context = get_context("spawn") if use_spawn else None
    else:
        use_spawn = False
        mp_context = None

    manager = Manager() if not use_spawn else get_context("spawn").Manager()
    log_listener = logger.start_log_listener(manager, log_level)

    if use_spawn:
        LOGGER.debug("Using spawn multiprocessing context for PyTorch/JAX compatibility")

    return log_listener, manager, mp_context


def _run_trials(  # noqa: PLR0917
    algorithms: list[Algorithm],
    n_trials: int,
    nw_init_state: Network,
    problem: BenchmarkProblem,
    progress_bar_ctrl: ProgressBarController,
    log_listener: QueueListener,
    max_processes: int | None,
    mp_context: "SpawnContext | None" = None,
    checkpoint_manager: "CheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
) -> dict[Algorithm, list[Network]]:
    results: dict[Algorithm, list[Network]] = defaultdict(list)
    progress_bar_handle = progress_bar_ctrl.get_handle()

    # Create centralized plotter process and queue for runtime metrics
    runtime_plotter = None
    runtime_plotter_queue = None
    if runtime_metrics:
        ctx = mp_context if mp_context is not None else get_context()
        runtime_plotter_queue = ctx.Manager().Queue()

        # Create and start plotter in separate process using the same context
        runtime_plotter = RuntimeMetricPlotter(runtime_plotter_queue, ctx)
        runtime_plotter.start()

    # Filter out completed trials if checkpoint manager is provided, and load their results, otherwise run all trials
    # Used when resuming from a previous benchmark run, so that only incomplete trials are run and completed trial
    # results are loaded from the checkpoint directory
    to_run: dict[Algorithm, list[int]] = defaultdict(list)
    if checkpoint_manager is not None:
        for alg_idx, alg in enumerate(algorithms):
            completed_trials = checkpoint_manager.get_completed_trials(alg_idx, n_trials)
            incompleted_trials = [t for t in range(n_trials) if t not in completed_trials]
            if len(incompleted_trials) > 0:
                to_run[alg] = incompleted_trials

            # load completed trials
            for trial in completed_trials:
                _, net = checkpoint_manager.load_trial_result(alg_idx, trial)
                results[alg].append(net)
                progress_bar_ctrl.mark_one_trial_as_complete(alg, trial)
                LOGGER.debug(f"Loaded completed trial {trial} for algorithm {alg.name} from checkpoint")
    else:
        to_run = {alg: list(range(n_trials)) for alg in algorithms}

    LOGGER.debug(
        f"Trials to run: { {alg.name: trials for alg, trials in to_run.items()} }, "
        f"Trials completed: { {alg.name: len(results[alg]) for alg in algorithms} }"
    )

    if len(to_run) == 0:
        LOGGER.info("No trials are left to run!")
        progress_bar_ctrl.stop()
        return results

    if max_processes == 1:
        partial_result = {
            alg: [
                _run_trial(
                    alg,
                    nw_init_state,
                    problem,
                    progress_bar_handle,
                    trial,
                    alg_idx,
                    checkpoint_manager,
                    runtime_metrics,
                    runtime_plotter_queue,
                )
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
                        _run_trial,
                        alg,
                        nw_init_state,
                        problem,
                        progress_bar_handle,
                        trial,
                        alg_idx,
                        checkpoint_manager,
                        runtime_metrics,
                        runtime_plotter_queue,
                    )
                    for trial in to_run[alg]
                ]
                for alg_idx, alg in enumerate(algorithms)
            }
            partial_result = {alg: [f.result() for f in as_completed(futures)] for alg, futures in all_futures.items()}

    progress_bar_ctrl.stop()
    for alg in partial_result:
        results[alg].extend(partial_result[alg])

    # Clean up runtime plotter process
    if runtime_plotter is not None:
        runtime_plotter.shutdown()

    return results


def _run_trial(  # noqa: PLR0917
    algorithm: Algorithm,
    nw_init_state: Network,
    problem: BenchmarkProblem,
    progress_bar_handle: "ProgressBarHandle",
    trial: int,
    alg_idx: int,
    checkpoint_manager: "CheckpointManager | None" = None,
    runtime_metrics: "list[RuntimeMetric] | None" = None,
    runtime_plotter_queue: "queue.Queue[Any] | None" = None,
) -> Network:
    if checkpoint_manager is not None:
        checkpoint = checkpoint_manager.load_checkpoint(alg_idx, trial)
        if checkpoint is not None:
            alg, network, last_completed_iteration = checkpoint
            # Set iterations in case it is updated
            alg.iterations = algorithm.iterations
            # Resume from the next iteration after the last completed one
            # The checkpoint at iteration N contains the state AFTER step(N) completes,
            # so we should resume from iteration N+1
            start_iteration = last_completed_iteration + 1
            LOGGER.debug(
                f"Resuming {algorithm.name} trial {trial} from iteration {start_iteration}/{algorithm.iterations} "
                f"(loaded checkpoint from iteration {last_completed_iteration})"
            )
        else:
            start_iteration = 0
            network = deepcopy(nw_init_state)
            alg = deepcopy(algorithm)
    else:
        start_iteration = 0
        network = deepcopy(nw_init_state)
        alg = deepcopy(algorithm)

    progress_bar_handle.start_progress_bar(algorithm, trial, start_iteration)

    trial_runtime_metrics = _get_runtime_metrics(runtime_metrics, algorithm, trial, runtime_plotter_queue)

    def progress_callback(iteration: int) -> None:
        progress_bar_handle.advance_progress_bar(algorithm, iteration)
        if checkpoint_manager is not None and checkpoint_manager.should_checkpoint(iteration):
            checkpoint_manager.save_checkpoint(alg_idx, trial, iteration, alg, network)

        for metric in trial_runtime_metrics:
            if metric.should_update(iteration):
                try:
                    metric.update_plot(problem, network.agents(), iteration)
                except Exception as e:
                    LOGGER.warning(
                        f"Failed to update runtime metric {metric.description} at iteration {iteration}: {e}"
                    )

    if not isinstance(network, P2PNetwork):
        # Update this when support for federated learning or other types of networks is added
        raise TypeError(f"Expected network to be a P2PNetwork, got {type(network)}")

    with warnings.catch_warnings(action="error"):
        try:
            alg.run(network, start_iteration, progress_callback, skip_finalize=True)
            if checkpoint_manager is not None:
                checkpoint_manager.mark_trial_complete(
                    alg_idx, trial, algorithm.iterations - 1, algorithm=alg, network=network
                )
            # Now that checkpoint is saved, we can finalize to clean up memory
            alg.finalize(network)
        except Exception as e:
            LOGGER.exception(f"An error or warning occurred when running {alg.name}: {type(e).__name__}: {e}")

    return network


def _get_runtime_metrics(
    runtime_metrics: "list[RuntimeMetric] | None",
    algorithm: Algorithm,
    trial: int,
    runtime_queue: "queue.Queue[Any] | None" = None,
) -> list["RuntimeMetric"]:
    # Initialize runtime metrics if provided
    if runtime_metrics:
        if runtime_queue is None:
            LOGGER.warning("Runtime metrics provided but no runtime queue available, metrics will not be plotted")
            return []

        # Create deep copies to avoid sharing state between trials
        trial_runtime_metrics = [deepcopy(metric) for metric in runtime_metrics]
        for metric in trial_runtime_metrics:
            try:
                metric.initialize_plot(algorithm.name, trial, runtime_queue)
            except Exception as e:
                LOGGER.warning(f"Failed to initialize runtime metric {metric.description}: {e}")
    else:
        trial_runtime_metrics = []

    return trial_runtime_metrics


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
