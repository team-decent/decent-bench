import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.benchmark import (
    BenchmarkProblem,
    MetricResult,
    benchmark,
    create_classification_problem,
    resume_benchmark,
)
from decent_bench.costs import LogisticRegressionCost, PyTorchCost
from decent_bench.distributed_algorithms import ADMM, ATC, DGD, Algorithm
from decent_bench.networks import Network, P2PNetwork
from decent_bench.schemes import GaussianNoise, Quantization, UniformActivationRate, UniformDropRate
from decent_bench.utils.checkpoint_manager import CheckpointManager

try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

# Suppress JAX debug logs that cause issues during cleanup
logging.getLogger("jax").setLevel(logging.WARNING)


@dataclass(eq=False)
class DummyAlg(DGD):
    def finalize(self, network: P2PNetwork) -> None:
        for agent in network.agents():
            if agent.id == 0:
                print(f"Agent {agent.id} finalizing with x: {agent.x}")  # noqa: T201
            agent.x = agent.x + 100


def _build_problem_and_algorithms(
    iterations: int,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
) -> tuple[BenchmarkProblem, list[Algorithm[Any]]]:
    # Keep n_agents low to avoid expensive optimization in tests.
    costs, x_optimal, test_data = create_classification_problem(
        cost_cls=cost_cls,
        n_agents=4,
    )
    agents = [Agent(i, cost, activation=UniformActivationRate(0.8)) for i, cost in enumerate(costs)]
    network = P2PNetwork(
        graph=nx.complete_graph(len(agents)),
        agents=agents,
        message_compression=Quantization(8),
        message_noise=GaussianNoise(0.0, 0.01),
        message_drop=UniformDropRate(0.1),
    )
    problem = BenchmarkProblem(network=network, x_optimal=x_optimal, test_data=test_data)
    algorithms: list[Algorithm[Any]] = [
        DGD(iterations=iterations),
        ATC(iterations=iterations),
        DummyAlg(iterations=iterations, name="DummyAlg"),
    ] + (
        # ADMM does not work with PyTorchCost due to no Proximal
        [ADMM(iterations=iterations)] if cost_cls is LogisticRegressionCost else []
    )
    return problem, algorithms


def test_init_validates_arguments(tmp_path: Path) -> None:  # noqa: D103
    with pytest.raises(ValueError, match="checkpoint_step must be a positive integer or None"):
        CheckpointManager(tmp_path / "ckpt", checkpoint_step=0)

    with pytest.raises(ValueError, match="keep_n_checkpoints must be a positive integer"):
        CheckpointManager(tmp_path / "ckpt", keep_n_checkpoints=0)


def test_initialize_saves_structure_and_metadata(tmp_path: Path) -> None:  # noqa: D103
    checkpoint_dir = tmp_path / "ckpt"
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(
        checkpoint_dir,
        checkpoint_step=2,
        benchmark_metadata={"seed": 123},
    )

    manager.initialize(algorithms=algorithms, problem=problem, n_trials=3)

    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "initial_algorithms.pkl").exists()
    assert (checkpoint_dir / "benchmark_problem.pkl").exists()
    assert (checkpoint_dir / "algorithm_0").exists()
    assert (checkpoint_dir / "algorithm_1").exists()
    assert (checkpoint_dir / "algorithm_2").exists()

    metadata = manager.load_metadata()
    assert metadata["n_trials"] == 3
    assert metadata["benchmark_metadata"] == {"seed": 123}
    assert [alg["name"] for alg in metadata["algorithms"]] == ["DGD", "ATC", "DummyAlg", "ADMM"]

    loaded_algs = manager.load_initial_algorithms()
    assert [alg.name for alg in loaded_algs] == ["DGD", "ATC", "DummyAlg", "ADMM"]

    loaded_problem = manager.load_benchmark_problem()
    assert len(loaded_problem.network.agents()) == 4


def test_append_metadata_merges_entries(tmp_path: Path) -> None:  # noqa: D103
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(tmp_path / "ckpt", benchmark_metadata={"seed": 7})
    manager.initialize(algorithms=algorithms, problem=problem, n_trials=1)

    updated = manager.append_metadata({"host": "runner-1", "seed": 99})

    assert updated["benchmark_metadata"] == {"seed": 99, "host": "runner-1"}


def test_should_checkpoint_logic(tmp_path: Path) -> None:  # noqa: D103
    manager = CheckpointManager(tmp_path / "ckpt", checkpoint_step=3)

    assert manager.should_checkpoint(0) is False
    assert manager.should_checkpoint(1) is False
    assert manager.should_checkpoint(2) is True
    assert manager.should_checkpoint(5) is True

    with pytest.raises(ValueError, match="Iteration number must be non-negative"):
        manager.should_checkpoint(-1)

    manager_no_step = CheckpointManager(tmp_path / "ckpt_no_step", checkpoint_step=None)
    assert manager_no_step.should_checkpoint(100) is False


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:  # noqa: D103
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(tmp_path / "ckpt", keep_n_checkpoints=5)
    manager.initialize(algorithms=algorithms, problem=problem, n_trials=1)

    assert manager.load_checkpoint(alg_idx=0, trial=0) is None

    rng_state = {"seed": 123, "python_random_state": random.getstate()}

    checkpoint_path = manager.save_checkpoint(
        alg_idx=0,
        trial=0,
        iteration=4,
        algorithm=algorithms[0],
        network=problem.network,
        rng_state=rng_state,
    )

    assert checkpoint_path.exists()
    progress_path = tmp_path / "ckpt" / "algorithm_0" / "trial_0" / "progress.json"
    with progress_path.open(encoding="utf-8") as f:
        progress = json.load(f)
    assert progress["last_completed_iteration"] == 4

    loaded = manager.load_checkpoint(alg_idx=0, trial=0)
    assert loaded is not None
    loaded_alg, loaded_net, last_iteration, rng_state = loaded
    assert loaded_alg.name == "DGD"
    assert len(loaded_net.agents()) == 4
    assert last_iteration == 4
    assert rng_state == {"seed": 123, "python_random_state": random.getstate()}


def test_mark_unmark_and_load_trial_result(tmp_path: Path) -> None:  # noqa: D103
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(tmp_path / "ckpt")
    manager.initialize(algorithms=algorithms, problem=problem, n_trials=1)

    final_checkpoint = manager.mark_trial_complete(
        alg_idx=0,
        trial=0,
        iteration=5,
        algorithm=algorithms[0],
        network=problem.network,
        rng_state={"seed": 123, "python_random_state": random.getstate()},
    )
    assert final_checkpoint.exists()
    assert manager.is_trial_complete(alg_idx=0, trial=0) is True

    loaded_alg, loaded_net = manager.load_trial_result(alg_idx=0, trial=0)
    assert loaded_alg.name == "DGD"
    assert len(loaded_net.agents()) == 4

    manager.unmark_trial_complete(alg_idx=0, trial=0)
    assert manager.is_trial_complete(alg_idx=0, trial=0) is False


def test_cleanup_old_checkpoints_keeps_latest_n(tmp_path: Path) -> None:  # noqa: D103
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(tmp_path / "ckpt", keep_n_checkpoints=2)
    manager.initialize(algorithms=algorithms, problem=problem, n_trials=1)

    for iteration in (1, 2, 3):
        manager.save_checkpoint(
            alg_idx=0,
            trial=0,
            iteration=iteration,
            algorithm=algorithms[0],
            network=problem.network,
            rng_state={"seed": iteration},
        )

    trial_dir = tmp_path / "ckpt" / "algorithm_0" / "trial_0"
    checkpoints = sorted(p.name for p in trial_dir.glob("checkpoint_*.pkl"))
    assert checkpoints == ["checkpoint_0000002.pkl", "checkpoint_0000003.pkl"]

    loaded = manager.load_checkpoint(alg_idx=0, trial=0)
    assert loaded is not None
    _, _, iteration, _ = loaded
    assert iteration == 3


def test_load_benchmark_result_skips_incomplete_algorithms(  # noqa: D103
    tmp_path: Path,
) -> None:
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(tmp_path / "ckpt")
    manager.initialize(algorithms=algorithms, problem=problem, n_trials=2)

    manager.mark_trial_complete(
        alg_idx=0,
        trial=0,
        iteration=5,
        algorithm=algorithms[0],
        network=problem.network,
        rng_state={"seed": 123, "python_random_state": random.getstate()},
    )
    manager.mark_trial_complete(
        alg_idx=0,
        trial=1,
        iteration=5,
        algorithm=algorithms[0],
        network=problem.network,
        rng_state={"seed": 123, "python_random_state": random.getstate()},
    )
    manager.mark_trial_complete(
        alg_idx=1,
        trial=0,
        iteration=4,
        algorithm=algorithms[1],
        network=problem.network,
        rng_state={"seed": 123, "python_random_state": random.getstate()},
    )

    result = manager.load_benchmark_result()

    assert result.problem is not None
    assert len(result.states) == 1
    only_algorithm = next(iter(result.states.keys()))
    assert only_algorithm.name == "DGD"
    assert len(result.states[only_algorithm]) == 2


def test_save_and_load_metrics_result(tmp_path: Path) -> None:  # noqa: D103
    manager = CheckpointManager(tmp_path / "ckpt")
    metrics_result = MetricResult(
        agent_metrics=None,
        table_metrics=None,
        plot_metrics=None,
        table_results=None,
        plot_results=None,
    )

    manager.save_metrics_result(metrics_result)
    loaded = manager.load_metrics_result()

    assert loaded == metrics_result
    assert (tmp_path / "ckpt" / "metric_computation.pkl").exists()


def test_create_backup_and_clear(tmp_path: Path) -> None:  # noqa: D103
    problem, algorithms = _build_problem_and_algorithms(iterations=5, cost_cls=LogisticRegressionCost)
    manager = CheckpointManager(tmp_path / "ckpt")
    manager.initialize(algorithms=algorithms, problem=problem, n_trials=1)

    backup_path = manager.create_backup()
    assert backup_path.exists()
    assert backup_path.suffix == ".zip"

    manager.clear()
    assert not (tmp_path / "ckpt").exists()


@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize(
    ("cost_cls", "max_processes"),
    [
        (LogisticRegressionCost, 1),
        (LogisticRegressionCost, 2),
        pytest.param(
            PyTorchCost,
            1,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_resume_from_checkpoint_with_additional_trials(
    tmp_path: Path,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
    max_processes: int,
    seed: int | None,
) -> None:
    if os.cpu_count() is not None and max_processes > os.cpu_count():
        pytest.skip(f"max_processes={max_processes} exceeds available CPU cores")

    if seed is not None:
        iop.set_seed(seed)

    problem_1, algorithms_1 = _build_problem_and_algorithms(10, cost_cls=cost_cls)
    problem_2, algorithms_2 = deepcopy(problem_1), deepcopy(algorithms_1)

    manager = CheckpointManager(tmp_path / "ckpt", checkpoint_step=2)
    bench_1 = benchmark(
        algorithms=algorithms_1,
        benchmark_problem=problem_1,
        n_trials=1,
        checkpoint_manager=manager,
        max_processes=max_processes,
    )
    bench_2 = benchmark(
        algorithms=algorithms_2,
        benchmark_problem=problem_2,
        n_trials=2,
        max_processes=max_processes,
    )
    assert bench_1 is not None
    assert bench_2 is not None

    resumed_bench = resume_benchmark(
        checkpoint_manager=manager,
        increase_trials=1,
        max_processes=max_processes,
    )
    assert resumed_bench is not None

    # Check that the resumed benchmark has the expected number of iterations and trials.
    for alg in resumed_bench.states:
        assert alg.iterations == 10
        assert len(resumed_bench.states[alg]) == 2

    # Check that the resumed benchmark's problem matches the original.
    assert len(resumed_bench.problem.network.agents()) == 4

    # Check that the agent states and history is correct
    resumed_results: dict[tuple[str, int], Network] = {}
    for alg in resumed_bench.states:
        for i, trial_result in enumerate(resumed_bench.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            resumed_results[(alg.name, i)] = trial_result

    full_results: dict[tuple[str, int], Network] = {}
    for alg in bench_2.states:
        for i, trial_result in enumerate(bench_2.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            full_results[(alg.name, i)] = trial_result

    for key in resumed_results:
        resumed_trial = resumed_results[key]
        full_trial = full_results[key]
        for resumed_agent, full_agent in zip(
            resumed_trial.agents(),
            full_trial.agents(),
            strict=True,
        ):
            for iteration in range(11):
                np.testing.assert_allclose(
                    iop.to_numpy(resumed_agent._x_history[iteration]),
                    iop.to_numpy(full_agent._x_history[iteration]),
                )

            with pytest.raises(AssertionError):  # noqa: PT012
                for iteration in range(11):
                    resumed_agent._x_history[iteration] += 1.0
                    np.testing.assert_allclose(
                        iop.to_numpy(resumed_agent._x_history[iteration]),
                        iop.to_numpy(full_agent._x_history[iteration]),
                    )


@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize(
    ("cost_cls", "max_processes"),
    [
        (LogisticRegressionCost, 1),
        (LogisticRegressionCost, 2),
        pytest.param(
            PyTorchCost,
            1,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_resume_from_checkpoint_with_additional_iterations(
    tmp_path: Path,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
    max_processes: int,
    seed: int | None,
) -> None:
    if os.cpu_count() is not None and max_processes > os.cpu_count():
        pytest.skip(f"max_processes={max_processes} exceeds available CPU cores")

    if seed is not None:
        iop.set_seed(seed)

    problem_5, algorithms_5 = _build_problem_and_algorithms(5, cost_cls=cost_cls)
    problem_10, algorithms_10 = deepcopy(problem_5), deepcopy(algorithms_5)
    for alg in algorithms_10:
        alg.iterations = 10

    manager = CheckpointManager(tmp_path / "ckpt", checkpoint_step=2)
    bench_5 = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=2,
        checkpoint_manager=manager,
        max_processes=max_processes,
    )
    bench_10 = benchmark(
        algorithms=algorithms_10,
        benchmark_problem=problem_10,
        n_trials=2,
        max_processes=max_processes,
    )
    assert bench_5 is not None
    assert bench_10 is not None

    resumed_bench = resume_benchmark(
        checkpoint_manager=manager,
        increase_iterations=5,
        max_processes=max_processes,
    )
    assert resumed_bench is not None

    # Check that the resumed benchmark has the expected number of iterations and trials.
    for alg in resumed_bench.states:
        assert alg.iterations == 10
        assert len(resumed_bench.states[alg]) == 2

    # Check that the resumed benchmark's problem matches the original.
    assert len(resumed_bench.problem.network.agents()) == 4

    # Check that the agent states and history is correct
    resumed_results: dict[tuple[str, int], Network] = {}
    for alg in resumed_bench.states:
        for i, trial_result in enumerate(resumed_bench.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            resumed_results[(alg.name, i)] = trial_result

    full_results: dict[tuple[str, int], Network] = {}
    for alg in bench_10.states:
        for i, trial_result in enumerate(bench_10.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            full_results[(alg.name, i)] = trial_result

    for key in resumed_results:
        resumed_trial = resumed_results[key]
        full_trial = full_results[key]
        for resumed_agent, full_agent in zip(
            resumed_trial.agents(),
            full_trial.agents(),
            strict=True,
        ):
            for iteration in range(11):
                np.testing.assert_allclose(
                    iop.to_numpy(resumed_agent._x_history[iteration]),
                    iop.to_numpy(full_agent._x_history[iteration]),
                )

            with pytest.raises(AssertionError):  # noqa: PT012
                for iteration in range(11):
                    resumed_agent._x_history[iteration] += 1.0
                    np.testing.assert_allclose(
                        iop.to_numpy(resumed_agent._x_history[iteration]),
                        iop.to_numpy(full_agent._x_history[iteration]),
                    )


@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize(
    ("cost_cls", "max_processes"),
    [
        (LogisticRegressionCost, 1),
        (LogisticRegressionCost, 2),
        pytest.param(
            PyTorchCost,
            1,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_resume_from_checkpoint_with_additional_iterations_and_trials(
    tmp_path: Path,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
    max_processes: int,
    seed: int | None,
) -> None:
    if os.cpu_count() is not None and max_processes > os.cpu_count():
        pytest.skip(f"max_processes={max_processes} exceeds available CPU cores")

    if seed is not None:
        iop.set_seed(seed)

    problem_5, algorithms_5 = _build_problem_and_algorithms(5, cost_cls=cost_cls)
    problem_10, algorithms_10 = deepcopy(problem_5), deepcopy(algorithms_5)
    for alg in algorithms_10:
        alg.iterations = 10

    manager = CheckpointManager(tmp_path / "ckpt", checkpoint_step=2)
    bench_5 = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=1,
        checkpoint_manager=manager,
        max_processes=max_processes,
    )
    bench_10 = benchmark(
        algorithms=algorithms_10,
        benchmark_problem=problem_10,
        n_trials=2,
        max_processes=max_processes,
    )
    assert bench_5 is not None
    assert bench_10 is not None

    resumed_bench = resume_benchmark(
        checkpoint_manager=manager,
        increase_iterations=5,
        increase_trials=1,
        max_processes=max_processes,
    )
    assert resumed_bench is not None

    # Check that the resumed benchmark has the expected number of iterations and trials.
    for alg in resumed_bench.states:
        assert alg.iterations == 10
        assert len(resumed_bench.states[alg]) == 2

    # Check that the resumed benchmark's problem matches the original.
    assert len(resumed_bench.problem.network.agents()) == 4

    # Check that the agent states and history is correct
    resumed_results: dict[tuple[str, int], Network] = {}
    for alg in resumed_bench.states:
        for i, trial_result in enumerate(resumed_bench.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            resumed_results[(alg.name, i)] = trial_result

    full_results: dict[tuple[str, int], Network] = {}
    for alg in bench_10.states:
        for i, trial_result in enumerate(bench_10.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            full_results[(alg.name, i)] = trial_result

    for key in resumed_results:
        resumed_trial = resumed_results[key]
        full_trial = full_results[key]
        for resumed_agent, full_agent in zip(
            resumed_trial.agents(),
            full_trial.agents(),
            strict=True,
        ):
            for iteration in range(11):
                np.testing.assert_allclose(
                    iop.to_numpy(resumed_agent._x_history[iteration]),
                    iop.to_numpy(full_agent._x_history[iteration]),
                )

            with pytest.raises(AssertionError):  # noqa: PT012
                for iteration in range(11):
                    resumed_agent._x_history[iteration] += 1.0
                    np.testing.assert_allclose(
                        iop.to_numpy(resumed_agent._x_history[iteration]),
                        iop.to_numpy(full_agent._x_history[iteration]),
                    )


@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize(
    ("cost_cls", "max_processes"),
    [
        (LogisticRegressionCost, 1),
        (LogisticRegressionCost, 2),
        pytest.param(
            PyTorchCost,
            1,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_resume_from_non_completed_checkpoint(
    tmp_path: Path,
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
    max_processes: int,
    seed: int | None,
) -> None:
    if os.cpu_count() is not None and max_processes > os.cpu_count():
        pytest.skip(f"max_processes={max_processes} exceeds available CPU cores")

    if seed is not None:
        iop.set_seed(seed)

    problem_5, algorithms_5 = _build_problem_and_algorithms(5, cost_cls=cost_cls)
    problem_10, algorithms_10 = deepcopy(problem_5), deepcopy(algorithms_5)
    for alg in algorithms_10:
        alg.iterations = 10

    manager = CheckpointManager(tmp_path / "ckpt", checkpoint_step=2)
    bench_5 = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=2,
        checkpoint_manager=manager,
        max_processes=max_processes,
    )

    bench_10 = benchmark(
        algorithms=algorithms_10,
        benchmark_problem=problem_10,
        n_trials=2,
        max_processes=max_processes,
    )
    assert bench_5 is not None, "Expected bench_5 to be created successfully"
    assert bench_10 is not None, "Expected bench_10 to be created successfully"

    # print files in checkpoint directory for debugging
    print("Before")
    for path in (tmp_path / "ckpt").rglob("*"):
        if path.is_file():
            print(path.relative_to(tmp_path / "ckpt"))
        if path.name == "progress.json":
            with path.open(encoding="utf-8") as f:
                progress = json.load(f)
            print(f"Progress for algorithm_0 trial_0: {progress}")

    # Modify the checkpoint to simulate an interrupted run that did not mark the trial as complete
    for alg in range(len(algorithms_5) - 1):
        for trial in range(2):
            progress_path = tmp_path / "ckpt" / f"algorithm_{alg}" / f"trial_{trial}" / "progress.json"
            with progress_path.open(encoding="utf-8") as f:
                progress = json.load(f)
            progress["last_completed_iteration"] = 1  # Set to 2 less than total iterations
            with progress_path.open("w", encoding="utf-8") as f:
                json.dump(progress, f)
            complete_marker = tmp_path / "ckpt" / f"algorithm_{alg}" / f"trial_{trial}" / "complete.json"
            if complete_marker.exists():
                complete_marker.unlink()
            else:
                raise FileNotFoundError(f"Expected complete marker not found at {complete_marker}")
            for i in [3, 4]:  # Remove checkpoints for iterations 3 and 4
                checkpoint_path = tmp_path / "ckpt" / f"algorithm_{alg}" / f"trial_{trial}" / f"checkpoint_{i:07d}.pkl"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                else:
                    raise FileNotFoundError(f"Expected checkpoint not found at {checkpoint_path}")

    print("After")
    # print files in checkpoint directory for debugging
    for path in (tmp_path / "ckpt").rglob("*"):
        if path.is_file():
            print(path.relative_to(tmp_path / "ckpt"))
        if path.name == "progress.json":
            with path.open(encoding="utf-8") as f:
                progress = json.load(f)
            print(f"Progress for {path.parent.name}: {progress}")

    resumed_bench = resume_benchmark(
        checkpoint_manager=manager,
        increase_iterations=5,
        max_processes=max_processes,
    )
    assert resumed_bench is not None

    # Check that the resumed benchmark has the expected number of iterations and trials.
    for alg in resumed_bench.states:
        assert alg.iterations == 10
        assert len(resumed_bench.states[alg]) == 2

    # Check that the resumed benchmark's problem matches the original.
    assert len(resumed_bench.problem.network.agents()) == 4

    # Check that the agent states and history is correct
    resumed_results: dict[tuple[str, int], Network] = {}
    for alg in resumed_bench.states:
        for i, trial_result in enumerate(resumed_bench.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            resumed_results[(alg.name, i)] = trial_result

    full_results: dict[tuple[str, int], Network] = {}
    for alg in bench_10.states:
        for i, trial_result in enumerate(bench_10.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 11, (
                    f"Expected 11 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 10 iterations + initial state
            full_results[(alg.name, i)] = trial_result

    for key in resumed_results:
        resumed_trial = resumed_results[key]
        full_trial = full_results[key]
        for resumed_agent, full_agent in zip(
            resumed_trial.agents(),
            full_trial.agents(),
            strict=True,
        ):
            for iteration in range(11):
                np.testing.assert_allclose(
                    iop.to_numpy(resumed_agent._x_history[iteration]),
                    iop.to_numpy(full_agent._x_history[iteration]),
                )

            with pytest.raises(AssertionError):  # noqa: PT012
                for iteration in range(11):
                    resumed_agent._x_history[iteration] += 1.0
                    np.testing.assert_allclose(
                        iop.to_numpy(resumed_agent._x_history[iteration]),
                        iop.to_numpy(full_agent._x_history[iteration]),
                    )


@pytest.mark.parametrize(
    ("cost_cls", "max_processes"),
    [
        (LogisticRegressionCost, 1),
        (LogisticRegressionCost, 2),
        pytest.param(
            PyTorchCost,
            1,
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called.*:RuntimeWarning"
)  # Suppress warnings about fork in JAX during cleanup, causes the test to fail
def test_back_to_back_benchmarks(
    cost_cls: type[LogisticRegressionCost | PyTorchCost],
    max_processes: int,
) -> None:
    if os.cpu_count() is not None and max_processes > os.cpu_count():
        pytest.skip(f"max_processes={max_processes} exceeds available CPU cores")

    iop.set_seed(123)
    problem_5, algorithms_5 = _build_problem_and_algorithms(5, cost_cls=cost_cls)

    bench_1 = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=2,
        max_processes=max_processes,
    )

    iop.set_seed(123)
    problem_5, algorithms_5 = _build_problem_and_algorithms(5, cost_cls=cost_cls)
    bench_2 = benchmark(
        algorithms=algorithms_5,
        benchmark_problem=problem_5,
        n_trials=2,
        max_processes=max_processes,
    )
    assert bench_1 is not None, "Expected bench_1 to be created successfully"
    assert bench_2 is not None, "Expected bench_2 to be created successfully"

    # Check that the agent states and history is correct
    bench_1_results: dict[tuple[str, int], Network] = {}
    for alg in bench_1.states:
        for i, trial_result in enumerate(bench_1.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 6, (
                    f"Expected 6 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 5 iterations + initial state
            bench_1_results[(alg.name, i)] = trial_result

    bench_2_results: dict[tuple[str, int], Network] = {}
    for alg in bench_2.states:
        for i, trial_result in enumerate(bench_2.states[alg]):
            assert len(trial_result.agents()) == 4
            for agent in trial_result.agents():
                assert len(agent._x_history) == 6, (
                    f"Expected 6 iterations for agent {agent.id}, got {len(agent._x_history)}"
                )  # 5 iterations + initial state
            bench_2_results[(alg.name, i)] = trial_result

    for key in bench_1_results:
        resumed_trial = bench_1_results[key]
        full_trial = bench_2_results[key]
        for resumed_agent, full_agent in zip(
            resumed_trial.agents(),
            full_trial.agents(),
            strict=True,
        ):
            print(f"Trial: {key}")
            for iteration in range(6):
                print(resumed_agent._x_history[iteration], full_agent._x_history[iteration])
                np.testing.assert_allclose(
                    iop.to_numpy(resumed_agent._x_history[iteration]),
                    iop.to_numpy(full_agent._x_history[iteration]),
                )

            with pytest.raises(AssertionError):  # noqa: PT012
                for iteration in range(6):
                    resumed_agent._x_history[iteration] += 1.0
                    np.testing.assert_allclose(
                        iop.to_numpy(resumed_agent._x_history[iteration]),
                        iop.to_numpy(full_agent._x_history[iteration]),
                    )
