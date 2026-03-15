from collections.abc import Sequence
from functools import reduce
from operator import add

import numpy as np

from decent_bench import centralized_algorithms as ca
from decent_bench.costs import Cost, LinearRegressionCost, LogisticRegressionCost, PyTorchCost, QuadraticCost
from decent_bench.datasets import SyntheticClassificationDatasetHandler, SyntheticRegressionDatasetHandler
from decent_bench.utils.array import Array
from decent_bench.utils.types import Dataset, EmpiricalRiskBatchSize, SupportedDevices, SupportedFrameworks

ran = np.random.default_rng()  # replace with iop tool


def create_classification_problem(
    cost_cls: type[LogisticRegressionCost | PyTorchCost] = LogisticRegressionCost,
    n_agents: int = 100,
    batch_size: EmpiricalRiskBatchSize = "all",
) -> tuple[Sequence[Cost], Array | None, Dataset]:
    """
    Create out-of-the-box classification problems.

    Args:
        cost_cls: type of cost function
        n_agents: number of agents
        batch_size: size of mini-batches for stochastic methods, or "all" for full-batch

    Note:
        If cost_cls is :class:`~decent_bench.costs.PyTorchCost`, x_optimal is not computed and set to None.
        Be aware that metrics that rely on x_optimal (e.g. :class:`~decent_bench.metrics.metric_library.Regret`)
        will not be available when using PyTorchCost.

    Raises:
        ValueError: if an unsupported cost class is provided
        ImportError: if PyTorchCost is selected but PyTorch is not installed

    """
    dataset = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_partitions=n_agents,
        n_samples_per_partition=10,
        n_features=3,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=SupportedDevices.CPU,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        squeeze_targets=cost_cls is PyTorchCost,  # PyTorchCost expects squeezed targets for CrossEntropyLoss
        seed=0,
    )
    test_data = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_partitions=1,
        n_samples_per_partition=100,  # 1 partition so this is number of samples in test set
        n_features=3,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=SupportedDevices.CPU,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        squeeze_targets=cost_cls is PyTorchCost,
        seed=12345,
    )

    if cost_cls is PyTorchCost:
        try:
            import torch  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("PyTorch must be installed to use PyTorchCost") from e

        from decent_bench.utils.pytorch_utils import ArgmaxActivation, SimpleLinearModel  # noqa: PLC0415

        def model_gen() -> torch.nn.Module:
            return SimpleLinearModel(
                input_size=3,
                hidden_sizes=[],
                activation=None,
                output_size=2,
            )

        # Mypy cannot infer that cost_cls is PyTorchCost here
        costs = [
            cost_cls(  # type: ignore[call-arg]
                dataset=p,
                model=model_gen(),
                loss_fn=torch.nn.CrossEntropyLoss(),
                final_activation=ArgmaxActivation(),
                batch_size=batch_size,
            )
            for p in dataset.get_partitions()
        ]
        x_optimal = None
    elif cost_cls is LogisticRegressionCost:
        costs = [cost_cls(dataset=p, batch_size=batch_size) for p in dataset.get_partitions()]  # type: ignore[call-arg]
        sum_cost = reduce(add, costs)
        x_optimal = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)
    else:
        raise ValueError(f"Unsupported cost class: {cost_cls}")

    return costs, x_optimal, test_data.get_datapoints()


def create_regression_problem(
    cost_cls: type[LinearRegressionCost | PyTorchCost] = LinearRegressionCost,
    n_agents: int = 100,
    batch_size: EmpiricalRiskBatchSize = "all",
) -> tuple[Sequence[Cost], Array | None, Dataset]:
    """
    Create out-of-the-box regression problems.

    Args:
        cost_cls: type of cost function
        n_agents: number of agents
        batch_size: size of mini-batches for stochastic methods, or "all" for full-batch

    Note:
        If cost_cls is :class:`~decent_bench.costs.PyTorchCost`, x_optimal is not computed and set to None.
        Be aware that metrics that rely on x_optimal (e.g. :class:`~decent_bench.metrics.metric_library.Regret`)
        will not be available when using PyTorchCost.

    Raises:
        ValueError: if an unsupported cost class is provided
        ImportError: if PyTorchCost is selected but PyTorch is not installed

    """
    dataset = SyntheticRegressionDatasetHandler(
        n_targets=1,
        n_partitions=n_agents,
        n_samples_per_partition=10,
        n_features=1,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=SupportedDevices.CPU,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        target_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        seed=0,
    )
    test_data = SyntheticRegressionDatasetHandler(
        n_targets=1,
        n_partitions=1,
        n_samples_per_partition=100,  # 1 partition so this is number of samples in test set
        n_features=1,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=SupportedDevices.CPU,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        target_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        seed=12345,
    )
    if cost_cls is PyTorchCost:
        try:
            import torch  # noqa: PLC0415
        except ImportError as e:
            raise ImportError("PyTorch must be installed to use PyTorchCost") from e

        from decent_bench.utils.pytorch_utils import SimpleLinearModel  # noqa: PLC0415

        def model_gen() -> torch.nn.Module:
            return SimpleLinearModel(
                input_size=1,
                hidden_sizes=[],
                activation=None,
                output_size=1,
            )

        costs = [
            cost_cls(dataset=p, model=model_gen(), loss_fn=torch.nn.MSELoss(), batch_size=batch_size)  # type: ignore[call-arg]
            for p in dataset.get_partitions()
        ]
        x_optimal = None
    elif cost_cls is LinearRegressionCost:
        costs = [cost_cls(dataset=p, batch_size=batch_size) for p in dataset.get_partitions()]  # type: ignore[call-arg]
        sum_cost = reduce(add, costs)
        x_optimal = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)
    else:
        raise ValueError(f"Unsupported cost class: {cost_cls}")

    return costs, x_optimal, test_data.get_datapoints()


def create_quadratic_problem(
    size: int = 10,
    n_agents: int = 100,
) -> tuple[Sequence[Cost], Array]:
    """
    Create out-of-the-box quadratic problems.

    Args:
        size: number of dimensions
        n_agents: number of agents

    """
    A, b = [], []  # noqa: N806
    for _ in range(n_agents):
        A_i = ran.random((size, size))  # noqa: N806
        A.append((A_i + A_i.T) / 2 + size * np.eye(size))
        b.append(ran.normal(scale=10, size=(size,)))

    costs = [QuadraticCost(Array(A[i]), Array(b[i])) for i in range(n_agents)]
    sum_cost = reduce(add, costs)
    x_optimal = ca.accelerated_gradient_descent(sum_cost, x0=None, max_iter=50000, stop_tol=1e-100, max_tol=1e-16)

    return costs, x_optimal
