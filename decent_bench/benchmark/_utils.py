from collections.abc import Sequence
from functools import reduce
from operator import add

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench import centralized_algorithms as ca
from decent_bench.costs import Cost, LinearRegressionCost, LogisticRegressionCost, PyTorchCost, QuadraticCost
from decent_bench.datasets import SyntheticClassificationDatasetHandler, SyntheticRegressionDatasetHandler
from decent_bench.utils import logger
from decent_bench.utils.array import Array
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import Dataset, EmpiricalRiskBatchSize, SupportedDevices, SupportedFrameworks

SOLVE_MAX_ITER = 10000
SOLVE_STOP_TOL = 1e-20
SOLVE_MAX_TOL = 1e-16


def create_classification_problem(
    cost_cls: type[LogisticRegressionCost | PyTorchCost] = LogisticRegressionCost,
    device: SupportedDevices = SupportedDevices.CPU,
    n_agents: int = 100,
    batch_size: EmpiricalRiskBatchSize = "all",
) -> tuple[Sequence[Cost], Array | None, Dataset]:
    """
    Create out-of-the-box classification problems.

    Args:
        cost_cls: type of cost function
        device: device to create the problem on (only relevant for PyTorchCost)
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
    if not LOGGER.handlers:
        logger.start_logger()
    LOGGER.info("Creating cost functions ...")
    dataset = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_partitions=n_agents,
        n_samples_per_partition=10,
        n_features=3,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        squeeze_targets=cost_cls is PyTorchCost,  # PyTorchCost expects squeezed targets for CrossEntropyLoss
    )
    test_data = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_partitions=1,
        n_samples_per_partition=100,  # 1 partition so this is number of samples in test set
        n_features=3,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        squeeze_targets=cost_cls is PyTorchCost,
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
        pytorch_costs: list[PyTorchCost] = [
            PyTorchCost(
                dataset=p,
                model=model_gen(),
                loss_fn=torch.nn.CrossEntropyLoss(),
                final_activation=ArgmaxActivation(),
                batch_size=batch_size,
                device=device,
            )
            for p in dataset.get_partitions()
        ]
        LOGGER.info("... done!")
        costs: Sequence[Cost] = pytorch_costs
        x_optimal = None
    elif cost_cls is LogisticRegressionCost:
        classification_costs: list[LogisticRegressionCost] = [
            LogisticRegressionCost(dataset=p, batch_size=batch_size) for p in dataset.get_partitions()
        ]
        LOGGER.info("... done!")
        sum_cost = reduce(add, classification_costs)
        if sum_cost.batch_size < sum_cost.n_samples:
            sum_cost._batch_size = sum_cost.n_samples  # noqa: SLF001
        x_optimal = ca.solve(sum_cost, max_iter=SOLVE_MAX_ITER, stop_tol=SOLVE_STOP_TOL, max_tol=SOLVE_MAX_TOL)
        costs = classification_costs
    else:
        raise ValueError(f"Unsupported cost class: {cost_cls}")

    return costs, x_optimal, test_data.get_datapoints()


def create_regression_problem(
    cost_cls: type[LinearRegressionCost | PyTorchCost] = LinearRegressionCost,
    device: SupportedDevices = SupportedDevices.CPU,
    n_agents: int = 100,
    batch_size: EmpiricalRiskBatchSize = "all",
) -> tuple[Sequence[Cost], Array | None, Dataset]:
    """
    Create out-of-the-box regression problems.

    Args:
        cost_cls: type of cost function
        device: device to create the problem on (only relevant for PyTorchCost)
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
    if not LOGGER.handlers:
        logger.start_logger()
    LOGGER.info("Creating cost functions ...")
    dataset = SyntheticRegressionDatasetHandler(
        n_targets=1,
        n_partitions=n_agents,
        n_samples_per_partition=10,
        n_features=1,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        target_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
    )
    test_data = SyntheticRegressionDatasetHandler(
        n_targets=1,
        n_partitions=1,
        n_samples_per_partition=100,  # 1 partition so this is number of samples in test set
        n_features=1,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        target_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
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

        pytorch_costs: list[PyTorchCost] = [
            PyTorchCost(dataset=p, model=model_gen(), loss_fn=torch.nn.MSELoss(), batch_size=batch_size, device=device)
            for p in dataset.get_partitions()
        ]
        LOGGER.info("... done!")
        costs: Sequence[Cost] = pytorch_costs
        x_optimal = None
    elif cost_cls is LinearRegressionCost:
        regression_costs: list[LinearRegressionCost] = [
            LinearRegressionCost(dataset=p, batch_size=batch_size) for p in dataset.get_partitions()
        ]
        LOGGER.info("... done!")
        sum_cost = reduce(add, regression_costs)
        if sum_cost.batch_size < sum_cost.n_samples:
            sum_cost._batch_size = sum_cost.n_samples  # noqa: SLF001
        x_optimal = ca.solve(sum_cost, max_iter=SOLVE_MAX_ITER, stop_tol=SOLVE_STOP_TOL, max_tol=SOLVE_MAX_TOL)
        costs = regression_costs
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
    if not LOGGER.handlers:
        logger.start_logger()
    LOGGER.info("Creating cost functions ...")
    A, b = [], []  # noqa: N806
    for _ in range(n_agents):
        A_i = iop.uniform(shape=(size, size), framework=SupportedFrameworks.NUMPY, device=SupportedDevices.CPU)  # noqa: N806
        A.append((A_i + iop.transpose(A_i)) / 2 + size * iop.eye_like(A_i))
        b.append(iop.normal(shape=(size,), std=10, framework=SupportedFrameworks.NUMPY, device=SupportedDevices.CPU))

    costs = [QuadraticCost(A[i], b[i]) for i in range(n_agents)]
    LOGGER.info("... done!")

    sum_cost = reduce(add, costs)
    x_optimal = ca.solve(sum_cost, max_iter=SOLVE_MAX_ITER, stop_tol=SOLVE_STOP_TOL, max_tol=SOLVE_MAX_TOL)

    return costs, x_optimal
