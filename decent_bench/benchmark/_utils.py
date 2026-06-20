from collections.abc import Sequence
from functools import reduce
from operator import add

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench import centralized_algorithms as ca
from decent_bench.costs import Cost, LinearRegressionCost, LogisticRegressionCost, PyTorchCost, QuadraticCost
from decent_bench.datasets import (
    SyntheticClassificationDatasetHandler,
    SyntheticRegressionDatasetHandler,
    split_iid,
)
from decent_bench.utils import logger
from decent_bench.utils.array import Array
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import Dataset, EmpiricalRiskBatchSize, SupportedDevices, SupportedFrameworks

SOLVE_MAX_ITER = 10000
SOLVE_STOP_TOL = 1e-20
SOLVE_MAX_TOL = 1e-16


def create_classification_problem(
    cost_cls: type[LogisticRegressionCost | PyTorchCost] = LogisticRegressionCost,
    *,
    device: SupportedDevices = SupportedDevices.CPU,
    n_agents: int = 100,
    batch_size: EmpiricalRiskBatchSize = "all",
    compute_x_optimal: bool = True,
    show_progress: bool = True,
) -> tuple[Sequence[Cost], Array | None, Dataset]:
    """
    Create out-of-the-box classification problems.

    Args:
        cost_cls: type of cost function
        device: device to create the problem on (only relevant for PyTorchCost)
        n_agents: number of agents
        batch_size: size of mini-batches for stochastic methods, or "all" for full-batch
        compute_x_optimal: if the optimal solution should be computed
            (using :func:`~decent_bench.centralized_algorithms.solve`). It is ignored when PyTorchCost is selected.
        show_progress: whether to display a progress bar while computing ``x_optimal``. Defaults to ``True``.

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
        n_samples=n_agents * 10,
        n_features=3,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        squeeze_targets=cost_cls is PyTorchCost,  # PyTorchCost expects squeezed targets for CrossEntropyLoss
    )
    test_data = SyntheticClassificationDatasetHandler(
        n_targets=2,
        n_samples=100,
        n_features=3,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        squeeze_targets=cost_cls is PyTorchCost,
    )
    local_datasets = dataset.split(partitions=split_iid(dataset, n_partitions=n_agents))

    x_optimal = None
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
            for p in local_datasets
        ]
        LOGGER.info("... done!")
        costs: Sequence[Cost] = pytorch_costs
    elif cost_cls is LogisticRegressionCost:
        classification_costs: list[LogisticRegressionCost] = [
            LogisticRegressionCost(dataset=p, batch_size=batch_size) for p in local_datasets
        ]
        LOGGER.info("... done!")
        if compute_x_optimal:
            # agents have the same n_samples, so minimizing a single logistic cost with all data is equivalent
            sum_cost = LogisticRegressionCost(dataset=dataset.get_datapoints(), batch_size="all")
            x_optimal = ca.solve(
                sum_cost,
                max_iter=SOLVE_MAX_ITER,
                stop_tol=SOLVE_STOP_TOL,
                max_tol=SOLVE_MAX_TOL,
                show_progress=show_progress,
            )
        costs = classification_costs
    else:
        raise ValueError(f"Unsupported cost class: {cost_cls}")

    return costs, x_optimal, test_data.get_datapoints()


def create_regression_problem(
    cost_cls: type[LinearRegressionCost | PyTorchCost] = LinearRegressionCost,
    *,
    device: SupportedDevices = SupportedDevices.CPU,
    n_agents: int = 100,
    batch_size: EmpiricalRiskBatchSize = "all",
    compute_x_optimal: bool = True,
) -> tuple[Sequence[Cost], Array | None, Dataset]:
    """
    Create out-of-the-box regression problems.

    Args:
        cost_cls: type of cost function
        device: device to create the problem on (only relevant for PyTorchCost)
        n_agents: number of agents
        batch_size: size of mini-batches for stochastic methods, or "all" for full-batch
        compute_x_optimal: if the optimal solution should be computed
            (by solving the linear system of equations). It is ignored when PyTorchCost is selected.

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
        n_samples=n_agents * 10,
        n_features=1,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        target_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
    )
    test_data = SyntheticRegressionDatasetHandler(
        n_targets=1,
        n_samples=100,
        n_features=1,
        framework=SupportedFrameworks.PYTORCH if cost_cls is PyTorchCost else SupportedFrameworks.NUMPY,
        device=device,
        feature_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
        target_dtype=np.float32 if cost_cls is PyTorchCost else np.float64,
    )
    local_datasets = dataset.split(partitions=split_iid(dataset, n_partitions=n_agents))

    x_optimal = None
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
            for p in local_datasets
        ]
        LOGGER.info("... done!")
        costs: Sequence[Cost] = pytorch_costs
    elif cost_cls is LinearRegressionCost:
        regression_costs: list[LinearRegressionCost] = [
            LinearRegressionCost(dataset=p, batch_size=batch_size) for p in local_datasets
        ]
        LOGGER.info("... done!")

        if compute_x_optimal:
            x_optimal = ca.solve(
                reduce(add, regression_costs),
                max_iter=SOLVE_MAX_ITER,
                stop_tol=SOLVE_STOP_TOL,
                max_tol=SOLVE_MAX_TOL,
                show_progress=False,
            )
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

    x_optimal = ca.solve(
        reduce(add, costs),
        max_iter=SOLVE_MAX_ITER,
        stop_tol=SOLVE_STOP_TOL,
        max_tol=SOLVE_MAX_TOL,
        show_progress=False,
    )

    return costs, x_optimal
