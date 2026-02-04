from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast, override

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.costs._empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskBatchSize,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)

if TYPE_CHECKING:
    import torch

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _IndexDataset:
    """A simple dataset wrapper to handle indexing when using a PyTorch dataloader."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        if not hasattr(self.dataset, "__len__"):
            raise ValueError("Dataset must implement __len__ method.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return cast("tuple[torch.Tensor, torch.Tensor, int]", (*self.dataset[idx], idx))


class PyTorchCost(EmpiricalRiskCost):
    """
    Cost function wrapper for PyTorch neural networks that integrates with the distributed optimization framework.

    Supports batch-based training and gradient computation for distributed learning scenarios.

    Note:
        It is generally recommended to set `agent_state_snapshot_period` in
        :class:`~decent_bench.benchmark_problem.BenchmarkProblem` to a value greater than 1
        when using PyTorchCost, as recording the full model parameters at every iteration can be expensive.

    """

    def __init__(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        final_activation: torch.nn.Module | None = None,
        *,
        batch_size: EmpiricalRiskBatchSize = "all",
        device: SupportedDevices = SupportedDevices.CPU,
        use_dataloader: bool = False,
        dataloader_kwargs: dict[str, Any] | None = None,
        load_dataset: bool = True,
        compile_model: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the PyTorch cost function.

        Args:
            dataset (Dataset):
                Dataset partition containing features and targets.
                Transformations should be applied beforehand such as converting to tensors.
                See torch.utils.data.Dataset for details.
            model (torch.nn.Module): PyTorch neural network model.
            loss_fn: (torch.nn.Module): PyTorch loss function.
            final_activation (torch.nn.Module | None): Optional final activation layer to apply after
                model output when predicting targets. E.g., argmax if classification and model outputs logits.
            batch_size (EmpiricalRiskBatchSize): Size of mini-batches for stochastic methods, or "all" for full-batch.
            device (SupportedDevices): Device to run computations on.
            use_dataloader (bool): Whether to use DataLoader for batching.
                Can be beneficial for large datasets to avoid loading all data into memory.
            dataloader_kwargs (dict | None): Additional arguments for the DataLoader.
            load_dataset (bool): If True, load the entire dataset into memory to optimize data access.
                This may lead to major speedups if the dataset is lazily loaded.
                May increase memory usage if the dataset is lazily loaded, set to False if memory is an issue.
            compile_model (bool): Whether to compile the model using torch.compile for performance.
                May improve speed after warm-up. Might need to try different modes based on the model and OS,
                use compile_kwargs.
            compile_kwargs (dict | None): Additional arguments for torch.compile. Commonly used mode is
                "reduce_overhead" for performance optimization. See
                https://pytorch.org/docs/stable/generated/torch.compile.html for details.

        Raises:
            ImportError: If PyTorch is not available
            ValueError: If batch_size is larger than the number of samples in the dataset

        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use PyTorchCost.")

        if isinstance(batch_size, int) and (batch_size <= 0 or batch_size > len(dataset)):
            raise ValueError(
                f"Batch size must be positive and at most the number of samples, "
                f"got: {batch_size} and number of samples is: {len(dataset)}."
            )
        if isinstance(batch_size, str) and batch_size != "all":
            raise ValueError(f"Invalid batch size string. Supported value is 'all', got {batch_size}.")

        if load_dataset:
            # Loads the dataset into memory in case it is lazily loaded
            self._dataset = _IndexDataset([(x, y) for x, y in dataset])
        else:
            self._dataset = _IndexDataset(dataset)

        self.model = model
        self.loss_fn = loss_fn
        self.final_activation = final_activation if final_activation is not None else torch.nn.Identity()
        self._batch_size = self.n_samples if batch_size == "all" else batch_size
        self._device = device
        self._use_dataloader = use_dataloader
        self._dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}
        self._load_dataset = load_dataset
        self._compile_model = compile_model
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        self._pytorch_device: str = iop.device_to_framework_device(device, framework=self.framework)
        self.model = self.model.to(self._pytorch_device)
        self.loss_fn = self.loss_fn.to(self._pytorch_device)

        if self._compile_model:
            torch.set_float32_matmul_precision("high")  # For better torch.compile performance
            self.model = cast("torch.nn.Module", torch.compile(self.model, **self._compile_kwargs))

        self._dataloader: torch.utils.data.DataLoader[Any] | None = None
        self._last_batch_used = []  # Pre-allocate list for last used batch for efficiency in _get_batch_data

        # Store parameter shapes for flattening/unflattening
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.param_sizes = [p.numel() for p in self.model.parameters()]
        self.total_params = sum(self.param_sizes)
        self.param_names = [n for n, _ in self.model.named_parameters()]
        self.param_offsets = torch.cumsum(torch.tensor([0, *self.param_sizes[:-1]]), dim=0).tolist()

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.total_params,)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.PYTORCH

    @property
    def device(self) -> SupportedDevices:
        return self._device

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dataset(self) -> Dataset:
        return self._dataset.dataset

    @property
    def m_smooth(self) -> float:
        return float("nan")

    @property
    def m_cvx(self) -> float:
        return float("nan")

    @cached_property
    @override
    def _rand(self) -> torch.Generator:  # type: ignore[override]
        return torch.Generator(device="cpu").manual_seed(0)  # Later replace with global rng

    def _clean(self) -> None:
        """Clean up cache."""
        self._last_batch_x = torch.empty(0)
        self._last_batch_y = torch.empty(0)

    def _set_model_parameters(self, x: torch.Tensor) -> None:
        """
        Set model parameters from a tensor.

        Args:
            x (torch.Tensor): Flattened parameter tensor.

        Raises:
            ValueError: If the size of x does not match the total number of model parameters.

        """
        if x.numel() != self.total_params:
            raise ValueError(
                f"Parameter vector size {x.numel()} does not match total model parameters {self.total_params}"
            )

        # Unflatten the parameter vector and set model parameters
        start_idx = 0
        with torch.no_grad():
            for param, size, shape in zip(self.model.parameters(), self.param_sizes, self.param_shapes, strict=True):
                end_idx = start_idx + size
                param.data = x[start_idx:end_idx].reshape(shape).to(param.device)
                start_idx = end_idx

    def _get_model_parameters(self) -> torch.Tensor:
        """Get model parameters as a flattened tensor."""
        params = [p.detach().flatten() for p in self.model.parameters()]
        return torch.cat(params).to(self._pytorch_device)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: torch.Tensor, data: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Make predictions at x on the given data.

        Args:
            x: Point to make predictions at.
            data: List of torch.Tensor containing features to make predictions on.

        Returns:
            Predicted targets as an array

        """
        self._set_model_parameters(x)
        self.model.eval()
        with torch.no_grad():
            inputs = torch.stack(data).to(self._pytorch_device)
            outputs: torch.Tensor = self.model(inputs)
            outputs = self.final_activation(outputs)

        return outputs.detach().cpu().tolist()

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> float:
        self._set_model_parameters(x)
        self.model.eval()

        batch_x, batch_y = self._get_batch_data(indices)

        with torch.no_grad():
            outputs = self.model(batch_x)
            loss: torch.Tensor = self.loss_fn(outputs, batch_y)

        return float(loss.cpu().item())

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(
        self,
        x: torch.Tensor,
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
    ) -> torch.Tensor:
        if reduction is None:
            return self._per_sample_gradients(x, indices)

        self._set_model_parameters(x)
        self.model.train()

        batch_x, batch_y = self._get_batch_data(indices)

        # Forward pass
        outputs = self.model(batch_x)
        loss = self.loss_fn(outputs, batch_y)

        # Compute gradients using torch.autograd.grad (doesn't modify model parameters)
        model_params = list(self.model.parameters())
        gradients = torch.autograd.grad(
            loss,
            model_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

        grads = [
            g.reshape(-1) if g is not None else torch.zeros_like(p)
            for p, g in zip(self.model.parameters(), gradients, strict=True)
        ]

        # Return concatenated gradient tensor
        return torch.cat(grads)

    def _per_sample_gradients(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> torch.Tensor:
        """Compute per-sample gradients for the specified indices. May need to batch calls due to memory constraints."""
        # Credit: https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html
        self._init_per_sample_grad()
        self._set_model_parameters(x)
        self.model.train()

        batch_x, batch_y = self._get_batch_data(indices)

        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        ft_per_sample_grads = self._ft_compute_sample_grad(params, buffers, batch_x, batch_y)

        # Collect gradients and flatten them into a single tensor
        batch_size = batch_x.shape[0]
        dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            flat_grads = torch.empty((batch_size, self.total_params), device=self._pytorch_device, dtype=dtype)
            for name, off, size in zip(self.param_names, self.param_offsets, self.param_sizes, strict=True):
                g = ft_per_sample_grads[name].reshape(batch_size, size)
                flat_grads[:, off : off + size] = g

        return flat_grads

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> torch.Tensor:
        """
        Compute the Hessian matrix.

        Note:
            This is computationally expensive for neural networks and typically not used.

        Raises:
            NotImplementedError: Always raised to indicate Hessian computation is not implemented.

        """
        raise NotImplementedError("Hessian computation is not implemented for PyTorchCost.")

    @iop.autodecorate_cost_method(EmpiricalRiskCost.proximal)
    def proximal(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Compute the proximal operator.

        Note:
            This is computationally expensive for neural networks and typically not used.

        Raises:
            NotImplementedError: Always raised to indicate proximal computation is not implemented.

        """
        raise NotImplementedError("Proximal operator is not implemented for NeuralNetworkCostFunction.")

    @override
    def _sample_batch_indices(self, indices: EmpiricalRiskIndices = "batch") -> list[int]:
        """Not used in PyTorchCost, implemented in _get_batch_data."""
        raise NotImplementedError("_sample_batch_indices is not used in PyTorchCost, implemented in _get_batch_data.")

    def _init_dataloader(self) -> tuple[torch.utils.data.DataLoader[Any], Iterator[torch.utils.data.DataLoader[Any]]]:
        def _collate_xy_idx(
            batch: list[tuple[torch.Tensor, torch.Tensor, int]],
        ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
            xs, ys, idx = zip(*batch, strict=True)
            return torch.stack(xs), torch.stack(ys), list(idx)

        self._dataloader_kwargs.setdefault("shuffle", True)
        dataloader = torch.utils.data.DataLoader(
            cast("torch.utils.data.Dataset[Any]", self._dataset),
            batch_size=self.batch_size,
            generator=self._rand,
            collate_fn=_collate_xy_idx,
            **self._dataloader_kwargs,
        )
        dataloader_iter = iter(dataloader)
        return dataloader, dataloader_iter

    def _handle_dataloader(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if self._dataloader is None:
            self._dataloader, self._dataloader_iter = self._init_dataloader()

        try:
            batch_x, batch_y, batch_idx = next(self._dataloader_iter)
        except StopIteration:
            # Restart the iterator if we reach the end
            self._dataloader_iter = iter(self._dataloader)
            batch_x, batch_y, batch_idx = next(self._dataloader_iter)

        return batch_x, batch_y, batch_idx

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> tuple[torch.Tensor, torch.Tensor]:
        batch_x: torch.Tensor | None = None
        batch_y: torch.Tensor | None = None
        batch_idx: list[int] | None = None
        if isinstance(indices, str):
            if indices == "batch":
                if self.batch_size < self.n_samples:
                    if self._use_dataloader:
                        batch_x, batch_y, batch_idx = self._handle_dataloader()
                    else:
                        indices = torch.randperm(self.n_samples, generator=self._rand)[: self.batch_size].tolist()
                else:
                    # Use full dataset
                    indices = list(range(self.n_samples))
            elif indices == "all":
                indices = list(range(self.n_samples))
            else:
                raise ValueError(f"Invalid indices string: {indices}. Only 'all' and 'batch' are supported.")

        if isinstance(indices, int):
            indices = [indices]

        if isinstance(indices, list):
            if len(indices) == len(self.batch_used) and indices == self.batch_used:
                # Use cached batch so we don't have to re-stack
                return self._last_batch_x, self._last_batch_y

            batch = (
                self._dataset
                if len(indices) == len(self._dataset) and self._load_dataset
                else [self._dataset[i] for i in indices]
            )
            list_batch_x, list_batch_y, batch_idx = zip(*batch, strict=True)  # type: ignore[misc, assignment]
            batch_x = torch.stack(list_batch_x)
            batch_y = torch.stack(list_batch_y)

        if batch_x is None or batch_y is None or batch_idx is None:
            raise RuntimeError("Batch data could not be retrieved. Please report this error.")

        self._last_batch_used = list(batch_idx)
        self._last_batch_x = batch_x.to(self._pytorch_device, non_blocking=True)
        self._last_batch_y = batch_y.to(self._pytorch_device, non_blocking=True)

        return self._last_batch_x, self._last_batch_y

    def _init_per_sample_grad(self) -> None:
        """Initialize per-sample gradient function using functorch."""
        if hasattr(self, "_ft_compute_sample_grad"):
            return  # Already initialized

        def compute_loss(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            sample: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = torch.func.functional_call(self.model, (params, buffers), (batch,))
            loss: torch.Tensor = self.loss_fn(predictions, targets)
            return loss

        self._ft_compute_grad = torch.func.grad(compute_loss)
        self._ft_compute_sample_grad = torch.func.vmap(self._ft_compute_grad, in_dims=(None, None, 0, 0))

        if not self._compile_model:
            return

        try:
            self._ft_compute_sample_grad = torch.compile(self._ft_compute_sample_grad, **self._compile_kwargs)
        except Exception as e:
            LOGGER.warning(f"Error compiling per-sample gradient function: {e}\n\nContinuing without compilation.")

    def __add__(self, other: Cost) -> Cost:
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        return SumCost([self, other])
