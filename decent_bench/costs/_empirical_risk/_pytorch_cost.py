from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, override

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils._tags import tags
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

    from decent_bench.agents import Agent

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


@tags("regression", "classification", "empirical-risk")
class PyTorchCost(EmpiricalRiskCost):
    """
    Cost function wrapper for PyTorch neural networks that integrates with the decentralized optimization framework.

    Supports batch-based training and gradient computation for decentralized learning scenarios.

    Note:
        It is generally recommended to set `agent_state_snapshot_period` to a value greater than 1
        when using PyTorchCost, as recording the full model parameters at every iteration can be expensive.

    """

    _NON_PICKLABLE_STATE_KEYS = (
        "_ft_compute_grad",
        "_ft_compute_sample_grad",
        "_last_batch_x",
        "_last_batch_y",
        "_params_list",
        "param_shapes",
        "param_sizes",
        "total_params",
        "param_names",
        "_dataloader_iter",
    )

    def __init__(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        final_activation: torch.nn.Module | None = None,
        *,
        batch_size: EmpiricalRiskBatchSize = "all",
        max_batch_size: int | None = None,
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
                model output when predicting targets using :meth:`predict`.
                E.g., argmax if classification and model outputs logits.
            batch_size (EmpiricalRiskBatchSize): Size of mini-batches for stochastic methods, or "all" for full-batch.
            max_batch_size (int | None): Optional maximum batch size to perform computations in, which can be used to
                avoid out-of-memory errors for large models/datasets. If specified, computations will be calculated in
                chunks of size at most max_batch_size. This limit will be applied to all computations irregardless of
                the batch_size or indices parameters; the result will still be the same. This is especially useful for
                when `indices` is set to "all" but the dataset is too large to fit in memory at once. If not specified,
                it will default to the batch_size (if batch_size is an int) or the total number of samples
                (if batch_size is "all").
            device (SupportedDevices): Device to run computations on. Make sure to test CPU vs GPU performance for your
                specific model and dataset, as it can vary.
            use_dataloader (bool): Whether to use DataLoader for batching.
                Can be beneficial for large datasets which can't fit into memory or when using an accelerator.
                Dataloaders cannot be pickled so resumption of iterrupted runs will start with a new random batch order.
            dataloader_kwargs (dict | None): Additional arguments for the DataLoader.
            load_dataset (bool): If True, loads the entire dataset into memory to optimize data access.
                This may lead to major speedups if the dataset is lazily loaded (e.g., loading data from disk), but it
                might increase memory usage so set to False if memory is an issue. Setting this to False might break
                checkpointing if the underlying dataset is not pickleable.
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
        self._max_batch_size = max_batch_size if max_batch_size is not None else self.n_samples
        self._device = device
        self._use_dataloader = use_dataloader
        self._dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}
        self._load_dataset = load_dataset
        self._compile_model = compile_model
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self._pytorch_device: str = iop.device_to_framework_device(device, framework=self.framework)
        self.model = self.model.to(self._pytorch_device)
        self.loss_fn = self.loss_fn.to(self._pytorch_device)

        if self._compile_model:
            torch.set_float32_matmul_precision("high")  # For better torch.compile performance
            self.model = cast("torch.nn.Module", torch.compile(self.model, **self._compile_kwargs))

        self._dataloader: torch.utils.data.DataLoader[Any] | None = None
        self._last_batch_used: list[int] = []
        self._remaining_batch_indices: list[int] = []  # Used for tracking remaining indices when not using dataloader

        self._init_param_caches()

    def __getstate__(self) -> dict[str, Any]:
        """Return a checkpoint-safe state for pickling."""
        try:
            state = self.__dict__.copy()
            for key in self._NON_PICKLABLE_STATE_KEYS:
                state.pop(key, None)
            state["model"] = self.model.to("cpu")
            state["loss_fn"] = self.loss_fn.to("cpu")
        finally:
            self.model = self.model.to(self._pytorch_device)
            self.loss_fn = self.loss_fn.to(self._pytorch_device)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state and clear transient runtime caches."""
        for key, value in state.items():
            setattr(self, key, value)
        self.model = self.model.to(self._pytorch_device)
        self.loss_fn = self.loss_fn.to(self._pytorch_device)
        if self._dataloader is not None:
            self._dataloader_iter = iter(self._dataloader)
        self._init_param_caches()

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

        # Move to target device once (assumes all params on same device)
        x = x.to(self._pytorch_device)

        # Split tensor efficiently and unflatten
        param_chunks = torch.split(x, self.param_sizes)
        with torch.no_grad():
            for param, chunk, shape in zip(self._params_list, param_chunks, self.param_shapes, strict=True):
                param.data = chunk.view(shape)

    def _get_model_parameters(self) -> torch.Tensor:
        """Get model parameters as a flattened tensor."""
        params = [p.detach().flatten() for p in self._params_list]
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

        Raises:
            TypeError: If data is not a list of torch.Tensor or a single torch.Tensor.

        """
        self._set_model_parameters(x)
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            if isinstance(data, list):
                inputs = torch.stack(data)
            elif isinstance(data, torch.Tensor):
                inputs = data
            else:
                raise TypeError(f"Data must be a list of torch.Tensor or a single torch.Tensor, got {type(data)}.")

            final_outputs: list[torch.Tensor] = []
            for i in range(0, inputs.shape[0], self._max_batch_size):
                inputs_chunk = inputs[i : i + self._max_batch_size].to(self._pytorch_device)
                outputs: torch.Tensor = self.model(inputs_chunk)
                outputs = self.final_activation(outputs)
                final_outputs.extend(outputs.detach().cpu().tolist())

        return final_outputs

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> float:
        self._set_model_parameters(x)
        if self.model.training:
            self.model.eval()

        batches = self._get_batch_data(indices)
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in batches:
                outputs = self.model(batch_x)
                loss: torch.Tensor = self.loss_fn(outputs, batch_y)
                total_loss += loss.item() * batch_x.shape[0]

        return float(total_loss / len(self.batch_used))

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
        if not self.model.training:
            self.model.train()

        batches = self._get_batch_data(indices)

        if len(batches) == 1:
            # Slightly faster path for single batch
            batch_x, batch_y = batches[0]
            outputs = self.model(batch_x)
            loss = self.loss_fn(outputs, batch_y)
            gradients = torch.autograd.grad(loss, self._params_list)
            return torch.cat([g.flatten() for g in gradients])

        # Accumulate chunk gradients with sample-count weighting so the final
        # result is the mean gradient over all selected samples.
        inv_total_samples = 1.0 / len(self.batch_used)
        grad_acc = [torch.zeros_like(param) for param in self._params_list]

        for batch_x, batch_y in batches:
            outputs = self.model(batch_x)
            loss = self.loss_fn(outputs, batch_y)

            gradients = torch.autograd.grad(loss, self._params_list)
            weight = batch_x.shape[0] * inv_total_samples
            for acc, grad in zip(grad_acc, gradients, strict=True):
                acc.add_(grad, alpha=weight)

        return torch.cat([g.flatten() for g in grad_acc])

    def _per_sample_gradients(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> torch.Tensor:
        """Compute per-sample gradients for the specified indices."""
        # Credit: https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html
        self._init_per_sample_grad()
        self._set_model_parameters(x)
        if not self.model.training:
            self.model.train()

        batches = self._get_batch_data(indices)

        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        if len(batches) == 1:
            batch_x, batch_y = batches[0]
            ft_per_sample_grads = self._ft_compute_sample_grad(params, buffers, batch_x, batch_y)
            batch_size = batch_x.shape[0]
            grad_list = [
                ft_per_sample_grads[name].view(batch_size, size)
                for name, size in zip(self.param_names, self.param_sizes, strict=True)
            ]
            return torch.cat(grad_list, dim=1)

        param_offsets = [0]
        for size in self.param_sizes:
            param_offsets.append(param_offsets[-1] + size)

        all_grads = torch.empty(
            (len(self.batch_used), self.total_params),
            dtype=self._params_list[0].dtype,
            device=self._pytorch_device,
        )
        row_start = 0

        for batch_x, batch_y in batches:
            ft_per_sample_grads = self._ft_compute_sample_grad(params, buffers, batch_x, batch_y)

            batch_size = batch_x.shape[0]
            row_end = row_start + batch_size
            for name, size, col_start, col_end in zip(
                self.param_names,
                self.param_sizes,
                param_offsets[:-1],
                param_offsets[1:],
                strict=True,
            ):
                all_grads[row_start:row_end, col_start:col_end] = ft_per_sample_grads[name].view(batch_size, size)
            row_start = row_end

        return all_grads

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
        raise NotImplementedError("Proximal operator is not implemented for PyTorchCost.")

    def init_local_training(
        self,
        opt_cls: type[torch.optim.Optimizer],
        opt_kwargs: dict[str, Any] | None = None,
        sched_cls: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        sched_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the optimizer and scheduler for local training.

        This method is required to be called before using :meth:`local_training` to set up the optimizer and scheduler.

        Args:
            opt_cls (type[torch.optim.Optimizer]): PyTorch optimizer class to use for local training.
            opt_kwargs (dict[str, Any] | None): Keyword arguments for initializing the optimizer. The model parameters
                will be passed as the first argument, so do not include them in opt_kwargs.
            sched_cls (type[torch.optim.lr_scheduler.LRScheduler] | None): Optional PyTorch learning rate scheduler
                class to use for local training. The scheduler will be stepped once at the end of each call to
                :meth:`local_training`.
            sched_kwargs (dict[str, Any] | None): Keyword arguments for initializing the scheduler. The optimizer will
                be passed as the first argument, so do not include it in sched_kwargs.

        Raises:
            RuntimeError: If the optimizer is already initialized. This method is intended to be called only once
                to set the optimizer for local training.

        """
        if self._optimizer is not None:
            raise RuntimeError(
                "Optimizer is already initialized. This method is intended to be called "
                "only once to set the optimizer for local training."
            )

        self._optimizer = opt_cls(self._params_list, **(opt_kwargs or {}))
        if sched_cls is not None:
            self._scheduler = sched_cls(self._optimizer, **(sched_kwargs or {}))

    def local_training(
        self,
        x: torch.Tensor,
        iterations: int,
        agent: Agent,
        regularization: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None,
        indices: EmpiricalRiskIndices = "batch",
    ) -> torch.Tensor:
        r"""
        Perform local training steps using the provided optimizer.

        Note:
            This method is intended to be used in decentralized algorithms that support local training.

        Args:
            x (torch.Tensor): Initial parameters to start local training from.
            iterations (int): Number of local training iterations to perform.
            agent (Agent): The agent performing the local training.
            regularization (torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None): Optional regularization.
                Two forms are supported:

                - Scalar tensor (or callable returning a scalar): interpreted as an additive loss penalty.
                - Flat tensor with the same number of elements as the flattened parameter vector: interpreted as a
                  parameter-space correction step applied after each optimizer step
                  (i.e., :math:`x \leftarrow x - r`).
            indices (EmpiricalRiskIndices): Indices of the samples to use for local training.

        Returns:
            torch.Tensor: Updated parameters after local training.

        Raises:
            RuntimeError: If no optimizer was provided during initialization.
            ValueError: If `regularization` is a non-scalar tensor but does not have the same number of elements as the
                flattened parameter vector.
            TypeError: If `regularization` is not a torch.Tensor or a callable returning a torch.Tensor.

        """
        if self._optimizer is None:
            raise RuntimeError(
                "Local training is not available because no optimizer was provided."
                " Please call init_local_training to set up the optimizer before using local_training."
            )

        self._set_model_parameters(x)
        if not self.model.training:
            self.model.train()

        for _ in range(iterations):
            batches = self._get_batch_data(indices)

            for batch_x, batch_y in batches:
                self._optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                reg_value: torch.Tensor | None = None
                if regularization is not None:
                    reg_value = (
                        regularization(torch.cat([p.flatten() for p in self._params_list]).to(self._pytorch_device))
                        if callable(regularization)
                        else regularization
                    )
                    if not isinstance(reg_value, torch.Tensor):
                        raise TypeError(
                            "`regularization` must be a torch.Tensor or a callable returning a torch.Tensor, "
                            f"got {type(reg_value)}."
                        )
                    reg_value = reg_value.to(
                        device=self._pytorch_device,
                        dtype=self._params_list[0].dtype,
                    )

                    # Two supported forms:
                    # 1) Scalar tensor: interpreted as an additive loss penalty.
                    # 2) Flat vector of model-parameter shape: interpreted as an additive parameter update step.
                    #    This is useful for algorithms that add a correction term directly in parameter space.
                    if reg_value.ndim == 0 or reg_value.numel() == 1:
                        loss += reg_value.reshape(())
                    elif reg_value.numel() != self.total_params:
                        raise ValueError(
                            "If `regularization` is non-scalar, it must have the same number of elements as the "
                            f"flattened parameter vector (expected {self.total_params}, got {reg_value.numel()})."
                        )

                loss.backward()
                self._optimizer.step()

                if reg_value is not None and reg_value.ndim != 0 and reg_value.numel() != 1:
                    # Apply parameter-space correction step: x <- x - reg_value
                    reg_value = reg_value.detach().flatten()
                    reg_chunks = torch.split(reg_value, self.param_sizes)
                    with torch.no_grad():
                        for param, chunk, shape in zip(
                            self._params_list,
                            reg_chunks,
                            self.param_shapes,
                            strict=True,
                        ):
                            param.data.sub_(chunk.view(shape))

            # Since we are not calling the gradient method,
            # we need to manually update the agent's gradient call count for benchmarking purposes.
            agent._n_gradient_calls += len(self.batch_used)  # noqa: SLF001

        if self._scheduler is not None:
            self._scheduler.step()

        return self._get_model_parameters()

    @override
    def _sample_batch_indices(self, indices: EmpiricalRiskIndices = "batch") -> list[int]:
        """Not used in PyTorchCost, implemented in _get_batch_data."""
        raise NotImplementedError("_sample_batch_indices is not used in PyTorchCost, implemented in _get_batch_data.")

    def _collate_xy_idx(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        xs, ys, idx = zip(*batch, strict=True)
        return torch.stack(xs), torch.stack(ys), list(idx)

    def _init_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        self._dataloader_kwargs.setdefault("shuffle", True)
        return torch.utils.data.DataLoader(
            cast("torch.utils.data.Dataset[Any]", self._dataset),
            batch_size=self.batch_size,
            generator=iop.rng_torch(SupportedDevices.CPU),  # DataLoader shuffling must be done on CPU
            collate_fn=self._collate_xy_idx,
            **self._dataloader_kwargs,
        )

    def _handle_dataloader(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if self._dataloader is None:
            self._dataloader = self._init_dataloader()
            self._dataloader_iter = iter(self._dataloader)

        try:
            batch_x, batch_y, batch_idx = next(self._dataloader_iter)
        except StopIteration:
            # Restart the iterator if we reach the end
            self._dataloader_iter = iter(self._dataloader)
            batch_x, batch_y, batch_idx = next(self._dataloader_iter)

        return batch_x, batch_y, batch_idx

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a list of batch data for the specified indices, each list item contains a tuple of (batch_x, batch_y).

        The max size of each batch is determined by self._max_batch_size.

        Raises:
            RuntimeError: If batch data could not be retrieved, which should not happen under normal circumstances

        """
        batch_x, batch_y, batch_idx = self._handle_indices(indices)

        if batch_x is None or batch_y is None or batch_idx is None:
            raise RuntimeError("Batch data could not be retrieved. Please report this error.")

        self._last_batch_used = list(batch_idx)
        self._last_batch_x = batch_x.to(self._pytorch_device)
        self._last_batch_y = batch_y.to(self._pytorch_device)

        batches = []
        for i in range(0, batch_x.shape[0], self._max_batch_size):
            batch_x_chunk = self._last_batch_x[i : i + self._max_batch_size]
            batch_y_chunk = self._last_batch_y[i : i + self._max_batch_size]
            batches.append((batch_x_chunk, batch_y_chunk))

        return batches

    def _handle_indices(
        self,
        indices: EmpiricalRiskIndices,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int] | None]:
        batch_x: torch.Tensor | None = None
        batch_y: torch.Tensor | None = None
        batch_idx: list[int] | None = None
        if isinstance(indices, str):
            if indices == "batch":
                if self.batch_size < self.n_samples:
                    if self._use_dataloader:
                        batch_x, batch_y, batch_idx = self._handle_dataloader()
                    else:
                        if len(self._remaining_batch_indices) < self.batch_size:
                            # Refill remaining indices if not enough left for a full batch
                            self._remaining_batch_indices = torch.randperm(self.n_samples).tolist()
                        indices = self._remaining_batch_indices[: self.batch_size]
                        self._remaining_batch_indices = self._remaining_batch_indices[self.batch_size :]
                else:
                    # Use full dataset
                    indices = list(range(self.n_samples))
            elif indices == "all":
                indices = list(range(self.n_samples))
            else:
                raise ValueError(f"Invalid indices string: {indices}. Only 'all' and 'batch' are supported.")
        elif isinstance(indices, int):
            indices = [indices]

        if isinstance(indices, list):
            if len(indices) == len(self.batch_used) and indices == self.batch_used and hasattr(self, "_last_batch_x"):
                # Use cached batch so we don't have to re-stack
                return self._last_batch_x, self._last_batch_y, self._last_batch_used

            batch = (
                self._dataset
                if len(indices) == len(self._dataset) and self._load_dataset
                else [self._dataset[i] for i in indices]
            )
            list_batch_x, list_batch_y, batch_idx = zip(*batch, strict=True)  # type: ignore[misc, assignment]
            batch_x = torch.stack(list_batch_x)
            batch_y = torch.stack(list_batch_y)

        return batch_x, batch_y, batch_idx

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

    def _init_param_caches(self) -> None:
        """Initialize parameter caches for flattening/unflattening."""
        # Store parameter shapes for flattening/unflattening
        self._params_list = list(self.model.parameters())  # Cache for faster access
        self.param_shapes = [p.shape for p in self._params_list]
        self.param_sizes = [p.numel() for p in self._params_list]
        self.total_params = sum(self.param_sizes)
        self.param_names = [n for n, _ in self.model.named_parameters()]

    def __add__(self, other: Cost) -> Cost:
        return super().__add__(other)
