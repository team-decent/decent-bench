from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt

TORCH_AVAILABLE = True

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:
        TORCH_AVAILABLE = False
        # Create mock torch module for documentation
        from types import ModuleType

        torch = ModuleType("torch")  # type: ignore[assignment]
        nn = ModuleType("torch.nn")

        class _MockModule:
            """Mock base class for PyTorch nn.Module."""

        _MockModule.__module__ = "torch.nn"
        _MockModule.__qualname__ = "Module"
        _MockModule.__name__ = "Module"

        nn.Module = _MockModule  # type: ignore[attr-defined]
        torch.nn = nn  # type: ignore[attr-defined]


class SimpleLinearModel(torch.nn.Module):
    """
    Simple feedforward neural network model with linear layers and optional activations.

    Args:
        input_size (int): The size of the input features.
        hidden_sizes (list[int]): A list of sizes for the hidden layers.
        output_size (int): The size of the output layer.
        activation (Literal["relu", "tanh", "sigmoid"] | None): The activation function to use for hidden layers.
        output_activation (Literal["relu", "tanh", "sigmoid"] | None): The final activation after the output layer.

    Raises:
        ImportError: If PyTorch is not installed.

    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: Literal["relu", "tanh", "sigmoid"] | None = "relu",
        output_activation: Literal["relu", "tanh", "sigmoid"] | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch must be installed to use SimpleLinearModel")

        super().__init__()

        layers: list[torch.nn.Module] = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "tanh":
                layers.append(torch.nn.Tanh())
            elif activation == "sigmoid":
                layers.append(torch.nn.Sigmoid())
            prev_size = hidden_size

        # Output layer
        layers.append(torch.nn.Linear(prev_size, output_size))
        if output_activation is not None:
            if output_activation == "relu":
                layers.append(torch.nn.ReLU())
            elif output_activation == "tanh":
                layers.append(torch.nn.Tanh())
            elif output_activation == "sigmoid":
                layers.append(torch.nn.Sigmoid())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        res: torch.Tensor = self.network(x)
        return res


class ArgmaxActivation(torch.nn.Module):
    """
    Applies the argmax function as an activation.

    Args:
        dim (int): The dimension along which to compute the argmax.

    Raises:
        ImportError: If PyTorch is not installed.

    """

    def __init__(self, dim: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch must be installed to use ArgmaxActivation")

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying argmax."""
        return torch.argmax(x, dim=self.dim)


class ArgminActivation(torch.nn.Module):
    """
    Applies the argmin function as an activation.

    Args:
        dim (int): The dimension along which to compute the argmin.

    Raises:
        ImportError: If PyTorch is not installed.

    """

    def __init__(self, dim: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch must be installed to use ArgminActivation")

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying argmin."""
        return torch.argmin(x, dim=self.dim)


def visualize_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler, num_iterations: int) -> None:
    """
    Visualize the learning rate schedule over a specified number of iterations.

    Args:
        scheduler (torch.optim.lr_scheduler.LRScheduler): The PyTorch learning rate scheduler to visualize.
        num_iterations (int): The number of iterations to visualize the learning rate for.

    """
    lrs = []
    for _ in range(num_iterations):
        lrs.append(scheduler.get_last_lr()[0])  # Assuming a single parameter group
        scheduler.step()

    plt.plot(lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid()
    plt.show()
