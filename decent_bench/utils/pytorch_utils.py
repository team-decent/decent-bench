from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SimpleLinearModel(torch.nn.Module):
    """Simple feedforward neural network model with linear layers and optional activations."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str | None = "relu",
        output_activation: str | None = None,
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
    """Applies the argmax function as an activation."""

    def __init__(self, dim: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch must be installed to use SimpleLinearModel")

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying argmax."""
        return torch.argmax(x, dim=self.dim)


class ArgminActivation(torch.nn.Module):
    """Applies the argmin function as an activation."""

    def __init__(self, dim: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch must be installed to use SimpleLinearModel")

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying argmin."""
        return torch.argmin(x, dim=self.dim)
