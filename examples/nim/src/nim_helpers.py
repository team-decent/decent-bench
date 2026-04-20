import torch
from torch import nn


class NimModel(nn.Module):  # noqa: D101
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()

        self.layers = nn.ModuleList()
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Output layer
        self.layers.append(torch.nn.Linear(prev_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Apply sine activation after the first layer to create a non-linear decision boundary,
        # then ReLU for the rest of the hidden layers
        x = torch.sin(self.layers[0](x))

        for layer in self.layers[1:-1]:
            x = torch.relu(layer(x))

        # No activation on the output layer since we'll be using BCEWithLogitsLoss which expects raw logits
        res: torch.Tensor = self.layers[-1](x)

        return res


class FinalActivation(nn.Module):  # noqa: D101
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        ret: torch.Tensor = (self.sigmoid(x) > self.threshold).long()
        return ret
