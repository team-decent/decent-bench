from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from decent_array import interoperability as iop
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from decent_bench import benchmark
from decent_bench.costs import PyTorchCost
from decent_bench.networks import Network


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


class ThresholdSigmoid(nn.Module):  # noqa: D101
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        ret: torch.Tensor = (self.sigmoid(x) > self.threshold).long()
        return ret


def create_heatmap_plots(
    image_file: str | Path,
    result: benchmark.BenchmarkResult,
    save_path: Path | None = None,
) -> list[Figure]:
    """Create heatmap plots for each agent in the networks."""
    with Image.open(image_file) as image:
        image_array = np.array(image.convert("L"))
    height, width = image_array.shape
    feature_norm = max(height, width)
    xs = np.arange(width) / feature_norm
    ys = np.arange(height) / feature_norm
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    inputs = list(torch.tensor(points, dtype=torch.float32).unbind())

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    def predict_heatmaps(network: Network) -> list[NDArray[Any]]:
        heatmaps = []
        for agent in network.agents():
            if not isinstance(agent.cost, PyTorchCost):
                continue
            original_activation = agent.cost.final_activation
            agent.cost.final_activation = nn.Sigmoid()
            try:
                predictions = agent.cost.predict(agent.x, inputs)  # type: ignore[arg-type]
            finally:
                agent.cost.final_activation = original_activation
            heatmaps.append(iop.to_numpy(predictions).reshape((height, width)))

        return heatmaps

    figs = []
    for alg, networks in result.states.items():
        for trial_index, network in enumerate(networks, start=1):
            heatmaps = predict_heatmaps(network)
            if not heatmaps:
                continue
            fig, axes = plt.subplots(1, len(heatmaps), figsize=(3 * len(heatmaps), 10), squeeze=False)
            ax = axes[0]
            fig.suptitle(f"{alg.name} - Trial {trial_index}")
            for agent_index, (axis, heatmap) in enumerate(zip(ax, heatmaps, strict=True), start=1):
                axis.imshow(image_array, cmap="gray")
                axis.imshow(heatmap, cmap="hot", interpolation="nearest", vmin=0, vmax=1, alpha=0.8)
                axis.invert_yaxis()
                axis.axis("off")
                axis.set_title(f"Agent {agent_index}")
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(save_path / f"heatmaps_{alg.name}_{trial_index}.png")
            figs.append(fig)
    return figs
