from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
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

    def heatmap_plot(
        network: Network,
        width: int,
        height: int,
        norm: float,
    ) -> list[np.ndarray]:
        xs = np.arange(0, width) / norm
        ys = np.arange(0, height) / norm
        X, Y = np.meshgrid(xs, ys, indexing="xy")  # noqa: N806
        nx, ny = X.shape
        points = np.column_stack((X.ravel(), Y.ravel()))
        mats = []
        for agent in network.agents():
            if not isinstance(agent.cost, PyTorchCost):
                continue
            agent_activation = agent.cost.final_activation
            agent.cost.final_activation = nn.Sigmoid()
            out = agent.cost.predict(agent.x, torch.tensor(points, dtype=torch.float32))  # type: ignore[arg-type]
            agent.cost.final_activation = agent_activation
            out_np = np.array(out)

            mats.append(out_np.reshape((nx, ny)))

        return mats

    image = Image.open(image_file).convert("L")
    image_array = np.array(image)
    height, width = image_array.shape
    feature_norm = max(height, width)

    figs = []
    for alg, networks in result.states.items():
        for j, n in enumerate(networks):
            heatmaps = heatmap_plot(n, width, height, feature_norm)
            fig, ax = plt.subplots(1, 5, figsize=(15, 10))
            fig.suptitle(f"{alg.name} - Trial {j + 1}")
            for i, hm in enumerate(heatmaps):
                ax[i].imshow(image_array, cmap="gray")
                ax[i].imshow(hm, cmap="hot", interpolation="nearest", vmin=0, vmax=1, alpha=0.8)
                ax[i].invert_yaxis()
                ax[i].axis("off")
                ax[i].set_title(f"Agent {i + 1}")
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path / f"heatmaps_{alg.name}_{j + 1}.png")
            figs.append(fig)
    return figs
