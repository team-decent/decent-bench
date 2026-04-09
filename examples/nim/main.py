import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.dataset import NIMDatasetHandler
from src.tools.visualize_dataset import visualize_nim_dataset

import decent_bench.utils.interoperability as iop
from decent_bench.utils.logger import start_logger

if __name__ == "__main__":
    # NIM path-based LIDAR sampling usage:
    import torch

    start_logger()
    iop.set_seed(0)
    nim_data = NIMDatasetHandler(
        image_file="data/kth_floorplan_sample.png",
        n_partitions=5,
        samples_per_partition=1000,
        transform=torch.tensor,  # type: ignore[arg-type]
        label_transform=torch.tensor,  # type: ignore[arg-type]
        label_balance=2.0,
        # leakage=0.1,
        # paths="data/kth_floorplan_path.json",
        # samples_per_pose=5,
        # num_beams=5,
        # fov=math.pi,
        # max_range=200,
        # scan_spacing=5.0,
        # add_empty_lidar_samples=True,
    )

    anim = visualize_nim_dataset(
        nim_data,
        animate=False,
        # path_index=[0, 1],
        fps=2,
        # save_path=f"test_1-2.png",
    )

    print(
        nim_data.get_partitions()[0][0],
        type(nim_data.get_partitions()[0][0]),
    )
    print(len(nim_data.get_partitions()[0]))

    plt.show()
