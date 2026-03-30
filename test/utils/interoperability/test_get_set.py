import pytest

import decent_bench.utils.interoperability as iop

from .test_core import create_array

try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
    TF_GPU_AVAILABLE = len(tf.config.list_physical_devices("GPU")) > 0
except (ImportError, ModuleNotFoundError):
    TF_AVAILABLE = False
    TF_GPU_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    JAX_GPU_AVAILABLE = len(jax.devices("gpu")) > 0
except (ImportError, ModuleNotFoundError):
    JAX_AVAILABLE = False
    JAX_GPU_AVAILABLE = False
except RuntimeError:
    # JAX raises RuntimeError if no GPU is available when querying devices
    JAX_GPU_AVAILABLE = False


@pytest.mark.parametrize(
    ("framework", "device"),
    [
        ("numpy", "cpu"),
        pytest.param(
            "pytorch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "pytorch",
            "gpu",
            marks=pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="PyTorch CUDA not available"),
        ),
    ],
)
def test_get_set_item_frameworks(framework: str, device: str) -> None:
    """Test get_item and set_item functions for all frameworks and devices."""
    data = [[1, 2, 3], [4, 5, 6]]
    arr = create_array(data, framework, device)

    # Test get_item
    item = iop.get_item(arr, (0, 1))
    assert iop.to_numpy(item) == 2

    # Test set_item
    val = create_array(99, framework, device)
    iop.set_item(arr, (0, 1), val)
    item = iop.get_item(arr, (0, 1))
    assert iop.to_numpy(item) == 99
