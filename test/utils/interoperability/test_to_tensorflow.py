import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedDevices

from .test_core import assert_arrays_equal, create_array

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


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
@pytest.mark.parametrize(
    "framework,device",
    [
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
        pytest.param(
            "tensorflow",
            "cpu",
            marks=pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available"),
        ),
        pytest.param(
            "tensorflow",
            "gpu",
            marks=pytest.mark.skipif(not TF_GPU_AVAILABLE, reason="TensorFlow GPU not available"),
        ),
        pytest.param(
            "jax",
            "cpu",
            marks=pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available"),
        ),
        pytest.param(
            "jax",
            "gpu",
            marks=pytest.mark.skipif(not JAX_GPU_AVAILABLE, reason="JAX GPU not available"),
        ),
    ],
)
def test_to_tensorflow_frameworks(framework: str, device: str):
    """Test to_tensorflow conversion for all frameworks and devices."""
    data = [1, 2, 3]
    arr = create_array(data, framework, device)
    out = iop.to_tensorflow(arr, SupportedDevices(device))

    assert isinstance(out, tf.Tensor), f"Expected tf.Tensor, got {type(out)}"
    assert ("gpu" if device == "gpu" and TF_GPU_AVAILABLE else "cpu") in out.device.lower(), (
        f"Expected device {device}, got {out.device}"
    )
    equals = create_array(data, "tensorflow", device)
    assert_arrays_equal(out, equals, "tensorflow")
