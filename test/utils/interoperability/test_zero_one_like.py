import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from .test_core import assert_arrays_equal, assert_same_type, assert_shapes_equal, create_array

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
    "framework,device",
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
@pytest.mark.parametrize(
    "shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_zeros_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test zeros_like function for all frameworks and devices."""
    data = [1, 2, 3, 4, 5, 6]
    arr = create_array(data, framework, device)
    arr = iop.reshape(arr, shape)
    zeros = iop.zeros_like(arr)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)
    expected = np.zeros_like(np_arr)

    assert_shapes_equal(zeros, expected, framework)
    assert_arrays_equal(zeros, expected, framework)
    assert_same_type(zeros, framework)


@pytest.mark.parametrize(
    "framework,device",
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
@pytest.mark.parametrize(
    "shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_ones_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test ones_like function for all frameworks and devices."""
    data = [1, 2, 3, 4, 5, 6]
    arr = create_array(data, framework, device)
    arr = iop.reshape(arr, shape)
    ones = iop.ones_like(arr)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)
    expected = np.ones_like(np_arr)

    assert_shapes_equal(ones, expected, framework)
    assert_arrays_equal(ones, expected, framework)
    assert_same_type(ones, framework)


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
@pytest.mark.parametrize(
    "shape",
    [(2, 3), (5,)],
)
def test_zeros_frameworks(framework: str, device: str, shape: tuple[int, ...]) -> None:
    """Test zeros function for all frameworks and devices."""
    zeros_arr = iop.zeros(framework=SupportedFrameworks(framework), shape=shape, device=SupportedDevices(device))
    expected = np.zeros(shape)

    assert_shapes_equal(zeros_arr, expected, framework)
    assert_arrays_equal(zeros_arr, expected, framework)
    assert_same_type(zeros_arr, framework)
