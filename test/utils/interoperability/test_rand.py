import numpy as np
import pytest

import decent_bench.utils.interoperability as iop
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from .test_core import assert_same_type, assert_shapes_equal, create_array

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
def test_rand_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test rand_like function for all frameworks and devices."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arr = create_array(data, framework, device)
    arr = iop.reshape(arr, shape)
    rand_arr = iop.rand_like(arr)

    # Compute expected shape using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)

    assert_shapes_equal(rand_arr, np_arr, framework)
    assert_same_type(rand_arr, framework)


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
def test_randn_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test randn_like function for all frameworks and devices."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arr = create_array(data, framework, device)
    arr = iop.reshape(arr, shape)
    rand_arr = iop.randn_like(arr)

    # Compute expected shape using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)

    assert_shapes_equal(rand_arr, np_arr, framework)
    assert_same_type(rand_arr, framework)


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
def test_randn_frameworks(framework: str, device: str, shape: tuple[int, ...]) -> None:
    """Test randn function for all frameworks and devices."""
    randn_arr = iop.randn(framework=SupportedFrameworks(framework), shape=shape, device=SupportedDevices(device))
    expected = np.random.default_rng().normal(size=shape)

    assert_shapes_equal(randn_arr, expected, framework)
    assert_same_type(randn_arr, framework)


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
def test_rand_frameworks(framework: str, device: str, shape: tuple[int, ...]) -> None:
    """Test rand function for all frameworks and devices."""
    rand_arr = iop.rand(framework=SupportedFrameworks(framework), shape=shape, device=SupportedDevices(device))
    expected = np.random.default_rng().random(size=shape)

    assert_shapes_equal(rand_arr, expected, framework)
    assert_same_type(rand_arr, framework)
