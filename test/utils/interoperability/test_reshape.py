import numpy as np
import pytest

import decent_bench.utils.interoperability as iop

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
    "new_shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_reshape_matrix_frameworks(framework: str, device: str, new_shape: tuple[int, ...]):
    """Test reshape function for all frameworks and devices."""
    data = [[1, 2, 3], [4, 5, 6]]
    arr = create_array(data, framework, device)
    reshaped = iop.reshape(arr, new_shape)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.reshape(np_arr, new_shape)

    assert_shapes_equal(reshaped, expected, framework)
    assert_arrays_equal(reshaped, expected, framework)
    assert_same_type(reshaped, framework)


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
    "new_shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_reshape_vector_frameworks(framework: str, device: str, new_shape: tuple[int, ...]):
    """Test reshape function for all frameworks and devices."""
    data = [1, 2, 3, 4, 5, 6]
    arr = create_array(data, framework, device)
    reshaped = iop.reshape(arr, new_shape)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.reshape(np_arr, new_shape)

    assert_shapes_equal(reshaped, expected, framework)
    assert_arrays_equal(reshaped, expected, framework)
    assert_same_type(reshaped, framework)
