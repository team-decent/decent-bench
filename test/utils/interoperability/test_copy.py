import pytest

import decent_bench.utils.interoperability as iop

from .test_core import assert_arrays_equal, assert_same_type, create_array

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
def test_copy_frameworks(framework: str, device: str):
    """Test copy function for all frameworks and devices."""
    data = [[1, 2, 3], [4, 5, 6]]
    arr = create_array(data, framework, device)
    arr_copy = iop.copy(arr)

    # Ensure the copied array is equal to the original
    assert_arrays_equal(arr_copy, arr, framework)

    # Modify the copy and ensure the original is unchanged
    if framework == "numpy":
        arr_copy[0, 0] = 999  # type: ignore
    elif framework == "pytorch":
        arr_copy[0, 0] = 999  # type: ignore
    elif framework == "tensorflow":
        arr_copy = tf.tensor_scatter_nd_update(arr_copy, [[0, 0]], [999])  # type: ignore
    elif framework == "jax":
        arr_copy = arr_copy.at[0, 0].set(999)  # type: ignore
    else:
        # For list and tuple, works on tuples because inner lists are mutable
        arr_copy[0][0] = 999  # type: ignore

    # Original should remain unchanged
    expected_original = create_array(data, framework, device)
    assert_arrays_equal(arr, expected_original, framework)
    assert_same_type(arr_copy, framework)

    # Assert not equal after modification
    with pytest.raises(AssertionError):
        assert_arrays_equal(arr_copy, arr, framework)
