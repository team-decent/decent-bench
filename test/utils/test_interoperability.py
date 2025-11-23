import numpy as np
import pytest
from numpy.testing import assert_array_equal as np_assert_equal
from typing import Any

from decent_bench.utils.interoperability import Interoperability

try:
    import torch
    from torch.testing import assert_close as torch_assert_close

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


# ============================================================================
# Helpers
# ============================================================================


def create_array(data: list, framework: str, device: str = "cpu"):
    """Factory function to create arrays in different frameworks and devices.

    Args:
        data: Python list/nested list to convert
        framework: One of 'numpy', 'torch', 'tensorflow', 'jax', 'list', 'tuple'
        device: 'cpu' or 'gpu'
    """
    if data is None:
        raise ValueError("Data cannot be None")

    if framework == "numpy":
        return np.array(data, dtype=np.float32)
    elif framework == "torch":
        array1 = torch.tensor(data, dtype=torch.float32)
        if device == "gpu" and TORCH_CUDA_AVAILABLE:
            array1 = array1.to("cuda")
        return array1
    elif framework == "tensorflow":
        device_str = "/GPU:0" if device == "gpu" and TF_GPU_AVAILABLE else "/CPU:0"
        with tf.device(device_str):
            array2: tf.Tensor = tf.constant(data, dtype=tf.float32)  # type: ignore
            return array2
    elif framework == "jax":
        array3 = jnp.array(data, dtype=jnp.float32)
        if device == "gpu" and JAX_GPU_AVAILABLE:
            gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
            if gpu_devices:
                array3 = jax.device_put(array3, device=gpu_devices[0])
        return array3
    elif framework == "list":
        return data
    elif framework == "tuple":
        return tuple(data)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def assert_arrays_equal(result, expected, framework: str):
    """Framework-agnostic assertion for array equality."""
    result_np = Interoperability.to_numpy(result)
    expected_np = Interoperability.to_numpy(expected)

    if framework == "torch" and isinstance(result, torch.Tensor):
        # For torch, use torch_assert_close if result is still a tensor
        expected_torch = torch.from_numpy(expected_np)
        if result.is_cuda:
            expected_torch = expected_torch.to("cuda")
        torch_assert_close(result, expected_torch)
    else:
        np_assert_equal(result_np, expected_np)


def assert_shapes_equal(result, expected, framework):
    if framework in ["list", "tuple"]:
        assert np.shape(result) == np.shape(expected)
    else:
        assert result.shape == expected.shape  # type: ignore


def assert_same_type(result: Any, framework: str):
    """Assert that the result is of the expected type based on the framework."""
    if framework == "numpy":
        assert "numpy" in str(type(result)), f"Expected numpy.ndarray, got {type(result)}"
    elif framework == "torch":
        assert "torch" in str(type(result)), f"Expected torch.Tensor, got {type(result)}"
    elif framework == "tensorflow":
        assert "tensorflow" in str(type(result)), f"Expected tf.Tensor, got {type(result)}"
    elif framework == "jax":
        assert "jax" in str(type(result)), f"Expected jnp.ndarray, got {type(result)}"
    elif framework == "list":
        assert isinstance(result, list) or isinstance(result, (int, float, complex)), (
            f"Expected list, got {type(result)}"
        )
    elif framework == "tuple":
        assert isinstance(result, tuple) or isinstance(result, (int, float, complex)), (
            f"Expected tuple, got {type(result)}"
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")


# ============================================================================
# Tests for Interoperability.to_numpy and from_numpy
# ============================================================================


def test_numpy_passthrough():
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = Interoperability.to_numpy(arr)
    # Should return the same numpy array object
    assert out is arr
    np_assert_equal(out, np.array([1, 2, 3], dtype=np.int32))


def test_scalars_and_none():
    # None becomes a 0-d object array containing None
    # None should not be an input to to_numpy but we test it anyway
    out = Interoperability.to_numpy(None)  # type: ignore
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.tolist() is None

    # Scalars become 0-d numpy arrays
    out = Interoperability.to_numpy(5)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.item() == 5

    out = Interoperability.to_numpy(3.14)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.item() == pytest.approx(3.14)


def test_list_of_scalars_conversion():
    out = Interoperability.to_numpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    np_assert_equal(out, np.array([1, 2, 3]))

    out = Interoperability.to_numpy([1.5, 2.43, 3.0])
    assert isinstance(out, np.ndarray)
    np_assert_equal(out, np.array([1.5, 2.43, 3.0]))


def test_dictionary_conversion():
    # Current implementation wraps unknown objects into an object-dtype 0-d array
    # Nested dictionaries should not be an input to to_numpy but we test it anyway
    nested = {
        "a": [np.array([1, 2]), 3],
        "b": (np.array([4]), {"c": np.array([5])}),
    }
    out = Interoperability.to_numpy(nested)  # type: ignore
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.dtype == object
    assert out.item() == nested


@pytest.mark.parametrize(
    "framework,device",
    [
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
def test_to_numpy_frameworks(framework: str, device: str):
    """Test to_numpy conversion for all frameworks and devices."""
    data = [1, 2, 3]
    arr = create_array(data, framework, device)
    out = Interoperability.to_numpy(arr)

    assert isinstance(out, np.ndarray)
    np_assert_equal(out, np.array(data, dtype=np.float32))


@pytest.mark.parametrize(
    "framework,device",
    [
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
def test_from_numpy_frameworks(framework, device: str):
    """Test from_numpy conversion for all frameworks and devices."""

    like = create_array([1, 2], framework, device)

    data = [1, 2, 3]
    np_arr = np.array(data, dtype=np.float32)
    out = Interoperability.from_numpy_like(np_arr, like)

    assert isinstance(out, type(like))


# ============================================================================
# Tests for reduction operations (sum, mean, min, max)
# ============================================================================


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim,keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_sum_all_combinations(framework: str, device: str, dim, keepdims):
    """Test sum with all parameter combinations across frameworks."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    arr = create_array(data, framework, device)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.sum(np_arr, axis=dim, keepdims=keepdims)

    result = Interoperability.sum(arr, dims=dim, keepdims=keepdims)
    assert_arrays_equal(result, expected, framework)
    assert_same_type(result, framework)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim,keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_mean_all_combinations(framework: str, device: str, dim, keepdims):
    """Test mean with all parameter combinations across frameworks."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    arr = create_array(data, framework, device)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.mean(np_arr, axis=dim, keepdims=keepdims)

    result = Interoperability.mean(arr, dims=dim, keepdims=keepdims)
    assert_arrays_equal(result, expected, framework)
    assert_same_type(result, framework)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim,keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_min_all_combinations(framework: str, device: str, dim, keepdims):
    """Test min with all parameter combinations across frameworks."""
    data = [[3.0, 1.0], [2.0, 4.0]]
    arr = create_array(data, framework, device)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.min(np_arr, axis=dim, keepdims=keepdims)

    result = Interoperability.min(arr, dims=dim, keepdims=keepdims)
    assert_arrays_equal(result, expected, framework)
    assert_same_type(result, framework)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim,keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_max_all_combinations(framework: str, device: str, dim, keepdims):
    """Test max with all parameter combinations across frameworks."""
    data = [[3.0, 1.0], [2.0, 4.0]]
    arr = create_array(data, framework, device)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.max(np_arr, axis=dim, keepdims=keepdims)

    result = Interoperability.max(arr, dims=dim, keepdims=keepdims)
    assert_arrays_equal(result, expected, framework)
    assert_same_type(result, framework)


# ============================================================================
# Tests for argmax and argmin
# ============================================================================


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim,keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ],
)
def test_argmax_all_combinations(framework: str, device: str, dim, keepdims):
    """Test argmax with all parameter combinations across frameworks."""
    data = [[1.0, 3.0], [4.0, 2.0]]
    arr = create_array(data, framework, device)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.argmax(np_arr, axis=dim, keepdims=keepdims)

    result = Interoperability.argmax(arr, dim=dim, keepdims=keepdims)
    assert_arrays_equal(result, expected, framework)
    assert_same_type(result, framework)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim,keepdims",
    [
        (None, False),
        (None, True),
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ],
)
def test_argmin_all_combinations(framework: str, device: str, dim, keepdims):
    """Test argmin with all parameter combinations across frameworks."""
    data = [[3.0, 1.0], [2.0, 4.0]]
    arr = create_array(data, framework, device)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.argmin(np_arr, axis=dim, keepdims=keepdims)

    result = Interoperability.argmin(arr, dim=dim, keepdims=keepdims)
    assert_arrays_equal(result, expected, framework)
    assert_same_type(result, framework)


# ============================================================================
# Tests for copy
# ============================================================================


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
def test_copy_frameworks(framework: str, device: str):
    """Test copy function for all frameworks and devices."""
    data = [[1, 2, 3], [4, 5, 6]]
    arr = create_array(data, framework, device)
    arr_copy = Interoperability.copy(arr)

    # Ensure the copied array is equal to the original
    assert_arrays_equal(arr_copy, arr, framework)

    # Modify the copy and ensure the original is unchanged
    if framework == "numpy":
        arr_copy[0, 0] = 999  # type: ignore
    elif framework == "torch":
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


# ============================================================================
# Tests for stack
# ============================================================================


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "dim",
    [0, 1],
)
def test_stack_frameworks(framework: str, device: str, dim: int):
    """Test stack function for all frameworks and devices."""
    data1 = [[1, 2, 3], [4, 5, 6]]
    data2 = [[7, 8, 9], [10, 11, 12]]
    arr1 = create_array(data1, framework, device)
    arr2 = create_array(data2, framework, device)
    stacked = Interoperability.stack([arr1, arr2], dim=dim)

    # Compute expected result using numpy
    np_arr1 = create_array(data1, "numpy")
    np_arr2 = create_array(data2, "numpy")
    expected = np.stack([np_arr1, np_arr2], axis=dim)

    assert_arrays_equal(stacked, expected, framework)
    assert_same_type(stacked, framework)


# ============================================================================
# Tests for reshape
# ============================================================================


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
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
    reshaped = Interoperability.reshape(arr, new_shape)

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
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
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
    reshaped = Interoperability.reshape(arr, new_shape)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    expected = np.reshape(np_arr, new_shape)

    assert_shapes_equal(reshaped, expected, framework)
    assert_arrays_equal(reshaped, expected, framework)
    assert_same_type(reshaped, framework)


# ============================================================================
# Tests for zeros_like, ones_like, rand_like and eye_like
# ============================================================================


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
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
    arr = Interoperability.reshape(arr, shape)
    zeros = Interoperability.zeros_like(arr)

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
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
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
    arr = Interoperability.reshape(arr, shape)
    ones = Interoperability.ones_like(arr)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)
    expected = np.ones_like(np_arr)

    assert_shapes_equal(ones, expected, framework)
    assert_arrays_equal(ones, expected, framework)
    assert_same_type(ones, framework)


@pytest.mark.parametrize(
    "framework,device",
    [
        ("numpy", "cpu"),
        pytest.param(
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(3, 2), (2, 3), (6,), (-1,), (2, 1, 3), (1, 6)],
)
def test_rand_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test rand_like function for all frameworks and devices."""
    data = [1, 2, 3, 4, 5, 6]
    arr = create_array(data, framework, device)
    arr = Interoperability.reshape(arr, shape)
    rand_arr = Interoperability.rand_like(arr)

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
            "torch",
            "cpu",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
        pytest.param(
            "torch",
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
        ("list", "cpu"),
        ("tuple", "cpu"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(4, 4), (2, 8), (1, 16), (-1,), (1, 2, 8)],
)
def test_eye_like_frameworks(framework: str, device: str, shape: tuple[int, ...]):
    """Test eye_like function for all frameworks and devices."""
    data = list(range(16))
    arr = create_array(data, framework, device)
    arr = Interoperability.reshape(arr, shape)
    eye_arr = Interoperability.eye_like(arr)

    # Compute expected result using numpy
    np_arr = create_array(data, "numpy")
    np_arr = np.reshape(np_arr, shape)
    expected = (
        np.eye(*np_arr.shape[-2:], dtype=np_arr.dtype)
        if len(np_arr.shape) >= 2
        else np.eye(np_arr.shape[0], dtype=np_arr.dtype)  # type: ignore
    )

    assert_shapes_equal(eye_arr, expected, framework)
    assert_arrays_equal(eye_arr, expected, framework)
    assert_same_type(eye_arr, framework)
