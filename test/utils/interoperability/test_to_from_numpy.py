import numpy as np
import pytest
from numpy.testing import assert_array_equal as np_assert_equal

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


def test_numpy_passthrough():
    arr = np.array([1, 2, 3], dtype=np.int32)
    out = iop.to_numpy(arr)
    # Should return the same numpy array object
    assert out is arr
    np_assert_equal(out, np.array([1, 2, 3], dtype=np.int32))


def test_scalars_and_none():
    # None becomes a 0-d object array containing None
    # None should not be an input to to_numpy but we test it anyway
    out = iop.to_numpy(None)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.tolist() is None

    # Scalars become 0-d numpy arrays
    out = iop.to_numpy(5)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.item() == 5

    out = iop.to_numpy(3.14)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.item() == pytest.approx(3.14)


def test_list_of_scalars_conversion():
    out = iop.to_numpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    np_assert_equal(out, np.array([1, 2, 3]))

    out = iop.to_numpy([1.5, 2.43, 3.0])
    assert isinstance(out, np.ndarray)
    np_assert_equal(out, np.array([1.5, 2.43, 3.0]))


def test_dictionary_conversion():
    # Current implementation wraps unknown objects into an object-dtype 0-d array
    # Nested dictionaries should not be an input to to_numpy but we test it anyway
    nested = {
        "a": [np.array([1, 2]), 3],
        "b": (np.array([4]), {"c": np.array([5])}),
    }
    out = iop.to_numpy(nested)
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.dtype == object
    assert out.item() == nested


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
def test_to_numpy_frameworks(framework: str, device: str):
    """Test to_numpy conversion for all frameworks and devices."""
    data = [1, 2, 3]
    arr = create_array(data, framework, device)
    out = iop.to_numpy(arr)

    assert isinstance(out, np.ndarray)
    np_assert_equal(out, np.array(data, dtype=np.float32))


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
def test_numpy_to_frameworks_like(framework, device: str):
    """Test from_numpy conversion for all frameworks and devices."""

    like = create_array([1, 2], framework, device)

    data = [1, 2, 3]
    np_arr = np.array(data, dtype=np.int16)
    out = iop.to_array_like(np_arr, like)

    assert isinstance(out, type(like.value)), f"Expected type {type(like.value)}, got {type(out)}"
    assert out.dtype == like.value.dtype, f"Expected dtype {like.value.dtype}, got {out.dtype}"
